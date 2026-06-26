#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.distance_metrics import calculate_ade, calculate_fde


class BivariateNLLLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred:   [B, H, 5] -> [mu_x, mu_y, log_sx, log_sy, rho_raw]
        target: [B, H, 2] -> [x, y]
        """
        mu_x = pred[..., 0]
        mu_y = pred[..., 1]
        log_sx = pred[..., 2]
        log_sy = pred[..., 3]
        rho = torch.tanh(pred[..., 4])

        sx = torch.exp(log_sx).clamp_min(self.eps)
        sy = torch.exp(log_sy).clamp_min(self.eps)

        x = target[..., 0]
        y = target[..., 1]

        norm_x = (x - mu_x) / sx
        norm_y = (y - mu_y) / sy

        z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
        rho_term = (1 - rho**2).clamp_min(self.eps)

        loss = (
            log_sx +
            log_sy +
            0.5 * torch.log(rho_term) +
            z / (2 * rho_term) +
            math.log(2 * math.pi)
        )

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class RNNPredictorBivariateNLL:

    class Encoder(nn.Module):
        def __init__(self, in_dim, h_dim, n_layers):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)

        def forward(self, x):
            _, (h, c) = self.lstm(x)
            return h, c

    class Decoder(nn.Module):
        def __init__(self, in_dim, h_dim, n_layers):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)
            self.out = nn.Linear(h_dim, 5)  # [mu_x, mu_y, log_sx, log_sy, rho_raw]

        def forward(self, x, h, c):
            y, (h, c) = self.lstm(x, (h, c))
            pred = self.out(y)  # [B, 1, 5]
            return pred, h, c

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc = enc
            self.dec = dec

        def forward(self, x, horizon, teacher_forcing_targets=None, teacher_forcing_ratio=0.0):
            """
            x: [B, T_obs, in_dim]
            teacher_forcing_targets: [B, H, 2] absolute future positions
            returns: [B, H, 5]
            """
            B = x.size(0)
            h, c = self.enc(x)

            # decoder input is previous position
            dec_in = x[:, -1:, :2]  # [B, 1, 2]

            preds = []
            for t in range(horizon):
                pred, h, c = self.dec(dec_in, h, c)
                preds.append(pred)

                mu = pred[..., :2]  # next decoder input uses predicted mean position

                if (
                    teacher_forcing_targets is not None
                    and torch.rand(1).item() < teacher_forcing_ratio
                ):
                    dec_in = teacher_forcing_targets[:, t:t+1, :]
                else:
                    dec_in = mu.detach()

            return torch.cat(preds, dim=1)

    def __init__(self, cfg: dict):
        self.params = [
            "prediction_horizon", "num_epochs", "learning_rate", "patience",
            "hidden_size", "num_layers", "input_size", "observation_length",
            "trained_fps"
        ]

        self.device = torch.device(cfg["device"])
        self.model_trained = False

        self.epsilon = float(cfg.get("epsilon", 0.1))
        self.adv_weight = float(cfg.get("adv_weight", 1.0))
        self.teacher_forcing_ratio = float(cfg.get("teacher_forcing_ratio", 0.0))

        ckpt_path = cfg.get("checkpoint")
        if ckpt_path:
            print("Loading checkpoint:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model_trained = True
            for k in self.params:
                setattr(self, k, ckpt[k])
            self.trained_fps = int(self.trained_fps)
            self.observation_length = int(self.observation_length)
            self.epsilon = ckpt.get("epsilon", self.epsilon)
            self.adv_weight = ckpt.get("adv_weight", self.adv_weight)
            self.teacher_forcing_ratio = ckpt.get("teacher_forcing_ratio", self.teacher_forcing_ratio)
        else:
            for k in self.params:
                setattr(self, k, cfg[k])

        self.pos_size = 2
        self.log_s_min = -3.0
        self.log_s_max = 5.0

        enc = self.Encoder(self.input_size, self.hidden_size, self.num_layers)
        dec = self.Decoder(self.pos_size, self.hidden_size, self.num_layers)
        self.model = self.Seq2Seq(enc, dec).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = BivariateNLLLoss()

        if ckpt_path:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def _prepare_inputs(self, obs, tgt):
        """
        obs expected shape: [B, T_obs, D]
        tgt expected shape: [B, H, D] or [B, H, 2]

        Encoder input:
            by default uses first input_size dims from obs

        Target for bivariate loss:
            absolute future positions [x, y]
        """
        enc_in = obs[:, :, :self.input_size]
        tgt_pos = tgt[:, :, :2]
        return enc_in, tgt_pos

    def _clamp_pred_params(self, pred):
        pred = pred.clone()
        pred[..., 2] = torch.clamp(pred[..., 2], self.log_s_min, self.log_s_max)
        pred[..., 3] = torch.clamp(pred[..., 3], self.log_s_min, self.log_s_max)
        return pred

    def _fgsm_perturb(self, enc_in, tgt_pos):
        x = enc_in.clone().detach().requires_grad_(True)
        pred = self.model(
            x,
            horizon=tgt_pos.size(1),
            teacher_forcing_targets=tgt_pos,
            teacher_forcing_ratio=0.0,
        )
        pred = self._clamp_pred_params(pred)
        loss = self.criterion(pred, tgt_pos)
        grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        x_adv = (x + self.epsilon * grad.sign()).detach()
        return x_adv

    def train(self, train_loader, valid_loader=None, save_path=None):
        print("Train batches:", len(train_loader))

        best_val = float("inf")
        patience_ctr = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            ep_loss = ep_ade = ep_fde = 0.0

            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)

            for batch in bar:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    cat, obs, tgt = batch
                else:
                    cat = None
                    obs, tgt = batch

                obs = obs.to(self.device)
                tgt = tgt.to(self.device)
                if cat is not None:
                    cat = cat.to(self.device)

                if obs.dim() == 2:
                    obs = obs.unsqueeze(0)
                    tgt = tgt.unsqueeze(0)
                    if cat is not None:
                        cat = cat.unsqueeze(0)

                enc_in, tgt_pos = self._prepare_inputs(obs, tgt)

                x_adv = self._fgsm_perturb(enc_in, tgt_pos)

                self.optimizer.zero_grad(set_to_none=True)

                pred_clean = self.model(
                    enc_in,
                    horizon=tgt_pos.size(1),
                    teacher_forcing_targets=tgt_pos,
                    teacher_forcing_ratio=self.teacher_forcing_ratio,
                )
                pred_clean = self._clamp_pred_params(pred_clean)
                loss_clean = self.criterion(pred_clean, tgt_pos)

                pred_adv = self.model(
                    x_adv,
                    horizon=tgt_pos.size(1),
                    teacher_forcing_targets=tgt_pos,
                    teacher_forcing_ratio=self.teacher_forcing_ratio,
                )
                pred_adv = self._clamp_pred_params(pred_adv)
                loss_adv = self.criterion(pred_adv, tgt_pos)

                loss = loss_clean + self.adv_weight * loss_adv
                loss.backward()
                self.optimizer.step()

                ep_loss += loss.item()

                with torch.no_grad():
                    pred_pos = pred_clean[..., :2]
                    ade_b = calculate_ade(pred_pos, tgt_pos)
                    fde_b = calculate_fde(pred_pos, tgt_pos)
                    ep_ade += ade_b
                    ep_fde += fde_b

                bar.set_postfix(loss=f"{loss.item():.6f}", ADE=f"{ade_b:.5f}", FDE=f"{fde_b:.5f}")

            print(
                f"\nEpoch {epoch}: Loss {ep_loss/len(train_loader):.6f}  "
                f"ADE {ep_ade/len(train_loader):.5f}  FDE {ep_fde/len(train_loader):.5f}"
            )

            if valid_loader:
                val_loss = self.validate(valid_loader)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_ctr = 0
                    if save_path:
                        self.save_checkpoint(save_path)
                else:
                    patience_ctr += 1

                if patience_ctr >= self.patience:
                    print("Early stopping.")
                    break
            elif save_path and epoch == self.num_epochs:
                self.save_checkpoint(save_path)

        self.model_trained = True

    def validate(self, loader):
        self.model.eval()
        loss_sum = 0.0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    _, obs, tgt = batch
                else:
                    obs, tgt = batch

                obs = obs.to(self.device)
                tgt = tgt.to(self.device)

                if obs.dim() == 2:
                    obs = obs.unsqueeze(0)
                    tgt = tgt.unsqueeze(0)

                enc_in, tgt_pos = self._prepare_inputs(obs, tgt)

                pred = self.model(
                    enc_in,
                    horizon=tgt_pos.size(1),
                    teacher_forcing_targets=None,
                    teacher_forcing_ratio=0.0,
                )
                pred = self._clamp_pred_params(pred)
                loss_sum += self.criterion(pred, tgt_pos).item()

        val_loss = loss_sum / len(loader)
        print(f"Validation loss (Bivariate NLL): {val_loss:.6f}")
        return val_loss

    def evaluate(self, loader):
        self.model.eval()
        ade = fde = 0.0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    cat, obs, tgt = batch
                else:
                    cat = None
                    obs, tgt = batch

                obs = obs.to(self.device)
                tgt = tgt.to(self.device)
                if cat is not None:
                    cat = cat.to(self.device)

                if obs.dim() == 2:
                    obs = obs.unsqueeze(0)
                    tgt = tgt.unsqueeze(0)
                    if cat is not None:
                        cat = cat.unsqueeze(0)

                enc_in, tgt_pos = self._prepare_inputs(obs, tgt)

                pred = self.model(
                    enc_in,
                    horizon=tgt_pos.size(1),
                    teacher_forcing_targets=None,
                    teacher_forcing_ratio=0.0,
                )
                pred = self._clamp_pred_params(pred)

                pred_pos = pred[..., :2]

                ade += calculate_ade(pred_pos, tgt_pos)
                fde += calculate_fde(pred_pos, tgt_pos)

        ade /= len(loader)
        fde /= len(loader)
        print(f"Test ADE {ade:.5f}  FDE {fde:.5f}")
        return ade, fde

    def predict(self, trajs, prediction_horizon):
        """
        trajs: list[np.ndarray], each [T_obs, input_size] or compatible with first input_size used
        Returns:
            predictions: list[np.ndarray], each [H, 2]
            covariances: list[np.ndarray], each [H, 2, 2]
        """
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")

        self.model.eval()
        obs_batch = []

        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)

            if t.size(0) > self.observation_length:
                t = t[-self.observation_length:]
            else:
                pad_len = self.observation_length - t.size(0)
                if pad_len > 0:
                    zpad = torch.zeros(pad_len, t.size(1), device=self.device)
                    t = torch.cat([zpad, t], dim=0)

            obs_batch.append(t[:, :self.input_size])

        obs_batch = torch.stack(obs_batch, dim=0)

        with torch.no_grad():
            pred = self.model(
                obs_batch,
                horizon=prediction_horizon,
                teacher_forcing_targets=None,
                teacher_forcing_ratio=0.0,
            )
            pred = self._clamp_pred_params(pred)

            mu = pred[..., :2]
            log_sx = pred[..., 2]
            log_sy = pred[..., 3]
            rho = torch.tanh(pred[..., 4])

            sx = torch.exp(log_sx)
            sy = torch.exp(log_sy)

            cov_xx = sx ** 2
            cov_yy = sy ** 2
            cov_xy = rho * sx * sy

            B, H, _ = mu.shape
            covs = torch.zeros(B, H, 2, 2, device=self.device)
            covs[..., 0, 0] = cov_xx
            covs[..., 1, 1] = cov_yy
            covs[..., 0, 1] = cov_xy
            covs[..., 1, 0] = cov_xy

        pred_np = mu.cpu().numpy()
        cov_np = covs.cpu().numpy()

        predictions = [pred_np[i] for i in range(pred_np.shape[0])]
        covariances = [cov_np[i] for i in range(cov_np.shape[0])]

        return predictions, covariances

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "adv_weight": self.adv_weight,
            "teacher_forcing_ratio": self.teacher_forcing_ratio,
        }
        for k in self.params:
            ckpt[k] = getattr(self, k)
        torch.save(ckpt, path)
        print("Saved best model to", path)