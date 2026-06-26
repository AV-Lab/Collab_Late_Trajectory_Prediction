#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:52 2024
@author: nadya

Clean RNN/LSTM predictor without uncertainty prediction.

This model predicts only future velocity means.
Any covariance/calibration should be computed outside this model,
for example in a calibration wrapper using prediction errors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.distance_metrics import calculate_ade, calculate_fde


class RNNPredictor:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~  sub-blocks  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    class Encoder(nn.Module):
        def __init__(self, in_dim, h_dim, n_layers):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)

        def forward(self, x):
            _, (h, c) = self.lstm(x)
            return h, c

    class Decoder(nn.Module):
        def __init__(self, in_dim, h_dim, out_dim, n_layers):
            """
            in_dim: decoder input size.
                    We feed back the predicted velocity, so usually in_dim = out_dim.

            out_dim: output velocity dimension.
            """
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)
            self.mean_head = nn.Linear(h_dim, out_dim)

        def forward(self, x, h, c):
            y, (h, c) = self.lstm(x, (h, c))
            mu = self.mean_head(y)      # [B, 1, out_dim]
            return mu, h, c

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc, self.dec = enc, dec

        def forward(self, x, horizon):
            """
            x: [B, T-1, enc_in_dim]

            returns:
                mu_seq: [B, horizon, out_dim]
            """
            B, _, _ = x.size()
            h, c = self.enc(x)

            dec_in = torch.zeros(
                B, 1, self.dec.lstm.input_size,
                device=x.device
            )

            mu_outs = []

            for _ in range(horizon):
                mu, h, c = self.dec(dec_in, h, c)
                mu_outs.append(mu)

                # Feed previous predicted velocity as next decoder input.
                dec_in = mu.detach()

            return torch.cat(mu_outs, dim=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~  constructor  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def __init__(self, cfg: dict):
        """
        cfg must contain at least:
            hidden_size
            num_layers
            input_size
            output_size
            observation_length
            prediction_horizon
            num_epochs
            learning_rate
            patience
            device
            trained_fps
            checkpoint or None

        Optional:
            epsilon
            adv_weight

        Notes:
            - input_size refers to the velocity-feature dimension used as encoder input.
            - output_size refers to the predicted velocity dimension.
            - The model itself does not predict covariance.
        """

        self.params = [
            "prediction_horizon",
            "num_epochs",
            "learning_rate",
            "patience",
            "hidden_size",
            "num_layers",
            "input_size",
            "output_size",
            "observation_length",
            "trained_fps",
        ]

        self.device = torch.device(cfg["device"])
        self.model_trained = False

        # Optional adversarial training hyperparameters.
        # Kept for consistency with the NLL version.
        self.epsilon = float(cfg.get("epsilon", 0.0))
        self.adv_weight = float(cfg.get("adv_weight", 0.0))

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

        else:
            for k in self.params:
                setattr(self, k, cfg[k])

        self.pos_size = 2
        self.pos_slice = slice(0, self.pos_size)

        # ---------- model ---------------------------------------------- #
        enc = self.Encoder(
            self.input_size,
            self.hidden_size,
            self.num_layers
        )

        dec = self.Decoder(
            self.output_size,
            self.hidden_size,
            self.output_size,
            self.num_layers
        )

        self.model = self.Seq2Seq(enc, dec).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # ---------- optional checkpoint -------------------------------- #
        if ckpt_path:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # --------------------------- losses ----------------------------------- #
    def prediction_loss(self, y_true, y_pred):
        """
        Standard MSE loss on predicted velocities.

        y_true: [B, H, D]
        y_pred: [B, H, D]
        """
        return F.mse_loss(y_pred, y_true)

    @staticmethod
    def _vel_to_pos(last_pos, vel_seq, pos_size):
        """
        Integrate predicted velocities into absolute positions.

        last_pos: [B, 2]
        vel_seq:  [B, H, D]
        """
        vel_seq = vel_seq[:, :, :pos_size]

        B, T, D = vel_seq.shape
        out = torch.zeros(B, T, D, device=vel_seq.device)

        out[:, 0] = last_pos + vel_seq[:, 0]

        for t in range(1, T):
            out[:, t] = out[:, t - 1] + vel_seq[:, t]

        return out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~  train  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def train(self, train_loader, valid_loader=None, save_path=None):
        print("Train batches:", len(train_loader))

        best_val = float("inf")
        patience_ctr = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()

            ep_loss = 0.0
            ep_ade = 0.0
            ep_fde = 0.0

            bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False
            )

            for batch in bar:
                # support (cat, obs, tgt) or (obs, tgt)
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
                    if cat is not None:
                        cat = cat.unsqueeze(0)
                    obs = obs.unsqueeze(0)
                    tgt = tgt.unsqueeze(0)

                # Encoder input: velocity features.
                enc_in_base = obs[
                    :, :, self.pos_size:self.pos_size + self.input_size
                ]

                # Target: future velocity.
                tgt_v = tgt[
                    :, :, self.pos_size:self.pos_size + self.output_size
                ]

                # ---------------- clean training ---------------- #
                if self.epsilon <= 0.0 or self.adv_weight <= 0.0:
                    self.optimizer.zero_grad(set_to_none=True)

                    mu = self.model(enc_in_base, tgt_v.size(1))
                    loss = self.prediction_loss(tgt_v, mu)

                    loss.backward()
                    self.optimizer.step()

                    mu_clean = mu

                # ---------------- optional FGSM training ---------------- #
                else:
                    enc_in_for_grad = (
                        enc_in_base
                        .clone()
                        .detach()
                        .requires_grad_(True)
                    )

                    mu_for_grad = self.model(enc_in_for_grad, tgt_v.size(1))
                    loss_for_grad = self.prediction_loss(tgt_v, mu_for_grad)

                    grad_enc_in = torch.autograd.grad(
                        loss_for_grad,
                        enc_in_for_grad,
                        retain_graph=False
                    )[0]

                    x_adv = (
                        enc_in_for_grad
                        + self.epsilon * grad_enc_in.sign()
                    ).detach()

                    self.optimizer.zero_grad(set_to_none=True)

                    mu_clean = self.model(enc_in_base, tgt_v.size(1))
                    loss_clean = self.prediction_loss(tgt_v, mu_clean)

                    mu_adv = self.model(x_adv, tgt_v.size(1))
                    loss_adv = self.prediction_loss(tgt_v, mu_adv)

                    loss = loss_clean + self.adv_weight * loss_adv

                    loss.backward()
                    self.optimizer.step()

                ep_loss += loss.item()

                # ADE/FDE from position mean.
                with torch.no_grad():
                    last_pos = obs[:, -1, self.pos_slice]

                    pred_pos = self._vel_to_pos(
                        last_pos,
                        mu_clean,
                        self.pos_size
                    )

                    tgt_pos = tgt[:, :, self.pos_slice]

                    ade_b = calculate_ade(pred_pos, tgt_pos)
                    fde_b = calculate_fde(pred_pos, tgt_pos)

                    ep_ade += ade_b
                    ep_fde += fde_b

                bar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    ADE=f"{ade_b:.5f}",
                    FDE=f"{fde_b:.5f}"
                )

            print(
                f"\nEpoch {epoch}: "
                f"Loss {ep_loss / len(train_loader):.6f}  "
                f"ADE {ep_ade / len(train_loader):.5f}  "
                f"FDE {ep_fde / len(train_loader):.5f}"
            )

            # -------------- validation & early stopping ---------------- #
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~  validate  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
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

                enc_in = obs[
                    :, :, self.pos_size:self.pos_size + self.input_size
                ]

                tgt_v = tgt[
                    :, :, self.pos_size:self.pos_size + self.output_size
                ]

                mu = self.model(enc_in, tgt_v.size(1))
                loss = self.prediction_loss(tgt_v, mu)

                loss_sum += loss.item()

        val_loss = loss_sum / len(loader)

        print(f"Validation loss (MSE): {val_loss:.6f}")

        return val_loss

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  evaluate  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def evaluate(self, loader):
        self.model.eval()

        ade = 0.0
        fde = 0.0

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
                    if cat is not None:
                        cat = cat.unsqueeze(0)
                    obs = obs.unsqueeze(0)
                    tgt = tgt.unsqueeze(0)

                enc_in = obs[
                    :, :, self.pos_size:self.pos_size + self.input_size
                ]

                mu = self.model(enc_in, tgt.size(1))

                last_pos = obs[:, -1, self.pos_slice]

                pred_pos = self._vel_to_pos(
                    last_pos,
                    mu,
                    self.pos_size
                )

                tgt_pos = tgt[:, :, self.pos_slice]

                ade += calculate_ade(pred_pos, tgt_pos)
                fde += calculate_fde(pred_pos, tgt_pos)

        ade /= len(loader)
        fde /= len(loader)

        print(f"Test ADE {ade:.5f}  FDE {fde:.5f}")

        return ade, fde

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~  predict  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def predict(self, trajs, prediction_horizon):
        """
        trajs: list of np.ndarray with shape [T_obs, input_size]

        Returns:
            predictions: list[np.ndarray]
                Each array has shape [H, pos_dim].

        Notes:
            - This model returns only predicted positions.
            - Covariance should be added outside this model by a wrapper.
        """
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")

        self.model.eval()

        vel_batch = []
        last_pos_batch = []

        for tr in trajs:
            t = torch.tensor(
                tr,
                dtype=torch.float32,
                device=self.device
            )

            last_pos_batch.append(t[-1, self.pos_slice])

            # Build velocity history from observed positions/features.
            vel = t[1:] - t[:-1]

            if vel.size(0) > self.observation_length - 1:
                vel = vel[-(self.observation_length - 1):]
            else:
                pad_len = self.observation_length - 1 - vel.size(0)

                if pad_len > 0:
                    zpad = torch.zeros(
                        pad_len,
                        vel.size(1),
                        device=self.device
                    )
                    vel = torch.cat([zpad, vel], dim=0)

            vel_batch.append(vel)

        vel_batch = torch.stack(vel_batch)          # [B, T-1, input_size]
        enc_in = vel_batch[:, :, :self.input_size]

        with torch.no_grad():
            mu_v = self.model(enc_in, prediction_horizon)  # [B, H, out_dim]

            pos_means = []

            for i in range(mu_v.size(0)):
                last_pos = last_pos_batch[i]
                mu_v_i = mu_v[i:i + 1, :, :self.pos_size]

                pos_mean_i = self._vel_to_pos(
                    last_pos,
                    mu_v_i,
                    self.pos_size
                )[0]

                pos_means.append(pos_mean_i)

            pos_means = torch.stack(pos_means, dim=0)

        pred_np = pos_means.cpu().numpy()

        predictions = [pred_np[i] for i in range(pred_np.shape[0])]

        return predictions

    # ~~~~~~~~~~~~~~~~~~~~~~~~  checkpoint  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "adv_weight": self.adv_weight,
        }

        for k in self.params:
            ckpt[k] = getattr(self, k)

        torch.save(ckpt, path)

        print("Saved best model to", path)