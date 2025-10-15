#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:52 2024
@author: nadya

RNNPredictor with Gaussian NLL (mean + variance) and FGSM adversarial training (epsilon=0.1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import math
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.distance_metrics import calculate_ade, calculate_fde


class RNNPredictorNLL:

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
            in_dim: decoder input size (we feed back predicted velocity -> out_dim)
            out_dim: velocity dimension for the mean/variance heads
            """
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)
            # Two heads: mean and log-variance (per-dim, heteroscedastic, diagonal)
            self.mean_head   = nn.Linear(h_dim, out_dim)
            self.logvar_head = nn.Linear(h_dim, out_dim)

        def forward(self, x, h, c):
            y, (h, c) = self.lstm(x, (h, c))
            mu     = self.mean_head(y)          # [B, 1, out_dim]
            logvar = self.logvar_head(y)        # [B, 1, out_dim]
            return mu, logvar, h, c

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc, self.dec = enc, dec

        def forward(self, x, horizon):
            """
            x: [B, T-1, enc_in_dim]  →  returns (mu_seq, logvar_seq)
            mu_seq/logvar_seq: [B, horizon, out_dim]
            """
            B, _, _ = x.size()
            h, c = self.enc(x)
            dec_in = torch.zeros(B, 1, self.dec.lstm.input_size, device=x.device)
            mu_outs, lv_outs = [], []
            for _ in range(horizon):
                mu, logvar, h, c = self.dec(dec_in, h, c)
                mu_outs.append(mu)
                lv_outs.append(logvar)
                # feed previous mean prediction as next input
                dec_in = mu.detach()
            return torch.cat(mu_outs, dim=1), torch.cat(lv_outs, dim=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~  constructor  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def __init__(self, cfg: dict):
        """
        cfg must contain at least
            hidden_size, num_layers,
            input_size  (pos dim of observations: 2, 3 or 4),
            output_size (pos dim of targets:  2 or 3),
            observation_length, prediction_horizon,
            num_epochs, learning_rate, patience,
            device, normalize, checkpoint (or None)

        Optional:
            epsilon (float)      FGSM step size (default 0.1)
            adv_weight (float)   weight on adversarial loss (default 1.0)
        """
        # ---------- copy config fields ----------------------------------- #
        self.params = [
            "prediction_horizon", "num_epochs", "learning_rate", "patience",
            "hidden_size", "num_layers", "input_size", "output_size",
            "observation_length", "trained_fps"
        ]

        self.device = torch.device(cfg["device"])
        self.model_trained = False

        # adversarial training hyperparams
        self.epsilon   = float(cfg.get("epsilon", 0.1))  # default 0.1
        self.adv_weight = float(cfg.get("adv_weight", 1.0))

        ckpt_path = cfg.get("checkpoint")
        if ckpt_path:
            print("Loading checkpoint:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model_trained = True
            for k in self.params:
                setattr(self, k, ckpt[k])
            # optional: restore adv params if present
            self.epsilon = ckpt.get("epsilon", self.epsilon)
            self.adv_weight = ckpt.get("adv_weight", self.adv_weight)
        else:
            for k in self.params:
                setattr(self, k, cfg[k])

        self.pos_size = 2
        self.pos_slice = slice(0, self.pos_size)  # x,y
        self.var_floor = 5e-3
        self.logvar_min = -2.0
        self.logvar_max = 6.0
        

        # ---------- model ---------------------------------------------- #
        enc = self.Encoder(self.input_size, self.hidden_size, self.num_layers)
        # decoder input size equals out_dim because we feed back predicted velocity
        dec = self.Decoder(self.output_size, self.hidden_size, self.output_size, self.num_layers)
        self.model = self.Seq2Seq(enc, dec).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # ---------- optional checkpoint -------------------------------- #
        if ckpt_path:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    @staticmethod
    def gaussian_nll_loss(
        y_true, mu, raw_var_head,
        var_floor=1e-2,              # keep ≥ your successful floor
        var_ceiling=None,            # optionally cap very large vars (e.g., 1e2)
        add_const=True              # add 0.5*log(2π) if you want the full NLL
    ):
        """
    #   Gaussian NLL with variance parameterized via softplus:
    #      var = softplus(raw) + var_floor  (strictly positive)
    #    Inputs:
    #      y_true, mu, raw_var_head: [B, H, D]
    #    Returns:
    #      scalar loss (mean over all dims)
    #    """
        # strictly positive variance, numerically stable around large inputs
        var = F.softplus(raw_var_head, beta=1.0, threshold=20.0) + var_floor
        if var_ceiling is not None:
            var = torch.clamp(var, max=var_ceiling)
    
        logvar = torch.log(var)
        nll = 0.5 * (logvar + (y_true - mu) ** 2 / var)
    
        if add_const:
            nll = nll + 0.5 * math.log(2.0 * math.pi)
    
        return nll.mean()

    @staticmethod
    def _vel_to_pos(last_pos, vel_seq, pos_size):
        """Integrate velocities to absolute positions."""
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

        best_val = float('inf')
        patience_ctr = 0

        # obs format [x, y, vx, vy, vyaw] (you slice velocities for encoder input)

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            ep_loss = ep_ade = ep_fde = 0.0

            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)

            for batch in bar:
                # support (cat, obs, tgt) or (obs, tgt)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    cat, obs, tgt = batch
                else:
                    cat = None
                    obs, tgt = batch

                obs, tgt = obs.to(self.device), tgt.to(self.device)
                if cat is not None:
                    cat = cat.to(self.device)

                if obs.dim() == 2:  # batch_size==1 edge-case
                    if cat is not None:
                        cat = cat.unsqueeze(0)
                    obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                # Build encoder input (velocities)
                enc_in_base = obs[:, :, self.pos_size:self.pos_size + self.input_size]  # [B, T-1, enc_in_dim]
                tgt_v = tgt[:, :, self.pos_size:self.pos_size + self.output_size]      # [B, H, out_dim]

                # ------------------- FGSM adversarial training ------------------- #
                # 1) Get gradient of clean loss wrt input enc_in
                enc_in_for_grad = enc_in_base.clone().detach().requires_grad_(True)
                mu_clean_g, lv_clean_g = self.model(enc_in_for_grad, tgt_v.size(1))
                loss_clean_for_grad = self.gaussian_nll_loss(tgt_v, mu_clean_g, lv_clean_g)
                grad_enc_in = torch.autograd.grad(loss_clean_for_grad, enc_in_for_grad, retain_graph=False)[0]

                # 2) Craft adversarial input
                x_adv = (enc_in_for_grad + self.epsilon * grad_enc_in.sign()).detach()

                # 3) Compute clean + adversarial losses for parameter update
                self.optimizer.zero_grad(set_to_none=True)

                mu_clean, lv_clean = self.model(enc_in_base, tgt_v.size(1))
                loss_clean = self.gaussian_nll_loss(tgt_v, mu_clean, lv_clean)

                mu_adv, lv_adv = self.model(x_adv, tgt_v.size(1))
                loss_adv = self.gaussian_nll_loss(tgt_v, mu_adv, lv_adv)

                loss = loss_clean + self.adv_weight * loss_adv
                loss.backward()
                self.optimizer.step()

                ep_loss += loss.item()

                # --- ADE/FDE computed from mean (clean pass) -------------------- #
                with torch.no_grad():
                    last_pos = obs[:, -1, self.pos_slice]
                    pred_pos = self._vel_to_pos(last_pos, mu_clean, self.pos_size)
                    tgt_pos = tgt[:, :, self.pos_slice]
                    ade_b = calculate_ade(pred_pos, tgt_pos)
                    fde_b = calculate_fde(pred_pos, tgt_pos)
                    ep_ade += ade_b
                    ep_fde += fde_b

                bar.set_postfix(loss=f"{loss.item():.6f}", ADE=f"{ade_b:.5f}", FDE=f"{fde_b:.5f}")

            print(f"\nEpoch {epoch}: Loss {ep_loss/len(train_loader):.6f}  "
                  f"ADE {ep_ade/len(train_loader):.5f}  FDE {ep_fde/len(train_loader):.5f}")

            # -------------- validation & early stop ------------------- #
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
                obs, tgt = obs.to(self.device), tgt.to(self.device)
                if obs.dim() == 2:
                    obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                enc_in = obs[:, :, self.pos_size:self.pos_size + self.input_size]
                tgt_v = tgt[:, :, self.pos_size:self.pos_size + self.output_size]
                mu, lv = self.model(enc_in, tgt_v.size(1))
                loss_sum += self.gaussian_nll_loss(tgt_v, mu, lv).item()
        val_loss = loss_sum / len(loader)
        print(f"Validation loss (Gaussian NLL): {val_loss:.6f}")
        return val_loss

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  evaluate  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
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

                obs, tgt = obs.to(self.device), tgt.to(self.device)
                if cat is not None:
                    cat = cat.to(self.device)
                if obs.dim() == 2:
                    if cat is not None:
                        cat = cat.unsqueeze(0)
                    obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                enc_in = obs[:, :, self.pos_size:self.pos_size + self.input_size]
                mu, _ = self.model(enc_in, tgt.size(1))  # mean only for metrics

                last_pos = obs[:, -1, self.pos_slice]
                pred_pos = self._vel_to_pos(last_pos, mu, self.pos_size)
                tgt_pos = tgt[:, :, self.pos_slice]

                ade += calculate_ade(pred_pos, tgt_pos)
                fde += calculate_fde(pred_pos, tgt_pos)

        ade /= len(loader)
        fde /= len(loader)
        print(f"Test ADE {ade:.5f}  FDE {fde:.5f}")
        return ade, fde

    def predict(self, trajs, prediction_horizon):
        """
        trajs: list[np.ndarray] each with shape [T_obs, input_size]
        Returns:
            predictions:  list[np.ndarray]  each [H, pos_dim]  (mean positions)
            covariances:  list[np.ndarray]  each [H, pos_dim, pos_dim] (diag cov)
        """
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")
    
        self.model.eval()
        vel_batch, last_pos_batch = [], []
        H = int(prediction_horizon * self.trained_fps)   # seconds → steps
    
        # Build velocity histories and last positions
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)     # [T_obs, input_size]
            last_pos_batch.append(t[-1, self.pos_slice])                      # last [x,y]
    
            vel = t[1:] - t[:-1]                                              # [T_obs-1, input_size]
            # pad/crop to observation_length - 1
            if vel.size(0) > self.observation_length - 1:
                vel = vel[-(self.observation_length - 1):]
            else:
                pad_len = self.observation_length - 1 - vel.size(0)
                if pad_len > 0:
                    zpad = torch.zeros(pad_len, vel.size(1), device=self.device)
                    vel = torch.cat([zpad, vel], dim=0)
    
            vel_batch.append(vel)
    
        vel_batch = torch.stack(vel_batch)                                    # [B, T-1, input_size]
        enc_in = vel_batch[:, :, :self.input_size]                            # velocities only
    
        predictions, covariances = [], []
        with torch.no_grad():
            # model outputs: mean velocities and *raw variance head* (not logvar!)
            mu_v, raw_v = self.model(enc_in, H)                               # [B, H, out_dim] each
    
            # --- VARIANCE TRANSFORM (match training!) -----------------------
            # var = softplus(raw) + floor  (strictly positive, stable)
            var_v = F.softplus(raw_v, beta=1.0, threshold=20.0) + self.var_floor
            # Optional ceiling if you decide to add one later:
            # if hasattr(self, "var_ceiling") and self.var_ceiling is not None:
            #     var_v = torch.clamp(var_v, max=self.var_ceiling)
    
            # integrate velocities to positions (mean) and propagate diagonal variance
            for i in range(len(trajs)):
                last_pos = last_pos_batch[i]                                   # [pos_dim]
                mu_v_i  = mu_v[i:i+1, :, :self.pos_size]                       # [1, H, pos_dim]
                var_v_i = var_v[i:i+1, :, :self.pos_size]                      # [1, H, pos_dim]
    
                # mean positions via cumulative sum of velocities
                pos_mean = self._vel_to_pos(last_pos, mu_v_i, self.pos_size)[0]  # [H, pos_dim]
    
                # diagonal covariance: Var(sum v) = sum Var(v)  (indep increments assumption)
                pos_var  = torch.cumsum(var_v_i, dim=1)[0]                       # [H, pos_dim]
                pos_cov  = torch.diag_embed(pos_var)                             # [H, pos_dim, pos_dim]
    
                predictions.append(pos_mean.cpu().numpy())
                covariances.append(pos_cov.cpu().numpy())
    
        return predictions, covariances



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
