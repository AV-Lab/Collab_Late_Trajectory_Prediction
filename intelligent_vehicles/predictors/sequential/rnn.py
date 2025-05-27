#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:52 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

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
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)
            self.fc   = nn.Linear(h_dim, out_dim)

        def forward(self, x, h, c):
            y, (h, c) = self.lstm(x, (h, c))
            return self.fc(y), h, c

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc, self.dec = enc, dec

        def forward(self, x, horizon):
            """x: [B, T-1, enc_in_dim]  →  [B, horizon, vel_dim]"""
            B, _, _ = x.size()
            h, c = self.enc(x)
            dec_in = torch.zeros(B, 1, self.dec.lstm.input_size,
                                 device=x.device)
            outs = []
            for _ in range(horizon):
                y, h, c = self.dec(dec_in, h, c)
                outs.append(y)
                dec_in = y.detach()
            return torch.cat(outs, dim=1)

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
        """
        # ---------- copy config fields ----------------------------------- #
        for k in ["hidden_size", "num_layers", "input_size", "output_size",
                  "observation_length", "prediction_horizon",
                  "num_epochs", "learning_rate", "patience",
                  "device", "normalize"]:
            setattr(self, k, cfg[k])

        self.device = torch.device(self.device)

        # ---------- slices & dims --------------------------------------- #
        self.pos_slice = slice(0, self.output_size)      # x y (z)
        self.has_yaw   = self.input_size > self.output_size
        self.yaw_idx   = self.output_size if self.has_yaw else None
        self.vel_dim   = self.output_size                # 2 or 3
        self.vel_slice = slice(-self.vel_dim, None)

        self.enc_in_dim = self.vel_dim + (1 if self.has_yaw else 0)

        # ---------- model ---------------------------------------------- #
        enc = self.Encoder(self.enc_in_dim, self.hidden_size,
                           self.num_layers)
        dec = self.Decoder(self.vel_dim,     self.hidden_size,
                           self.vel_dim,     self.num_layers)
        self.model = self.Seq2Seq(enc, dec).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # ---------- optional checkpoint -------------------------------- #
        self.model_trained = False
        ckpt_path = cfg.get("checkpoint")
        if ckpt_path:
            print("Loading checkpoint:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "mean" in ckpt:
                self.mean = ckpt["mean"].to(self.device)
                self.std  = ckpt["std"].to(self.device)
                self.normalize = True
            self.model_trained = True

    # ~~~~~~~~~~~~~~~~~~~~~  private helpers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def _norm(self, vel):   return (vel - self.mean[self.vel_slice]) / self.std[self.vel_slice]
    def _denorm(self, vel): return vel * self.std[self.vel_slice] + self.mean[self.vel_slice]

    @staticmethod
    def _vel_to_pos(last_pos, vel_seq):
        """Integrate velocities to absolute positions."""
        B, T, D = vel_seq.shape
        out = torch.zeros(B, T, D, device=vel_seq.device)
        out[:, 0] = last_pos + vel_seq[:, 0]
        for t in range(1, T):
            out[:, t] = out[:, t-1] + vel_seq[:, t]
        return out

    def _make_enc_input(self, obs):
        """Return [yaw?] + velocities, shape [B, T-1, enc_in_dim]."""
        yaw_col = (obs[:, 1:, self.yaw_idx:self.yaw_idx+1]
                   if self.has_yaw else None)
        vel     = obs[:, 1:, self.vel_slice]
        return torch.cat([yaw_col, vel], dim=-1) if self.has_yaw else vel

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~  train  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def train(self, train_loader, valid_loader=None, save_path=None):
        print("Train batches:", len(train_loader))

        # fetch global stats from loader (attached by our custom Dataset)
        if self.normalize:
            self.mean = train_loader.dataset.mean.to(self.device)
            self.std  = train_loader.dataset.std.to(self.device)

        best_val = float('inf'); patience_ctr = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            ep_loss = ep_ade = ep_fde = 0.0

            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}",
                       leave=False)
            for obs, tgt in bar:
                obs, tgt = obs.to(self.device), tgt.to(self.device)
                if obs.dim() == 2:            # batch_size==1 edge-case
                    obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                enc_in = self._make_enc_input(obs)        # [B, T-1, enc_in]
                tgt_v  = tgt[:, :, self.vel_slice]        # [B, T_tgt, vel_dim]

                if self.normalize:
                    enc_in = enc_in.clone()
                    enc_in[:, :, -self.vel_dim:] = self._norm(enc_in[:, :, -self.vel_dim:])
                    tgt_v  = self._norm(tgt_v)

                self.optimizer.zero_grad()
                pred_v = self.model(enc_in, tgt_v.size(1))
                loss = self.criterion(pred_v, tgt_v)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                ep_loss += loss.item()

                # --- ADE/FDE in full pos space ------------------------- #
                with torch.no_grad():
                    if self.normalize:
                        pred_v_abs = self._denorm(pred_v)
                    else:
                        pred_v_abs = pred_v
                    last_pos = obs[:, -1, self.pos_slice]
                    pred_pos = self._vel_to_pos(last_pos, pred_v_abs)
                    tgt_pos  = tgt[:, :, self.pos_slice]

                    ade_b = calculate_ade(pred_pos, tgt_pos)
                    fde_b = calculate_fde(pred_pos, tgt_pos)
                    ep_ade += ade_b; ep_fde += fde_b

                bar.set_postfix(loss=f"{loss.item():.4f}",
                                ADE=f"{ade_b:.3f}", FDE=f"{fde_b:.3f}")

            print(f"\nEpoch {epoch}: Loss {ep_loss/len(train_loader):.4f}  "
                  f"ADE {ep_ade/len(train_loader):.3f}  "
                  f"FDE {ep_fde/len(train_loader):.3f}")

            # -------------- validation & early stop ------------------- #
            if valid_loader:
                val_loss = self.validate(valid_loader)
                if val_loss < best_val:
                    best_val = val_loss; patience_ctr = 0
                    if save_path: self.save_checkpoint(save_path)
                else:
                    patience_ctr += 1
                if patience_ctr >= self.patience:
                    print("Early stopping."); break
            elif save_path and epoch == self.num_epochs:
                self.save_checkpoint(save_path)

        self.model_trained = True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~  validate  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def validate(self, loader):
        self.model.eval(); loss_sum = 0.0
        with torch.no_grad():
            for obs, tgt in loader:
                obs, tgt = obs.to(self.device), tgt.to(self.device)
                if obs.dim() == 2: obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                enc_in = self._make_enc_input(obs)
                tgt_v  = tgt[:, :, self.vel_slice]
                if self.normalize:
                    enc_in[:, :, -self.vel_dim:] = self._norm(enc_in[:, :, -self.vel_dim:])
                    tgt_v = self._norm(tgt_v)

                pred_v = self.model(enc_in, tgt_v.size(1))
                loss_sum += self.criterion(pred_v, tgt_v).item()
        val_loss = loss_sum / len(loader)
        print(f"Validation loss: {val_loss:.4f}")
        return val_loss

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  evaluate  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def evaluate(self, loader):
        self.model.eval(); ade = fde = 0.0
        with torch.no_grad():
            for obs, tgt in loader:
                obs, tgt = obs.to(self.device), tgt.to(self.device)
                if obs.dim() == 2: obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                enc_in = self._make_enc_input(obs)
                if self.normalize:
                    enc_in[:, :, -self.vel_dim:] = self._norm(enc_in[:, :, -self.vel_dim:])

                pred_v = self.model(enc_in, tgt.size(1))
                if self.normalize: pred_v = self._denorm(pred_v)

                last_pos = obs[:, -1, self.pos_slice]
                pred_pos = self._vel_to_pos(last_pos, pred_v)
                tgt_pos  = tgt[:, :, self.pos_slice]

                ade += calculate_ade(pred_pos, tgt_pos)
                fde += calculate_fde(pred_pos, tgt_pos)

        ade /= len(loader); fde /= len(loader)
        print(f"Test ADE {ade:.3f}  FDE {fde:.3f}")
        return ade, fde

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  predict  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def predict(self, trajs):
        """trajs: list of np.ndarray with shape [T_obs, input_size]"""
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")

        self.model.eval()
        vel_batch, last_pos_batch = [], []
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)
            last_pos_batch.append(t[-1, self.pos_slice])
            vel = t[1:, self.pos_slice] - t[:-1, self.pos_slice]
            # pad if needed
            pad_len = self.observation_length - 1 - vel.size(0)
            if pad_len > 0:
                vel = torch.cat([torch.zeros(pad_len, self.vel_dim,
                                             device=self.device), vel])
            vel_batch.append(vel)
        vel_batch = torch.stack(vel_batch)            # [B, T-1, vel_dim]
        yaw_batch = None
        if self.has_yaw:
            yaw_batch = torch.stack([
                torch.tensor(tr[1:, self.yaw_idx:self.yaw_idx+1],
                             dtype=torch.float32, device=self.device)
                for tr in trajs
            ])
            enc_in = torch.cat([yaw_batch, vel_batch], dim=-1)
        else:
            enc_in = vel_batch

        with torch.no_grad():
            if self.normalize:
                enc_in[..., -self.vel_dim:] = self._norm(enc_in[..., -self.vel_dim:])
                pred_v = self._denorm(self.model(enc_in, self.prediction_horizon))
            else:
                pred_v = self.model(enc_in, self.prediction_horizon)

        predictions = []
        for i in range(len(trajs)):
            pos_seq = self._vel_to_pos(last_pos_batch[i], pred_v[i:i+1])[0]
            predictions.append(pos_seq.cpu().numpy())
        return predictions

    # ~~~~~~~~~~~~~~~~~~~~~~~~  checkpoint  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.normalize:
            ckpt["mean"] = self.mean.cpu()
            ckpt["std"]  = self.std.cpu()
        torch.save(ckpt, path)
        print("Saved best model to", path)


