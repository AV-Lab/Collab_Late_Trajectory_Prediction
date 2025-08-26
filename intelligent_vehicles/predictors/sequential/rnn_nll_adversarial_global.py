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
            super().__init__()
            self.lstm = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True)
            self.fc   = nn.Linear(h_dim, out_dim * 2)          # ← double the channels
            nn.init.constant_(self.fc.bias[out_dim:], 0.0) 
    
        def forward(self, x, h, c):
            y, (h, c) = self.lstm(x, (h, c))
            y = self.fc(y)                                     # [B, 1, 2*out_dim]
            mu, log_var = torch.chunk(y, 2, dim=-1)            # split along feature dim
            return (mu, log_var), h, c

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.enc, self.dec = enc, dec
    
        def forward(self, x, horizon):
            B, _, _ = x.size()
            h, c = self.enc(x)
    
            dec_in = torch.zeros(B, 1, self.dec.lstm.input_size, device=x.device)
            mus, log_vars = [], []
    
            for _ in range(horizon):
                (mu, log_var), h, c = self.dec(dec_in, h, c)
                mus.append(mu)
                log_vars.append(log_var)
                dec_in = mu.detach()                           # feed mean back
    
            return torch.cat(mus, 1), torch.cat(log_vars, 1)   # [B,H,D] each


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
        self.params = ["prediction_horizon", "num_epochs", "learning_rate", "patience", 
                  "hidden_size", "num_layers", "input_size", "output_size", "observation_length", "trained_fps"] 
    
        self.device = torch.device(cfg["device"])
        self.model_trained = False
        
        ckpt_path = cfg.get("checkpoint")
        if ckpt_path: 
            print("Loading checkpoint:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model_trained = True
            for k in self.params:
                setattr(self, k, ckpt[k])
        else:
            for k in self.params:
                setattr(self, k, cfg[k])

        self.pos_size = 2
        self.pos_slice = slice(0, self.pos_size)  # x,y
        
        # ---------- model ---------------------------------------------- #
        enc = self.Encoder(self.input_size, self.hidden_size, self.num_layers)
        dec = self.Decoder(self.output_size, self.hidden_size, self.output_size, self.num_layers)
        self.model = self.Seq2Seq(enc, dec).to(self.device)

        #self.criterion = nn.SmoothL1Loss()
        #self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # ---------- optional checkpoint -------------------------------- #
        if ckpt_path: 
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    @staticmethod
    def _vel_to_pos(last_pos, vel_seq, pos_size):
        """Integrate velocities to absolute positions."""
        vel_seq = vel_seq[:,:,:pos_size]
        B, T, D = vel_seq.shape
        out = torch.zeros(B, T, D, device=vel_seq.device)
        out[:, 0] = last_pos + vel_seq[:, 0]
        for t in range(1, T):
            out[:, t] = out[:, t-1] + vel_seq[:, t]
        return out
    
    @staticmethod
    def gaussian_nll(y, mu, log_var, clamp=(-4.5, 4.0)):
        """ L = 0.5 * (log σ² + (y-μ)² / σ²) [+ 0.5*log(2π)]
            where σ² = exp(log_var).  log_var is clamped for stability. """
        log_var = torch.clamp(log_var, *clamp)                # numerical safety
        inv_var = torch.exp(-log_var)
        return 0.5 * (log_var + (y - mu)**2 * inv_var)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~  train  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def train(self, train_loader, valid_loader=None, save_path=None):
        """
        Mixed-batch FGSM training:
            • loss_total = ½ (clean NLL + adversarial NLL)
            • ε comes from cfg[ "epsilon" ]  (default 0.03)
            • monitors mean / min / max log-variance per epoch
        """
        print("Train batches:", len(train_loader))
    
        eps        = getattr(self, "epsilon", 0.03)            # set in cfg or default
        best_val   = float("inf")
        patience_c = 0
    
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            ep_loss = ep_ade = ep_fde = 0.0
            logv_sum = logv_min = logv_max = 0.0
    
            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)
    
            for cat, obs, tgt in bar:
                cat, obs, tgt = cat.to(self.device), obs.to(self.device), tgt.to(self.device)
                if obs.dim() == 2:                               # batch_size == 1 edge-case
                    cat, obs, tgt = cat.unsqueeze(0), obs.unsqueeze(0), tgt.unsqueeze(0)
    
                # ---------- prepare inputs ----------------------------- #
                enc_in = obs[:, :, self.pos_size : self.pos_size + self.input_size]
                tgt_v  = tgt[:, :, self.pos_size : self.pos_size + self.output_size]
    
                # ---------------- clean pass (retain graph for grad-x) -- #
                enc_in.requires_grad_(True)
                mu, log_var = self.model(enc_in, tgt_v.size(1))
                loss_clean  = self.gaussian_nll(tgt_v, mu, log_var).mean()
    
                # gradient w.r.t. inputs (no param grads yet)
                grad_x = torch.autograd.grad(
                    loss_clean, enc_in, retain_graph=True, create_graph=False
                )[0].detach().sign()
    
                # ---------- adversarial input & second forward ---------- #
                enc_adv = (enc_in + eps * grad_x).detach()
                mu_adv, log_var_adv = self.model(enc_adv, tgt_v.size(1))
                loss_adv = self.gaussian_nll(tgt_v, mu_adv, log_var_adv).mean()
    
                # ---------------- combined optimisation ---------------- #
                total_loss = 0.5 * (loss_clean + loss_adv)
    
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
    
                # ---------------- logging & metrics -------------------- #
                ep_loss += total_loss.item()
    
                with torch.no_grad():
                    last_pos = obs[:, -1, self.pos_slice]
                    pred_pos = self._vel_to_pos(last_pos, mu, self.pos_size)
                    tgt_pos  = tgt[:, :, self.pos_slice]
                    ade_b = calculate_ade(pred_pos, tgt_pos)
                    fde_b = calculate_fde(pred_pos, tgt_pos)
    
                    ep_ade += ade_b
                    ep_fde += fde_b
    
                    # variance stats
                    logv_sum += log_var.mean().item()
                    logv_min = min(logv_min, log_var.min().item()) if ep_loss > 0 else log_var.min().item()
                    logv_max = max(logv_max, log_var.max().item()) if ep_loss > 0 else log_var.max().item()
    
                bar.set_postfix(
                    loss=f"{total_loss.item():.6f}",
                    ADE=f"{ade_b:.4f}",
                    FDE=f"{fde_b:.4f}"
                )
    
            n_batches = len(train_loader)
            print(
                f"\nEpoch {epoch}: Loss {ep_loss/n_batches:.6f}  "
                f"ADE {ep_ade/n_batches:.4f}  FDE {ep_fde/n_batches:.4f} | "
                f"logσ² μ={logv_sum/n_batches:+.3f}  "
                f"min={logv_min:+.3f}  max={logv_max:+.3f}"
            )
    
            # -------------- validation & early stop ------------------- #
            if valid_loader:
                val_loss = self.validate(valid_loader)
                if val_loss < best_val:
                    best_val = val_loss; patience_c = 0
                    if save_path: self.save_checkpoint(save_path)
                else:
                    patience_c += 1
                if patience_c >= self.patience:
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
                
                if obs.dim() == 2: 
                    obs, tgt = obs.unsqueeze(0), tgt.unsqueeze(0)

                enc_in = obs[:, :, self.pos_size:self.pos_size+self.input_size]      
                tgt_v  = tgt[:, :, self.pos_size:self.pos_size+self.output_size]     
                mu, log_var = self.model(enc_in, tgt_v.size(1))
                loss_sum += self.gaussian_nll(tgt_v, mu, log_var).mean().item()
        val_loss = loss_sum / len(loader)
        print(f"Validation loss: {val_loss:.4f}")
        return val_loss

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  evaluate  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def evaluate(self, loader):
        """
        Returns ADE, FDE, MSNE.
        MSNE = E[ ((y-μ)/σ)² ]  where  σ = exp(½·log_var)
        Ideal calibration ⇒ MSNE ≈ 1.
        """
        self.model.eval()
        ade = fde = msne_sum = 0.0
    
        with torch.no_grad():
            for cat, obs, tgt in loader:
                cat, obs, tgt = cat.to(self.device), obs.to(self.device), tgt.to(self.device)
    
                if obs.dim() == 2:                             # batch_size==1 edge-case
                    cat, obs, tgt = cat.unsqueeze(0), obs.unsqueeze(0), tgt.unsqueeze(0)
    
                enc_in = obs[:, :, self.pos_size : self.pos_size + self.input_size]   # [B, T-1, in]
                tgt_v  = tgt[:, :, self.pos_size : self.pos_size + self.output_size]  # [B, H, out]
    
                mu, log_var = self.model(enc_in, tgt_v.size(1))                       # [B, H, out] each
                sigma = torch.exp(0.5 * log_var)                                      # std
    
                # ---------- MSNE --------------------------------------------- #
                z   = (tgt_v - mu) / sigma
                msne_sum += z.pow(2).mean().item()
    
                # ---------- ADE / FDE ---------------------------------------- #
                last_pos = obs[:, -1, self.pos_slice]
                pred_pos = self._vel_to_pos(last_pos, mu, self.pos_size)
                tgt_pos  = tgt[:, :, self.pos_slice]
    
                ade += calculate_ade(pred_pos, tgt_pos)
                fde += calculate_fde(pred_pos, tgt_pos)
    
        n_batches = len(loader)
        ade  /= n_batches
        fde  /= n_batches
        msne = msne_sum / n_batches
    
        print(f"Test ADE {ade:.5f}  FDE {fde:.5f}  MSNE {msne:.3f}")
        return ade, fde, msne


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~  predict  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def predict(self, trajs, prediction_horizon):
        """trajs: list of np.ndarray with shape [T_obs, input_size]"""
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")

        self.model.eval()
        vel_batch, last_pos_batch = [], []
        prediction_horizon *= self.trained_fps   # seconds → steps if cfg gave FPS
        
        # ----- build velocity tensors ---------------------------------------
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)      # [T_obs, input_size]
            last_pos_batch.append(t[-1, self.pos_slice])                       # store last x,y
            vel = t[1:] - t[:-1]                                               # [T_obs-1, input_size]
            
            if vel.size(0) > self.observation_length - 1:
                vel = vel[-(self.observation_length - 1):]   # keep the latest part
            else:
                pad_len = self.observation_length - 1 - vel.size(0)
                if pad_len > 0:
                    zpad = torch.zeros(pad_len, self.input_size, device=self.device)
                    vel = torch.cat([zpad, vel], dim=0)
            vel_batch.append(vel)

        vel_batch = torch.stack(vel_batch)                                   # [B, T-1, input_size]

        # no yaw-branch, no normalisation
        enc_in = vel_batch[:, :, :self.input_size ]                                                      # velocities only

        with torch.no_grad():
            pred_v, _ = self.model(enc_in, prediction_horizon)        # predict()

        # ----- integrate velocities back to absolute positions ---------------
        predictions = []
        for i in range(len(trajs)):
            pos_seq = self._vel_to_pos(last_pos_batch[i], pred_v[i:i+1], self.pos_size)[0]    # [H, output_size]
            predictions.append(pos_seq.cpu().numpy())

        return predictions

    # ~~~~~~~~~~~~~~~~~~~~~~~~  checkpoint  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
 
        for k in self.params:
            ckpt[k] = getattr(self, k)

        torch.save(ckpt, path)
        print("Saved best model to", path)


