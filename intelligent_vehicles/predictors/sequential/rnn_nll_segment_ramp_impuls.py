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
    
    @staticmethod
    def _adv_segment_ramp(enc_in, grad_x, eps_base, p_tail=0.5, min_len=1, max_frac=0.35, L_cap=5):
        """
        Unified 'segment-ramp' adversary.
        - Picks a contiguous segment of length L (1..min(5, ceil(max_frac*T))).
        - Places it at tail with prob p_tail, else anywhere.
        - Inside the segment, per-step epsilon ramps 1×..min(L, m_cap)× (m_cap∈[1..5]).
        """
        B, Tm1, D = enc_in.shape
        delta = torch.zeros_like(enc_in)
        sgn   = grad_x.sign()
    
        # Max segment length allowed by window and cap
        Lmax_frac = max(min_len, int(Tm1 * max_frac))          # floor; OK for sampling
        Lmax      = max(min_len, min(L_cap, Lmax_frac))
    
        # Per-sample segment + ramp
        for b in range(B):
            # sample length
            L = int(torch.randint(low=min_len, high=Lmax + 1, size=(1,), device=enc_in.device).item())
            # sample cap for the ramp (varies 1..5, not always hitting 5)
            m_cap = int(torch.randint(low=1, high=L_cap + 1, size=(1,), device=enc_in.device).item())
    
            # choose start index: tail or anywhere
            tail = bool(torch.rand(1, device=enc_in.device).item() < p_tail)
            s = Tm1 - L if tail else int(torch.randint(low=0, high=max(1, Tm1 - L + 1), size=(1,), device=enc_in.device).item())
            e = s + L
    
            # ramp multipliers: [1, 2, ..., L] clipped by m_cap
            ramp = torch.arange(1, L + 1, device=enc_in.device, dtype=enc_in.dtype).clamp_max(m_cap).view(L, 1)  # [L,1]
            seg_grad = sgn[b, s:e, :]  # [L, D]
    
            # per-step: delta = eps_base * ramp_j * sign(grad)
            delta[b, s:e, :] = eps_base * (seg_grad * ramp)
    
        return (enc_in + delta).detach()

    @staticmethod
    def _adv_impulse_tail(enc_in, grad_x, eps, tail_len=2, decay=(1.0, 0.6)):
        """
        Localized impulse with a short decaying tail (1–2 steps).
        - Choose a random index t; apply eps*sign at t and decayed at t+1..t+tail_len-1.
        """
        B, Tm1, D = enc_in.shape
        delta = torch.zeros_like(enc_in)
        sgn   = grad_x.sign()
    
        t0 = torch.randint(low=0, high=Tm1, size=(B,), device=enc_in.device)
        for b in range(B):
            t = t0[b].item()
            # main impulse
            delta[b, t] += eps * sgn[b, t]
            # short tail
            for k in range(1, min(tail_len, Tm1 - t)):
                delta[b, t + k] += eps * decay[k - 1] * sgn[b, t + k]
    
        return (enc_in + delta).detach()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~  train  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def train(self, train_loader, valid_loader=None, save_path=None):
        """
        Mixture: clean + {global FGSM, unified segment-ramp (A′/A), impulse(+short tail)}
        """
        print("Train batches:", len(train_loader))
    
        eps_base   = getattr(self, "epsilon", 0.03)   # base ε
        p_tail     = getattr(self, "p_tail", 0.5)     # prob segment at tail
        w_clean    = getattr(self, "w_clean", 0.5)
        # split remaining weight equally across 3 adversaries by default
        rem = 1.0 - w_clean
        w_global   = getattr(self, "w_global", rem / 3.0)
        w_seg      = getattr(self, "w_seg",    rem / 3.0)
        w_impulse  = getattr(self, "w_impulse",rem / 3.0)
    
        best_val   = float("inf")
        patience_c = 0
    
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            ep_loss = ep_ade = ep_fde = 0.0
            logv_sum = logv_min = logv_max = 0.0
    
            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)
    
            for cat, obs, tgt in bar:
                cat, obs, tgt = cat.to(self.device), obs.to(self.device), tgt.to(self.device)
                if obs.dim() == 2:
                    cat, obs, tgt = cat.unsqueeze(0), obs.unsqueeze(0), tgt.unsqueeze(0)
    
                enc_in = obs[:, :, self.pos_size : self.pos_size + self.input_size]
                tgt_v  = tgt[:, :, self.pos_size : self.pos_size + self.output_size]
    
                # ----- clean -----
                enc_in = enc_in.requires_grad_(True)
                mu_c, logv_c = self.model(enc_in, tgt_v.size(1))
                loss_clean   = self.gaussian_nll(tgt_v, mu_c, logv_c).mean()
    
                grad_x = torch.autograd.grad(loss_clean, enc_in, retain_graph=True, create_graph=False)[0]
    
                # ----- adversaries -----
                enc_adv_global = (enc_in + eps_base * grad_x.sign()).detach()
                enc_adv_seg    = self._adv_segment_ramp(enc_in, grad_x, eps_base, p_tail=p_tail)
                enc_adv_imp    = self._adv_impulse_tail(enc_in, grad_x, eps_base, tail_len=2, decay=(1.0, 0.6))
    
                # forwards
                mu_g, lv_g = self.model(enc_adv_global, tgt_v.size(1))
                mu_s, lv_s = self.model(enc_adv_seg,    tgt_v.size(1))
                mu_i, lv_i = self.model(enc_adv_imp,    tgt_v.size(1))
    
                loss_g = self.gaussian_nll(tgt_v, mu_g, lv_g).mean()
                loss_s = self.gaussian_nll(tgt_v, mu_s, lv_s).mean()
                loss_i = self.gaussian_nll(tgt_v, mu_i, lv_i).mean()
    
                # mixture
                total_loss = w_clean * loss_clean + w_global * loss_g + w_seg * loss_s + w_impulse * loss_i
    
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
    
                # ----- logging from clean pass -----
                ep_loss += total_loss.item()
                with torch.no_grad():
                    last_pos = obs[:, -1, self.pos_slice]
                    pred_pos = self._vel_to_pos(last_pos, mu_c, self.pos_size)
                    tgt_pos  = tgt[:, :, self.pos_slice]
                    ade_b = calculate_ade(pred_pos, tgt_pos)
                    fde_b = calculate_fde(pred_pos, tgt_pos)
                    ep_ade += ade_b; ep_fde += fde_b
    
                    logv_mean = logv_c.mean().item()
                    logv_sum += logv_mean
                    lv_min = logv_c.min().item(); lv_max = logv_c.max().item()
                    if ep_loss == total_loss.item():
                        logv_min, logv_max = lv_min, lv_max
                    else:
                        logv_min = min(logv_min, lv_min); logv_max = max(logv_max, lv_max)
    
                bar.set_postfix(loss=f"{total_loss.item():.6f}", ADE=f"{ade_b:.4f}", FDE=f"{fde_b:.4f}")
    
            n_batches = len(train_loader)
            print(
                f"\nEpoch {epoch}: Loss {ep_loss/n_batches:.6f}  "
                f"ADE {ep_ade/n_batches:.4f}  FDE {ep_fde/n_batches:.4f} | "
                f"logσ² μ={logv_sum/n_batches:+.3f}  min={logv_min:+.3f}  max={logv_max:+.3f}"
            )
    
            # ----- validation / early stop -----
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
        """trajs: list of np.ndarray with shape [T_obs, input_size]
           Returns:
             predictions : list[dict]
                 For each trajectory i:
                   {
                     "mean": np.ndarray [H, pos_size],          # mean positions μ_pos
                     "cov":  np.ndarray [H, pos_size, pos_size] # per-step covariance Σ_pos (diagonal here)
                   }
        """
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")
    
        self.model.eval()
        vel_batch, last_pos_batch = [], []
        steps = int(round(prediction_horizon * self.trained_fps))  # seconds → steps
    
        # ----- build velocity tensors ---------------------------------------
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)      # [T_obs, input_size]
            last_pos_batch.append(t[-1, self.pos_slice])                       # store last x,y
            vel = t[1:] - t[:-1]                                               # [T_obs-1, input_size]
    
            if vel.size(0) > self.observation_length - 1:
                vel = vel[-(self.observation_length - 1):]                     # keep latest part
            else:
                pad_len = self.observation_length - 1 - vel.size(0)
                if pad_len > 0:
                    zpad = torch.zeros(pad_len, self.input_size, device=self.device)
                    vel = torch.cat([zpad, vel], dim=0)
            vel_batch.append(vel)
    
        vel_batch = torch.stack(vel_batch)                                      # [B, T-1, input_size]
        enc_in = vel_batch[:, :, :self.input_size]                              # velocities only
    
        # ----- model forward -------------------------------------------------
        with torch.no_grad():
            mu_v, log_var_v = self.model(enc_in, steps)                         # [B,H,out], [B,H,out]
            var_v = torch.exp(log_var_v)                                        # velocity variance
    
        # ----- integrate means & propagate covariance -----------------------
        predictions = []
        for i in range(len(trajs)):
            # mean positions from mean velocities
            mu_pos = self._vel_to_pos(last_pos_batch[i], mu_v[i:i+1], self.pos_size)[0]   # [H, pos_size]
    
            # diagonal position covariance: Var(sum v) = sum Var(v) (assume step-independence)
            var_v_i = var_v[i, :, :self.pos_size]                                          # [H, pos_size]
            var_pos = torch.cumsum(var_v_i, dim=0)                                         # [H, pos_size]
            cov_pos = torch.diag_embed(var_pos)                                            # [H, pos_size, pos_size]
    
            predictions.append({
                "mean": mu_pos.cpu().numpy(),
                "cov":  cov_pos.cpu().numpy(),
            })
    
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


