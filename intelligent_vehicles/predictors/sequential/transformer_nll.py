#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based seq-2-seq predictor with Gaussian NLL (mean + variance)
and covariance propagation to positions (similar style to RNNPredictorNLL).

- The model outputs velocity mean + log-variance.
- Training uses Gaussian NLL on velocities.
- ADE/FDE are computed from the mean (causal greedy decode).
- predict(...) returns per-step position means and diagonal covariances.
"""

import math, os, sys
import numpy as np
from pathlib import Path
from typing import List

import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.distance_metrics import calculate_ade, calculate_fde


# ──────────────────────────── Scheduler ────────────────────────────── #
class ScheduledOptim:
    """Transformer LR schedule from *Attention is All You Need*."""
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self.opt = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.warm = n_warmup_steps
        self.n_steps = 0

    def _lr(self):
        s = max(1, self.n_steps)
        return self.lr_mul * (self.d_model ** -0.5) * min(s ** -0.5,
                                                          s * self.warm ** -1.5)

    def step_and_update_lr(self):
        self.n_steps += 1
        lr = self._lr()
        for g in self.opt.param_groups:
            g['lr'] = lr
        self.opt.step()

    def zero_grad(self): 
        self.opt.zero_grad()


# ════════════════════════  Predictor  ════════════════════════════════ #
class TransformerPredictorNLL:

    # ───────────────── helper sub-modules ────────────────── #
    class Linear_Embeddings(nn.Module):
        def __init__(self, in_features: int, d_model: int):
            super().__init__()
            self.proj = nn.Linear(in_features, d_model)
            self.scale = math.sqrt(d_model)

        def forward(self, x):                    # [B,T,in_features]
            return self.proj(x) * self.scale

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.,
                     max_len: int = 5000, batch_first: bool = True):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.batch_first = batch_first

            pos = torch.arange(max_len).unsqueeze(1)          # [L,1]
            div = torch.exp(
                torch.arange(0, d_model, 2) *
                (-math.log(10000.0) / d_model)
            )                                                 # [d_model/2]

            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)

            if batch_first:
                pe = pe.unsqueeze(0)         # [1,L,D]
            else:
                pe = pe.unsqueeze(1)         # [L,1,D]
            self.register_buffer('pe', pe, persistent=False)

        def forward(self, x):                 # [B,T,D] or [T,B,D]
            x = x + (self.pe[:, :x.size(1)] if self.batch_first
                     else self.pe[:x.size(0)])
            return self.dropout(x)

    class Seq2Seq(nn.Module):
        """
        Teacher-forcing pass only.  Autoregressive decoding is handled
        by outer `_greedy_decode()`.
        """
        def __init__(self, enc_in, pos_enc, encoder,
                     decoder, dec_in, out_mu, out_logvar):
            super().__init__()
            self.enc_in, self.dec_in = enc_in, dec_in
            self.pos_enc = pos_enc
            self.encoder, self.decoder = encoder, decoder
            self.out_mu = out_mu
            self.out_logvar = out_logvar

        def forward(self, src, tgt_shifted, tgt_mask):
            # src : [B,T_enc,in_feat]     tgt_shifted : [B,T_dec,out_feat]
            memory = self.encoder(self.pos_enc(self.enc_in(src)))
            dec_in = self.pos_enc(self.dec_in(tgt_shifted))
            dec_out = self.decoder(tgt=dec_in, memory=memory, tgt_mask=tgt_mask)
            mu     = self.out_mu(dec_out)        # [B,T_dec,out_feat]
            logvar = self.out_logvar(dec_out)    # [B,T_dec,out_feat]
            return mu, logvar

    # ─────────────────── constructor (same keys) ─────────────────── #
    def __init__(self, cfg: dict):

        # keep exactly same param list you used before
        self.params = ["past_trajectory", "future_trajectory",
                       "in_features", "out_features",
                       "num_heads", "num_encoder_layers",
                       "num_decoder_layers", "embedding_size",
                       "dropout_encoder", "dropout_decoder",
                       "batch_first", "actn",
                       "lr_mul", "n_warmup_steps",
                       "optimizer_betas", "optimizer_eps",
                       "num_epochs", "trained_fps",
                       "early_stopping_patience", "early_stopping_delta"]

        self.device = torch.device(cfg["device"])
        self.model_trained = False
        
        self.pos_size = 2
        self.pos_slice = slice(0, self.pos_size)  # x,y

        ckpt = None
        if cfg.get("checkpoint"):
            ckpt = torch.load(cfg["checkpoint"], map_location=self.device)
            self.model_trained = True

        # copy attributes
        source = ckpt if ckpt else cfg
        for k in self.params:
            setattr(self, k, source[k])

        max_len = max(self.past_trajectory, self.future_trajectory)
        d_ff = 4 * self.embedding_size

        # layers
        enc_il  = self.Linear_Embeddings(self.in_features,  self.embedding_size)
        dec_il  = self.Linear_Embeddings(self.out_features, self.embedding_size)
        pos_enc = self.PositionalEncoding(self.embedding_size,
                                          dropout=self.dropout_encoder,
                                          max_len=max_len,
                                          batch_first=self.batch_first)

        enc_layer = nn.TransformerEncoderLayer(self.embedding_size,
                                               self.num_heads, d_ff,
                                               self.dropout_encoder,
                                               batch_first=self.batch_first,
                                               activation=self.actn)
        encoder = nn.TransformerEncoder(enc_layer, self.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(self.embedding_size,
                                               self.num_heads, d_ff,
                                               self.dropout_decoder,
                                               batch_first=self.batch_first,
                                               activation=self.actn)
        decoder = nn.TransformerDecoder(dec_layer, self.num_decoder_layers)

        # two heads: mean and log-variance of velocity
        out_mu     = nn.Linear(self.embedding_size, self.out_features)
        out_logvar = nn.Linear(self.embedding_size, self.out_features)

        self.model = self.Seq2Seq(enc_il, pos_enc, encoder,
                                  decoder, dec_il, out_mu, out_logvar).to(self.device)

        # optimiser
        base_opt = optim.Adam(self.model.parameters(),
                              betas=self.optimizer_betas,
                              eps=self.optimizer_eps)
        self.optimizer = ScheduledOptim(base_opt, self.lr_mul,
                                        self.embedding_size, self.n_warmup_steps)

        if ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.opt.load_state_dict(ckpt["optimizer_state_dict"])

        # Gaussian NLL params (same style as RNN)
        self.var_floor  = 5e-3
        self.logvar_min = -2.0
        self.logvar_max = 6.0

        self.pos_size  = 2
        self.pos_slice = slice(0, self.pos_size)

    # ──────────────────────── utilities ──────────────────────── #
    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        m = torch.triu(torch.ones(sz, sz, device=device), 1)
        return m.masked_fill(m == 1, float('-inf'))

    @staticmethod
    def _vel_to_pos(last_xy: torch.Tensor, vel_seq: torch.Tensor, pos_size: int):
        vel_xy = vel_seq[:, :, :pos_size]
        out = torch.zeros_like(vel_xy)
        out[:, 0] = last_xy + vel_xy[:, 0]
        for t in range(1, vel_xy.size(1)):
            out[:, t] = out[:, t-1] + vel_xy[:, t]
        return out

    def gaussian_nll_loss(self, y_true, mu, logvar):
        """
        Diagonal Gaussian NLL with per-dim heteroscedastic variance.
        y_true, mu, logvar: [B, H, D]
        """
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        var = torch.exp(logvar).clamp_min(self.var_floor)
        nll = 0.5 * (logvar + (y_true - mu) ** 2 / var)
        return nll.mean()

    # ─────────────────── greedy autoregressive ─────────────────── #
    def _greedy_decode(self, src_vel: torch.Tensor, steps: int):
        """
        src_vel : [B,T_enc,in_feat]
        Returns:
          mu_seq  : [B,steps,out_features]
          logvar_seq : [B,steps,out_features]
        """
        B = src_vel.size(0)
        memory = self.model.encoder(
            self.model.pos_enc(self.model.enc_in(src_vel))
        )
        ys = torch.zeros(B, 1, self.out_features, device=self.device)  # BOS in velocity space
        mu_outs, lv_outs = [], []

        for _ in range(steps):
            dec_in = self.model.pos_enc(self.model.dec_in(ys))
            tgt_mask = self._causal_mask(dec_in.size(1), self.device)
            dec_out = self.model.decoder(tgt=dec_in, memory=memory,
                                         tgt_mask=tgt_mask)
            mu     = self.model.out_mu(dec_out[:, -1:, :])      # [B,1,F]
            logvar = self.model.out_logvar(dec_out[:, -1:, :])  # [B,1,F]
            mu_outs.append(mu)
            lv_outs.append(logvar)
            ys = torch.cat([ys, mu.detach()], dim=1)

        mu_seq  = torch.cat(mu_outs, dim=1)
        lv_seq  = torch.cat(lv_outs, dim=1)
        return mu_seq, lv_seq

    # ───────────────────────  training  ───────────────────────── #
    def train(self, train_loader, valid_loader=None, save_path=None):
        best_val, patience = float('inf'), 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = running_ade = running_fde = 0.0

            pbar = tqdm(train_loader,
                        desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)

            for batch in pbar:
                # loader may yield (cat, obs, tgt) or (obs, tgt)
                if len(batch) == 3:
                    _, obs, tgt = batch
                else:
                    obs, tgt = batch
                obs, tgt = obs.to(self.device), tgt.to(self.device)

                src_vel = obs[:, :, self.pos_size:self.pos_size+self.in_features]
                tgt_vel = tgt[:, :, self.pos_size:self.pos_size+self.out_features]

                tgt_in = torch.zeros_like(tgt_vel)
                tgt_in[:, 1:] = tgt_vel[:, :-1]
                tgt_mask = self._causal_mask(tgt_in.size(1), self.device)

                self.optimizer.zero_grad()
                mu_tf, lv_tf = self.model(src_vel, tgt_in, tgt_mask)
                loss = self.gaussian_nll_loss(tgt_vel, mu_tf, lv_tf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step_and_update_lr()
                running_loss += loss.item()

                # greedy metrics (use velocity mean)
                mu_gd, _ = self._greedy_decode(src_vel, tgt_vel.size(1))
                last_xy = obs[:, -1, self.pos_slice]
                pred_pos = self._vel_to_pos(last_xy, mu_gd, self.pos_size)
                tgt_pos  = tgt[:, :, self.pos_slice]
                ade = calculate_ade(pred_pos, tgt_pos)
                fde = calculate_fde(pred_pos, tgt_pos)
                running_ade += ade; running_fde += fde
                pbar.set_postfix(loss=f"{loss.item():.5f}",
                                 ADE=f"{ade:.4f}", FDE=f"{fde:.4f}")

            print(f"\nEpoch {epoch}:  "
                  f"Loss {running_loss/len(train_loader):.5f}  "
                  f"ADE {running_ade/len(train_loader):.4f}  "
                  f"FDE {running_fde/len(train_loader):.4f}")

            # ─── validation & early-stop
            if valid_loader:
                val = self.evaluate(valid_loader, silent=True)[0]
                if val < best_val - self.early_stopping_delta:
                    best_val, patience = val, 0
                    if save_path: self.save_checkpoint(save_path)
                else:
                    patience += 1
                if patience >= self.early_stopping_patience:
                    print("Early stopping."); break
            elif save_path and epoch == self.num_epochs:
                self.save_checkpoint(save_path)

        self.model_trained = True

    # ─────────────────────── evaluation  ──────────────────────── #
    def evaluate(self, loader, silent=False):
        self.model.eval(); ade = fde = 0.0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    _, obs, tgt = batch
                else:
                    obs, tgt = batch
                obs, tgt = obs.to(self.device), tgt.to(self.device)

                src_vel = obs[:, :, self.pos_size:self.pos_size+self.in_features]
                mu_gd, _ = self._greedy_decode(src_vel, tgt.size(1))

                last_xy = obs[:, -1, self.pos_slice]
                pred_pos = self._vel_to_pos(last_xy, mu_gd, self.pos_size)
                tgt_pos  = tgt[:, :, self.pos_slice]
                ade += calculate_ade(pred_pos, tgt_pos)
                fde += calculate_fde(pred_pos, tgt_pos)

        ade /= len(loader); fde /= len(loader)
        if not silent:
            print(f"Eval ADE {ade:.5f}  FDE {fde:.5f}")
        return ade, fde

    # ─────────────────────── inference  ──────────────────────── #
    def predict(self, trajs, prediction_horizon):
        """
        Make this Transformer predictor behave like the LSTM predictor.
    
        trajs : list of np.ndarray, each [T_obs, in_features_total]
        prediction_horizon : *number of steps* (same semantics as LSTM)
    
        Returns:
          predictions  : list[np.ndarray] each [H, pos_dim] position means
          covariances  : list[np.ndarray] each [H, pos_dim, pos_dim] diag covs
        """
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")
    
        self.model.eval()
    
        # LSTM treats prediction_horizon as "H" directly (not seconds)
        H = int(round(prediction_horizon))
    
        vel_batch, last_pos_batch = [], []
    
        # ---- build velocity histories exactly like LSTM version ----
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)  # [T_obs, in_features_total]
    
            # last observed position (x,y or x,y,z etc.) using the same slice as LSTM
            last_pos_batch.append(t[-1, self.pos_slice])                   # [pos_dim]
    
            # frame-to-frame velocity
            vel = t[1:] - t[:-1]                                           # [T_obs-1, in_features_total]
    
            # truncate/pad to fixed history length: observation_length - 1
            if vel.size(0) > self.past_trajectory - 1:
                vel = vel[-(self.past_trajectory - 1):]
            else:
                pad_len = self.past_trajectory - 1 - vel.size(0)
                if pad_len > 0:
                    zpad = torch.zeros(pad_len, vel.size(1), device=self.device)
                    vel = torch.cat([zpad, vel], dim=0)
    
            vel_batch.append(vel)
    
        vel_batch     = torch.stack(vel_batch)              # [B, T_enc, in_features_total]
        last_pos_batch = torch.stack(last_pos_batch)        # [B, pos_dim]
    
        # use the first self.in_features channels of velocity as model input,
        # analogous to enc_in = vel_batch[:, :, :self.input_size] in the LSTM code
        src_vel_batch = vel_batch[:, :, :self.in_features]  # [B, T_enc, in_features]
    
        with torch.no_grad():
            # ---- Transformer decoding, but with the same semantics as LSTM ----
            # outputs mean velocities and log-variance
            mu_v, lv_v = self._greedy_decode(src_vel_batch, H)   # [B, H, out_dim]
    
            # map log-variance -> variance as in training: exp(clamp) + floor
            lv_v = torch.clamp(lv_v, self.logvar_min, self.logvar_max)
            var_v = torch.exp(lv_v).clamp_min(self.var_floor)     # [B, H, out_dim]
    
            B = mu_v.size(0)
            pos_means, pos_covs = [], []
    
            for i in range(B):
                # last_pos: [pos_dim]
                last_pos = last_pos_batch[i]
    
                # predicted mean velocities for position dims
                mu_v_i = mu_v[i:i+1, :, :self.pos_size]          # [1, H, pos_dim]
    
                # integrate velocities -> positions (same helper as LSTM)
                pos_mean_i = self._vel_to_pos(last_pos, mu_v_i, self.pos_size)[0]  # [H, pos_dim]
                pos_means.append(pos_mean_i)
    
                # diagonal covariance: cumulative sum of per-step velocity variances
                var_v_i = var_v[i:i+1, :, :self.pos_size]        # [1, H, pos_dim]
                pos_var = torch.cumsum(var_v_i, dim=1)[0]        # [H, pos_dim]
                pos_cov_i = torch.diag_embed(pos_var)            # [H, pos_dim, pos_dim]
                pos_covs.append(pos_cov_i)
    
        # convert to lists-of-arrays (like LSTM predict)
        predictions = [pos_means[i].cpu().numpy() for i in range(B)]   # each [H, pos_dim]
        covariances = [pos_covs[i].cpu().numpy()  for i in range(B)]   # each [H, pos_dim, pos_dim]
    
        return predictions, covariances


    # ───────────────────── checkpoint I/O ─────────────────────── #
    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {k: getattr(self, k) for k in self.params}
        ckpt["model_state_dict"] = self.model.state_dict()
        ckpt["optimizer_state_dict"] = self.optimizer.opt.state_dict()
        torch.save(ckpt, path)
        print("Saved best model to", path)
