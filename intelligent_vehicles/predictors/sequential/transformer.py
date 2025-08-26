#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 09:33:13 2025

@author: nadya
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based seq-2-seq predictor **without data leakage in metrics**.

Only three logical additions to the original file:

  ▸ `_greedy_decode()` – causal inference helper  
  ▸ ADE/FDE paths (train / validate / evaluate / predict) call that helper  
  ▸ `save_checkpoint()` identical to RNN version

Everything else (imports, parameter names, nested sub-modules, scheduled
optimizer wrapper, etc.) stays intact so existing integration code keeps
working unchanged.
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

    def zero_grad(self): self.opt.zero_grad()


# ════════════════════════  Predictor  ════════════════════════════════ #
class TransformerPredictor:

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
        by outer `_greedy_decode()` to avoid changing call-sites.
        """
        def __init__(self, enc_in, pos_enc, encoder,
                     decoder, dec_in, out_proj):
            super().__init__()
            self.enc_in, self.dec_in = enc_in, dec_in
            self.pos_enc = pos_enc
            self.encoder, self.decoder = encoder, decoder
            self.out_proj = out_proj

        def forward(self, src, tgt_shifted, tgt_mask):
            # src : [B,T_enc,in_feat]     tgt_shifted : [B,T_dec,out_feat]
            memory = self.encoder(self.pos_enc(self.enc_in(src)))
            dec_in = self.pos_enc(self.dec_in(tgt_shifted))
            dec_out = self.decoder(tgt=dec_in, memory=memory, tgt_mask=tgt_mask)
            return self.out_proj(dec_out)                     # [B,T_dec,out_feat]

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

        out_proj = nn.Linear(self.embedding_size, self.out_features)

        self.model = self.Seq2Seq(enc_il, pos_enc, encoder,
                                  decoder, dec_il, out_proj).to(self.device)

        # optimiser
        base_opt = optim.Adam(self.model.parameters(),
                              betas=self.optimizer_betas,
                              eps=self.optimizer_eps)
        self.optimizer = ScheduledOptim(base_opt, self.lr_mul,
                                        self.embedding_size, self.n_warmup_steps)

        if ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.opt.load_state_dict(ckpt["optimizer_state_dict"])

        self.criterion  = nn.MSELoss()
        self.pos_size = 2
        self.pos_slice  = slice(0, self.pos_size)

    # ──────────────────────── utilities ──────────────────────── #
    @staticmethod
    def _causal_mask(sz: int, device) -> torch.Tensor:
        m = torch.triu(torch.ones(sz, sz, device=device), 1)
        return m.masked_fill(m == 1, float('-inf'))

    @staticmethod
    def _vel_to_pos(last_xy: torch.Tensor, vel_seq: torch.Tensor, pos_size):
        vel_xy = vel_seq[:, :, :pos_size]
        out = torch.zeros_like(vel_xy)
        out[:, 0] = last_xy + vel_xy[:, 0]
        for t in range(1, vel_xy.size(1)):
            out[:, t] = out[:, t-1] + vel_xy[:, t]
        return out

    # ─────────────────── greedy autoregressive ─────────────────── #
    def _greedy_decode(self, src_vel: torch.Tensor, steps: int) -> torch.Tensor:
        """
        src_vel : [B,T_enc,in_feat]   returns [B,steps,out_feat]
        """
        B = src_vel.size(0)
        memory = self.model.encoder(
            self.model.pos_enc(self.model.enc_in(src_vel))
        )
        ys = torch.zeros(B, 1, self.out_features, device=self.device)  # BOS
        out = []
        for _ in range(steps):
            dec_in = self.model.pos_enc(self.model.dec_in(ys))
            tgt_mask = self._causal_mask(dec_in.size(1), self.device)
            dec_out = self.model.decoder(tgt=dec_in, memory=memory,
                                         tgt_mask=tgt_mask)
            next_tok = self.model.out_proj(dec_out[:, -1:, :])          # [B,1,F]
            out.append(next_tok)
            ys = torch.cat([ys, next_tok.detach()], dim=1)
        return torch.cat(out, dim=1)                                    # [B,steps,F]

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

                src_vel = obs[:, :, self.pos_size:self.pos_size+self.in_features]            # teacher forcing
                tgt_vel = tgt[:, :, self.pos_size:self.pos_size+self.out_features]

                tgt_in = torch.zeros_like(tgt_vel)
                tgt_in[:, 1:] = tgt_vel[:, :-1]
                tgt_mask = self._causal_mask(tgt_in.size(1), self.device)

                self.optimizer.zero_grad()
                pred_tf = self.model(src_vel, tgt_in, tgt_mask)
                loss = self.criterion(pred_tf, tgt_vel)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step_and_update_lr()
                running_loss += loss.item()

                # greedy metrics
                pred_gd = self._greedy_decode(src_vel, tgt_vel.size(1))
                last_xy = obs[:, -1, self.pos_slice]
                ade = calculate_ade(self._vel_to_pos(last_xy, pred_gd, self.pos_size),
                                    tgt[:, :, self.pos_slice])
                fde = calculate_fde(self._vel_to_pos(last_xy, pred_gd, self.pos_size),
                                    tgt[:, :, self.pos_slice])
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
                pred_gd = self._greedy_decode(src_vel, tgt.size(1))

                last_xy = obs[:, -1, self.pos_slice]
                ade += calculate_ade(self._vel_to_pos(last_xy, pred_gd, self.pos_size),
                                     tgt[:, :, self.pos_slice])
                fde += calculate_fde(self._vel_to_pos(last_xy, pred_gd, self.pos_size),
                                     tgt[:, :, self.pos_slice])

        ade /= len(loader); fde /= len(loader)
        if not silent:
            print(f"Eval ADE {ade:.5f}  FDE {fde:.5f}")
        return ade, fde

    # ─────────────────────── inference  ──────────────────────── #
    def predict(self, trajs: List[np.ndarray], prediction_horizon: float):
        """
        trajs : list of np.ndarray, each [T_obs, in_features]
        prediction_horizon : seconds  (converted with trained_fps)
        returns list[np.ndarray] with shape [H, out_features]
        """
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")

        self.model.eval()
        steps = int(round(prediction_horizon * self.trained_fps))
        src_vel_batch, last_xy_batch = [], []
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)
            last_xy_batch.append(t[-1, self.pos_slice])
            vel = t[1:] - t[:-1]
            pad = self.past_trajectory - 1 - vel.size(0)
            if pad > 0:
                vel = torch.cat([torch.zeros(pad, self.in_features,
                                             device=self.device), vel])
            src_vel_batch.append(vel[:, self.pos_size:self.pos_size+self.in_features])
        src_vel_batch = torch.stack(src_vel_batch)         # [B,T_enc,4]
        last_xy_batch = torch.stack(last_xy_batch)         # [B,2]

        with torch.no_grad():
            pred_vel = self._greedy_decode(src_vel_batch, steps)

        preds = []
        for i in range(len(trajs)):
            pos = self._vel_to_pos(last_xy_batch[i:i+1], pred_vel[i:i+1], self.pos_size)[0]
            preds.append(pos.cpu().numpy())
        return preds

    # ───────────────────── checkpoint I/O ─────────────────────── #
    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {k: getattr(self, k) for k in self.params}
        ckpt["model_state_dict"] = self.model.state_dict()
        ckpt["optimizer_state_dict"] = self.optimizer.opt.state_dict()
        torch.save(ckpt, path)
        print("Saved best model to", path)
