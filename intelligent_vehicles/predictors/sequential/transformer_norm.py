#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer seq-to-seq predictor with optional velocity normalisation.

Set cfg["normalize"] = True to enable z-scoring on (vx, vy, vyaw).
Stats are copied once from train_loader.dataset and stored in the
checkpoint for reproducibility.
"""
from __future__ import annotations
import math, os, sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.distance_metrics import calculate_ade, calculate_fde      # noqa: E402


# ───────────────────────── Noam scheduler ────────────────────────── #
class ScheduledOptim:
    def __init__(self, opt, lr_mul, d_model, warm):
        self.opt, self.lr_mul, self.d_model, self.warm = opt, lr_mul, d_model, warm
        self.n_steps = 0

    def _lr(self):
        s = max(1, self.n_steps)
        return self.lr_mul * self.d_model ** -0.5 * min(s ** -0.5,
                                                        s * self.warm ** -1.5)

    def step_and_update_lr(self):
        self.n_steps += 1
        for g in self.opt.param_groups:
            g["lr"] = self._lr()
        self.opt.step()

    def zero_grad(self): self.opt.zero_grad()


# ═════════════════════ Predictor ═══════════════════════ #
class TransformerPredictor:

    # ───── helper blocks ───── #
    class LinearEmb(nn.Module):
        def __init__(self, d_in, d_model):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.scale = math.sqrt(d_model)

        def forward(self, x): return self.proj(x) * self.scale

    class PosEnc(nn.Module):
        def __init__(self, d_model, dropout=0., max_len=5000, batch_first=True):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.batch_first = batch_first

            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            pe = pe.unsqueeze(0) if batch_first else pe.unsqueeze(1)
            self.register_buffer("pe", pe, persistent=False)

        def forward(self, x):
            bias = self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(0)]
            return self.dropout(x + bias)

    class Seq2Seq(nn.Module):
        def __init__(self, enc_in, pos, enc, dec, dec_in, out):
            super().__init__()
            self.enc_in, self.dec_in = enc_in, dec_in
            self.pos, self.enc, self.dec, self.out = pos, enc, dec, out

        def forward(self, src, tgt_shifted, tgt_mask):
            mem = self.enc(self.pos(self.enc_in(src)))
            dec_in = self.pos(self.dec_in(tgt_shifted))
            return self.out(self.dec(tgt=dec_in, memory=mem, tgt_mask=tgt_mask))

    # ───── constructor ───── #
    def __init__(self, cfg: dict):
        self.params = ["past_trajectory", "future_trajectory",
                       "in_features", "out_features",
                       "num_heads", "num_encoder_layers", "num_decoder_layers",
                       "embedding_size", "dropout_encoder", "dropout_decoder",
                       "batch_first", "actn", "lr_mul", "n_warmup_steps",
                       "optimizer_betas", "optimizer_eps", "num_epochs",
                       "trained_fps", "early_stopping_patience",
                       "early_stopping_delta", "normalize", "mean", "std"]

        self.device = torch.device(cfg["device"])
        ckpt = torch.load(cfg["checkpoint"], map_location=self.device) \
               if cfg.get("checkpoint") else None
        src = ckpt if ckpt else cfg
        
        for p in self.params:
            setattr(self, p, src.get(p))
            
        if ckpt and self.normalize:
            self._restore_stats_from_ckpt()

        # model --------------------------------------------------------------
        max_len = max(self.past_trajectory, self.future_trajectory)
        d_ff = 4 * self.embedding_size

        enc_in  = self.LinearEmb(self.in_features,  self.embedding_size)
        dec_in  = self.LinearEmb(self.out_features, self.embedding_size)
        pos_enc = self.PosEnc(self.embedding_size, self.dropout_encoder,
                              max_len, self.batch_first)

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
        self.model = self.Seq2Seq(enc_in, pos_enc, encoder,
                                  decoder, dec_in, out_proj).to(self.device)

        opt = optim.Adam(self.model.parameters(),
                         betas=self.optimizer_betas, eps=self.optimizer_eps)
        self.optimizer = ScheduledOptim(opt, self.lr_mul,
                                        self.embedding_size, self.n_warmup_steps)

        if ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.opt.load_state_dict(ckpt["optimizer_state_dict"])

        self.criterion  = nn.MSELoss()
        self.pos_size   = 2
        self.pos_slice  = slice(0, 2)
        self.model_trained = bool(ckpt)


    def _norm(self, t):   # ensure float32
        return ((t - self.mean) / self.std).float() if self.normalize else t

    def _denorm(self, t):
        return (t * self.std + self.mean).float() if self.normalize else t

    # ───── misc ───── #
    @staticmethod
    def _mask(sz, device):
        m = torch.triu(torch.ones(sz, sz, device=device), 1)
        return m.masked_fill(m == 1, float("-inf"))

    def _vel_to_pos(self, last_xy, vel_seq):
        vel_xy = self._denorm(vel_seq)[:, :, :self.pos_size]
        out = torch.zeros_like(vel_xy)
        out[:, 0] = last_xy + vel_xy[:, 0]
        for t in range(1, vel_xy.size(1)):
            out[:, t] = out[:, t-1] + vel_xy[:, t]
        return out

    # ───── greedy decode ───── #
    def _greedy(self, src_vn, steps):
        B = src_vn.size(0)
        mem = self.model.enc(self.model.pos(self.model.enc_in(src_vn)))
        ys = torch.zeros(B, 1, self.out_features, device=self.device)
        outs = []
        for _ in range(steps):
            dec_in = self.model.pos(self.model.dec_in(ys))
            dec_out = self.model.dec(tgt=dec_in, memory=mem,
                                     tgt_mask=self._mask(dec_in.size(1),
                                                         self.device))
            next_tok = self.model.out(dec_out[:, -1:, :])
            outs.append(next_tok)
            ys = torch.cat([ys, next_tok.detach()], dim=1)
        return torch.cat(outs, dim=1)

    # ───── train ───── #
    def train(self, train_loader, valid_loader=None, save_path=None):
        if self.normalize:
            self.mean = train_loader.dataset.mean[self.pos_size:self.pos_size+self.in_features] if self.mean == None else self.mean
            self.std = train_loader.dataset.std[self.pos_size:self.pos_size+self.in_features] if self.std == None else self.std
            self.mean, self.std = self.mean.to(self.device), self.std.to(self.device)    

        best, pat = float("inf"), 0
        for ep in range(1, self.num_epochs + 1):
            self.model.train()
            run_l = run_a = run_f = 0.
            bar = tqdm(train_loader, desc=f"Ep {ep}/{self.num_epochs}", leave=False)
            
            for _, obs, tgt in bar:
                obs, tgt = obs.to(self.device), tgt.to(self.device)

                src_v = obs[:, :, self.pos_size:self.pos_size+self.in_features]
                tgt_v = tgt[:, :, self.pos_size:self.pos_size+self.out_features]

                if self.normalize:
                    src_v = self._norm(src_v)
                    tgt_v = self._norm(tgt_v)
                
                tgt_in = torch.zeros_like(tgt_v) 
                tgt_in[:, 1:] = tgt_v[:, :-1]

                self.optimizer.zero_grad()
                pred = self.model(src_v, tgt_in, self._mask(tgt_in.size(1), self.device))
                loss = self.criterion(pred, tgt_v) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step_and_update_lr()
                run_l += loss.item()

                with torch.no_grad():
                    pred_g = self._greedy(src_v, tgt_v.size(1))
                    if self.normalize == True:
                        pred_g = self._denorm(pred_g)
                    last   = obs[:, -1, self.pos_slice]
                    ade = calculate_ade(self._vel_to_pos(last, pred_g), tgt[:, :, self.pos_slice])
                    fde = calculate_fde(self._vel_to_pos(last, pred_g), tgt[:, :, self.pos_slice])
                    run_a += ade; run_f += fde
                bar.set_postfix(Loss=f"{loss.item():.4f}",
                                ADE=f"{ade:.3f}", FDE=f"{fde:.3f}")

            n = len(train_loader)
            print(f"\nEp {ep}: Loss {run_l/n:.4f}  ADE {run_a/n:.3f}  FDE {run_f/n:.3f}")

            if valid_loader:
                val = self.evaluate(valid_loader, silent=True)[0]
                if val < best - self.early_stopping_delta:
                    best, pat = val, 0
                    if save_path: self.save_checkpoint(save_path)
                else:
                    pat += 1
                if pat >= self.early_stopping_patience:
                    print("Early stopping"); break
            elif save_path and ep == self.num_epochs:
                self.save_checkpoint(save_path)
        self.model_trained = True

    def validate(self, loader): return self.evaluate(loader, silent=True)[0]

    # ───── evaluate ───── #
    def evaluate(self, loader, silent=False):
        self.model.eval(); ade = fde = 0.
        with torch.no_grad():
            for _, obs, tgt in loader:
                obs, tgt = obs.to(self.device), tgt.to(self.device)
                src_vn = self._norm(obs[:, :, self.pos_size:self.pos_size+self.in_features])
                pred = self._greedy(src_vn, tgt.size(1))
                if self.normalize == True:
                    pred = self._denorm(pred) 
                last = obs[:, -1, self.pos_slice]
                ade += calculate_ade(self._vel_to_pos(last, pred), tgt[:, :, self.pos_slice])
                fde += calculate_fde(self._vel_to_pos(last, pred), tgt[:, :, self.pos_slice])
        ade /= len(loader); fde /= len(loader)
        if not silent:
            print(f"Eval ADE {ade:.4f} FDE {fde:.4f}")
        return ade, fde

    # ───── predict ───── #
    def predict(self, trajs: List[np.ndarray], horizon: float):
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")
        self.model.eval(); steps = int(round(horizon * self.trained_fps))

        src_b, last_b = [], []
        for tr in trajs:
            t = torch.tensor(tr, dtype=torch.float32, device=self.device)
            last_b.append(t[-1, self.pos_slice])
            vel = t[1:] - t[:-1]
            pad = self.past_trajectory - 1 - vel.size(0)
            if pad > 0:
                vel = torch.cat([torch.zeros(pad, self.in_features, device=self.device), vel])
            src_b.append(vel[:, self.pos_size:self.pos_size+self.in_features])
        src_b = torch.stack(src_b); last_b = torch.stack(last_b)

        with torch.no_grad():
            pred_n = self._greedy(self._norm(src_b), steps)

        preds = []
        for i in range(len(trajs)):
            pos = self._vel_to_pos(last_b[i:i+1], pred_n[i:i+1])[0]
            preds.append(pos.cpu().numpy())
        return preds

    # ───── checkpoint ───── #
    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {k: getattr(self, k) for k in self.params}
        if self.normalize:
            ckpt["normalize"] = True
            ckpt["mean_full"] = self.mean.cpu().numpy()
            ckpt["std_full"]  = self.std.cpu().numpy()
        ckpt["model_state_dict"] = self.model.state_dict()
        ckpt["optimizer_state_dict"] = self.optimizer.opt.state_dict()
        torch.save(ckpt, path); print("Saved model to", path)
