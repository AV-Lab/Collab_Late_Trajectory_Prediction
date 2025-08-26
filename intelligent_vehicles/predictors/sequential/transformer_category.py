#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer seq-to-seq predictor **with optional category embedding**.

Config additions
----------------
    "category":        True | False
    "num_categories":  <int>   # required if category True
    "cat_embed_dim":   8       # optional (defaults to 8)

Data-loader contract
--------------------
    • category = True  → batches are (cat_idx, obs, tgt)
      cat_idx : int64 [B]            category id per agent
      obs     : float32 [B, Tobs, Din]
      tgt     : float32 [B, Th,   Dout]

    • category = False → batches remain (obs, tgt) unchanged.

The category embedding (proj → model dimension) is **added to every token
AFTER Linear projection**, so shapes always match.
"""

import math, os, sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.distance_metrics import calculate_ade, calculate_fde           # noqa: E402


# ─────────────────────── Noam LR scheduler ────────────────────────── #
class ScheduledOptim:
    def __init__(self, opt, lr_mul, d_model, n_warm):
        self.opt, self.lr_mul, self.d_model, self.warm = opt, lr_mul, d_model, n_warm
        self.n_steps = 0

    def _lr(self):
        s = max(1, self.n_steps)
        return self.lr_mul * self.d_model ** -0.5 * min(s ** -0.5,
                                                        s * self.warm ** -1.5)

    def step_and_update_lr(self):
        self.n_steps += 1
        lr = self._lr()
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()

    def zero_grad(self): self.opt.zero_grad()


# ═══════════════════  Predictor  ════════════════════════════════════ #
class TransformerPredictorWithCategory:

    # ───── helper blocks ───── #
    class LinearEmb(nn.Module):
        def __init__(self, d_in, d_model):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.scale = math.sqrt(d_model)

        def forward(self, x):
            return self.proj(x) * self.scale          # [B,T,d_model]

    class PosEnc(nn.Module):
        def __init__(self, d_model, dropout, max_len=5000, batch_first=True):
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
        """
        Teacher-forcing forward pass.
        Expects **category biases already added**.
        """
        def __init__(self, enc_in, pos, enc, dec, dec_in, out_proj):
            super().__init__()
            self.enc_in, self.dec_in = enc_in, dec_in
            self.pos = pos
            self.enc, self.dec = enc, dec
            self.out = out_proj

        def forward(self, src, tgt_shifted, tgt_mask,
                    cb_enc: torch.Tensor, cb_dec: torch.Tensor):
            mem = self.enc(self.pos(self.enc_in(src) + cb_enc))
            dec_in = self.pos(self.dec_in(tgt_shifted) + cb_dec)
            dec_out = self.dec(tgt=dec_in, memory=mem, tgt_mask=tgt_mask)
            return self.out(dec_out)

    # ───── constructor ───── #
    def __init__(self, cfg: dict):
        keys = ["past_trajectory", "future_trajectory", "in_features", "out_features",
                "num_heads", "num_encoder_layers", "num_decoder_layers",
                "embedding_size", "dropout_encoder", "dropout_decoder",
                "batch_first", "actn", "lr_mul", "n_warmup_steps",
                "optimizer_betas", "optimizer_eps", "num_epochs",
                "trained_fps", "early_stopping_patience", "early_stopping_delta",
                "category", "num_categories", "cat_embed_dim"]
        self.params = keys

        self.device = torch.device(cfg["device"])
        ckpt = torch.load(cfg["checkpoint"], map_location=self.device) \
               if cfg.get("checkpoint") else None
        src = ckpt if ckpt else cfg
        for k in keys:
            setattr(self, k, src.get(k))

        self.category = bool(getattr(self, "category", False))
        self.cat_embed_dim = getattr(self, "cat_embed_dim", 8)

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

        self.model = self.Seq2Seq(enc_in, pos_enc, encoder, decoder, dec_in, out_proj).to(self.device)

        # category embedding → model dim
        self.cat_emb  = nn.Embedding(self.num_categories, self.cat_embed_dim).to(self.device)
        self.cat_proj = nn.Linear(self.cat_embed_dim, self.embedding_size).to(self.device)

        base_opt = optim.Adam(self.model.parameters(), betas=self.optimizer_betas, eps=self.optimizer_eps)
        self.optimizer = ScheduledOptim(base_opt, self.lr_mul, self.embedding_size, self.n_warmup_steps)
        if ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.opt.load_state_dict(ckpt["optimizer_state_dict"])

        self.criterion = nn.MSELoss()
        self.pos_size  = 2
        self.pos_slice = slice(0, 2)
        self.model_trained = bool(ckpt)

    # ───── helpers ───── #
    @staticmethod
    def _causal_mask(sz, device):
        m = torch.triu(torch.ones(sz, sz, device=device), 1)
        return m.masked_fill(m == 1, float("-inf"))

    @staticmethod
    def _vel_to_pos(last_xy, vel_seq, pos_size):
        vel = vel_seq[:, :, :pos_size]
        out = torch.zeros_like(vel)
        out[:, 0] = last_xy + vel[:, 0]
        for t in range(1, vel.size(1)):
            out[:, t] = out[:, t-1] + vel[:, t]
        return out

    def _cat_bias(self, cat_idx: torch.Tensor, seq_len: int, batch: int) -> torch.Tensor:
        if self.category and cat_idx is not None:
            emb = self.cat_proj(self.cat_emb(cat_idx))      # [B,D]
            return emb.unsqueeze(1).expand(-1, seq_len, -1) # [B,T,D]
        return torch.zeros(batch, seq_len,
                           self.embedding_size, device=self.device)

    def _greedy(self, src_vel, steps, cb_enc):
        """
        src_vel : [B, Tenc, Din]   (already raw velocities)
        cb_enc  : [B, Tenc, Dmodel] category bias for encoder
        """
        B = src_vel.size(0)
    
        # encoder
        mem = self.model.enc(
            self.model.pos(self.model.enc_in(src_vel) + cb_enc)
        )
    
        # decoder setup
        ys = torch.zeros(B, 1, self.out_features, device=self.device)
        cat_vec = cb_enc[:, 0, :].unsqueeze(1)              # [B,1,Dmodel]
        outs = []
    
        for _ in range(steps):
            cb_step = cat_vec.expand(-1, ys.size(1), -1)    # [B, t, Dmodel]
            dec_in = self.model.pos(self.model.dec_in(ys) + cb_step)
            dec_out = self.model.dec(
                tgt=dec_in,
                memory=mem,
                tgt_mask=self._causal_mask(dec_in.size(1), self.device)
            )
            next_tok = self.model.out(dec_out[:, -1:, :])   # [B,1,Dout]
            outs.append(next_tok)
            ys = torch.cat([ys, next_tok.detach()], dim=1)  # grow sequence
    
        return torch.cat(outs, dim=1)                       # [B, steps, Dout]


    # ───── training ───── #
    def train(self, train_loader, valid_loader=None, save_path=None):
        best, pat = float("inf"), 0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train(); run_l = run_a = run_f = 0.
            bar = tqdm(train_loader, desc=f"Ep {epoch}/{self.num_epochs}", leave=False)
            for batch in bar:
                cat_idx, obs, tgt = batch
                cat_idx = cat_idx.to(self.device)
                obs, tgt = obs.to(self.device), tgt.to(self.device)

                src_vel = obs[:, :, self.pos_size:self.pos_size+self.in_features]
                tgt_vel = tgt[:, :, self.pos_size:self.pos_size+self.out_features]

                B = src_vel.size(0)
                cb_enc = self._cat_bias(cat_idx, src_vel.size(1), B)
                cb_dec = self._cat_bias(cat_idx, tgt_vel.size(1), B)

                tgt_in = torch.zeros_like(tgt_vel)
                tgt_in[:, 1:] = tgt_vel[:, :-1]
                self.optimizer.zero_grad()
                pred_tf = self.model(src_vel, tgt_in,
                                     self._causal_mask(tgt_in.size(1), self.device),
                                     cb_enc, cb_dec)
                loss = self.criterion(pred_tf, tgt_vel); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step_and_update_lr(); run_l += loss.item()

                pred = self._greedy(src_vel, tgt_vel.size(1), cb_enc)
                last = obs[:, -1, self.pos_slice]
                ade = calculate_ade(self._vel_to_pos(last, pred, self.pos_size),
                                    tgt[:, :, self.pos_slice])
                fde = calculate_fde(self._vel_to_pos(last, pred, self.pos_size),
                                    tgt[:, :, self.pos_slice])
                run_a += ade; run_f += fde
                bar.set_postfix(Loss=f"{loss.item():.4f}",
                                ADE=f"{ade:.3f}", FDE=f"{fde:.3f}")

            n = len(train_loader)
            print(f"\nEpoch {epoch}: Loss {run_l/n:.4f} ADE {run_a/n:.3f} FDE {run_f/n:.3f}")

            if valid_loader:
                val = self.evaluate(valid_loader, silent=True)[0]
                if val < best - self.early_stopping_delta:
                    best, pat = val, 0
                    if save_path: self.save_checkpoint(save_path)
                else:
                    pat += 1
                if pat >= self.early_stopping_patience:
                    print("Early stopping"); break
            elif save_path and epoch == self.num_epochs:
                self.save_checkpoint(save_path)

        self.model_trained = True

    # validate alias
    def validate(self, loader): return self.evaluate(loader, silent=True)[0]

    # ───── evaluation ───── #
    def evaluate(self, loader, silent=False):
        self.model.eval(); ade = fde = 0.
        with torch.no_grad():
            for batch in loader:
                cat_idx, obs, tgt = batch
                cat_idx = cat_idx.to(self.device)
                obs, tgt = obs.to(self.device), tgt.to(self.device)

                src_vel = obs[:, :, self.pos_size:self.pos_size+self.in_features]
                cb_enc = self._cat_bias(cat_idx, src_vel.size(1), src_vel.size(0))
                pred = self._greedy(src_vel, tgt.size(1), cb_enc)
                last = obs[:, -1, self.pos_slice]
                ade += calculate_ade(self._vel_to_pos(last, pred, self.pos_size),
                                     tgt[:, :, self.pos_slice])
                fde += calculate_fde(self._vel_to_pos(last, pred, self.pos_size),
                                     tgt[:, :, self.pos_slice])
        ade /= len(loader); fde /= len(loader)
        if not silent:
            print(f"Eval ADE {ade:.4f} FDE {fde:.4f}")
        return ade, fde

    # ───── predict ───── #
    def predict(self, trajs: List[np.ndarray], prediction_horizon: float, categories= None):
        if not self.model_trained:
            raise RuntimeError("Model not trained / loaded.")
        self.model.eval(); steps = int(round(prediction_horizon * self.trained_fps))

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

        cat_idx = torch.tensor(categories, dtype=torch.long, device=self.device)
        cb_enc = self._cat_bias(cat_idx, src_b.size(1), src_b.size(0))

        with torch.no_grad():
            pred_v = self._greedy(src_b, steps, cb_enc)

        preds = []
        for i in range(len(trajs)):
            pos = self._vel_to_pos(last_b[i:i+1], pred_v[i:i+1], self.pos_size)[0]
            preds.append(pos.cpu().numpy())
        return preds

    # ───── checkpoint ───── #
    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {k: getattr(self, k) for k in self.params}
        ckpt["model_state_dict"] = self.model.state_dict()
        ckpt["optimizer_state_dict"] = self.optimizer.opt.state_dict()
        torch.save(ckpt, path); print("Saved model to", path)
