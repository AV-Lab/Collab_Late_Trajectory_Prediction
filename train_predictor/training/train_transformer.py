#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from intelligent_vehicles.predictors.sequential.transformer_nll import TransformerPredictorNLL
from intelligent_vehicles.predictors.dataloaders.seq_loader import SeqDataset
from torch.utils.data import DataLoader
import argparse
import os
import re
from pathlib import Path


def parse_LHSF_from_path(data_path):
    """Parse prefix, L, H, S, F from a folder name in the given path."""
    pat = re.compile(r"(?P<prefix>.*?)(?:_|^)L(?P<L>\d+)_H(?P<H>\d+)_S(?P<S>\d+)_F(?P<F>\d+)")
    for part in Path(data_path).parts:
        m = pat.fullmatch(part)
        if m:
            return {k: v for k, v in m.groupdict().items()}
    raise ValueError(
        f"Could not find folder of format L##_H##_S##_F## in path: {data_path}"
    )


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Training Transformer predictor.")
    ap.add_argument("data_path", help="Path to train data folder.")
    ap.add_argument("checkpoint_path", help="Path to checkpoints folder.")
    ap.add_argument("--device", type=str, default="cuda:0", help="Training device")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    ap.add_argument("--in_features", type=int, default=2)
    ap.add_argument("--out_features", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_encoder_layers", type=int, default=3)
    ap.add_argument("--num_decoder_layers", type=int, default=3)
    ap.add_argument("--embedding_size", type=int, default=256)
    ap.add_argument("--dropout_encoder", type=float, default=0.25)
    ap.add_argument("--dropout_decoder", type=float, default=0.25)
    ap.add_argument("--actn", type=str, default="relu", help="Activation function")
    ap.add_argument("--cat_embed_dim", type=int, default=8)
    ap.add_argument("--num_categories", type=int, default=6)
    ap.add_argument("--lr_mul", type=float, default=0.2)
    ap.add_argument("--n_warmup_steps", type=int, default=3500)
    ap.add_argument("--optimizer_beta1", type=float, default=0.9)
    ap.add_argument("--optimizer_beta2", type=float, default=0.98)
    ap.add_argument("--optimizer_eps", type=float, default=1e-9)
    ap.add_argument("--early_stopping_patience", type=int, default=20)
    ap.add_argument("--early_stopping_delta", type=float, default=0.01)
    args = ap.parse_args()

    data_path = args.data_path
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Provided path does not exist or is not a directory: {data_path}")
    if not os.path.isdir(args.checkpoint_path):
        raise FileNotFoundError(f"Provided checkpoint path does not exist or is not a directory: {args.checkpoint_path}")

    # Parse L, H, S, F values from the folder name
    lhsf = parse_LHSF_from_path(data_path)
    prediction_config = {}
    
    prediction_config["data_path"] = os.path.abspath(args.data_path)

    # L, H, F parsed from folder
    prediction_config["past_trajectory"] = int(lhsf["L"])
    prediction_config["future_trajectory"] = int(lhsf["H"])
    prediction_config["trained_fps"] = int(lhsf["F"])
    
    # Transformer architecture
    prediction_config["in_features"] = args.in_features
    prediction_config["out_features"] = args.out_features
    prediction_config["num_heads"] = args.num_heads
    prediction_config["num_encoder_layers"] = args.num_encoder_layers
    prediction_config["num_decoder_layers"] = args.num_decoder_layers
    prediction_config["embedding_size"] = args.embedding_size
    prediction_config["dropout_encoder"] = args.dropout_encoder
    prediction_config["dropout_decoder"] = args.dropout_decoder
    prediction_config["batch_first"] = True
    prediction_config["actn"] = args.actn
    
    # Extra categorical embedding settings
    prediction_config["cat_embed_dim"] = args.cat_embed_dim
    prediction_config["num_categories"] = args.num_categories
    
    # Training settings
    prediction_config["num_epochs"] = args.num_epochs
    prediction_config["normalize"] = False
    prediction_config["checkpoint"] = None
    prediction_config["device"] = args.device
    
    # Optimizer
    prediction_config["lr_mul"] = args.lr_mul
    prediction_config["n_warmup_steps"] = args.n_warmup_steps
    prediction_config["optimizer_betas"] = (args.optimizer_beta1, args.optimizer_beta2)
    prediction_config["optimizer_eps"] = args.optimizer_eps
    
    # Early stopping
    prediction_config["early_stopping_patience"] = args.early_stopping_patience
    prediction_config["early_stopping_delta"] = args.early_stopping_delta

    print("\nResolved Transformer training configuration:")
    for k, v in prediction_config.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_path = os.path.join(data_path, "train.pkl")
    valid_path = os.path.join(data_path, "valid.pkl")
    test_path  = os.path.join(data_path, "test.pkl")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"train.pkl is missing in: {data_path}")

    train_loader = DataLoader(SeqDataset(train_path), batch_size=args.batch_size, shuffle=True)
    test_loader = None
    valid_loader = None

    if test_path and os.path.isfile(test_path):
        test_loader = DataLoader(SeqDataset(test_path), batch_size=args.batch_size, shuffle=False)

    if valid_path and os.path.isfile(valid_path):
        if test_loader is None:
            test_loader = DataLoader(SeqDataset(valid_path), batch_size=args.batch_size, shuffle=False)
        else:
            valid_loader = DataLoader(SeqDataset(valid_path), batch_size=args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Train + evaluate
    # ------------------------------------------------------------------
    save_path = os.path.join(
        args.checkpoint_path,
        f"{lhsf['prefix']}_transformer_nll_{int(lhsf['L'])}_{int(lhsf['H'])}_{int(lhsf['F'])}.pth"
    )

    predictor = TransformerPredictorNLL(prediction_config)
    predictor.train(train_loader, valid_loader, save_path)

    if test_loader is not None:
        predictor.evaluate(test_loader)
