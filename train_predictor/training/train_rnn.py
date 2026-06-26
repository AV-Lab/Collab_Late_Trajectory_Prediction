from intelligent_vehicles.predictors.sequential.rnn_nll import RNNPredictorNLL
from intelligent_vehicles.predictors.sequential.rnn_bivariate_nll import RNNPredictorBivariateNLL
from intelligent_vehicles.predictors.sequential.rnn import RNNPredictor
from intelligent_vehicles.predictors.dataloaders.seq_loader import SeqDataset
from torch.utils.data import DataLoader
import argparse
import os
import re
from pathlib import Path


    
def parse_LHSF_from_path(data_path):
    """Parse L, H, S, F integers from a folder name in the given path."""
    pat = re.compile(r"(?P<prefix>.*?)(?:_|^)L(?P<L>\d+)_H(?P<H>\d+)_S(?P<S>\d+)_F(?P<F>\d+)")
    for part in Path(data_path).parts:
        m = pat.fullmatch(part)
        if m:
            return {k: v for k, v in m.groupdict().items()}
    raise ValueError(
        f"Could not find folder of format L##_H##_S##_F## in path: {data_path}"
    )

if __name__ == "__main__":
        
    ap = argparse.ArgumentParser(description="Training predictor.")
    ap.add_argument("data_path", help="Path to train data folder.")
    ap.add_argument("checkpoint_path", help="Path to checkpoints folder.")
    ap.add_argument("--model", choices=['lstm_nll', 'lstm', 'lstm_bivariate_nll'], default='lstm_nll', help='Training Model')
    ap.add_argument("--device", type=str, default='cuda:1', help='Training device')
    ap.add_argument("--hidden-size", type=int, default=128, help="Hidden size of the model.")
    ap.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers.")
    ap.add_argument("--input-size", type=int, default=2, help="Model input feature size.")
    ap.add_argument("--output-size", type=int, default=2, help="Model output size.")
    ap.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs.")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    ap.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    ap.add_argument("--normalize", action="store_true", help="Apply normalization to input data.")
    args = ap.parse_args()

    data_path = args.data_path
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Provided path does not exist or is not a directory: {data_path}")
    if not os.path.isdir(args.checkpoint_path):
        raise FileNotFoundError(f"Provided checkpoint path does not exist or is not a directory: {args.checkpoint_path}")

    # Parse L, H, S, F values from the folder name
    lhsf = parse_LHSF_from_path(data_path)

    # Build the configuration dictionary
    prediction_config = {
        "data_path": os.path.abspath(data_path),
        "observation_length": lhsf["L"],
        "prediction_horizon": lhsf["H"],
        "trained_fps": lhsf["F"],
        "hidden_size": args.hidden_size,
        "device": args.device,
        "num_layers": args.num_layers,
        "input_size": args.input_size,
        "output_size": args.output_size,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "normalize": args.normalize,
    }

    print("\nResolved training configuration:")
    for k, v in prediction_config.items():
        print(f"{k}: {v}")
      
    train_path = os.path.join(data_path, "train.pkl")
    valid_path = os.path.join(data_path, "valid.pkl")
    test_path  = os.path.join(data_path, "test.pkl")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"train.pkl is missing in: {ap.data_path}")
    
    train_loader = DataLoader(SeqDataset(train_path), batch_size=args.batch_size, shuffle=True)
    test_loader = None
    valid_loader = None
    
    if test_path and os.path.isfile(test_path):
        test_loader = DataLoader(SeqDataset(test_path), batch_size=args.batch_size, shuffle=False)
        
    if os.path.isfile(valid_path):
        if test_loader is None:
            test_loader = DataLoader(SeqDataset(valid_path), batch_size=args.batch_size, shuffle=False)
        else:
            valid_loader = DataLoader(SeqDataset(valid_path), batch_size=args.batch_size, shuffle=True)
    
    save_path = os.path.join(args.checkpoint_path, f"{lhsf['prefix']}_{args.model}_{int(lhsf['L'])}_{int(lhsf['H'])}_{int(lhsf['F'])}.pth")
    
    if args.model == "lstm_nll":
        predictor = RNNPredictorNLL(prediction_config)
    elif args.model == "lstm_bivariate_nll":
        predictor = RNNPredictorBivariateNLL(prediction_config)
    else:
        predictor = RNNPredictor(prediction_config)
        
    predictor.train(train_loader, valid_loader, save_path)
    predictor.evaluate(test_loader)