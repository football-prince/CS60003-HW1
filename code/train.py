from __future__ import annotations

import argparse

from config import Config
from trainer import train_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-layer NumPy MLP on EuroSAT_RGB.")
    parser.add_argument("--image-size", type=int, default=32, help="Input size used by the MLP after resizing raw 64x64 EuroSAT images.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--hidden-dim2", type=int, default=128)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        activation=args.activation,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    result = train_once(cfg, run_name=args.run_name)
    print(f"Best checkpoint saved to: {result['checkpoint_path']}")
    print(f"Training history saved to: {result['history_path']}")


if __name__ == "__main__":
    main()
