from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import Config
from data_utils import prepare_data
from evaluate import evaluate_split, summarize_test_results
from model import ThreeLayerMLP
from utils import load_checkpoint, timestamp_name
from visualize import plot_confusion_matrix, visualize_misclassified_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the EuroSAT test split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    state_dict, metadata = load_checkpoint(checkpoint_path)

    cfg = Config(
        seed=int(metadata["config"]["seed"]),
        image_size=int(metadata["config"]["image_size"]),
        batch_size=args.batch_size,
        train_ratio=float(metadata["config"]["train_ratio"]),
        val_ratio=float(metadata["config"]["val_ratio"]),
        test_ratio=float(metadata["config"]["test_ratio"]),
        normalize=bool(metadata["config"]["normalize"]),
    )
    prepared = prepare_data(cfg)
    splits = prepared["splits"]
    class_names = metadata["class_names"]

    model = ThreeLayerMLP(
        input_dim=int(metadata["input_dim"]),
        hidden_dim1=int(metadata["hidden_dim1"]),
        hidden_dim2=int(metadata["hidden_dim2"]),
        num_classes=int(metadata["num_classes"]),
        activation=str(metadata["activation"]),
        seed=cfg.seed,
    )
    model.load_state_dict(state_dict)

    test_metrics = evaluate_split(model, splits["test"].X, splits["test"].y, batch_size=args.batch_size)
    run_name = args.run_name or timestamp_name("test")
    summary_path = Path(cfg.errors_dir) / f"{run_name}_summary.json"
    summary = summarize_test_results(
        splits["test"].y,
        test_metrics["predictions"],
        class_names,
        summary_path,
    )

    cm = np.asarray(summary["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, Path(cfg.errors_dir), run_name)
    visualize_misclassified_samples(
        splits["test"].images,
        splits["test"].y,
        test_metrics["predictions"],
        class_names,
        Path(cfg.errors_dir),
        run_name,
        max_samples=cfg.num_error_samples,
    )

    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Saved test summary to: {summary_path}")


if __name__ == "__main__":
    main()
