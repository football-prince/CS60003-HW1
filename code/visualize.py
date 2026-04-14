from __future__ import annotations

import argparse
from math import ceil, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from utils import ensure_dir, load_checkpoint, load_json


def plot_training_curves(history: dict, save_dir: Path, run_name: str) -> None:
    ensure_dir(save_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["val_loss"], label="val loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["val_accuracy"], label="val acc", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_curves.png", dpi=200)
    plt.close()


def visualize_first_layer_weights(
    weight_matrix: np.ndarray,
    image_shape: tuple[int, int, int],
    save_dir: Path,
    run_name: str,
    max_filters: int = 64,
) -> None:
    ensure_dir(save_dir)
    num_filters = min(weight_matrix.shape[1], max_filters)
    side = ceil(sqrt(num_filters))
    h, w, c = image_shape

    fig, axes = plt.subplots(side, side, figsize=(side * 2.2, side * 2.2))
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= num_filters:
            continue
        filt = weight_matrix[:, idx].reshape(h, w, c)
        filt = filt - filt.min()
        filt = filt / (filt.max() + 1e-8)
        ax.imshow(filt)
        ax.set_title(f"n{idx}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_first_layer_weights.png", dpi=200)
    plt.close(fig)


def visualize_misclassified_samples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    run_name: str,
    max_samples: int = 24,
) -> None:
    ensure_dir(save_dir)
    wrong_idx = np.where(y_true != y_pred)[0][:max_samples]
    if len(wrong_idx) == 0:
        return

    cols = 4
    rows = ceil(len(wrong_idx) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_2d(axes)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= len(wrong_idx):
            continue
        sample_idx = wrong_idx[idx]
        ax.imshow(np.clip(images[sample_idx], 0.0, 1.0))
        true_name = class_names[int(y_true[sample_idx])]
        pred_name = class_names[int(y_pred[sample_idx])]
        ax.set_title(f"T:{true_name}\nP:{pred_name}", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_misclassified.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_dir: Path, run_name: str) -> None:
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots from saved training/test artifacts.")
    parser.add_argument("--history", type=str, help="Path to history json file.")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint npz file.")
    parser.add_argument("--test-summary", type=str, help="Path to test summary json file.")
    parser.add_argument("--run-name", type=str, default="manual")
    args = parser.parse_args()

    cfg = Config()
    if args.history:
        history = load_json(Path(args.history))
        plot_training_curves(history, cfg.curves_dir, args.run_name)

    if args.checkpoint:
        state, metadata = load_checkpoint(Path(args.checkpoint))
        image_shape = tuple(metadata["image_shape"])
        visualize_first_layer_weights(state["fc1_W"], image_shape, cfg.weights_dir, args.run_name)

    if args.test_summary:
        summary = load_json(Path(args.test_summary))
        cm = np.asarray(summary["confusion_matrix"])
        class_names = list(summary["class_accuracy"].keys())
        plot_confusion_matrix(cm, class_names, cfg.errors_dir, args.run_name)


if __name__ == "__main__":
    main()
