from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from trainer import train_once
from utils import ensure_dir, save_csv, save_json, timestamp_name


def parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def configure_search_space(
    cfg: Config,
    learning_rates: list[float] | None = None,
    hidden_dims: list[int] | None = None,
    weight_decays: list[float] | None = None,
    activations: list[str] | None = None,
) -> Config:
    space = dict(cfg.search_space)
    if learning_rates is not None:
        space["learning_rate"] = learning_rates
    if hidden_dims is not None:
        space["hidden_dim"] = hidden_dims
    if weight_decays is not None:
        space["weight_decay"] = weight_decays
    if activations is not None:
        space["activation"] = activations
    return replace(cfg, search_space=space)


def generate_search_candidates(cfg: Config) -> list[dict[str, Any]]:
    space = cfg.search_space
    if cfg.search_mode == "grid":
        keys = ["learning_rate", "hidden_dim", "weight_decay", "activation"]
        values = [space[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    rng = np.random.default_rng(cfg.seed)
    candidates = []
    for _ in range(cfg.search_max_trials):
        candidates.append(
            {
                "learning_rate": float(rng.choice(space["learning_rate"])),
                "hidden_dim": int(rng.choice(space["hidden_dim"])),
                "weight_decay": float(rng.choice(space["weight_decay"])),
                "activation": str(rng.choice(space["activation"])),
            }
        )
    return candidates


def summarize_grouped_results(results: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[float]] = {}
    for row in results:
        grouped.setdefault(row[field], []).append(float(row["best_val_accuracy"]))

    summary_rows = []
    for key, values in grouped.items():
        summary_rows.append(
            {
                field: key,
                "num_trials": len(values),
                "mean_best_val_accuracy": float(np.mean(values)),
                "std_best_val_accuracy": float(np.std(values)),
                "max_best_val_accuracy": float(np.max(values)),
                "min_best_val_accuracy": float(np.min(values)),
            }
        )

    summary_rows.sort(key=lambda row: row["mean_best_val_accuracy"], reverse=True)
    return summary_rows


def build_heatmap_matrix(
    results: list[dict[str, Any]],
    x_field: str,
    y_field: str,
) -> tuple[np.ndarray, list[Any], list[Any]]:
    x_values = sorted({row[x_field] for row in results})
    y_values = sorted({row[y_field] for row in results})
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)

    for yi, y_val in enumerate(y_values):
        for xi, x_val in enumerate(x_values):
            matches = [
                float(row["best_val_accuracy"])
                for row in results
                if row[x_field] == x_val and row[y_field] == y_val
            ]
            if matches:
                matrix[yi, xi] = float(np.mean(matches))
    return matrix, x_values, y_values


def format_float_for_label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 0.01:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text
    return f"{value:.0e}"


def pad_label_text(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    if len(text) >= width:
        return text
    if align == "right":
        return " " * (width - len(text)) + text
    if align == "center":
        left = (width - len(text)) // 2
        right = width - len(text) - left
        return " " * left + text + " " * right
    return text + " " * (width - len(text))


def format_config_label(row: dict[str, Any]) -> str:
    lr = pad_label_text(format_float_for_label(float(row["learning_rate"])), width=5, align="right")
    hidden = pad_label_text(f"{int(row['hidden_dim1'])}/{int(row['hidden_dim2'])}", width=7, align="right")
    wd = pad_label_text(format_float_for_label(float(row["weight_decay"])), width=5, align="right")
    act = pad_label_text(str(row["activation"]), width=7, align="left")
    return f"lr={lr} | h={hidden} | wd={wd} | {act}"


def plot_search_results(results: list[dict[str, Any]], save_path: Path) -> None:
    ensure_dir(save_path.parent)
    sorted_results = sorted(results, key=lambda row: row["best_val_accuracy"], reverse=True)
    labels = [format_config_label(row) for row in sorted_results]
    scores = [row["best_val_accuracy"] for row in sorted_results]

    fig_height = max(5.5, len(labels) * 0.38)
    plt.figure(figsize=(12.5, fig_height))
    y_pos = np.arange(len(labels))
    bars = plt.barh(y_pos, scores, color="tab:blue", alpha=0.88)
    plt.yticks(y_pos, labels, fontsize=8, fontfamily="monospace")
    plt.xlabel("Best Validation Accuracy")
    plt.ylabel("Hyperparameter Configuration")
    plt.title("Hyperparameter Search Results")
    plt.grid(True, axis="x", alpha=0.3)
    plt.xlim(0, min(1.0, max(scores) + 0.05))
    plt.gca().invert_yaxis()

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_width() + 0.004,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_grouped_summary(
    summary_rows: list[dict[str, Any]],
    field: str,
    save_path: Path,
) -> None:
    ensure_dir(save_path.parent)
    labels = [str(row[field]) for row in summary_rows]
    means = [row["mean_best_val_accuracy"] for row in summary_rows]
    stds = [row["std_best_val_accuracy"] for row in summary_rows]

    plt.figure(figsize=(max(6, len(labels) * 1.4), 4.2))
    plt.bar(np.arange(len(labels)), means, yerr=stds, capsize=4, color="tab:green", alpha=0.85)
    plt.xticks(np.arange(len(labels)), labels, fontsize=9)
    plt.xlabel(field.replace("_", " ").title())
    plt.ylabel("Mean Best Validation Accuracy")
    plt.title(f"Grouped Search Summary by {field}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    x_values: list[Any],
    y_values: list[Any],
    x_field: str,
    y_field: str,
    save_path: Path,
) -> None:
    ensure_dir(save_path.parent)
    plt.figure(figsize=(max(5, len(x_values) * 1.2), max(4, len(y_values) * 0.9)))
    im = plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Mean Best Validation Accuracy")
    plt.xticks(np.arange(len(x_values)), [str(v) for v in x_values])
    plt.yticks(np.arange(len(y_values)), [str(v) for v in y_values])
    plt.xlabel(x_field.replace("_", " ").title())
    plt.ylabel(y_field.replace("_", " ").title())
    plt.title(f"{y_field.replace('_', ' ').title()} vs {x_field.replace('_', ' ').title()}")

    for yi in range(len(y_values)):
        for xi in range(len(x_values)):
            if np.isfinite(matrix[yi, xi]):
                plt.text(xi, yi, f"{matrix[yi, xi]:.3f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_search(
    cfg: Config,
    search_name: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    candidates = generate_search_candidates(cfg)
    results: list[dict[str, Any]] = []
    best_result = None
    search_name = search_name or timestamp_name("search")

    print(f"Search mode: {cfg.search_mode}")
    print(f"Number of candidate trials: {len(candidates)}")
    print(f"Search space: {cfg.search_space}")

    for idx, candidate in enumerate(candidates, start=1):
        run_name = f"{search_name}_trial_{idx:03d}"
        hidden_dim1 = int(candidate["hidden_dim"])
        hidden_dim2 = max(hidden_dim1 // 2, 32)
        trial_cfg = replace(
            cfg,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            activation=str(candidate["activation"]),
        )

        print(f"=== Search trial {idx}/{len(candidates)}: {candidate} ===")
        train_result = train_once(trial_cfg, run_name=run_name)
        history = train_result["history"]
        row = {
            "trial": idx,
            "run_name": run_name,
            "learning_rate": float(candidate["learning_rate"]),
            "hidden_dim1": hidden_dim1,
            "hidden_dim2": hidden_dim2,
            "weight_decay": float(candidate["weight_decay"]),
            "activation": str(candidate["activation"]),
            "best_val_accuracy": float(history["best_val_accuracy"]),
            "best_epoch": int(history["best_epoch"]),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_val_accuracy": float(history["val_accuracy"][-1]),
            "checkpoint_path": train_result["checkpoint_path"],
            "history_path": train_result["history_path"],
        }
        results.append(row)

        if best_result is None or row["best_val_accuracy"] > best_result["best_val_accuracy"]:
            best_result = row

    sorted_results = sorted(results, key=lambda row: row["best_val_accuracy"], reverse=True)
    top_results = sorted_results[: min(top_k, len(sorted_results))]

    csv_path = Path(cfg.search_dir) / f"{search_name}_results.csv"
    json_path = Path(cfg.search_dir) / f"{search_name}_summary.json"
    topk_csv_path = Path(cfg.search_dir) / f"{search_name}_top{min(top_k, len(sorted_results))}.csv"
    ranking_fig_path = Path(cfg.search_dir) / f"{search_name}_best_val_accuracy.png"

    save_csv(csv_path, results)
    save_csv(topk_csv_path, top_results)

    grouped_summaries = {}
    for field in ["learning_rate", "hidden_dim1", "weight_decay", "activation"]:
        rows = summarize_grouped_results(results, field)
        grouped_summaries[field] = rows
        save_csv(Path(cfg.search_dir) / f"{search_name}_groupby_{field}.csv", rows)
        plot_grouped_summary(rows, field, Path(cfg.search_dir) / f"{search_name}_groupby_{field}.png")

    lr_hidden_matrix, lr_hidden_x, lr_hidden_y = build_heatmap_matrix(results, "learning_rate", "hidden_dim1")
    plot_heatmap(
        lr_hidden_matrix,
        lr_hidden_x,
        lr_hidden_y,
        "learning_rate",
        "hidden_dim1",
        Path(cfg.search_dir) / f"{search_name}_heatmap_lr_hidden.png",
    )

    wd_hidden_matrix, wd_hidden_x, wd_hidden_y = build_heatmap_matrix(results, "weight_decay", "hidden_dim1")
    plot_heatmap(
        wd_hidden_matrix,
        wd_hidden_x,
        wd_hidden_y,
        "weight_decay",
        "hidden_dim1",
        Path(cfg.search_dir) / f"{search_name}_heatmap_wd_hidden.png",
    )

    plot_search_results(results, ranking_fig_path)

    summary_payload = {
        "search_name": search_name,
        "mode": cfg.search_mode,
        "num_trials": len(results),
        "top_k": min(top_k, len(sorted_results)),
        "search_space": cfg.search_space,
        "best_result": best_result,
        "top_results": top_results,
        "grouped_summaries": grouped_summaries,
        "results": results,
    }
    save_json(json_path, summary_payload)

    return {
        "csv_path": str(csv_path),
        "topk_csv_path": str(topk_csv_path),
        "json_path": str(json_path),
        "best_result": best_result,
        "top_results": top_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for EuroSAT MLP.")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=32, help="Input size used by the MLP after resizing raw 64x64 EuroSAT images.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--search-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--learning-rates", type=str, default=None, help="Comma-separated learning rates.")
    parser.add_argument("--hidden-dims", type=str, default=None, help="Comma-separated hidden dimensions.")
    parser.add_argument("--weight-decays", type=str, default=None, help="Comma-separated weight decays.")
    parser.add_argument("--activations", type=str, default=None, help="Comma-separated activations.")
    args = parser.parse_args()

    cfg = Config(
        search_mode=args.mode,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        search_max_trials=args.max_trials,
    )
    cfg = configure_search_space(
        cfg,
        learning_rates=parse_float_list(args.learning_rates),
        hidden_dims=parse_int_list(args.hidden_dims),
        weight_decays=parse_float_list(args.weight_decays),
        activations=parse_str_list(args.activations),
    )

    result = run_search(cfg, search_name=args.search_name, top_k=args.top_k)
    print("Best hyperparameter combination:")
    print(result["best_result"])
    print(f"Saved full results to: {result['csv_path']}")
    print(f"Saved top-k results to: {result['topk_csv_path']}")
    print(f"Saved search summary to: {result['json_path']}")


if __name__ == "__main__":
    main()
