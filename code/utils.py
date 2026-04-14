from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs(cfg: Any) -> None:
    for path in [
        cfg.results_dir,
        cfg.curves_dir,
        cfg.weights_dir,
        cfg.errors_dir,
        cfg.search_dir,
        cfg.checkpoints_dir,
    ]:
        ensure_dir(Path(path))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def serialize_config(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        data = asdict(cfg)
    else:
        data = dict(cfg)
    return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


def save_checkpoint(
    path: Path,
    model_state: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    arrays = {name: value.astype(np.float32) for name, value in model_state.items()}
    arrays["__metadata__"] = np.array(json.dumps(metadata), dtype=object)
    np.savez(path, **arrays)


def load_checkpoint(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    ckpt = np.load(path, allow_pickle=True)
    metadata = json.loads(str(ckpt["__metadata__"].item()))
    state = {key: ckpt[key] for key in ckpt.files if key != "__metadata__"}
    return state, metadata


def timestamp_name(prefix: str) -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

