from __future__ import annotations

from pathlib import Path

import numpy as np

from losses import l2_regularization
from losses import softmax_cross_entropy_loss
from utils import accuracy_score, confusion_matrix, save_json, softmax


def evaluate_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
    weight_decay: float = 0.0,
) -> dict[str, np.ndarray | float]:
    logits_list = []
    loss_sum = 0.0
    total = 0

    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model.forward(xb)
        loss, _ = softmax_cross_entropy_loss(logits, yb)
        if weight_decay > 0.0:
            reg_loss, _ = l2_regularization([model.fc1.W, model.fc2.W, model.fc3.W], weight_decay)
            loss += reg_loss
        logits_list.append(logits)
        loss_sum += loss * len(xb)
        total += len(xb)

    logits_all = np.vstack(logits_list)
    probs = softmax(logits_all)
    preds = np.argmax(probs, axis=1)
    return {
        "loss": float(loss_sum / max(total, 1)),
        "accuracy": accuracy_score(y, preds),
        "predictions": preds,
        "probabilities": probs,
    }


def summarize_test_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        class_accuracy[class_name] = 0.0 if total == 0 else correct / total

    summary = {
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "confusion_matrix": cm.tolist(),
        "class_accuracy": class_accuracy,
    }
    save_json(save_path, summary)
    return summary
