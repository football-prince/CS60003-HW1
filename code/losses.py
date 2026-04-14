from __future__ import annotations

import numpy as np


def softmax_cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    batch_size = logits.shape[0]

    correct_log_probs = -np.log(probs[np.arange(batch_size), targets] + 1e-12)
    loss = float(np.mean(correct_log_probs))

    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1.0
    grad /= batch_size
    return loss, grad.astype(np.float32)


def l2_regularization(parameters: list[np.ndarray], weight_decay: float) -> tuple[float, list[np.ndarray]]:
    reg_loss = 0.0
    reg_grads = []
    for param in parameters:
        reg_loss += 0.5 * weight_decay * float(np.sum(param * param))
        reg_grads.append(weight_decay * param)
    return reg_loss, reg_grads

