from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, parameters: list[np.ndarray], gradients: list[np.ndarray], lr: float):
        self.parameters = parameters
        self.gradients = gradients
        self.lr = lr

    def step(self) -> None:
        for param, grad in zip(self.parameters, self.gradients):
            param -= self.lr * grad

    def set_lr(self, lr: float) -> None:
        self.lr = lr

