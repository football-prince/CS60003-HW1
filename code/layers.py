from __future__ import annotations

import numpy as np


class Linear:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        limit = np.sqrt(2.0 / in_dim)
        self.W = (rng.standard_normal((in_dim, out_dim)) * limit).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.cache: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = (
                x.astype(np.float64, copy=False) @ self.W.astype(np.float64, copy=False)
                + self.b.astype(np.float64, copy=False)
            )
        return out.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("forward must be called before backward")
        x = self.cache
        # Keep gradient arrays stable so the optimizer always sees fresh values.
        grad_output64 = grad_output.astype(np.float64, copy=False)
        x64 = x.astype(np.float64, copy=False)
        w64 = self.W.astype(np.float64, copy=False)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.dW[...] = (x64.T @ grad_output64).astype(np.float32)
        self.db[...] = grad_output64.sum(axis=0).astype(np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            grad_input = grad_output64 @ w64.T
        return grad_input.astype(np.float32)


class ReLU:
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self.mask


class Sigmoid:
    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self.out * (1.0 - self.out)


class Tanh:
    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * (1.0 - self.out ** 2)


def build_activation(name: str):
    name = name.lower()
    if name == "relu":
        return ReLU()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    raise ValueError(f"Unsupported activation: {name}")
