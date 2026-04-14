from __future__ import annotations

import numpy as np

from layers import Linear, build_activation


class ThreeLayerMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.fc1 = Linear(input_dim, hidden_dim1, rng)
        self.act1 = build_activation(activation)
        self.fc2 = Linear(hidden_dim1, hidden_dim2, rng)
        self.act2 = build_activation(activation)
        self.fc3 = Linear(hidden_dim2, num_classes, rng)
        self.activation_name = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.fc1.forward(x)
        out = self.act1.forward(out)
        out = self.fc2.forward(out)
        out = self.act2.forward(out)
        out = self.fc3.forward(out)
        return out

    def backward(self, grad_logits: np.ndarray, weight_decay: float = 0.0) -> None:
        grad = self.fc3.backward(grad_logits)
        grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.act1.backward(grad)
        self.fc1.backward(grad)

        if weight_decay > 0.0:
            self.fc1.dW += weight_decay * self.fc1.W
            self.fc2.dW += weight_decay * self.fc2.W
            self.fc3.dW += weight_decay * self.fc3.W

    def parameters(self) -> list[np.ndarray]:
        return [self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]

    def gradients(self) -> list[np.ndarray]:
        return [self.fc1.dW, self.fc1.db, self.fc2.dW, self.fc2.db, self.fc3.dW, self.fc3.db]

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "fc1_W": self.fc1.W,
            "fc1_b": self.fc1.b,
            "fc2_W": self.fc2.W,
            "fc2_b": self.fc2.b,
            "fc3_W": self.fc3.W,
            "fc3_b": self.fc3.b,
        }

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        self.fc1.W = state_dict["fc1_W"].astype(np.float32)
        self.fc1.b = state_dict["fc1_b"].astype(np.float32)
        self.fc2.W = state_dict["fc2_W"].astype(np.float32)
        self.fc2.b = state_dict["fc2_b"].astype(np.float32)
        self.fc3.W = state_dict["fc3_W"].astype(np.float32)
        self.fc3.b = state_dict["fc3_b"].astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

