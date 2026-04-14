from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "EuroSAT_RGB"
RESULTS_DIR = ROOT_DIR / "results"
CURVES_DIR = RESULTS_DIR / "curves"
WEIGHTS_DIR = RESULTS_DIR / "weights"
ERRORS_DIR = RESULTS_DIR / "errors"
SEARCH_DIR = RESULTS_DIR / "search"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    curves_dir: Path = CURVES_DIR
    weights_dir: Path = WEIGHTS_DIR
    errors_dir: Path = ERRORS_DIR
    search_dir: Path = SEARCH_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    seed: int = 42
    image_size: int = 32
    batch_size: int = 128
    epochs: int = 30
    hidden_dim1: int = 256
    hidden_dim2: int = 128
    activation: str = "relu"
    learning_rate: float = 0.01
    lr_decay: float = 0.95
    min_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: bool = True
    num_classes: int = 10
    save_every_epoch: bool = False
    num_error_samples: int = 24
    search_mode: str = "grid"
    search_max_trials: int = 12
    search_space: dict = field(
        default_factory=lambda: {
            "learning_rate": [0.01, 0.005, 0.001],
            "hidden_dim": [128, 256, 512],
            "weight_decay": [0.0, 1e-4, 1e-3],
            "activation": ["relu", "tanh", "sigmoid"],
        }
    )
