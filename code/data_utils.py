from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from utils import set_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray
    images: np.ndarray
    paths: np.ndarray


def discover_classes(data_dir: Path) -> list[str]:
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


def load_eurosat_dataset(
    data_dir: Path,
    image_size: int = 32,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    class_names = discover_classes(data_dir)
    features = []
    labels = []
    images = []
    paths = []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if image.size != (image_size, image_size):
                    image = image.resize((image_size, image_size), Image.BILINEAR)
                image_arr = np.asarray(image, dtype=np.float32)
            if normalize:
                image_arr /= 255.0
            images.append(image_arr)
            features.append(image_arr.reshape(-1))
            labels.append(label)
            paths.append(str(image_path))

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    raw_images = np.asarray(images, dtype=np.float32)
    file_paths = np.asarray(paths)
    return X, y, raw_images, file_paths, class_names


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_total = len(cls_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train : n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val : n_train + n_val + n_test])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def build_splits(
    X: np.ndarray,
    y: np.ndarray,
    images: np.ndarray,
    paths: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, DataSplit]:
    train_idx, val_idx, test_idx = stratified_split_indices(
        y=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    return {
        "train": DataSplit(X[train_idx], y[train_idx], images[train_idx], paths[train_idx]),
        "val": DataSplit(X[val_idx], y[val_idx], images[val_idx], paths[val_idx]),
        "test": DataSplit(X[test_idx], y[test_idx], images[test_idx], paths[test_idx]),
    }


def compute_normalization_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 1, 2), keepdims=True)
    std = X.std(axis=(0, 1, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def prepare_data(cfg) -> dict[str, object]:
    set_seed(cfg.seed)
    X, y, images, paths, class_names = load_eurosat_dataset(
        cfg.data_dir,
        image_size=cfg.image_size,
        normalize=cfg.normalize,
    )
    splits = build_splits(
        X=X,
        y=y,
        images=images,
        paths=paths,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )
    train_mean, train_std = compute_normalization_stats(splits["train"].images)
    for split in splits.values():
        standardized_images = apply_standardization(split.images, train_mean, train_std)
        split.X = standardized_images.reshape(len(standardized_images), -1).astype(np.float32)

    return {
        "splits": splits,
        "class_names": class_names,
        "input_dim": splits["train"].X.shape[1],
        "image_shape": splits["train"].images.shape[1:],
        "mean": train_mean.reshape(-1),
        "std": train_std.reshape(-1),
    }


class MiniBatchIterator:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.X))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
