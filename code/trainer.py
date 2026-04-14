from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from data_utils import MiniBatchIterator, prepare_data
from evaluate import evaluate_split
from losses import l2_regularization, softmax_cross_entropy_loss
from model import ThreeLayerMLP
from optim import SGD
from utils import (
    ensure_project_dirs,
    save_checkpoint,
    save_json,
    serialize_config,
    set_seed,
    timestamp_name,
)
from visualize import plot_training_curves, visualize_first_layer_weights


class Trainer:
    def __init__(self, cfg, run_name: str | None = None):
        self.cfg = cfg
        self.run_name = run_name or timestamp_name("mlp")
        ensure_project_dirs(cfg)
        set_seed(cfg.seed)

        prepared = prepare_data(cfg)
        self.splits = prepared["splits"]
        self.class_names = prepared["class_names"]
        self.input_dim = prepared["input_dim"]
        self.image_shape = tuple(prepared["image_shape"])
        self.mean = prepared["mean"]
        self.std = prepared["std"]

        self.model = ThreeLayerMLP(
            input_dim=self.input_dim,
            hidden_dim1=cfg.hidden_dim1,
            hidden_dim2=cfg.hidden_dim2,
            num_classes=len(self.class_names),
            activation=cfg.activation,
            seed=cfg.seed,
        )
        self.optimizer = SGD(self.model.parameters(), self.model.gradients(), lr=cfg.learning_rate)
        self.best_checkpoint_path = Path(cfg.checkpoints_dir) / f"{self.run_name}_best.npz"
        self.history_path = Path(cfg.curves_dir) / f"{self.run_name}_history.json"

    def train(self) -> dict:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_val_acc = -1.0
        best_epoch = -1

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = MiniBatchIterator(
                self.splits["train"].X,
                self.splits["train"].y,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                seed=self.cfg.seed + epoch,
            )
            epoch_loss_sum = 0.0
            num_samples = 0

            for xb, yb in train_iterator:
                logits = self.model.forward(xb)
                if not np.isfinite(logits).all():
                    raise FloatingPointError(
                        "Non-finite logits detected during training. "
                        "Try a smaller learning rate or inspect data normalization."
                    )
                data_loss, grad_logits = softmax_cross_entropy_loss(logits, yb)
                reg_loss, _ = l2_regularization([self.model.fc1.W, self.model.fc2.W, self.model.fc3.W], self.cfg.weight_decay)
                self.model.backward(grad_logits, weight_decay=self.cfg.weight_decay)
                self.optimizer.step()
                epoch_loss_sum += (data_loss + reg_loss) * len(xb)
                num_samples += len(xb)

            train_loss = epoch_loss_sum / max(num_samples, 1)
            val_metrics = evaluate_split(
                self.model,
                self.splits["val"].X,
                self.splits["val"].y,
                batch_size=self.cfg.batch_size,
                weight_decay=self.cfg.weight_decay,
            )

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_accuracy"].append(float(val_metrics["accuracy"]))
            history["learning_rate"].append(float(self.optimizer.lr))

            print(
                f"[Epoch {epoch:03d}/{self.cfg.epochs:03d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"lr={self.optimizer.lr:.6f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = float(val_metrics["accuracy"])
                best_epoch = epoch
                self._save_best_checkpoint(epoch=epoch, best_val_acc=best_val_acc, history=history)

            if self.cfg.save_every_epoch:
                self._save_epoch_snapshot(epoch=epoch)

            next_lr = max(self.optimizer.lr * self.cfg.lr_decay, self.cfg.min_learning_rate)
            self.optimizer.set_lr(next_lr)

        history["best_val_accuracy"] = best_val_acc
        history["best_epoch"] = best_epoch
        save_json(self.history_path, history)
        plot_training_curves(history, Path(self.cfg.curves_dir), self.run_name)
        visualize_first_layer_weights(
            self.model.fc1.W,
            self.image_shape,
            Path(self.cfg.weights_dir),
            self.run_name,
        )
        return history

    def _save_best_checkpoint(self, epoch: int, best_val_acc: float, history: dict) -> None:
        metadata = {
            "epoch": epoch,
            "best_val_accuracy": best_val_acc,
            "class_names": self.class_names,
            "image_shape": list(self.image_shape),
            "input_dim": self.input_dim,
            "hidden_dim1": self.cfg.hidden_dim1,
            "hidden_dim2": self.cfg.hidden_dim2,
            "num_classes": len(self.class_names),
            "activation": self.cfg.activation,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "config": serialize_config(self.cfg),
            "history_path": str(self.history_path),
        }
        save_checkpoint(self.best_checkpoint_path, self.model.named_parameters(), metadata)
        save_json(self.history_path, history)

    def _save_epoch_snapshot(self, epoch: int) -> None:
        snapshot_path = Path(self.cfg.checkpoints_dir) / f"{self.run_name}_epoch_{epoch:03d}.npz"
        metadata = {
            "epoch": epoch,
            "class_names": self.class_names,
            "image_shape": list(self.image_shape),
            "activation": self.cfg.activation,
            "config": serialize_config(self.cfg),
        }
        save_checkpoint(snapshot_path, self.model.named_parameters(), metadata)


def train_once(cfg, run_name: str | None = None) -> dict:
    trainer = Trainer(cfg, run_name=run_name)
    history = trainer.train()
    return {
        "history": history,
        "checkpoint_path": str(trainer.best_checkpoint_path),
        "history_path": str(trainer.history_path),
        "class_names": trainer.class_names,
        "image_shape": trainer.image_shape,
    }
