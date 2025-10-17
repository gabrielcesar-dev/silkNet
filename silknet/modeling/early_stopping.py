from pathlib import Path
from typing import Optional, Union

from loguru import logger
import torch
import torch.nn as nn


class EarlyStopping:
    """Monitor validation loss and stop training when it ceases to improve."""

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: Union[str, Path] = "./checkpoint.pt",
    ) -> None:
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.delta = float(delta)
        self.path = Path(path)

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min: float = float("inf")

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Update internal state given the current validation loss and model."""
        try:
            val_loss = float(val_loss)
        except Exception:
            logger.warning("EarlyStopping received a non-convertible val_loss value")
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: no improvement (counter={self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Save the model state_dict to the configured path."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(self.path))

            if self.verbose:
                logger.info(f"Validation loss decreased ({val_loss:.6f}). Saved model to {self.path}")

            self.val_loss_min = val_loss
        except Exception:
            logger.exception("Failed to save checkpoint in EarlyStopping")