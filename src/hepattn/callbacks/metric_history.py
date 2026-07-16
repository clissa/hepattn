import json
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


class MetricHistory(Callback):
    def __init__(self, filename: str = "metric_history.jsonl") -> None:
        super().__init__()
        self.filename = filename

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        log_dir = trainer.log_dir
        if log_dir is None:
            return

        row: dict[str, Any] = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        for key, value in trainer.callback_metrics.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                row[str(key)] = scalar

        path = Path(log_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def _to_scalar(value: Any) -> float | int | None:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return value
        return None
