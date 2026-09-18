from pathlib import Path
from typing import Literal

import torch
from lightning import LightningModule
from lion_pytorch import Lion
from torch import nn
from torch.optim import AdamW

# from torchjd import mtl_backward
# from torchjd.aggregation import UPGrad


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        mtl: bool = False,
        init_ckpt_path: str | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        self.mtl = mtl
        self.init_ckpt_path = init_ckpt_path

        if init_ckpt_path is not None:
            self._load_init_ckpt(init_ckpt_path)

        # If we are doing multi-task-learning, optimisation step must be done manually
        if mtl:
            self.automatic_optimization = False

    def _load_init_ckpt(self, init_ckpt_path: str) -> None:
        init_ckpt_path = Path(init_ckpt_path)
        checkpoint = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        model_state_dict = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
        if not model_state_dict:
            raise KeyError(f"No model weights found in checkpoint: {init_ckpt_path}")
        self.model.load_state_dict(model_state_dict)
        print(f"Loaded initial model weights from {init_ckpt_path.resolve()!r}")

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, outputs):
        return self.model.predict(outputs)

    def log_losses(self, losses, stage):
        total_loss = 0

        for layer_name, layer_losses in losses.items():
            layer_loss = 0
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value, sync_dist=True)
                    layer_loss += loss_value
                    total_loss += loss_value
            self.log(f"{stage}/{layer_name}_loss", layer_loss, sync_dist=True)

        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_task_metrics(self, preds, targets, stage):
        # Log any task specific metrics
        for task in self.model.tasks:
            # Check that the task actually has some metrics to log
            if not hasattr(task, "metrics"):
                continue

            # Just log the predictions from the final layer for now
            task_metrics = task.metrics(preds["final"][task.name], targets)

            # If the task returned a non-empty metrics dict, log it
            if task_metrics:
                self.log_dict({f"{stage}/final_{task.name}_{k}": v for k, v in task_metrics.items()}, sync_dist=True)

    def log_metrics(self, preds, targets, stage):
        # First log any task metrics
        self.log_task_metrics(preds, targets, stage)

        # Log any custom metrics implemented by subclass
        if hasattr(self, "log_custom_metrics"):
            self.log_custom_metrics(preds, targets, stage)

    def detach_nested(self, d):
        if isinstance(d, dict):
            return {k: self.detach_nested(v) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d.detach().cpu()
        return d

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "train")

        # if total_loss is None or total_loss > 100:
        #     losses_per_element = self.model.loss_per_element(outputs, targets)

        #     from datetime import datetime
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     torch.save(self.detach_nested(losses_per_element),
        #         f"/srv01/agrp/nilotpal/projects/glow_atlas/hepattn/src/hepattn/experiments/atlas/logs/losses_per_element_{timestamp}.pt")

        #     idxs = targets["getitem_idx"].cpu().numpy().tolist()
        #     print("\n\nLarge loss detected, problematic indices:\n", idxs, "\n\n")
        #     exit(1)

        # Get the predictions from the model
        if batch_idx % self.trainer.log_every_n_steps == 0:  # avoid calling predict if possible
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        # Use Jacobian Descent for Multi Task Learning https://arxiv.org/abs/2406.16232
        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None

        # if self.global_step > 50_000:
        #     total_loss = torch.clamp(total_loss, max=5.5)
        self.log("train/clamped_loss", total_loss, sync_dist=True)

        return total_loss

    def validation_step(self, batch):
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "val")

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return total_loss

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        losses = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self):
        if self.lrs_config.get("resume_cosine_lr"):
            if not self.trainer.ckpt_path or self.lrs_config.get("skip_scheduler") or self.lrs_config.get("resume_constant_lr"):
                raise ValueError("resume_cosine_lr requires a full checkpoint resume, skip_scheduler: false and resume_constant_lr: false")
            remaining_steps = int(self.trainer.estimated_stepping_batches) - self.global_step
            if remaining_steps <= 0:
                raise ValueError("The cosine continuation requires remaining optimizer steps")
            for config in self.trainer.lr_scheduler_configs:
                restored = config.scheduler.state_dict()
                opt = config.scheduler.optimizer
                end_lr = float(self.lrs_config["end"])
                if not 0 < end_lr <= min(group["lr"] for group in opt.param_groups):
                    raise ValueError("The continuation end LR must be positive and no greater than the restored LR")
                continuing = restored.get("hepattn_cosine_tail", False)
                if continuing and (restored["T_max"] != restored["last_epoch"] + remaining_steps or restored["eta_min"] != end_lr):
                    raise ValueError("Resuming a cosine continuation requires the same step horizon and end LR")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining_steps, eta_min=end_lr)
                # OneCycle leaves initial_lr in optimizer groups. Anchor this tail
                # to the actual restored LR instead, without changing LR or betas.
                scheduler.base_lrs = [group["lr"] for group in opt.param_groups]
                if continuing:
                    scheduler.load_state_dict(restored)
                scheduler.hepattn_cosine_tail = True
                config.scheduler = scheduler
            return

        if self.lrs_config.get("resume_constant_lr"):
            if not self.trainer.ckpt_path or self.lrs_config.get("skip_scheduler"):
                raise ValueError("resume_constant_lr requires a full --ckpt_path resume and skip_scheduler: false")
            # Lightning has restored model, optimizer and scheduler state by this hook.
            # factor=1 leaves the restored LR unchanged and never modifies betas.
            for config in self.trainer.lr_scheduler_configs:
                config.scheduler = torch.optim.lr_scheduler.ConstantLR(config.scheduler.optimizer, factor=1.0, total_iters=1)
            return

        # Manually overwride the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]
        print("Skipping learning rate scheduler.")
        return opt

    # def mlt_opt(self, losses, outputs):
    #     opt = self.optimizers()
    #     opt.zero_grad()

    #     for layer_name, layer_losses in losses.items():
    #         # Get a list of the features that are used by all of the tasks
    #         layer_feature_names = set()
    #         for task in self.model.tasks:
    #             layer_feature_names.update(task.inputs)

    #         # Remove any duplicate features that are used by multiple tasks
    #         layer_features = [outputs[layer_name][feature_name] for feature_name in layer_feature_names]

    #         # Perform the backward pass for this layer
    #         # For each layer we sum the losses from each task, so we get one loss per task
    #         layer_losses = [sum(losses[layer_name][task.name].values()) for task in self.model.tasks]

    #         mtl_backward(losses=layer_losses, features=layer_features, aggregator=UPGrad())

    #     opt.step()
