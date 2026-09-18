from copy import deepcopy
from math import cos, pi

import pytest
import torch
from lightning import Callback, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hepattn.models.wrapper import ModelWrapper


class ResumeModel(ModelWrapper):
    def training_step(self, batch, batch_idx):
        self.lr_trace.append(self.trainer.optimizers[0].param_groups[0]["lr"])
        return self.model(batch[0]).square().mean()

    def on_train_start(self):
        super().on_train_start()
        self.restored_weights = deepcopy(self.model.state_dict())
        self.restored_optimizer = deepcopy(self.trainer.optimizers[0].state_dict())
        self.start_epoch = self.current_epoch
        self.lr_trace = []


def test_full_resume_preserves_optimizer_and_replaces_exhausted_scheduler(tmp_path):
    torch.manual_seed(7)
    loader = DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)
    lrs_config = {
        "initial": 1e-6,
        "max": 8e-5,
        "end": 1e-6,
        "pct_start": 0.3,
        "weight_decay": 1e-4,
        "skip_scheduler": False,
    }
    checkpoint_path = None
    previous = None
    for max_epochs in (3, 5, 7):
        model = ResumeModel("resume-test", nn.Linear(2, 1), dict(lrs_config), optimizer="Lion")
        if previous is not None:
            model.lrs_config["resume_constant_lr"] = True
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_path,
        )
        trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint_path)
        if previous is not None:
            # These checkpoints are saved after fit, when Lightning has already
            # advanced the epoch counter. Check completed optimizer steps.
            assert model.start_epoch == previous["global_step"] // len(loader)
            for key, value in model.restored_weights.items():
                torch.testing.assert_close(value, previous["state_dict"][f"model.{key}"], rtol=0, atol=0)
            expected = previous["optimizer_states"][0]
            assert model.restored_optimizer["param_groups"] == expected["param_groups"]
            for key, state in expected["state"].items():
                torch.testing.assert_close(model.restored_optimizer["state"][key]["exp_avg"], state["exp_avg"], rtol=0, atol=0)
            group = trainer.optimizers[0].param_groups[0]
            assert group["lr"] == expected["param_groups"][0]["lr"]
            assert group["betas"] == expected["param_groups"][0]["betas"]
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.ConstantLR)
        assert trainer.global_step == max_epochs * len(loader)
        checkpoint_path = tmp_path / f"epoch-{max_epochs}.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


class StopAfterEpoch(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 4:
            trainer.should_stop = True


def test_cosine_resume_matches_uninterrupted_tail(tmp_path):
    torch.manual_seed(7)
    loader = DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2)
    config = {"initial": 1e-6, "max": 8e-5, "end": 1e-6, "pct_start": 0.3, "weight_decay": 1e-4}

    def run(name, epochs, checkpoint=None, callbacks=None):
        model = ResumeModel(name, nn.Linear(2, 1), dict(config), optimizer="Lion")
        if checkpoint is not None:
            model.lrs_config.update(resume_cosine_lr=True, end=1e-7)
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_path,
            callbacks=callbacks,
        )
        trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint)
        path = tmp_path / f"{name}.ckpt"
        trainer.save_checkpoint(path)
        return model, trainer, path

    _, _, source_path = run("source", 3)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    full, full_trainer, _ = run("full", 7, source_path)
    partial, _, partial_path = run("partial", 7, source_path, [StopAfterEpoch()])
    resumed, resumed_trainer, _ = run("resumed", 7, partial_path)
    saved_partial = torch.load(partial_path, map_location="cpu", weights_only=False)

    for model, saved in ((full, source), (resumed, saved_partial)):
        assert model.start_epoch == saved["global_step"] // len(loader)
        expected = saved["optimizer_states"][0]
        assert model.restored_optimizer["param_groups"] == expected["param_groups"]
        for key, state in expected["state"].items():
            torch.testing.assert_close(model.restored_optimizer["state"][key]["exp_avg"], state["exp_avg"], rtol=0, atol=0)
        for key, value in model.restored_weights.items():
            torch.testing.assert_close(value, saved["state_dict"][f"model.{key}"], rtol=0, atol=0)

    start_lr = source["optimizer_states"][0]["param_groups"][0]["lr"]
    expected_curve = [1e-7 + (start_lr - 1e-7) * (1 + cos(pi * step / 8)) / 2 for step in range(8)]
    assert full.lr_trace == pytest.approx(expected_curve)
    assert partial.lr_trace + resumed.lr_trace == pytest.approx(full.lr_trace)
    assert all(left > right for left, right in zip(full.lr_trace, full.lr_trace[1:], strict=False))
    for trainer in (full_trainer, resumed_trainer):
        assert trainer.global_step == 14
        group = trainer.optimizers[0].param_groups[0]
        assert group["lr"] == pytest.approx(1e-7)
        assert group["betas"] == source["optimizer_states"][0]["param_groups"][0]["betas"]
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
