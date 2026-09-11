from copy import deepcopy

import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hepattn.models.wrapper import ModelWrapper


class ResumeModel(ModelWrapper):
    def training_step(self, batch, batch_idx):
        return self.model(batch[0]).square().mean()

    def on_train_start(self):
        super().on_train_start()
        self.restored_weights = deepcopy(self.model.state_dict())
        self.restored_optimizer = deepcopy(self.trainer.optimizers[0].state_dict())
        self.start_epoch = self.current_epoch


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
            assert model.start_epoch == previous["epoch"] + 1
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
