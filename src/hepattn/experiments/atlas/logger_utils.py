from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class FixedCometLogger(CometLogger):
    """CometLogger that:
    - maps `offline_directory` -> `save_dir` for back-compat
    - forces the experiment display name to `experiment_name`/`name` instead of
      Comet's auto-generated `adjective_noun_NNNN`.
    """

    def __init__(self, save_dir=None, offline_directory=None, experiment_name=None, name=None, **kwargs):
        if save_dir is None:
            save_dir = offline_directory

        self._forced_name = experiment_name or name
        if self._forced_name is not None:
            kwargs["name"] = self._forced_name

        super().__init__(save_dir=save_dir, **kwargs)

    @rank_zero_only
    def _create_experiment(self) -> None:
        super()._create_experiment()
        if self._forced_name and self._experiment is not None:
            try:
                self._experiment.set_name(self._forced_name)
            except Exception as e:
                print(f"FixedCometLogger: failed to set experiment name: {e}")
