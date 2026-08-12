from lightning.pytorch.loggers import CometLogger


class FixedCometLogger(CometLogger):
    """CometLogger that:
    - maps `offline_directory` -> `save_dir` for back-compat
    - forces the experiment display name to `experiment_name`/`name` instead of
      Comet's auto-generated `adjective_noun_NNNN`.
    """

    def __init__(self, save_dir=None, offline_directory=None, experiment_name=None, name=None, **kwargs):
        if save_dir is None:
            save_dir = offline_directory

        super().__init__(save_dir=save_dir, experiment_name=experiment_name or name, **kwargs)
