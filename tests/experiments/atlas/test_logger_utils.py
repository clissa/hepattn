import comet_ml
import pytest

from hepattn.experiments.atlas.logger_utils import FixedCometLogger


class FakeOfflineExperiment:
    alive = True

    def __init__(self, **_kwargs):
        self.display_name = None

    def log_other(self, *_args):
        pass

    def set_name(self, name):
        self.display_name = name


@pytest.mark.parametrize("name_arg", ["experiment_name", "name"])
def test_comet_display_name_is_forwarded(tmp_path, monkeypatch, name_arg):
    monkeypatch.setattr(comet_ml, "OfflineExperiment", FakeOfflineExperiment)
    logger = FixedCometLogger(save_dir=tmp_path, offline=True, **{name_arg: "qs00_ref_detw60"})

    assert logger.experiment.display_name == "qs00_ref_detw60"


def test_offline_directory_alias_is_preserved(tmp_path):
    logger = FixedCometLogger(offline_directory=tmp_path)

    assert logger.save_dir == tmp_path
