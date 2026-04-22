"""Lightweight check that Hydra configs resolve (no training, no data)."""

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def test_train_compose_f6_expflow() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["experiment=f6_expflow"])
    assert cfg["model_name"] == "flowmatchingSwin6hr"
    assert cfg["exp_name"] == "f6_w1c"
    GlobalHydra.instance().clear()


def test_train_compose_f6_exppangu() -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train", overrides=["experiment=f6_exppangu"])
    assert cfg["model_name"] == "pangu"
    GlobalHydra.instance().clear()
