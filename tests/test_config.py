from pathlib import Path

import pytest
import yaml

from safeland.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_roundtrip_and_digest_stable():
    cfg = Config()
    again = Config.from_dict(yaml.safe_load(cfg.to_yaml()))
    assert again == cfg
    assert again.digest() == cfg.digest()


def test_shipped_default_yaml_matches_code_defaults():
    assert Config.load(ROOT / "configs" / "default.yaml") == Config()


def test_overrides_are_typed():
    cfg = Config.load(None, ["rl.episodes=7", "world.pads=[[0,0]]", "shield.prob_lambda=0.5"])
    assert cfg.rl.episodes == 7
    assert cfg.world.pads == ((0, 0),)
    assert cfg.shield.prob_lambda == 0.5


@pytest.mark.parametrize(
    "data",
    [
        {"rl": {"episodez": 1}},
        {"nonsense": {}},
        {"world": {"pads": [[3, 3]], "nfz": [[3, 3]]}},
        {"world": {"wind_prob": 1.5}},
        {"world": {"cost_hover": 0}},
        {"experiment": {"methods": ["magic"]}},
        {"rl": {"eps_start": 0.1, "eps_end": 0.5}},
    ],
)
def test_invalid_configs_rejected(data):
    with pytest.raises((ValueError, TypeError)):
        Config.from_dict(data)


def test_bad_override_syntax():
    with pytest.raises(ValueError):
        Config.load(None, ["episodes=3"])
