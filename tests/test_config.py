import dataclasses

import pytest
import torch

from mnist_flow.config import Config


def test_defaults():
    c = Config()
    assert c.base_channels == 64 and c.channel_mults == (1, 2, 2)
    assert c.ddpm_T == 1000 and c.rf_time_sampling == "uniform"


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().lr = 1.0


def test_round_trip():
    c = Config()
    assert Config.from_dict(c.to_dict()) == c
    t = Config.tiny()
    assert Config.from_dict(t.to_dict()) == t


def test_to_dict_builtins_only():
    d = Config().to_dict()
    for v in d.values():
        assert type(v) in (int, float, str, bool, list)


def test_to_dict_survives_torch_save(tmp_path):
    p = tmp_path / "c.pt"
    torch.save(Config().to_dict(), p)
    loaded = torch.load(p, weights_only=True)
    assert Config.from_dict(loaded) == Config()


def test_tiny():
    t = Config.tiny()
    assert (t.base_channels, t.time_embed_dim, t.batch_size, t.epochs) == (8, 32, 8, 1)


def test_from_dict_ignores_unknown_keys():
    d = Config().to_dict()
    d["bogus"] = 1
    assert Config.from_dict(d) == Config()
