import copy

import torch
from torch import nn

from mnist_flow.ema import EMA


def test_update_formula():
    m = nn.Linear(3, 2)
    d = 0.9
    ema = EMA(m, d)
    old = copy.deepcopy([p.detach() for p in m.parameters()])
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    new = [p.detach().clone() for p in m.parameters()]
    ema.update(m)
    for e, o, n in zip(ema.model.parameters(), old, new):
        assert torch.allclose(e, d * o + (1 - d) * n, atol=1e-6)


def test_ema_is_independent_copy_without_grad():
    m = nn.Linear(2, 2)
    ema = EMA(m, 0.5)
    assert all(not p.requires_grad for p in ema.model.parameters())
    with torch.no_grad():
        m.weight.zero_()
    assert not torch.equal(ema.model.weight, m.weight)


def test_state_dict_round_trip():
    m = nn.Linear(2, 2)
    ema = EMA(m, 0.5)
    sd = ema.state_dict()
    other = EMA(nn.Linear(2, 2), 0.5)
    other.load_state_dict(sd)
    assert torch.equal(other.model.weight, ema.model.weight)


def test_copy_to():
    m = nn.Linear(2, 2)
    ema = EMA(m, 0.5)
    with torch.no_grad():
        m.weight.zero_()
    ema.copy_to(m)
    assert torch.equal(m.weight, ema.model.weight)


def test_buffers_are_copied():
    m = nn.BatchNorm1d(2)
    ema = EMA(m, 0.5)
    m.running_mean.fill_(3.0)
    ema.update(m)
    assert torch.equal(ema.model.running_mean, m.running_mean)
