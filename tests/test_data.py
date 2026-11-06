import pytest
import torch

from mnist_flow.data import iterate_batches, to_minus_one_one


def test_scaling_range_and_dtype():
    x = torch.tensor([0, 127, 255], dtype=torch.uint8)
    y = to_minus_one_one(x)
    assert y.dtype == torch.float32
    assert y[0] == -1.0 and y[2] == 1.0
    assert torch.isclose(y[1], torch.tensor(127 / 127.5 - 1))


def test_iterate_batches_drops_last_and_covers():
    x = torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1)
    g = torch.Generator().manual_seed(0)
    batches = list(iterate_batches(x, 4, g))
    assert len(batches) == 2
    assert all(b.shape == (4, 1, 1, 1) for b in batches)
    seen = torch.cat(batches).flatten()
    assert len(set(seen.tolist())) == 8


def test_iterate_batches_reshuffles_and_is_seeded():
    x = torch.arange(32, dtype=torch.float32).view(32, 1)
    g = torch.Generator().manual_seed(1)
    a = torch.cat(list(iterate_batches(x, 8, g)))
    b = torch.cat(list(iterate_batches(x, 8, g)))
    assert not torch.equal(a, b)
    g2 = torch.Generator().manual_seed(1)
    assert torch.equal(a, torch.cat(list(iterate_batches(x, 8, g2))))


def test_iterate_batches_smaller_than_batch_yields_nothing():
    x = torch.zeros(3, 1)
    assert list(iterate_batches(x, 4, torch.Generator())) == []


def test_load_mnist_uses_dataset(monkeypatch, tmp_path):
    from mnist_flow import data

    class Fake:
        def __init__(self, root, train, download):
            self.data = torch.randint(0, 256, (5, 28, 28), dtype=torch.uint8)
            self.targets = torch.arange(5)

    monkeypatch.setattr(data.datasets, "MNIST", Fake)
    x, y = data.load_mnist(tmp_path, True, torch.device("cpu"))
    assert x.shape == (5, 1, 28, 28) and x.dtype == torch.float32
    assert x.min() >= -1 and x.max() <= 1
    assert y.dtype == torch.int64 and y.shape == (5,)
