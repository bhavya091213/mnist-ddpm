import math

import pytest
import torch

from mnist_flow.metrics import (
    class_histogram,
    compute_generation_metrics,
    coverage_entropy,
    frechet_distance,
    mean_confidence,
)


def _gauss(n, d, mean=0.0, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * scale + mean


def test_frechet_identical_is_zero():
    x = _gauss(500, 8)
    assert frechet_distance(x, x) == pytest.approx(0.0, abs=1e-6)


def test_frechet_mean_shift_1d():
    a = _gauss(20000, 1, seed=1)
    a = (a - a.mean()) / a.std()
    b = a + 3.0
    assert frechet_distance(a, b) == pytest.approx(9.0, rel=1e-4)


def test_frechet_symmetric():
    a, b = _gauss(300, 6, seed=1), _gauss(300, 6, mean=0.5, scale=1.5, seed=2)
    assert frechet_distance(a, b) == pytest.approx(frechet_distance(b, a), rel=1e-6)


def test_frechet_diagonal_closed_form():
    a = _gauss(5000, 4, seed=3)
    b = _gauss(5000, 4, seed=4) * torch.tensor([1.0, 2.0, 0.5, 3.0])
    a64, b64 = a.double(), b.double()
    va, vb = a64.var(0, unbiased=True), b64.var(0, unbiased=True)
    # sample covariances are not exactly diagonal; compare against diagonal-only data
    a_d = a64 - a64.mean(0)
    b_d = b64 - b64.mean(0)
    # Make covariances exactly diagonal by whitening then scaling
    def diag_data(x, scales):
        xc = x - x.mean(0)
        cov = xc.T @ xc / (x.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        w = xc @ evecs / evals.sqrt()
        return w * scales

    sa = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    sb = torch.tensor([1.2, 1.0, 0.7, 3.0], dtype=torch.float64)
    xa, xb = diag_data(a_d, sa), diag_data(b_d, sb)
    expected = float(((sa - sb) ** 2).sum())
    assert frechet_distance(xa.float(), xb.float()) == pytest.approx(expected, rel=1e-3)
    assert va.shape == vb.shape


def test_frechet_shape_validation():
    with pytest.raises(ValueError):
        frechet_distance(torch.zeros(5, 3), torch.zeros(5, 4))
    with pytest.raises(ValueError):
        frechet_distance(torch.zeros(1, 3), torch.zeros(5, 3))


def test_coverage_entropy_uniform_and_single():
    uniform = torch.eye(10).repeat(5, 1) * 10
    assert coverage_entropy(uniform) == pytest.approx(math.log(10), abs=1e-9)
    single = torch.zeros(20, 10)
    single[:, 3] = 5.0
    assert coverage_entropy(single) == pytest.approx(0.0, abs=1e-12)


def test_mean_confidence_range():
    g = torch.Generator().manual_seed(0)
    c = mean_confidence(torch.randn(100, 10, generator=g))
    assert 0.0 <= c <= 1.0
    assert mean_confidence(torch.zeros(4, 10)) == pytest.approx(0.1)


def test_class_histogram():
    logits = torch.zeros(5, 10)
    logits[torch.arange(5), torch.tensor([0, 0, 1, 9, 9])] = 1.0
    h = class_histogram(logits)
    assert len(h) == 10 and h[0] == 2 and h[1] == 1 and h[9] == 2 and sum(h) == 5


def test_compute_generation_metrics_keys():
    from mnist_flow.classifier import MNISTClassifier

    torch.manual_seed(0)
    clf = MNISTClassifier()
    real = torch.rand(40, 1, 28, 28) * 2 - 1
    gen = torch.rand(30, 1, 28, 28) * 2 - 1
    m = compute_generation_metrics(clf, real, gen)
    assert set(m) == {
        "fid", "coverage_entropy", "mean_confidence", "class_histogram", "n_real", "n_gen",
    }
    assert m["n_real"] == 40 and m["n_gen"] == 30
    assert isinstance(m["fid"], float) and sum(m["class_histogram"]) == 30
