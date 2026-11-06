import torch

from mnist_flow.classifier import (
    MNISTClassifier,
    evaluate_accuracy,
    extract_features,
    load_classifier,
    predict_logits,
    save_classifier,
    train_classifier,
)


def _toy(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = -torch.ones(n, 1, 28, 28)
    x[y == 0, :, 4:12, 4:12] = 1.0
    x[y == 1, :, 16:24, 16:24] = 1.0
    x = (x + 0.1 * torch.randn(x.shape, generator=g)).clamp(-1, 1)
    return x, y


def test_shapes():
    m = MNISTClassifier()
    x = torch.randn(4, 1, 28, 28)
    assert m.feature_dim == 128
    assert m.features(x).shape == (4, 128)
    assert (m.features(x) >= 0).all()
    assert m(x).shape == (4, 10)


def test_train_reaches_high_accuracy(cpu):
    x, y = _toy()
    m = train_classifier(x, y, device=cpu, epochs=2, batch_size=64, seed=0)
    assert evaluate_accuracy(m, x, y) > 0.9


def test_extract_and_predict_batched(cpu):
    m = MNISTClassifier()
    x = torch.randn(25, 1, 28, 28)
    f = extract_features(m, x, batch_size=10)
    p = predict_logits(m, x, batch_size=10)
    assert f.shape == (25, 128) and f.dtype == torch.float32 and f.device.type == "cpu"
    assert p.shape == (25, 10)
    assert torch.allclose(p, predict_logits(m, x, batch_size=25), atol=1e-5)


def test_save_load_roundtrip(tmp_path, cpu):
    m = MNISTClassifier().eval()
    path = tmp_path / "clf.pt"
    save_classifier(m, path)
    sd = torch.load(path, weights_only=True)
    assert set(sd) == set(m.state_dict())
    m2 = load_classifier(path, cpu)
    x = torch.randn(3, 1, 28, 28)
    assert torch.allclose(m(x), m2(x), atol=1e-6)
    assert not m2.training
