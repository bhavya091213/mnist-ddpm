"""Small MNIST CNN used as a feature extractor for generation-quality metrics."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mnist_flow.data import iterate_batches
from mnist_flow.utils import set_seed


class MNISTClassifier(nn.Module):
    """conv32-conv64-pool-fc128-fc10; input (B,1,28,28) in [-1,1]."""

    feature_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, self.feature_dim)
        self.fc2 = nn.Linear(self.feature_dim, 10)

    def features(self, x: Tensor) -> Tensor:
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        return F.relu(self.fc1(h.flatten(1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.features(x))


def train_classifier(
    x_train: Tensor,
    y_train: Tensor,
    *,
    device: torch.device | str,
    epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> MNISTClassifier:
    """Train with Adam + cross-entropy; returns the model in eval mode."""
    set_seed(seed)
    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_train, y_train = x_train.to(device), y_train.to(device)
    idx = torch.arange(x_train.shape[0], device=device)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for batch_idx in iterate_batches(idx, batch_size, gen):
            loss = F.cross_entropy(model(x_train[batch_idx]), y_train[batch_idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def _batched(fn, model: MNISTClassifier, x: Tensor, batch_size: int) -> Tensor:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out = [fn(x[i : i + batch_size].to(device)).float().cpu() for i in range(0, x.shape[0], batch_size)]
    model.train(was_training)
    return torch.cat(out) if out else torch.empty(0)


def extract_features(model: MNISTClassifier, x: Tensor, batch_size: int = 500) -> Tensor:
    """(N,128) CPU float32 penultimate activations."""
    return _batched(model.features, model, x, batch_size)


def predict_logits(model: MNISTClassifier, x: Tensor, batch_size: int = 500) -> Tensor:
    """(N,10) CPU float32 logits."""
    return _batched(model, model, x, batch_size)


def evaluate_accuracy(model: MNISTClassifier, x: Tensor, y: Tensor) -> float:
    pred = predict_logits(model, x).argmax(1)
    return float((pred == y.cpu()).float().mean())


def save_classifier(model: MNISTClassifier, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def load_classifier(path: Path | str, device: torch.device | str) -> MNISTClassifier:
    model = MNISTClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
