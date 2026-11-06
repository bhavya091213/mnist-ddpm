"""Generation-quality metrics (classifier-feature Frechet distance, coverage)."""
from __future__ import annotations


import torch
from torch import Tensor

from mnist_flow.classifier import MNISTClassifier, extract_features, predict_logits


def _mean_cov(feat: Tensor) -> tuple[Tensor, Tensor]:
    f = feat.detach().cpu().to(torch.float64)
    if f.ndim != 2 or f.shape[0] < 2:
        raise ValueError("features must be (N, D) with N >= 2")
    mu = f.mean(0)
    fc = f - mu
    return mu, fc.T @ fc / (f.shape[0] - 1)


def _psd_sqrt(m: Tensor) -> Tensor:
    evals, evecs = torch.linalg.eigh((m + m.T) / 2)
    return (evecs * evals.clamp_min(0).sqrt()) @ evecs.T


def frechet_distance(feat_a: Tensor, feat_b: Tensor) -> float:
    """||mu_a-mu_b||^2 + Tr(Sa + Sb - 2 (Sa Sb)^(1/2)), float64 on CPU.

    Tr((Sa Sb)^(1/2)) equals the sum of sqrt of eigenvalues of the symmetric
    PSD matrix Sa^(1/2) Sb Sa^(1/2), so no complex arithmetic is needed.
    """
    if feat_a.ndim != 2 or feat_b.ndim != 2 or feat_a.shape[1] != feat_b.shape[1]:
        raise ValueError(f"feature shapes incompatible: {tuple(feat_a.shape)} vs {tuple(feat_b.shape)}")
    mu_a, sa = _mean_cov(feat_a)
    mu_b, sb = _mean_cov(feat_b)
    root_a = _psd_sqrt(sa)
    eig = torch.linalg.eigvalsh(root_a @ sb @ root_a).clamp_min(0)
    tr_sqrt = eig.sqrt().sum()
    d = (mu_a - mu_b).pow(2).sum() + sa.trace() + sb.trace() - 2 * tr_sqrt
    return float(d)


def class_histogram(logits: Tensor) -> list[int]:
    return torch.bincount(logits.argmax(1).cpu(), minlength=10).tolist()


def coverage_entropy(logits: Tensor) -> float:
    """Shannon entropy (nats) of the predicted-class histogram; max ln 10."""
    h = torch.tensor(class_histogram(logits), dtype=torch.float64)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum()) if p.numel() else 0.0


def mean_confidence(logits: Tensor) -> float:
    return float(logits.detach().cpu().to(torch.float64).softmax(1).max(1).values.mean())


def compute_generation_metrics(classifier: MNISTClassifier, real_x: Tensor, gen_x: Tensor) -> dict:
    gen_logits = predict_logits(classifier, gen_x)
    return {
        "fid": frechet_distance(extract_features(classifier, real_x), extract_features(classifier, gen_x)),
        "coverage_entropy": coverage_entropy(gen_logits),
        "mean_confidence": mean_confidence(gen_logits),
        "class_histogram": class_histogram(gen_logits),
        "n_real": int(real_x.shape[0]),
        "n_gen": int(gen_x.shape[0]),
    }
