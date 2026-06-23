"""Training helpers: augmentation, threshold tuning, metrics."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def augment_curves(
    global_s: torch.Tensor,
    local_s: torch.Tensor,
    *,
    noise_std: float = 0.0,
    scale_std: float = 0.02,
    max_roll: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Light 1D augmentations on folded flux (train only)."""
    if noise_std > 0:
        global_s = global_s + torch.randn_like(global_s) * noise_std
        local_s = local_s + torch.randn_like(local_s) * noise_std

    if scale_std > 0:
        g_scale = 1.0 + torch.randn(global_s.size(0), 1, 1, device=global_s.device) * scale_std
        l_scale = 1.0 + torch.randn(local_s.size(0), 1, 1, device=local_s.device) * scale_std
        global_s = global_s * g_scale
        local_s = local_s * l_scale

    if max_roll > 0:
        for b in range(global_s.size(0)):
            shift_g = int(torch.randint(-max_roll, max_roll + 1, (1,)).item())
            shift_l = int(torch.randint(-max_roll, max_roll + 1, (1,)).item())
            global_s[b] = torch.roll(global_s[b], shifts=shift_g, dims=-1)
            local_s[b] = torch.roll(local_s[b], shifts=shift_l, dims=-1)

    return global_s, local_s


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> tuple[float, dict]:
    """Scan thresholds on validation probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    best_t, best_score, best_stats = 0.5, -1.0, {}

    for t in np.linspace(0.2, 0.8, 61):
        pred = (probs >= t).astype(np.int64)
        f1 = f1_score(labels, pred, average="weighted", zero_division=0)
        prec = precision_score(labels, pred, average="weighted", zero_division=0)
        rec = recall_score(labels, pred, average="weighted", zero_division=0)
        score = {"f1": f1, "precision": prec, "recall": rec}[metric]
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_stats = {"f1": f1, "precision": prec, "recall": rec}

    return best_t, best_stats


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(np.float64)
