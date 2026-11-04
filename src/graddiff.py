"""Device-independent GradDiff causal-LM objective helpers."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _scalar_finite(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError(f"{name} must be a scalar torch.Tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).item()):
        raise FloatingPointError(f"{name} must be a finite floating point scalar")


def causal_lm_nll(logits: torch.Tensor, labels: torch.Tensor,
                  ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Mean next-token cross entropy over non-ignored targets (HF causal-LM convention)."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    batch, seq_len, vocab = logits.shape
    if labels.shape != (batch, seq_len) or batch < 1 or seq_len < 2 or vocab < 2:
        raise ValueError("logits and labels must have matching nontrivial causal-LM shapes")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("logits must be finite floating point values")
    if labels.is_floating_point() or labels.is_complex():
        raise ValueError("labels must use an integer dtype")
    shifted_labels = labels[:, 1:].to(device=logits.device, dtype=torch.long)
    valid = shifted_labels.ne(ignore_index)
    if not bool(valid.any().item()):
        raise ValueError("labels contain no next-token targets after causal shift")
    result = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab), shifted_labels.reshape(-1),
        ignore_index=ignore_index, reduction="mean",
    )
    _scalar_finite("causal NLL", result)
    return result


def _coefficient(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def graddiff_objective(forget_nll: torch.Tensor, retain_nll: torch.Tensor,
                       gamma: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """Return the minimized objective -gamma*mean_NLL(F)+alpha*mean_NLL(R)."""
    _scalar_finite("forget NLL", forget_nll)
    _scalar_finite("retain NLL", retain_nll)
    gamma, alpha = _coefficient("gamma", gamma), _coefficient("alpha", alpha)
    if gamma == 0 and alpha == 0:
        raise ValueError("gamma and alpha cannot both be zero")
    objective = -gamma * forget_nll + alpha * retain_nll
    _scalar_finite("GradDiff objective", objective)
    return objective
