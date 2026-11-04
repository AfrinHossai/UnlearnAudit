"""Synthetic, offline GradDiff training-step prototype (never loads real models)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Mapping
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # requirements.txt intentionally has no ML dependencies
    raise SystemExit("GradDiff prototype needs PyTorch; install requirements-ml.txt first") from exc

from src.graddiff import causal_lm_nll, graddiff_objective


def select_device(requested: str = "auto", torch_module: Any = torch) -> Any:
    """Select once; an MPS execution error is surfaced and never silently retried on CPU."""
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    built = bool(backend and backend.is_built())
    available = bool(backend and backend.is_available())
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "mps":
        if not built or not available:
            raise RuntimeError("MPS was requested but is not built and available in this process")
        return torch_module.device("mps")
    return torch_module.device("mps" if built and available else "cpu")


def assert_single_process(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = env.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid WORLD_SIZE={raw!r}; this prototype is single-process only") from exc
    if world_size != 1:
        raise RuntimeError(f"WORLD_SIZE={world_size}; this prototype supports only one process")
    return world_size


class TinySyntheticCausalLM(nn.Module):
    """Two independent task rows make forget/retain gradient behavior controllable."""

    def __init__(self) -> None:
        super().__init__()
        self.task_logits = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("synthetic input_ids must have shape [batch, sequence>=2]")
        batch_logits = self.task_logits[input_ids[:, 0].long()]
        return batch_logits[:, None, :].expand(-1, input_ids.shape[1], -1)


def synthetic_batches(device: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """One forget and one retain target, each labeled only as a next-token target."""
    return (
        {"input_ids": torch.tensor([[0, 0]], dtype=torch.long, device=device),
         "labels": torch.tensor([[-100, 1]], dtype=torch.long, device=device)},
        {"input_ids": torch.tensor([[1, 0]], dtype=torch.long, device=device),
         "labels": torch.tensor([[-100, 0]], dtype=torch.long, device=device)},
    )


def _batch_nll(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return causal_lm_nll(model(batch["input_ids"]), batch["labels"])


def _finite_gradients(model: nn.Module) -> float:
    squared_norm = 0.0
    saw_gradient = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        saw_gradient = True
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError("non-finite parameter gradient")
        # MPS does not support float64 tensors; FP32 norm accumulation is portable.
        squared_norm += float(parameter.grad.detach().float().square().sum().item())
    norm = math.sqrt(squared_norm)
    if not saw_gradient or not math.isfinite(norm):
        raise FloatingPointError("missing or non-finite aggregate gradient")
    return norm


def _check_finite_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError("optimizer produced non-finite model parameters")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not bool(torch.isfinite(value).all().item()):
                    raise FloatingPointError("optimizer produced non-finite state")


def graddiff_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    forget_batch: Mapping[str, torch.Tensor],
    retain_batch: Mapping[str, torch.Tensor],
    *, gamma: float = 1.0, alpha: float = 1.0,
) -> dict[str, float]:
    """Minimize -gamma*forget mean NLL + alpha*retain mean NLL for one update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    forget_before = _batch_nll(model, forget_batch)
    retain_before = _batch_nll(model, retain_batch)
    objective_before = graddiff_objective(forget_before, retain_before, gamma, alpha)
    objective_before.backward()
    gradient_norm = _finite_gradients(model)
    optimizer.step()
    _check_finite_model_and_optimizer(model, optimizer)
    with torch.no_grad():
        forget_after = _batch_nll(model, forget_batch)
        retain_after = _batch_nll(model, retain_batch)
        objective_after = graddiff_objective(forget_after, retain_after, gamma, alpha)
    return {
        "forget_nll_before": float(forget_before.detach().item()),
        "forget_nll_after": float(forget_after.item()),
        "retain_nll_before": float(retain_before.detach().item()),
        "retain_nll_after": float(retain_after.item()),
        "objective_before": float(objective_before.detach().item()),
        "objective_after": float(objective_after.item()),
        "gradient_norm": gradient_norm,
    }


def run_synthetic_dry_run(
    *, device: str = "auto", steps: int = 3, learning_rate: float = 0.05,
    seed: int = 2026, gamma: float = 1.0, alpha: float = 1.0,
) -> dict[str, Any]:
    assert_single_process()
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if (not math.isfinite(gamma) or gamma <= 0 or
            not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("synthetic dry-run gamma and alpha must be finite and positive")
    random.seed(seed)
    torch.manual_seed(seed)
    chosen = select_device(device)
    if chosen.type == "mps":
        torch.mps.manual_seed(seed)
    model = TinySyntheticCausalLM().to(chosen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    forget, retain = synthetic_batches(chosen)
    records = [graddiff_training_step(model, optimizer, forget, retain,
                                      gamma=gamma, alpha=alpha)
               for _ in range(steps)]
    if not all(row["forget_nll_after"] > row["forget_nll_before"] for row in records):
        raise AssertionError("controlled synthetic fixture did not increase forget NLL")
    if not all(row["retain_nll_after"] < row["retain_nll_before"] for row in records):
        raise AssertionError("controlled synthetic fixture did not decrease retain NLL")
    if not all(row["objective_after"] < row["objective_before"] for row in records):
        raise AssertionError("optimizer did not decrease the minimized GradDiff objective")
    return {
        "mode": "synthetic_dry_run",
        "actual_device": str(chosen),
        "world_size": 1,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "alpha": alpha,
        "loss_reduction": "mean over non-ignored shifted next-token labels, separately per split",
        "checkpoint_loaded": False,
        "external_data_loaded": False,
        "records": records,
        "assertions": {
            "finite_losses_gradients_parameters_optimizer_state": True,
            "forget_nll_increased_each_step": True,
            "retain_nll_decreased_each_step": True,
            "minimized_objective_decreased_each_step": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="required: execute only the built-in synthetic fixture")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("only --dry-run is supported; real model/data training is not implemented")
    report = run_synthetic_dry_run(
        device=args.device, steps=args.steps, learning_rate=args.learning_rate,
        seed=args.seed, gamma=args.gamma, alpha=args.alpha,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
