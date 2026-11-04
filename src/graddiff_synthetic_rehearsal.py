"""Tiny random-model backward rehearsal; never loads pretrained assets or steps."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.graddiff_model_smoke import memory_snapshot
from src.graddiff_real_train import (
    _seed_runtime, _write_report, _validate_output_path, backward_pair, build_parser,
    configure_training_model, resolve_training_device,
)
from src.graddiff_train import assert_single_process


def tiny_model() -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
        tie_word_embeddings=True, eos_token_id=2, pad_token_id=2,
        attention_dropout=0.0, attn_implementation="sdpa",
    ))


def synthetic_batches(device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    batches = []
    for row in ([1, 3, 4, 5, 2], [1, 6, 7, 8, 2]):
        ids = torch.tensor([row], device=device)
        labels = ids.clone()
        labels[:, :2] = -100
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    return batches[0], batches[1]


def run_rehearsal(*, device_name: str, seed: int = 2026) -> dict[str, Any]:
    import torch

    assert_single_process()
    device = resolve_training_device(device_name, torch_module=torch)
    _seed_runtime(torch, device, seed)
    model = tiny_model().to(device)
    settings = configure_training_model(model)
    originals = [p.detach().clone() for p in model.parameters()]
    versions = tuple(p._version for p in model.parameters())
    snapshots = [{"stage": "before_backward", **memory_snapshot(torch, device)}]

    def observe(stage: str) -> None:
        snapshots.append({"stage": stage, **memory_snapshot(torch, device)})

    args = build_parser().parse_args(["--optimizer", "sgd", "--seed", str(seed)])
    losses = backward_pair(model, *synthetic_batches(device), args, observe=observe)
    gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    gradient_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None
    ])).item()
    unchanged = versions == tuple(p._version for p in model.parameters()) and all(
        torch.equal(before, after) for before, after in zip(originals, model.parameters())
    )
    if not unchanged:
        raise RuntimeError("synthetic backward unexpectedly changed weights")
    model.zero_grad(set_to_none=True)
    observe("after_gradient_clear")
    return {
        "mode": "synthetic_backward_only", "actual_device": str(device), "seed": seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "observed_gradient_tensor_bytes": gradient_bytes, "observed_gradient_l2_norm": gradient_norm,
        "checkpointing": settings, "losses": losses, "observed_memory_snapshots": snapshots,
        "process_peak_rss_is_lifetime_high_water_not_incremental_peak": True,
        "snapshots_are_not_continuous_peaks": True,
        "weights_unchanged": unchanged, "optimizer_created": False, "optimizer_steps": 0,
        "pretrained_weights_loaded": False, "external_data_loaded": False,
        "full_model_memory_estimate_is_not_validated_by_this_fixture": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report-path")
    args = parser.parse_args()
    if args.report_path:
        _validate_output_path(args.report_path, kind="synthetic report")
    report = run_rehearsal(device_name=args.device, seed=args.seed)
    if args.report_path:
        _write_report(args.report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
