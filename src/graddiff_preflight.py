"""Optional FP32 GradDiff load/forward preflight; default prints a plan only.

Run --run-preflight only in a normal macOS Terminal. This module has no
training or checkpoint-saving path. --synthetic validates forward losses only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

from src.evaluate_tofu_paired import (
    DEFAULT_MANIFEST_PATH, DEFAULT_RETAIN_DATA_PATH, load_pinned_split,
)
from src.inspect_tofu import DEFAULT_DATA_PATH
from src.graddiff_integration import (
    evaluate_readonly_pair, prepare_tofu_batch, validate_model_tokenizer,
)
from src.graddiff_model_smoke import memory_snapshot, SNAPSHOT_PATH
from src.graddiff_real_train import (
    MODEL_REVISION, EXPECTED_PARAMETER_COUNT, FP32_ADAMW_STATE_FLOOR_BYTES,
    _check_local_checkpoint, _validate_output_path, _write_report,
    resolve_training_device,
)
from src.graddiff_train import assert_single_process

GIB = 1024**3
MANIFEST_FILE_SHA256 = "83581108e06c7b73c722269e52a055af3accdbe9b8e07d24422191e5a196938b"


def training_memory_estimate(total_bytes: int = 48 * GIB) -> dict[str, Any]:
    # Deliberately more conservative than the earlier 27.1--31.1 GiB envelope.
    required = FP32_ADAMW_STATE_FLOOR_BYTES + 24 * GIB
    return {
        "is_estimate_not_measurement": True,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "fp32_weights_gradients_adam_moments_bytes": FP32_ADAMW_STATE_FLOOR_BYTES,
        "activation_and_workspace_allowance_bytes": 8 * GIB,
        "os_and_other_process_reserve_bytes": 12 * GIB,
        "additional_uncertainty_margin_bytes": 4 * GIB,
        "total_planning_bytes": required,
        "host_capacity_bytes": total_bytes,
        "capacity_headroom_bytes": total_bytes - required,
        "fits_total_capacity_on_paper": total_bytes >= required,
        "training_readiness_proven": False,
    }


def parse_vm_stat(output: str) -> int:
    """Conservative physical availability: free + speculative pages only.

    Inactive, compressed and purgeable pages are deliberately excluded. This
    lower bound can reject a host whose caches could in fact be reclaimed.
    """
    page_size = re.search(r"page size of (\d+) bytes", output)
    if page_size is None:
        raise RuntimeError("vm_stat page size unavailable")
    pages = []
    for name in ("Pages free", "Pages speculative"):
        match = re.search(rf"^{name}:\s+(\d+)\.?$", output, re.MULTILINE)
        if match is None:
            raise RuntimeError("vm_stat free/speculative page reading unavailable")
        pages.append(int(match.group(1)))
    return int(page_size.group(1)) * sum(pages)


def _command(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout


def host_memory_snapshot() -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("physical preflight requires macOS memory telemetry")
    available = parse_vm_stat(_command(["/usr/bin/vm_stat"]))
    pressure = _command(["/usr/bin/memory_pressure", "-Q"])
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if match is None:
        raise RuntimeError("system memory-pressure reading unavailable")
    rss = int(_command(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) * 1024
    return {
        "available_physical_memory_lower_bound_bytes": available,
        "availability_basis": "vm_stat free + speculative pages; excludes reclaimable caches",
        "system_memory_free_percentage": int(match.group(1)),
        "process_current_rss_bytes": rss,
    }


def check_memory_thresholds(snapshot: dict[str, Any], *, before_load: bool) -> None:
    minimum = (20 if before_load else 6) * GIB
    available = snapshot.get("available_physical_memory_lower_bound_bytes")
    free_percentage = snapshot.get("system_memory_free_percentage")
    total = snapshot.get("total_physical_memory_bytes")
    if any(not isinstance(v, int) or isinstance(v, bool)
           for v in (available, free_percentage, total)):
        raise RuntimeError("STOP: required memory telemetry is unavailable")
    if not 0 <= free_percentage <= 100 or available > total or available < 0:
        raise RuntimeError("STOP: inconsistent memory telemetry")
    if available < minimum or free_percentage < 20:
        raise RuntimeError("STOP: insufficient available physical memory or high memory pressure")
    if before_load and total < 48 * GIB:
        raise RuntimeError("STOP: this conservative profile requires at least 48 GiB total memory")


def select_manifest_examples() -> tuple[dict[str, Any], list[tuple[str, int, dict[str, Any]]]]:
    raw = DEFAULT_MANIFEST_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_FILE_SHA256:
        raise ValueError("pinned manifest checksum mismatch")
    manifest = json.loads(raw)
    examples = []
    sources = {}
    for split, path in (("forget01", DEFAULT_DATA_PATH), ("retain99", DEFAULT_RETAIN_DATA_PATH)):
        records, source = load_pinned_split(split, path, download_if_missing=False)
        if source != manifest["datasets"][split]:
            raise ValueError("manifest dataset provenance mismatch")
        case = manifest["evaluation_sets"][split]["cases"][0]
        index = case["record_index"]
        record = records[index]
        for field, digest in (("question", "question_sha256"), ("answer", "target_answer_sha256")):
            if hashlib.sha256(record[field].encode("utf-8")).hexdigest() != case[digest]:
                raise ValueError("manifest example hash mismatch")
        examples.append((split, index, record))
        sources[split] = source
    return {"manifest_sha256": MANIFEST_FILE_SHA256, "splits": sources}, examples


def run_preflight(*, enabled: bool, device_name: str = "mps") -> dict[str, Any]:
    if not enabled:
        raise ValueError("full-model loading requires explicit --run-preflight")
    assert_single_process()
    os.environ["HF_HUB_OFFLINE"] = "1"
    checkpoint = _check_local_checkpoint()
    dataset, examples = select_manifest_examples()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = resolve_training_device(device_name, torch_module=torch)
    snapshots = []

    def observe(stage: str, *, before_load: bool = False) -> None:
        snapshot = {**memory_snapshot(torch, device), **host_memory_snapshot()}
        snapshot["stage"] = stage
        snapshots.append(snapshot)
        check_memory_thresholds(snapshot, before_load=before_load)

    observe("before_load", before_load=True)
    config = AutoConfig.from_pretrained(str(SNAPSHOT_PATH), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(SNAPSHOT_PATH), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    validate_model_tokenizer(config, tokenizer, max_sequence_length=512)
    batches = [prepare_tofu_batch([record], tokenizer, device=device, max_sequence_length=512)
               for _split, _index, record in examples]
    observe("immediately_before_model_load", before_load=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT_PATH), config=config, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    observe("after_load")
    load_seconds = time.monotonic() - started
    validate_model_tokenizer(model.config, tokenizer, max_sequence_length=512, model=model)
    timings = []
    forward_start = 0.0

    def before_forward(_model: Any, _inputs: Any) -> None:
        nonlocal forward_start
        observe(f"before_forward_{len(timings) + 1}")
        forward_start = time.monotonic()

    def after_forward(_model: Any, _inputs: Any, _outputs: Any) -> None:
        observe(f"after_forward_{len(timings) + 1}_logits_live")
        timings.append(time.monotonic() - forward_start)

    pre_hook = model.register_forward_pre_hook(before_forward)
    post_hook = model.register_forward_hook(after_forward)
    try:
        losses = evaluate_readonly_pair(model, batches[0], batches[1])
    finally:
        pre_hook.remove()
        post_hook.remove()
    observe("after_loss_validation")
    return {
        "mode": "readonly_fp32_full_model_preflight", "actual_device": str(device),
        "model": {**checkpoint, "weights_loaded": True, "precision": "float32", "revision": MODEL_REVISION},
        "dataset": dataset,
        "examples": [{"split": split, "record_index": index} for split, index, _record in examples],
        "losses": losses, "memory_snapshots": snapshots,
        "load_seconds_including_transfer_and_snapshot": load_seconds,
        "forward_seconds_including_telemetry": timings,
        "snapshots_are_not_continuous_peak_measurements": True,
        "training_memory_estimate": training_memory_estimate(snapshots[0]["total_physical_memory_bytes"]),
        "training_authorized": False,
    }


def run_synthetic_validation() -> dict[str, Any]:
    """Two tiny CPU forwards under inference_mode; no autograd or optimizer."""
    import torch
    from src.graddiff import causal_lm_nll

    class Fixture(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

        def forward(self, input_ids: Any, attention_mask: Any, labels: Any, use_cache: bool) -> Any:
            logits = self.values.expand(1, input_ids.shape[1], 2)
            return SimpleNamespace(logits=logits, loss=causal_lm_nll(logits, labels))

    batch = {"input_ids": torch.tensor([[0, 1, 0]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
             "labels": torch.tensor([[-100, 1, 0]])}
    return {"mode": "synthetic_readonly_validation", "actual_device": "cpu",
            "checkpoint_loaded": False, "losses": evaluate_readonly_pair(Fixture(), batch, batch)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run-preflight", action="store_true")
    modes.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--report-path")
    args = parser.parse_args(argv)
    try:
        if args.report_path:
            _validate_output_path(args.report_path, kind="preflight report")
        if args.run_preflight:
            report = run_preflight(enabled=True, device_name=args.device)
        elif args.synthetic:
            report = run_synthetic_validation()
        else:
            report = {"mode": "plan_only", "weights_loaded": False,
                      "required_opt_in": "--run-preflight", "training_memory_estimate": training_memory_estimate()}
        if args.report_path:
            _write_report(args.report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, RuntimeError, ValueError, FloatingPointError, MemoryError, subprocess.SubprocessError) as exc:
        # Avoid printing arbitrary third-party exception messages or dataset text.
        reason = str(exc) if str(exc).startswith("STOP:") else f"STOP: {type(exc).__name__} in preflight"
        print(reason + ". Check assets, telemetry, memory thresholds, loss parity and allocation. "
              "No retry or device fallback. Check Activity Monitor; no report indicates an incomplete run.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
