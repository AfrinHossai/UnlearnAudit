"""Read-only, local-cache-only GradDiff loss smoke with the TOFU Llama model.

This command loads the pinned checkpoint and tokenizer, prepares one forget and
one retain example, and compares the model's loss with the local causal NLL.
It does not create an optimizer, run backward, update weights, or write outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
from typing import Any

from src.evaluate_tofu_baseline import CHAT_DATE, MODEL_CACHE_DIR, MODEL_ID, MODEL_REVISION
from src.evaluate_tofu_paired import (
    DEFAULT_RETAIN_DATA_PATH,
    load_pinned_split,
)
from src.inspect_tofu import DEFAULT_DATA_PATH
from src.graddiff_integration import (
    choose_readonly_forget_retain_pair,
    evaluate_readonly_pair,
    prepare_tofu_batch,
    resolve_smoke_device,
    validate_model_tokenizer,
)
from src.graddiff_train import assert_single_process

SNAPSHOT_PATH = (
    MODEL_CACHE_DIR
    / f"models--{MODEL_ID.replace('/', '--')}"
    / "snapshots"
    / MODEL_REVISION
)
EXPECTED_MODEL_FILE_BYTES = 2_471_645_608
DEFAULT_MAX_SEQUENCE_LENGTH = 512


def _total_physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _process_peak_rss_bytes() -> int | None:
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # Darwin reports ru_maxrss in bytes; Linux and most BSDs report KiB.
    return peak if sys.platform == "darwin" else peak * 1024


def memory_snapshot(torch_module: Any, device: Any) -> dict[str, int | None]:
    """Report host capacity, process high-water RSS, and MPS allocator use."""
    report: dict[str, int | None] = {
        "total_physical_memory_bytes": _total_physical_memory_bytes(),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "mps_tensor_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
    }
    if getattr(device, "type", None) == "mps":
        mps = getattr(torch_module, "mps", None)
        if mps is not None:
            synchronize = getattr(mps, "synchronize", None)
            if callable(synchronize):
                synchronize()
            for key, method_name in (
                ("mps_tensor_allocated_bytes", "current_allocated_memory"),
                ("mps_driver_allocated_bytes", "driver_allocated_memory"),
            ):
                method = getattr(mps, method_name, None)
                if callable(method):
                    try:
                        report[key] = int(method())
                    except (RuntimeError, NotImplementedError):
                        report[key] = None
    return report


def _check_local_checkpoint() -> dict[str, Any]:
    """Require the exact cached snapshot and model weight file; never download."""
    if not SNAPSHOT_PATH.is_dir():
        raise FileNotFoundError(
            f"Pinned model snapshot is not cached at {SNAPSHOT_PATH}; this smoke test "
            "is local-only and will not download it."
        )
    config_path = SNAPSHOT_PATH / "config.json"
    weights_path = SNAPSHOT_PATH / "model.safetensors"
    for path in (config_path, weights_path, SNAPSHOT_PATH / "tokenizer_config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required pinned checkpoint file is missing: {path}")
    observed_bytes = weights_path.stat().st_size
    if observed_bytes != EXPECTED_MODEL_FILE_BYTES:
        raise ValueError(
            f"Cached model.safetensors has {observed_bytes} bytes; expected the pinned "
            f"{EXPECTED_MODEL_FILE_BYTES}-byte file."
        )
    if SNAPSHOT_PATH.name != MODEL_REVISION:
        raise ValueError("resolved checkpoint directory does not match the pinned revision")
    return {
        "snapshot_path": str(SNAPSHOT_PATH),
        "revision": MODEL_REVISION,
        "model_file": str(weights_path),
        "model_file_bytes": observed_bytes,
        "local_files_only": True,
    }


def run_readonly_smoke(
    *,
    device: str = "mps",
    allow_cpu_fallback: bool = False,
    seed: int = 2026,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    """Load cached local assets and compute one paired read-only loss."""
    assert_single_process()
    if allow_cpu_fallback and device != "auto":
        raise ValueError("--allow-cpu-fallback can only be used with --device auto")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")

    # Prevent any Hub request from this process, even if a library path changes.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    selected_device = resolve_smoke_device(
        device, allow_cpu_fallback=allow_cpu_fallback, torch_module=torch
    )
    checkpoint = _check_local_checkpoint()
    config = AutoConfig.from_pretrained(
        str(SNAPSHOT_PATH), local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(SNAPSHOT_PATH), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    tokenizer_compatibility = validate_model_tokenizer(
        config, tokenizer, max_sequence_length=max_sequence_length
    )

    forget_records, forget_source = load_pinned_split(
        "forget01", DEFAULT_DATA_PATH, download_if_missing=False
    )
    retain_records, retain_source = load_pinned_split(
        "retain99", DEFAULT_RETAIN_DATA_PATH, download_if_missing=False
    )
    forget_record, retain_record, selected_indices = choose_readonly_forget_retain_pair(
        forget_records, retain_records, seed=seed
    )
    forget_batch = prepare_tofu_batch(
        [forget_record], tokenizer, device=selected_device,
        max_sequence_length=max_sequence_length,
    )
    retain_batch = prepare_tofu_batch(
        [retain_record], tokenizer, device=selected_device,
        max_sequence_length=max_sequence_length,
    )

    memory_before_model_load = memory_snapshot(torch, selected_device)
    model_dtype = torch.float16 if selected_device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT_PATH),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.to(selected_device)
    model.eval()
    loaded_compatibility = validate_model_tokenizer(
        model.config,
        tokenizer,
        max_sequence_length=max_sequence_length,
        model=model,
    )
    memory_after_model_load = memory_snapshot(torch, selected_device)

    loss_report = evaluate_readonly_pair(model, forget_batch, retain_batch)
    memory_after_forwards = memory_snapshot(torch, selected_device)

    return {
        "mode": "read_only_real_model_graddiff_loss_smoke",
        "read_only": True,
        "device": str(selected_device),
        "model_dtype": str(model_dtype).removeprefix("torch."),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model": {
            "repo_id": MODEL_ID,
            **checkpoint,
            "config_model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "attention_implementation": "sdpa",
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            **tokenizer_compatibility,
            "chat_date": CHAT_DATE,
            "assistant_mask_strategy": "generation-prefix token boundary; tokenizer template has no generation spans",
        },
        "loaded_model_compatibility": loaded_compatibility,
        "dataset": {
            "forget": forget_source,
            "retain": retain_source,
            "sampling_seed": seed,
            "sampling": "forget row 0; one retain row selected by random.Random(seed), uniformly with replacement semantics for this single draw",
            **selected_indices,
        },
        "batch": {
            "forget_batch_size": int(forget_batch["input_ids"].shape[0]),
            "retain_batch_size": int(retain_batch["input_ids"].shape[0]),
            "max_sequence_length": max_sequence_length,
            "padding_side": tokenizer.padding_side,
        },
        "memory": {
            "before_model_load": memory_before_model_load,
            "after_model_load": memory_after_model_load,
            "after_forwards": memory_after_forwards,
            "peak_rss_is_process_high_water_mark": True,
        },
        "losses": loss_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("mps", "cpu", "auto"), default="mps",
        help="default requires MPS; CPU must be selected explicitly or allowed as auto fallback",
    )
    parser.add_argument(
        "--allow-cpu-fallback", action="store_true",
        help="with --device auto only, permit CPU if MPS is unavailable",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    args = parser.parse_args(argv)
    report = run_readonly_smoke(
        device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
