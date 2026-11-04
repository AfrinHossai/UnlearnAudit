"""Guarded, local-only GradDiff runner for the pinned TOFU Llama checkpoint.

The default command validates pinned local files and answer-only tokenization;
it does not load model weights or construct an optimizer. Weight updates need
an explicit ``--apply-updates``, device selection, and revision confirmation.
This small Apple Silicon port is not an OpenUnlearning benchmark reproduction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluate_tofu_baseline import (
    MODEL_CACHE_DIR,
    MODEL_ID,
    MODEL_REVISION,
)
from src.evaluate_tofu_paired import (
    DEFAULT_RETAIN_DATA_PATH,
    DATASET_REVISION,
    load_pinned_split,
)
from src.inspect_tofu import DEFAULT_DATA_PATH
from src.graddiff import causal_lm_nll
from src.graddiff_integration import (
    prepare_tofu_batch,
    validate_model_tokenizer,
)
from src.graddiff_model_smoke import (
    EXPECTED_MODEL_FILE_BYTES,
    SNAPSHOT_PATH,
    memory_snapshot,
)
from src.graddiff_train import assert_single_process

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_SEED = 2026
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_GAMMA = 1.0
DEFAULT_ALPHA = 1.0
FIXED_BATCH_SIZE = 1
MAX_STEPS_PER_INVOCATION = 40
EXPECTED_PARAMETER_COUNT = 1_235_814_400
FP32_ADAMW_STATE_FLOOR_BYTES = 19_773_030_400
PLANNING_ENVELOPE_GIB = {"minimum": 27.1, "maximum": 31.1}
FINITE_CHECK_CHUNK_ELEMENTS = 16_777_216


class GradDiffOutOfMemoryError(RuntimeError):
    """A local accelerator or host-memory failure with an explicit no-fallback report."""

    def __init__(self, phase: str, device: str, snapshot: Mapping[str, Any]):
        self.phase = phase
        self.device = device
        self.snapshot = dict(snapshot)
        super().__init__(
            f"Out of memory during {phase} on {device}. No device fallback was "
            "attempted and no checkpoint was saved. If the failure occurred in "
            "optimizer.step, discard this process because a partial in-memory update "
            f"cannot be ruled out. Memory snapshot: {json.dumps(self.snapshot, sort_keys=True)}"
        )


def _validate_numeric_options(args: argparse.Namespace) -> None:
    if isinstance(args.max_steps, bool) or not 1 <= args.max_steps <= MAX_STEPS_PER_INVOCATION:
        raise ValueError(
            f"--max-steps must be between 1 and {MAX_STEPS_PER_INVOCATION}; "
            "the runner allows at most one pass over forget01 per invocation."
        )
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        raise ValueError("--seed must be an integer")
    for name, value in (
        ("learning rate", args.learning_rate),
        ("max gradient norm", args.max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("gamma", args.gamma), ("alpha", args.alpha)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.gamma == 0 and args.alpha == 0:
        raise ValueError("gamma and alpha cannot both be zero")
    if not 2 <= args.max_sequence_length <= DEFAULT_MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"--max-sequence-length must be between 2 and {DEFAULT_MAX_SEQUENCE_LENGTH}"
        )


def resolve_training_device(requested: str, *, torch_module: Any | None = None) -> Any:
    """Resolve only explicit MPS or CPU requests; never fall back between them."""
    if requested not in {"mps", "cpu"}:
        raise ValueError("device must be selected explicitly as 'mps' or 'cpu'")
    if torch_module is None:
        import torch as torch_module
    if requested == "cpu":
        return torch_module.device("cpu")
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if not (mps and mps.is_built() and mps.is_available()):
        raise RuntimeError(
            "MPS was requested but is unavailable in this process; no CPU fallback "
            "was attempted. Rerun with --device cpu only if CPU execution is intended."
        )
    return torch_module.device("mps")


def _seed_runtime(torch_module: Any, device: Any, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if getattr(device, "type", None) == "mps":
        manual_seed = getattr(getattr(torch_module, "mps", None), "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)


def _check_local_checkpoint() -> dict[str, Any]:
    """Check the immutable cached snapshot without deserializing model weights."""
    if not SNAPSHOT_PATH.is_dir() or SNAPSHOT_PATH.name != MODEL_REVISION:
        raise FileNotFoundError("The pinned TOFU checkpoint snapshot is not cached locally.")
    config_path = SNAPSHOT_PATH / "config.json"
    weights_path = SNAPSHOT_PATH / "model.safetensors"
    tokenizer_path = SNAPSHOT_PATH / "tokenizer_config.json"
    if not all(path.is_file() for path in (config_path, weights_path, tokenizer_path)):
        raise FileNotFoundError("A required file is missing from the pinned local checkpoint.")
    observed_size = weights_path.stat().st_size
    if observed_size != EXPECTED_MODEL_FILE_BYTES:
        raise ValueError("The cached model weight file does not match the pinned size.")
    return {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "local_snapshot": str(SNAPSHOT_PATH.relative_to(PROJECT_ROOT)),
        "weight_file_bytes": observed_size,
        "local_files_only": True,
        "weights_loaded": False,
    }


def _load_local_inputs(
    *, seed: int, max_sequence_length: int
) -> tuple[Any, Any, Any, Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load local config, tokenizer, and checksum-pinned split files only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

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
    # Match the one-step integration smoke: forget row zero and a seeded
    # uniform retain draw. Additional steps advance through forget01 once.
    retain_rng = random.Random(seed)
    choices = [
        (step, step, retain_rng.randrange(len(retain_records)))
        for step in range(min(MAX_STEPS_PER_INVOCATION, len(forget_records)))
    ]
    data = {
        "forget_records": forget_records,
        "retain_records": retain_records,
        "forget_source": forget_source,
        "retain_source": retain_source,
        "choices": choices,
    }
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
    }
    return (
        torch,
        transformers,
        config,
        tokenizer,
        forget_records,
        retain_records,
        checkpoint,
        tokenizer_compatibility,
        {"data": data, "runtime": runtime},
    )


def _example_log(split: str, record_index: int, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable row references and counts without storing QA text or hashes."""
    labels = batch["labels"]
    return {
        "split": split,
        "record_index": record_index,
        "input_tokens": int(batch["attention_mask"].sum().item()),
        "supervised_target_tokens": int((labels[:, 1:] != -100).sum().item()),
    }


def _data_log(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_id": "locuslab/TOFU",
        "revision": DATASET_REVISION,
        "splits": {
            "forget01": dict(data["forget_source"]),
            "retain99": dict(data["retain_source"]),
        },
        "selected_examples_are_referenced_by_split_and_row_only": True,
        "raw_question_or_answer_text_logged": False,
    }


def _batch_metadata(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": int(batch["input_ids"].shape[0]),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "supervised_target_tokens": int((batch["labels"][:, 1:] != -100).sum().item()),
    }


def run_local_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Validate local assets and tokenization without loading weights or an optimizer."""
    assert_single_process()
    _validate_numeric_options(args)
    if args.device is not None:
        import torch

        selected_device = resolve_training_device(args.device, torch_module=torch)
        validated_device = str(selected_device)
    else:
        selected_device = None
        validated_device = None

    (
        torch,
        _transformers,
        config,
        tokenizer,
        forget_records,
        retain_records,
        checkpoint,
        tokenizer_compatibility,
        loaded,
    ) = _load_local_inputs(seed=args.seed, max_sequence_length=args.max_sequence_length)
    data = loaded["data"]
    step, forget_index, retain_index = data["choices"][0]
    forget_batch = prepare_tofu_batch(
        [forget_records[forget_index]], tokenizer,
        max_sequence_length=args.max_sequence_length,
    )
    retain_batch = prepare_tofu_batch(
        [retain_records[retain_index]], tokenizer,
        max_sequence_length=args.max_sequence_length,
    )
    memory_device = selected_device if selected_device is not None else torch.device("cpu")
    return {
        "schema_version": "1.0",
        "mode": "local_validation_only",
        "status": "validated_no_model_weights_loaded",
        "requested_device": args.device,
        "validated_device": validated_device,
        "actual_device": "not_allocated_validation_only",
        "model": checkpoint,
        "tokenizer": tokenizer_compatibility,
        "dataset": _data_log(data),
        "selected_examples": [
            _example_log("forget01", forget_index, forget_batch),
            _example_log("retain99", retain_index, retain_batch),
        ],
        "batches": {
            "forget01": _batch_metadata(forget_batch),
            "retain99": _batch_metadata(retain_batch),
        },
        "hyperparameters": {
            "seed": args.seed,
            "max_optimizer_steps": args.max_steps,
            "batch_size": FIXED_BATCH_SIZE,
            "max_sequence_length": args.max_sequence_length,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
        },
        "optimizer_created": False,
        "weight_updates_enabled": False,
        "memory_snapshots": {
            "validation_complete": memory_snapshot(torch, memory_device),
        },
        "runtime": loaded["runtime"],
    }


def _is_finite_in_chunks(torch_module: Any, tensor: Any) -> bool:
    flat = tensor.detach().reshape(-1)
    for offset in range(0, flat.numel(), FINITE_CHECK_CHUNK_ELEMENTS):
        chunk = flat[offset : offset + FINITE_CHECK_CHUNK_ELEMENTS]
        if not bool(torch_module.isfinite(chunk).all().item()):
            return False
    return True


def _assert_finite_gradients(model: Any, torch_module: Any) -> None:
    observed = False
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        observed = True
        if not _is_finite_in_chunks(torch_module, gradient):
            raise FloatingPointError("model produced a non-finite gradient")
    if not observed:
        raise RuntimeError("model produced no gradients; refusing optimizer step")


def _assert_finite_parameters(model: Any, torch_module: Any) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad and not _is_finite_in_chunks(torch_module, parameter):
            raise FloatingPointError("optimizer produced a non-finite model parameter")


def checked_optimizer_step(
    model: Any,
    optimizer: Any,
    *,
    max_grad_norm: float,
    torch_module: Any | None = None,
) -> dict[str, float | bool]:
    """Reject bad gradients, clip in place, then perform one optimizer update."""
    if torch_module is None:
        import torch as torch_module
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    _assert_finite_gradients(model, torch_module)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch_module.nn.utils.clip_grad_norm_(
        parameters,
        max_grad_norm,
        error_if_nonfinite=True,
        foreach=False,
    )
    norm_value = float(norm.detach().item())
    if not math.isfinite(norm_value):
        raise FloatingPointError("gradient norm is non-finite")
    _assert_finite_gradients(model, torch_module)
    optimizer.step()
    _assert_finite_parameters(model, torch_module)
    return {
        "gradient_norm_before_clipping": norm_value,
        "gradient_clipping_applied": norm_value > max_grad_norm,
    }


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "mps backend out of memory",
            "failed to allocate",
            "cannot allocate memory",
        )
    )


def _validate_output_path(path_value: str | Path, *, kind: str) -> Path:
    """Allow new artifacts only beneath the Git-ignored repository outputs/ tree."""
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    output_root = OUTPUT_ROOT.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    try:
        relative = candidate_resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"{kind} path must be inside the local outputs/ directory") from exc
    if not relative.parts or output_root != OUTPUT_ROOT:
        raise ValueError("outputs/ must be a real repository-local directory, not a symlink")
    if candidate.exists():
        raise FileExistsError(f"refusing to overwrite existing {kind} output")
    original = SNAPSHOT_PATH.resolve(strict=False)
    if candidate_resolved == original or original in candidate_resolved.parents:
        raise ValueError("refusing to write inside the original pinned TOFU checkpoint")
    if candidate_resolved in original.parents:
        raise ValueError("refusing to replace a parent of the original TOFU checkpoint")
    try:
        relative_to_repo = candidate_resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("output must resolve inside this repository") from exc
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", str(relative_to_repo)],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise ValueError("output path is not Git-ignored; refusing to write model artifacts")
    return candidate_resolved


def _write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = _validate_output_path(path, kind="report")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _loss_value(torch_module: Any, model: Any, batch: Mapping[str, Any], split: str) -> tuple[Any, float, float]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    )
    model_loss = getattr(outputs, "loss", None)
    logits = getattr(outputs, "logits", None)
    if model_loss is None or logits is None:
        raise RuntimeError(f"{split} model output did not include both loss and logits")
    local_loss = causal_lm_nll(logits, batch["labels"])
    if not bool(torch_module.isclose(model_loss.detach().float(), local_loss.detach().float(), atol=1e-5, rtol=1e-4).item()):
        raise ValueError(f"{split} model loss does not match the validated local causal NLL")
    difference = float((model_loss.detach().float() - local_loss.detach().float()).abs().item())
    return local_loss, float(local_loss.detach().item()), difference


def _train_one_step(
    *,
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    forget_record: Mapping[str, Any],
    retain_record: Mapping[str, Any],
    forget_index: int,
    retain_index: int,
    device: Any,
    args: argparse.Namespace,
    torch_module: Any,
) -> dict[str, Any]:
    forget_batch = prepare_tofu_batch(
        [forget_record], tokenizer, device=device,
        max_sequence_length=args.max_sequence_length,
    )
    retain_batch = prepare_tofu_batch(
        [retain_record], tokenizer, device=device,
        max_sequence_length=args.max_sequence_length,
    )
    optimizer.zero_grad(set_to_none=True)
    forget_nll, forget_value, forget_parity = _loss_value(
        torch_module, model, forget_batch, "forget01"
    )
    (-args.gamma * forget_nll).backward()
    _assert_finite_gradients(model, torch_module)
    del forget_nll

    retain_nll, retain_value, retain_parity = _loss_value(
        torch_module, model, retain_batch, "retain99"
    )
    (args.alpha * retain_nll).backward()
    _assert_finite_gradients(model, torch_module)
    objective_value = float((-args.gamma * forget_value + args.alpha * retain_value))
    if not math.isfinite(objective_value):
        raise FloatingPointError("GradDiff objective is non-finite")
    del retain_nll

    gradient_report = checked_optimizer_step(
        model,
        optimizer,
        max_grad_norm=args.max_grad_norm,
        torch_module=torch_module,
    )
    optimizer.zero_grad(set_to_none=True)
    return {
        "step": forget_index + 1,
        "examples": [
            _example_log("forget01", forget_index, forget_batch),
            _example_log("retain99", retain_index, retain_batch),
        ],
        "losses": {
            "forget_nll": forget_value,
            "retain_nll": retain_value,
            "forget_objective_term": -args.gamma * forget_value,
            "retain_objective_term": args.alpha * retain_value,
            "objective": objective_value,
            "forget_model_loss_abs_difference": forget_parity,
            "retain_model_loss_abs_difference": retain_parity,
        },
        "gradient": {
            **gradient_report,
            "max_grad_norm": args.max_grad_norm,
        },
        "batch_size": FIXED_BATCH_SIZE,
        "sequence_lengths": {
            "forget01": int(forget_batch["input_ids"].shape[1]),
            "retain99": int(retain_batch["input_ids"].shape[1]),
        },
    }


def _save_checkpoint_safely(model: Any, tokenizer: Any, path: str | Path) -> str:
    target = _validate_output_path(path, kind="checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    return str(target.relative_to(PROJECT_ROOT))


def _update_mode_requirements(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if not getattr(args, "apply_updates", False) or getattr(args, "dry_run", False):
        raise ValueError("weight updates require --apply-updates and cannot use --dry-run")
    _validate_numeric_options(args)
    if args.device is None:
        raise ValueError("weight updates require an explicit --device mps or --device cpu")
    if args.confirm_model_revision != MODEL_REVISION:
        raise ValueError(
            "weight updates require --confirm-model-revision followed by the exact pinned revision"
        )
    if args.report_path is None:
        raise ValueError("weight updates require --report-path inside Git-ignored outputs/")
    report_path = _validate_output_path(args.report_path, kind="report")
    checkpoint_path = None
    if args.save_checkpoint is not None:
        checkpoint_path = _validate_output_path(args.save_checkpoint, kind="checkpoint")
        if report_path == checkpoint_path or report_path in checkpoint_path.parents or checkpoint_path in report_path.parents:
            raise ValueError("report and checkpoint outputs must use separate paths")
    return report_path, checkpoint_path


def run_real_updates(args: argparse.Namespace) -> dict[str, Any]:
    """Load the pinned local model and run at most one forget/retain pair per step."""
    report_path, checkpoint_path = _update_mode_requirements(args)
    assert_single_process()
    import torch
    from transformers import AutoModelForCausalLM

    device = resolve_training_device(args.device, torch_module=torch)
    _seed_runtime(torch, device, args.seed)
    phase = "local data and tokenizer validation"
    model = None
    optimizer = None
    try:
        (
            torch,
            transformers,
            config,
            tokenizer,
            forget_records,
            retain_records,
            checkpoint,
            tokenizer_compatibility,
            loaded,
        ) = _load_local_inputs(seed=args.seed, max_sequence_length=args.max_sequence_length)
        data = loaded["data"]
        memory = {
            "before_model_load": memory_snapshot(torch, device),
        }
        phase = "pinned FP32 model load"
        model = AutoModelForCausalLM.from_pretrained(
            str(SNAPSHOT_PATH),
            config=config,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        model.to(device)
        loaded_model_compatibility = validate_model_tokenizer(
            model.config,
            tokenizer,
            max_sequence_length=args.max_sequence_length,
            model=model,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.train()
        memory["after_model_load_and_gradient_checkpointing"] = memory_snapshot(torch, device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        records: list[dict[str, Any]] = []
        for step_index, (_step, forget_index, retain_index) in enumerate(
            data["choices"][: args.max_steps]
        ):
            before_step = memory_snapshot(torch, device)
            phase = (
                f"forward/backward, gradient clipping, or optimizer.step for step "
                f"{step_index + 1}"
            )
            record = _train_one_step(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                forget_record=forget_records[forget_index],
                retain_record=retain_records[retain_index],
                forget_index=forget_index,
                retain_index=retain_index,
                device=device,
                args=args,
                torch_module=torch,
            )
            record["memory_before_step"] = before_step
            record["memory_after_step"] = memory_snapshot(torch, device)
            records.append(record)

        saved_to = None
        if checkpoint_path is not None:
            phase = "optional local checkpoint save"
            saved_to = _save_checkpoint_safely(model, tokenizer, checkpoint_path)
        memory["after_training_and_optional_save"] = memory_snapshot(torch, device)
        report = {
            "schema_version": "1.0",
            "mode": "real_model_graddiff_training",
            "status": "completed_local_update",
            "actual_device": str(device),
            "model": {
                **checkpoint,
                "weights_loaded": True,
                "precision": "float32",
                "attention_implementation": "sdpa",
                "gradient_checkpointing": True,
            },
            "tokenizer": tokenizer_compatibility,
            "loaded_model_compatibility": loaded_model_compatibility,
            "dataset": _data_log(data),
            "runtime": loaded["runtime"],
            "hyperparameters": {
                "seed": args.seed,
                "max_optimizer_steps": args.max_steps,
                "optimizer_steps_completed": len(records),
                "batch_size": FIXED_BATCH_SIZE,
                "max_sequence_length": args.max_sequence_length,
                "gamma": args.gamma,
                "alpha": args.alpha,
                "learning_rate": args.learning_rate,
                "weight_decay": 0.0,
                "max_grad_norm": args.max_grad_norm,
                "optimizer": "torch.optim.AdamW(foreach=False, fused=False)",
                "precision": "float32",
                "gradient_checkpointing": True,
            },
            "steps": records,
            "memory_snapshots": memory,
            "memory_estimate": {
                "estimated_trainable_parameters": EXPECTED_PARAMETER_COUNT,
                "fp32_parameter_gradient_two_moment_adamw_floor_bytes": FP32_ADAMW_STATE_FLOOR_BYTES,
                "fp32_state_floor_gib": round(FP32_ADAMW_STATE_FLOOR_BYTES / (1024**3), 2),
                "planning_envelope_gib": PLANNING_ENVELOPE_GIB,
                "is_measured_peak": False,
                "excludes_activations_transients_and_other_processes": True,
            },
            "checkpoint": {
                "saved": saved_to is not None,
                "local_ignored_output": saved_to,
                "original_checkpoint_overwritten": False,
            },
            "weight_updates_enabled": True,
            "raw_question_or_answer_text_logged": False,
            "benchmark_reproduction_claim": False,
        }
        return report
    except (RuntimeError, MemoryError) as exc:
        if _is_oom_error(exc):
            if optimizer is not None:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
            try:
                snapshot = memory_snapshot(torch, device)
            except Exception:
                snapshot = {"snapshot_unavailable": True}
            raise GradDiffOutOfMemoryError(phase, str(device), snapshot) from exc
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="explicitly run the default local-only validation (no model weights or updates)",
    )
    parser.add_argument(
        "--apply-updates",
        action="store_true",
        help="explicitly load the pinned real model and enable optimizer updates",
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--confirm-model-revision", default=None)
    parser.add_argument("--max-steps", "--max-optimizer-steps", dest="max_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--save-checkpoint", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.apply_updates and args.dry_run:
            raise ValueError("--dry-run and --apply-updates cannot be used together")
        if args.apply_updates:
            report_path, _checkpoint_path = _update_mode_requirements(args)
            report = run_real_updates(args)
            _write_report(report_path, report)
        else:
            if args.confirm_model_revision is not None:
                raise ValueError("--confirm-model-revision is only valid with --apply-updates")
            if args.save_checkpoint is not None:
                raise ValueError("--save-checkpoint requires --apply-updates")
            if args.report_path is not None:
                _validate_output_path(args.report_path, kind="report")
            report = run_local_validation(args)
            if args.report_path is not None:
                _write_report(args.report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except GradDiffOutOfMemoryError as exc:
        print(
            json.dumps(
                {"status": "failed_out_of_memory", "error": str(exc), "no_device_fallback": True},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, FileExistsError, FloatingPointError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "failed_safe", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
