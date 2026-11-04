"""Guarded, optional SmolLM2 backward-only memory rehearsal; default validates.

No optimizer, checkpoint writer, download option or training entrypoint exists.
The full-model --run-backward-rehearsal command is for a normal Mac Terminal,
never the Codex sandbox. A pass is not training clearance or unlearning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import small_model_preflight as readonly
from src.evaluate_tofu_paired import load_pinned_split
from src.graddiff_integration import collate_causal_examples
from src.inspect_tofu import DATASET_ID, DATASET_REVISION, sha256_file
from src.small_model_experiment import MANIFEST, MODEL_ID, PROFILE, REVISION, ROOT, digest, encode_plain_qa
from src.small_model_memory import GIB, select_memory_guard

GUARD_ID = "smollm2_135m_fp32_mps_one_pair_backward_only_v1"
MANIFEST_FILE_SHA256 = "0ba5d80715dd9fb0c542d16f5167c23a0464da89a2f413d6a6bbd18235f2ea21"
MAX_TOKENS = 128
PASS = "backward_rehearsal_passed_not_training_clearance"


def rehearsal_memory_policy(profile: Mapping[str, Any], *, device: str,
                            baseline_peak_rss_bytes: int | None = None) -> dict[str, Any]:
    """Uncalibrated estimates for this exact operation, not a training guard."""
    select_memory_guard(profile)
    if device != "mps":
        raise RuntimeError("backward rehearsal requires mps; no fallback")
    if baseline_peak_rss_bytes is not None and (
            type(baseline_peak_rss_bytes) is not int or baseline_peak_rss_bytes < 0):
        raise RuntimeError("invalid measured baseline RSS")
    weight = 4 * profile["model"]["parameter_count"]
    terms = {
        "three_fp32_weight_copies_estimate": 3 * weight,
        "fp32_gradients_estimate": weight,
        "source_file_residency_allowance_estimate": profile["model"]["weight_file_metadata_only"]["size_bytes"],
        "two_forward_graphs_and_backward_workspace_allowance_estimate": 4 * GIB,
        "tokenizer_host_cache_and_integrity_copy_allowance_estimate": GIB,
        "metal_runtime_cache_allowance_estimate": GIB,
        "additional_process_headroom_estimate": max(GIB, baseline_peak_rss_bytes or 0),
        "uncertainty_reserve_estimate": 4 * GIB,
    }
    incremental = sum(terms.values())
    return {"guard_id": GUARD_ID, "scope": "one_pair_backward_only", "model_id": MODEL_ID,
            "revision": REVISION, "profile_sha256": digest(profile), "device": device,
            "precision": "float32", "batch_size_per_split": 1, "max_sequence_length": MAX_TOKENS,
            "forward_count": 2, "backward_count": 1, "gradient_checkpointing": "non-reentrant",
            "use_cache": False, "optimizer_state_bytes": 0,
            "baseline_process_peak_rss_bytes_observed": baseline_peak_rss_bytes,
            "estimated_incremental_terms_bytes": terms, "estimated_incremental_total_bytes": incremental,
            "preload_available_required_bytes": 6 * GIB + ((incremental + GIB - 1) // GIB) * GIB,
            "continuing_available_required_bytes": 6 * GIB, "minimum_total_host_bytes": 48 * GIB,
            "minimum_system_memory_free_percentage": 20, "required_kernel_pressure_level": 1,
            "measured_backward_requirement": False, "training_authorized": False,
            "approved_small_model_training_threshold_bytes": None}


def select_pair(profile: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the pinned forget/retain files; sealed holdout stays unopened."""
    select_memory_guard(profile)
    readonly._safe_path(MANIFEST, ROOT / "experiments")
    if sha256_file(MANIFEST) != MANIFEST_FILE_SHA256:
        raise ValueError("frozen manifest checksum mismatch")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["profile_sha256"] != digest(profile):
        raise ValueError("manifest/profile mismatch")
    pair = manifest["unlearning_and_retain_control_pairs"][0]
    if pair != {"forget": "forget01:0000", "retain": "retain99:0501"}:
        raise ValueError("unexpected deterministic pair")
    records, provenance = {}, {"repo_id": DATASET_ID, "revision": DATASET_REVISION,
                               "license": "MIT", "manifest_file_sha256": MANIFEST_FILE_SHA256,
                               "selection": "first frozen manifest pair; no adaptive selection", "cases": {}}
    for role, set_name in (("forget", "forget_direct"), ("retain", "retain_direct")):
        case = next(c for c in manifest["sets"][set_name] if c["case_id"] == pair[role])
        split = case["source_split"]
        path = readonly._safe_path(ROOT / ".cache/tofu" / profile["dataset"]["files"][split]["filename"],
                                   ROOT / ".cache")
        rows, _ = load_pinned_split(split, path, download_if_missing=False)
        row = rows[case["record_index"]]
        for field in ("question", "answer"):
            if hashlib.sha256(row[field].encode()).hexdigest() != case[f"{field}_sha256"]:
                raise ValueError("selected QA checksum mismatch")
        records[role] = {field: row[field] for field in ("question", "answer")}
        provenance["cases"][role] = {**case, "file_sha256": profile["dataset"]["files"][split]["sha256"]}
    return records, provenance


def validation_plan() -> dict[str, Any]:
    """Default performs local profile/source validation without importing Torch."""
    profile = json.loads(PROFILE.read_text())
    _, source = select_pair(profile)
    return {"status": "validation_only", "model_loaded": False, "backward_performed": False,
            "optimizer_created": False, "weight_updates": 0, "checkpoint_writes": 0,
            "training_authorized": False, "downloads_enabled": False, "source": source,
            "rehearsal_memory_policy": rehearsal_memory_policy(profile, device="mps"),
            "explicit_opt_in_required": "--run-backward-rehearsal"}


def validate_sample(sample: Mapping[str, Any], *, profile: Mapping[str, Any], device: str,
                    before_load: bool = False, baseline: Mapping[str, Any] | None = None,
                    active_monitor: bool = False) -> None:
    readonly._validate_reading(sample)
    policy = rehearsal_memory_policy(profile, device=device,
                                     baseline_peak_rss_bytes=sample["process_peak_rss_bytes"])
    now = time.monotonic_ns()
    if not sample["timestamp_ns"] + sample["sample_duration_ns"] <= now <= sample["timestamp_ns"] + 2_000_000_000:
        raise RuntimeError("stale or future memory telemetry")
    if before_load and sample["available_physical_memory_lower_bound_bytes"] < policy["preload_available_required_bytes"]:
        raise RuntimeError(f"backward preload floor failed ({policy['preload_available_required_bytes'] // GIB} GiB required)")
    if baseline is not None:
        readonly._validate_reading(baseline)
        gap = sample["timestamp_ns"] - baseline["timestamp_ns"]
        if gap <= 0 or (active_monitor and gap > 2_000_000_000):
            raise RuntimeError("unordered readings or memory watchdog sampling gap exceeded 2 seconds")
        if (sample["total_physical_memory_bytes"] != baseline["total_physical_memory_bytes"]
                or sample["swap_used_bytes"] > baseline["swap_used_bytes"]
                or sample["pageouts"] != baseline["pageouts"]):
            raise RuntimeError("capacity changed or swap/page-outs grew")


class RehearsalMonitor(readonly.Monitor):
    """Reuse direct counters/watchdog; admission uses only the rehearsal policy."""

    def __init__(self, torch: Any, *, profile: Mapping[str, Any], device: str):
        rehearsal_memory_policy(profile, device=device)
        super().__init__(torch, profile=profile, device=device)

    def capture(self, stage: str, *, before_load: bool = False) -> dict[str, Any]:
        with self.lock:
            self.check()
            sample = readonly.read_sample(self.torch, stage)
            self.samples.append(sample)
            validate_sample(sample, profile=self.profile, device=self.device, before_load=before_load,
                            baseline=self.samples[-2] if len(self.samples) > 1 else None,
                            active_monitor=self.thread is not None)
            return sample


def prepare_batches(tokenizer: Any, records: Mapping[str, Any], device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if (type(tokenizer).__name__ != "GPT2TokenizerFast" or not tokenizer.is_fast
            or len(tokenizer) != 49152 or tokenizer.eos_token_id != 0 or tokenizer.chat_template
            or tokenizer.pad_token_id is not None):
        raise ValueError("incompatible pinned tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    batches, encoding = {}, {}
    for role in ("forget", "retain"):
        row = records[role]
        example = encode_plain_qa(tokenizer, row["question"], row["answer"], max_sequence_length=MAX_TOKENS)
        batches[role] = collate_causal_examples([example], tokenizer, device=device, max_sequence_length=MAX_TOKENS)
        encoding[role] = {"tokens": len(example.input_ids), "assistant_target_start": example.assistant_target_start,
                          "supervised_tokens": len(example.input_ids) - example.assistant_target_start,
                          "input_and_labels_sha256": digest([example.input_ids, example.labels])}
    return batches, encoding


def parameter_fingerprint(model: Any) -> str:
    """Hash actual parameter bytes, one host parameter copy at a time; no clone of model."""
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        hasher.update(digest([name, list(parameter.shape), str(parameter.dtype)]).encode())
        hasher.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def validate_backward_model(model: Any, torch: Any, profile: Mapping[str, Any]) -> None:
    readonly.validate_config(model.config.to_dict(), profile)
    params = list(model.parameters())
    if (type(model).__name__ != "LlamaForCausalLM" or not model.training or model.config.use_cache
            or not model.is_gradient_checkpointing or model.config._attn_implementation != "sdpa"
            or sum(p.numel() for p in params) != profile["model"]["parameter_count"]):
        raise RuntimeError("backward model identity/checkpointing/cache mismatch")
    if any(not p.requires_grad or p.dtype != torch.float32 or p.device.type != "mps" for p in params):
        raise RuntimeError("backward parameters must be trainable FP32 on mps")
    if any(buffer.device.type != "mps" for buffer in model.buffers()):
        raise RuntimeError("backward model buffer device mismatch")


def validate_gradients(model: Any, torch: Any, *, device: str) -> dict[str, Any]:
    count, nonzero = 0, False
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if (gradient is None or gradient.device.type != device or gradient.dtype != torch.float32
                or gradient.shape != parameter.shape or gradient.is_sparse):
            raise RuntimeError(f"missing or incompatible gradient: {name}")
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError(f"nonfinite gradient: {name}")
        nonzero = nonzero or bool(gradient.ne(0).any().item())
        count += parameter.numel()
    if count == 0 or not nonzero:
        raise RuntimeError("no nonzero gradients observed")
    return {"finite": True, "nonzero": True, "parameter_elements_with_gradients": count}


def backward_once(model: Any, batches: Mapping[str, Any], torch: Any, monitor: Any,
                  result: dict[str, Any]) -> None:
    from src.graddiff import causal_lm_nll, graddiff_objective

    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("existing gradients forbidden")
    versions = tuple(p._version for p in model.parameters())
    before = parameter_fingerprint(model)
    result["parameter_sha256_before"] = before
    losses, loss_report = {}, {}
    for role in ("forget", "retain"):
        batch = batches[role]
        if set(batch) != {"input_ids", "attention_mask", "labels"} or any(t.device.type != "mps" for t in batch.values()):
            raise RuntimeError("batch device/fields mismatch")
        if batch["input_ids"].shape[0] != 1 or not 2 <= batch["input_ids"].shape[1] <= MAX_TOKENS:
            raise RuntimeError("batch/sequence safety cap exceeded")
        monitor.stage = f"{role}_forward"
        monitor.boundary(f"before_{role}_forward")
        output = model(**batch, use_cache=False, return_dict=True, output_attentions=False, output_hidden_states=False)
        logits, loss = output.logits, output.loss
        if (logits.device.type != "mps" or loss.device.type != "mps" or logits.dtype != torch.float32
                or loss.dtype != torch.float32 or loss.ndim != 0 or not loss.requires_grad
                or not logits.requires_grad or tuple(logits.shape) != (*batch["input_ids"].shape, 49152)
                or getattr(output, "past_key_values", None) is not None):
            raise RuntimeError("forward output device/dtype/shape/graph/cache mismatch")
        with torch.no_grad():
            local = causal_lm_nll(logits.detach(), batch["labels"])
        nll, local_nll = float(loss.item()), float(local.item())
        if not math.isfinite(nll) or nll < 0 or not math.isclose(nll, local_nll, rel_tol=1e-4, abs_tol=1e-4):
            raise FloatingPointError("nonfinite loss or causal NLL parity failure")
        losses[role] = loss
        loss_report[role] = {"model_nll": nll, "local_nll": local_nll, "parity_passed": True}
        monitor.boundary(f"after_{role}_forward_graph_live")
        del output, logits, local
    objective = graddiff_objective(losses["forget"], losses["retain"], gamma=1.0, alpha=1.0)
    result.update(losses=loss_report, objective=float(objective.item()),
                  objective_definition="-mean_NLL(forget)+mean_NLL(retain); gamma=alpha=1", forward_count=2)
    monitor.stage = "backward"
    monitor.boundary("before_backward_graphs_live")
    result["backward_started"] = True
    objective.backward()
    result.update(backward_performed=True, backward_count=1)
    monitor.boundary("after_backward_gradients_live")
    monitor.stage = "gradient_and_parameter_integrity"
    result["gradients"] = validate_gradients(model, torch, device="mps")
    after = parameter_fingerprint(model)
    result["parameter_sha256_after"] = after
    if before != after or versions != tuple(p._version for p in model.parameters()):
        raise RuntimeError("parameter values or versions changed")
    result.update(parameters_unchanged=True, parameter_versions_unchanged=True)
    del objective, losses
    model.zero_grad(set_to_none=True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("gradient cleanup failed")
    result["gradients_cleared"] = True
    monitor.boundary("after_gradient_cleanup")


def validate_report(report: Mapping[str, Any]) -> None:
    """Reject misleading success and non-JSON/nonfinite output before file creation."""
    for key, expected in {"training_authorized": False, "optimizer_created": False, "weight_updates": 0,
                          "optimizer_steps": 0, "checkpoint_writes": 0, "downloads_enabled": False}.items():
        if type(report.get(key)) is not type(expected) or report[key] != expected:
            raise ValueError(f"unsafe report invariant: {key}")
    if report["status"] == PASS:
        result = report["computation"]
        if (report.get("error") or report["model_loaded"] is not True
                or report["actual_device"] != "mps" or report["telemetry_device"] != "mps"
                or report["scope"] != "one_pair_backward_only" or report["model_id"] != MODEL_ID
                or report["revision"] != REVISION or not report["checkpoint_unchanged"]
                or not result["backward_performed"] or result["backward_count"] != 1 or result["forward_count"] != 2
                or not result["gradients"]["finite"] or not result["gradients"]["nonzero"]
                or not result["parameters_unchanged"] or not result["parameter_versions_unchanged"]
                or not result["gradients_cleared"] or result["parameter_sha256_before"] != result["parameter_sha256_after"]):
            raise ValueError("incomplete backward rehearsal success report")
        stages = {s["stage"] for s in report["measurements"]}
        if not {"baseline", "before_load", "after_load", "before_backward_graphs_live",
                "after_backward_gradients_live", "after_gradient_cleanup", "after_checkpoint_integrity"} <= stages:
            raise ValueError("missing required measurement boundaries")
    elif report["status"] != "STOP":
        raise ValueError("unknown rehearsal status")
    json.dumps(report, allow_nan=False)


def run_rehearsal(*, enabled: bool, report_path: Path) -> dict[str, Any]:
    if enabled is not True:
        raise ValueError("requires explicit --run-backward-rehearsal")
    path = readonly.validate_report_path(report_path)
    profile = json.loads(PROFILE.read_text())
    policy = rehearsal_memory_policy(profile, device="mps")
    records, source = select_pair(profile)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback must be disabled before Torch import")
    os.environ.update(HF_HOME=str(ROOT / ".cache/smollm2_hub"), HF_HUB_CACHE=str(ROOT / ".cache/smollm2_hub/hub"),
                      HF_XET_CACHE=str(ROOT / ".cache/smollm2_hub/xet"), HF_HUB_DISABLE_XET="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        readonly._safe_path(Path(os.environ[key]), ROOT / ".cache")
    import torch
    import transformers
    import huggingface_hub

    for name, module in (("torch", torch), ("transformers", transformers), ("huggingface_hub", huggingface_hub)):
        if module.__version__ != readonly.VERSIONS[name]:
            raise RuntimeError(f"requires pinned {name}=={readonly.VERSIONS[name]}")
    device = readonly.validate_environment(torch)
    monitor = RehearsalMonitor(torch, profile=profile, device=device.type)
    report: dict[str, Any] = {
        "schema_version": "1.0", "status": "STOP", "scope": "one_pair_backward_only", "model_id": MODEL_ID,
        "revision": REVISION, "profile_sha256": digest(profile), "requested_device": "mps", "actual_device": None,
        "telemetry_device": str(device), "model_loaded": False, "training_authorized": False,
        "approved_small_model_training_threshold_bytes": None, "optimizer_created": False, "optimizer_steps": 0,
        "weight_updates": 0, "checkpoint_writes": 0, "checkpoint_unchanged": False, "downloads_enabled": False,
        "precision": "float32", "source": source, "rehearsal_memory_policy": policy,
        "environment": {**readonly.VERSIONS, "os": platform.mac_ver()[0], "machine": platform.machine()},
        "measurement_kind": "direct_counters_and_sampled_maxima_separate_from_policy_estimates",
        "measurement_sources": {"rss": "ps RSS KiB; getrusage Darwin lifetime peak bytes",
                                "mps": "Torch current/driver allocations; sampled maxima are lower bounds",
                                "availability": "vm_stat free+speculative pages; excludes reclaimable caches",
                                "pressure": "memory_pressure -Q plus kernel NORMAL=1",
                                "swap": "sysctl rounded MiB; sub-resolution growth may be missed", "sample_atomic": False},
        "computation": {"backward_started": False, "backward_performed": False, "backward_count": 0,
                        "forward_count": 0, "parameters_unchanged": None, "gradients_cleared": False},
    }
    model = None
    try:
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        torch.mps.synchronize()
        monitor.capture("baseline", before_load=True)
        time.sleep(readonly.SAMPLE_INTERVAL_SECONDS)
        monitor.capture("before_checkpoint", before_load=True)
        report["checkpoint"] = readonly.ensure_checkpoint(profile, allow_download=False)
        config = transformers.AutoConfig.from_pretrained(readonly.CHECKPOINT, local_files_only=True, trust_remote_code=False)
        readonly.validate_config(config.to_dict(), profile)
        config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(readonly.CHECKPOINT, local_files_only=True,
                                                              trust_remote_code=False, use_fast=True)
        batches, encoding = prepare_batches(tokenizer, records, device)
        report["encoding"] = encoding
        before_load = monitor.capture("before_load", before_load=True)
        report["rehearsal_memory_policy"] = rehearsal_memory_policy(
            profile, device=device.type, baseline_peak_rss_bytes=before_load["process_peak_rss_bytes"])
        monitor.stage = "loading_cpu"
        monitor.start()
        model, info = transformers.AutoModelForCausalLM.from_pretrained(
            readonly.CHECKPOINT, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        report["model_loaded"] = True
        monitor.check()
        if not isinstance(info, dict) or any(info.get(k) != [] for k in
                                           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("incomplete/incompatible checkpoint loading")
        model.requires_grad_(False)
        model.eval()
        readonly.validate_model(model, torch, profile, device="cpu")
        report["actual_device"] = "cpu"
        monitor.boundary("after_cpu_load")
        monitor.stage = "transfer_to_mps"
        model.to(device)
        readonly.validate_model(model, torch, profile, device="mps")
        report["actual_device"] = "mps"
        monitor.boundary("after_load")
        model.requires_grad_(True)
        model.train()  # activates checkpointing; no optimizer or parameter update
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
        validate_backward_model(model, torch, profile)
        monitor.stage = "parameter_integrity_before"
        backward_once(model, batches, torch, monitor, report["computation"])
        validate_backward_model(model, torch, profile)
        monitor.stage = "checkpoint_integrity"
        if readonly.ensure_checkpoint(profile, allow_download=False) != report["checkpoint"]:
            raise RuntimeError("checkpoint provenance changed")
        report["checkpoint_unchanged"] = True
        monitor.boundary("after_checkpoint_integrity")
        report["status"] = PASS
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "reason": str(monitor.error or exc) or "interrupted"}
    finally:
        try:
            monitor.close()
        except Exception as exc:
            report.update(status="STOP", error={"type": type(exc).__name__, "reason": str(exc)})
        if model is not None:
            model.zero_grad(set_to_none=True)
        report["measurements"] = monitor.samples
        report["peaks"] = readonly.summarize_samples(monitor.samples)
    validate_report(report)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    readonly.validate_report_path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-backward-rehearsal", action="store_true")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)
    if bool(args.report_path) != args.run_backward_rehearsal:
        parser.error("--run-backward-rehearsal requires a new --report-path under outputs/; report option requires opt-in")
    try:
        report = run_rehearsal(enabled=True, report_path=args.report_path) if args.run_backward_rehearsal else validation_plan()
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["status"] in {PASS, "validation_only"} else 2
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOP: backward rehearsal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
