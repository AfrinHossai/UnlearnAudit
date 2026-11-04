"""Task 53: offline plan by default; separately authorized native LoRA mechanics.

No PEFT, downloads, resume, full-base saving, device/optimizer overrides or
unlearning path. Physical execution belongs in an owner-run normal Mac Terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import small_model_acquisition as acquisition
from src import small_model_preflight as telemetry
from src.inspect_tofu import sha256_file
from src.small_model_experiment import MODEL_ID, REVISION, ROOT, PROFILE, digest
from src.small_model_memory import GIB

SPEC = ROOT / "experiments/smollm2_135m_lora_pilot.json"
SPEC_SHA256 = "df8d4d9c354a5126acf53b834c31c5680de9915c77c21e262f4e00155af81443"
LOCK = ROOT / "requirements-ml.lock"
LOCK_SHA256 = "675f3d49828d17d0f55253e2b106f73e9aad9474bf2757f45f603c7abdf8f52a"
HISTORICAL = acquisition.RUNS / "baseline-002.json"
HISTORICAL_SHA256 = "61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff"
BACKWARD = ROOT / "outputs/smollm2_backward_task37.json"
BACKWARD_SHA256 = "f71c8d1e673067676708c33be9cfdb30b7545f96ed691768b7b183652bde80b3"
FULL_RUNNER_SHA256 = "92c3896e03d1599b007131fc255810fa32b5893a396b21203ef783004b8c01cd"
BASE_SHA256 = "709264bff244d021ac31b1fcf9e1733b9af31f75032e854e28db62287b8cd072"
RUNS = ROOT / "outputs/smollm2_lora"
AUTHORIZATION = "SmolLM2-135M:development:LoRA-r4-qv:one-SGD-step"
GUARD_ID = "smollm2_135m_fp32_mps_lora_r4_qv_one_step_v1"
BASELINE_GUARD_ID = "smollm2_135m_fp32_mps_lora_r4_qv_baseline_v2"
FORMAT = "unlearnguard_native_lora_v1"
PARAMETERS = 230400
MAX_SECONDS = 1800


def expected_targets() -> dict[str, tuple[int, int]]:
    return {f"model.layers.{i}.self_attn.{target}": (output, 576)
            for i in range(30) for target, output in (("q_proj", 576), ("v_proj", 192))}


def expected_adapter_shapes() -> dict[str, tuple[int, int]]:
    return {f"{name}.{suffix}": shape for name, (output, width) in sorted(expected_targets().items())
            for suffix, shape in (("lora_A", (4, width)), ("lora_B", (output, 4)))}


def frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Byte hashing and metadata validation only; no ML imports or tensor loads."""
    for path, expected in ((SPEC, SPEC_SHA256), (LOCK, LOCK_SHA256),
                           (Path(acquisition.__file__), FULL_RUNNER_SHA256),
                           (HISTORICAL, HISTORICAL_SHA256), (BACKWARD, BACKWARD_SHA256)):
        telemetry._safe_path(path, ROOT)
        if sha256_file(path) != expected:
            raise ValueError(f"protected input hash mismatch: {path.name}")
    spec = json.loads(SPEC.read_text())
    manifest = acquisition.frozen_manifest()
    acquisition.artifact_path(HISTORICAL, new=False)
    historical = json.loads(HISTORICAL.read_text())
    acquisition.validate_baseline(historical, manifest)
    backward = json.loads(BACKWARD.read_text())
    if (backward["status"] != "backward_rehearsal_passed_not_training_clearance"
            or backward["computation"]["parameter_sha256_after"] != BASE_SHA256
            or historical["base_parameter_sha256"] != BASE_SHA256
            or manifest["epoch_zero_order"][0] != spec["pilot_case_id"]):
        raise ValueError("historical model/pilot evidence mismatch")
    return spec, manifest, historical


def metadata_targets() -> dict[str, Any]:
    config = json.loads((telemetry.CHECKPOINT / "config.json").read_text())
    telemetry.validate_config(config, json.loads(PROFILE.read_text()))
    # Header only: never safe_open/get_tensor or torch.load on original weights.
    with (telemetry.CHECKPOINT / "model.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    observed = {k.removesuffix(".weight"): tuple(v["shape"]) for k, v in tensors.items()
                if k.endswith((".q_proj.weight", ".v_proj.weight"))}
    if (observed != expected_targets() or len(tensors) != 272
            or sum(math.prod(v["shape"]) for v in tensors.values()) != 134515008
            or any(v["dtype"] != "BF16" for v in tensors.values())):
        raise ValueError("checkpoint target shapes/count/dtype mismatch")
    return {"validation": "safetensors header and pinned Llama config; no tensor load",
            "targets": {k: list(v) for k, v in observed.items()},
            "runtime_verification_required": "exact bias-free Linear types/shapes/identities, ties, device, gradients"}


def dependency_versions() -> dict[str, str]:
    if sha256_file(LOCK) != LOCK_SHA256:
        raise ValueError("runtime lock changed")
    versions = {}
    for line in LOCK.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"requires locked {name}=={expected}; found {actual}")
        versions[name] = actual
    return versions


def provenance() -> dict[str, Any]:
    """Bind baselines to every local source file, evaluator, profile and lock."""
    paths = [*sorted((ROOT / "src").glob("*.py")), SPEC, LOCK]
    llama = importlib.metadata.distribution("transformers").locate_file("transformers/models/llama/modeling_llama.py")
    return {"files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
            "versions": dependency_versions(), "python": platform.python_version(),
            "installed_llama_source_sha256": sha256_file(llama)}


def memory_policy(baseline_peak: int = 0, *, mode: str = "pilot") -> dict[str, Any]:
    if mode not in ("pilot", "baseline") or type(baseline_peak) is not int or baseline_peak < 0:
        raise ValueError("invalid LoRA admission mode/RSS")
    if mode == "baseline":
        return baseline_memory_policy(baseline_peak)
    terms = {"three_fp32_base_copies": 3 * 538060032,
             "adapter_weights_and_gradients": 2 * 4 * PARAMETERS,
             "source_residency": 269060552, "forward_backward_workspace": 4 * GIB,
             "tokenizer_integrity": GIB, "metal_cache": GIB,
             "additional_process_headroom": max(GIB, baseline_peak),
             "adapter_clipping_step_save": GIB, "uncertainty": 4 * GIB}
    incremental = sum(terms.values())
    return {"guard_id": GUARD_ID if mode == "pilot" else BASELINE_GUARD_ID,
            "model_id": MODEL_ID, "revision": REVISION, "device": "mps", "precision": "float32",
            "max_sequence_length": 128, "batch_size": 1,
            "max_optimizer_steps": 1 if mode == "pilot" else 0,
            "backward_permitted": mode == "pilot", "optimizer_state_bytes": 0,
            "estimated_incremental_terms_bytes": terms,
            "estimated_incremental_total_bytes": incremental,
            "preload_available_required_bytes": math.ceil(incremental / GIB) * GIB + 6 * GIB,
            "continuing_available_required_bytes": 6 * GIB,
            "minimum_total_host_bytes": 48 * GIB, "max_process_peak_rss_bytes": 8 * GIB,
            "max_mps_driver_bytes": 8 * GIB, "calibrated_training_requirement": False,
            "scope": "native rank4 q/v one-step mechanics only; baseline conservatively uses same envelope"}


def baseline_memory_policy(baseline_peak: int = 0) -> dict[str, Any]:
    """Task 58 frozen inference only; estimates, never training clearance.

    Reuse the reviewed 76-case baseline's forward/generation allowances,
    retaining LoRA's larger 4 GiB uncertainty and all continuing safeguards.
    Three adapter copies cover live weights and transfer/hash staging; no
    gradients, optimizer or tensor checkpoint serialization is permitted.
    """
    if type(baseline_peak) is not int or baseline_peak < 0:
        raise ValueError("invalid LoRA baseline RSS")
    terms = {"three_fp32_base_copies": 3 * 538060032,
             "adapter_weights_and_staging": 3 * 4 * PARAMETERS,
             "source_residency": 269060552, "forward_workspace": 2 * GIB,
             "tokenizer_integrity": GIB, "metal_cache": GIB,
             "additional_process_headroom": max(GIB, baseline_peak),
             "sequential_generation": 2 * GIB, "uncertainty": 4 * GIB}
    incremental = sum(terms.values())
    return {"guard_id": BASELINE_GUARD_ID,
            "model_id": MODEL_ID, "revision": REVISION, "device": "mps", "precision": "float32",
            "max_sequence_length": 128, "batch_size": 1,
            "max_optimizer_steps": 0, "backward_permitted": False,
            "optimizer_state_bytes": 0, "adapter_gradient_bytes": 0,
            "checkpoint_serialization_bytes": 0,
            "estimated_incremental_terms_bytes": terms,
            "estimated_incremental_total_bytes": incremental,
            "preload_available_required_bytes": math.ceil(incremental / GIB) * GIB + 6 * GIB,
            "continuing_available_required_bytes": 6 * GIB,
            "minimum_total_host_bytes": 48 * GIB, "max_process_peak_rss_bytes": 8 * GIB,
            "max_mps_driver_bytes": 8 * GIB, "calibrated_training_requirement": False,
            "calibrated_inference_requirement": False,
            "scope": "frozen zero-effect native rank4 q/v; 76 sequential probes; no backward/update/checkpoint"}


def counter_checks(sample, *, previous=None, first=None, active=False, before_load=False, mode="pilot"):
    telemetry._validate_reading(sample)
    policy = memory_policy(sample["process_peak_rss_bytes"], mode=mode)
    if before_load and sample["available_physical_memory_lower_bound_bytes"] < policy["preload_available_required_bytes"]:
        raise RuntimeError(f"LoRA preload floor failed ({policy['preload_available_required_bytes'] // GIB} GiB required)")
    if sample["process_peak_rss_bytes"] > 8 * GIB or sample["mps_driver_allocated_bytes"] > 8 * GIB:
        raise RuntimeError("LoRA RSS/driver 8 GiB stop cap exceeded")
    if previous is not None:
        gap = sample["timestamp_ns"] - previous["timestamp_ns"]
        if gap <= 0 or (active and gap > 2_000_000_000):
            raise RuntimeError("unordered telemetry or watchdog gap exceeds 2 seconds")
        if sample["process_peak_rss_bytes"] < previous["process_peak_rss_bytes"]:
            raise RuntimeError("process lifetime peak RSS decreased")
    if first is not None and (
            sample["total_physical_memory_bytes"] != first["total_physical_memory_bytes"]
            or sample["swap_used_bytes"] > first["swap_used_bytes"] or sample["pageouts"] != first["pageouts"]):
        raise RuntimeError("capacity changed or cumulative swap/page-out growth")


def validate_sample(sample, **kwargs):
    counter_checks(sample, **kwargs)
    now = time.monotonic_ns()
    if not sample["timestamp_ns"] + sample["sample_duration_ns"] <= now <= sample["timestamp_ns"] + 2_000_000_000:
        raise RuntimeError("stale/future LoRA telemetry")


class LoRAMonitor(telemetry.Monitor):
    def __init__(self, torch, *, profile, device, mode):
        if device != "mps":
            raise RuntimeError("LoRA requires MPS")
        super().__init__(torch, profile=profile, device=device)
        memory_policy(mode=mode)
        self.mode, self.started = mode, time.monotonic()

    def check(self):
        super().check()
        if time.monotonic() - self.started > MAX_SECONDS:
            raise RuntimeError("LoRA 30-minute budget exceeded")
        if (self.thread is not None and not self.stop_event.is_set() and self.samples
                and time.monotonic_ns() - self.samples[-1]["timestamp_ns"] > 2_000_000_000):
            raise RuntimeError("LoRA watchdog's latest observation is stale")

    def capture(self, stage, *, before_load=False):
        with self.lock:
            self.check()
            sample = telemetry.read_sample(self.torch, stage)
            previous = self.samples[-1] if self.samples else None
            first = self.samples[0] if self.samples else None
            self.samples.append(sample)  # retain even the failing observation
            validate_sample(sample, previous=previous, first=first, before_load=before_load,
                            active=self.thread is not None, mode=self.mode)
            return sample

    def boundary(self, stage):
        self.stage = stage
        super().boundary(stage)

    def _loop(self):
        import _thread
        try:
            while not self.stop_event.wait(telemetry.SAMPLE_INTERVAL_SECONDS):
                self.capture(self.stage)
        except Exception as exc:
            self.error = exc
            _thread.interrupt_main()


def new_run_path(path: Path, mode: str) -> Path:
    path = telemetry._safe_path(path, RUNS)
    if (path.parent != RUNS or not re.fullmatch(mode + r"-[A-Za-z0-9_-]+", path.name)
            or path.exists()):
        raise ValueError("requires a new direct baseline-* or pilot-* directory in outputs/smollm2_lora")
    acquisition.require_ignored(path)
    return path


def require_authorization(enabled: bool, authorization: str | None):
    if enabled is not True or authorization != AUTHORIZATION:
        raise ValueError("requires --run-one-step and exact LoRA --authorize-weight-update token")


def fingerprint(parameters: Mapping[str, Any]) -> str:
    """Canonical names; one host parameter staging at a time; no base clone."""
    h = hashlib.sha256()
    for name, p in parameters.items():
        h.update(digest([name, list(p.shape), str(p.dtype)]).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_layout(model, originals, wrappers, torch, *, training: bool):
    from src.lora_adapter import LoRALinear, adapter_parameters
    adapters = adapter_parameters(wrappers)
    telemetry.validate_config(model.config.to_dict(), json.loads(PROFILE.read_text()))
    if (type(model).__name__ != "LlamaForCausalLM" or model.training is not training
            or model.config.use_cache or model.config._attn_implementation != "sdpa"
            or set(wrappers) != set(expected_targets())
            or sum(p.numel() for p in originals.values()) != 134515008
            or set(adapters) != set(expected_adapter_shapes())):
        raise RuntimeError("LoRA model/config/coverage mismatch")
    for name, module in wrappers.items():
        if (type(module) is not LoRALinear or model.get_submodule(name) is not module
                or module.rank != 4 or module.alpha != 8 or module.scale != 2.0
                or module.base_layer.weight is not originals[name + ".weight"]
                or tuple(module.base_layer.weight.shape) != expected_targets()[name]):
            raise RuntimeError("LoRA wrapper identity/scaling/base mismatch")
    all_params = list(model.parameters())
    ids = [id(p) for p in [*originals.values(), *adapters.values()]]
    if len(set(ids)) != len(ids) or {id(p) for p in all_params} != set(ids):
        raise RuntimeError("unaccounted, aliased or missing parameters")
    if any(p.dtype != torch.float32 or p.device.type != "mps" for p in all_params):
        raise RuntimeError("all parameters must be FP32 on MPS")
    if any(p.requires_grad or p.grad is not None for p in originals.values()):
        raise RuntimeError("pretrained weights must be frozen and gradient-free")
    if any(tuple(p.shape) != expected_adapter_shapes()[name] or p.requires_grad is not training
           for name, p in adapters.items()):
        raise RuntimeError("adapter shape/trainability mismatch")
    if sum(p.numel() for p in adapters.values()) != PARAMETERS or len(adapters) != 120:
        raise RuntimeError("trainable parameter/tensor count mismatch")
    if any(b.device.type != "mps" for b in model.buffers()):
        raise RuntimeError("buffer device mismatch")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise RuntimeError("pretrained embedding/head tie lost")
    if training:
        functions = [getattr(m, "_gradient_checkpointing_func", None) for m in model.modules()
                     if getattr(m, "gradient_checkpointing", False)]
        if not model.is_gradient_checkpointing or not functions or any(
                getattr(fn, "keywords", {}).get("use_reentrant") is not False
                or getattr(fn, "keywords", {}).get("preserve_rng_state") is not True for fn in functions):
            raise RuntimeError("non-reentrant/RNG checkpoint settings not propagated")
    elif any(p.grad is not None for p in adapters.values()):
        raise RuntimeError("evaluation must be gradient-free")
    return adapters


def validate_gradients(adapters, originals, torch, *, device="mps"):
    if any(p.requires_grad or p.grad is not None for p in originals.values()):
        raise RuntimeError("base trainability/gradient violation")
    nonzero_b = False
    for name, p in adapters.items():
        g = p.grad
        if (not p.requires_grad or g is None or g.device.type != device or g.dtype != torch.float32
                or g.shape != p.shape or g.is_sparse):
            raise RuntimeError(f"missing/incompatible adapter gradient: {name}")
        if not bool(torch.isfinite(g).all().item()):
            raise FloatingPointError(f"nonfinite adapter gradient: {name}")
        if name.endswith(".lora_B"):
            nonzero_b = nonzero_b or bool(g.ne(0).any().item())
    if not adapters or not nonzero_b:
        raise RuntimeError("no nonzero B-side adapter gradient")
    return {"finite": True, "nonzero_B": True, "A_zero_allowed_at_initialization": True,
            "elements_with_gradients": sum(p.numel() for p in adapters.values())}


def validate_optimizer(optimizer, adapters, torch):
    if type(optimizer) is not torch.optim.SGD or optimizer.state or len(optimizer.param_groups) != 1:
        raise RuntimeError("exact stateless SGD with one adapter group required")
    group = optimizer.param_groups[0]
    expected = {"lr": 0.0001, "momentum": 0.0, "dampening": 0.0, "weight_decay": 0.0,
                "nesterov": False, "maximize": False, "foreach": False, "fused": False, "differentiable": False}
    if any(type(group.get(k)) is not type(v) or group[k] != v for k, v in expected.items()):
        raise RuntimeError("LoRA SGD configuration drift")
    if [id(p) for p in group["params"]] != [id(p) for p in adapters.values()]:
        raise RuntimeError("SGD membership must be exactly the adapters, once each")


def one_step(model, batch, originals, wrappers, torch, monitor, report, *, enabled, authorization, pre_step, run_path):
    require_authorization(enabled, authorization)
    if (monitor.mode != "pilot" or type(report.get("optimizer_steps")) is not int
            or report["optimizer_steps"] != 0 or report.get("step_attempted") is not False
            or report.get("backward_attempted", False) is not False):
        raise RuntimeError("LoRA admission/one-step budget violation")
    adapters = validate_layout(model, originals, wrappers, torch, training=True)
    report["trainable_parameter_count"] = sum(p.numel() for p in adapters.values() if p.requires_grad)
    if any(p.grad is not None for p in adapters.values()):
        raise RuntimeError("preexisting adapter gradients")
    optimizer = None
    try:
        monitor.boundary("before_training_forward")
        loss = acquisition.checked_loss(model, batch, torch)
        if not loss.requires_grad:
            raise RuntimeError("adapter loss has no graph")
        report["training_nll_before"] = float(loss.item())
        monitor.boundary("before_backward")
        report["backward_attempted"] = True
        loss.backward()
        report["backward_count"] = 1
        del loss
        monitor.boundary("after_backward")
        report["gradients_before_clip"] = validate_gradients(adapters, originals, torch)
        norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1.0, foreach=False, error_if_nonfinite=True)
        before_norm = float(norm.item())
        if not math.isfinite(before_norm):
            raise FloatingPointError("nonfinite clipping norm")
        report["gradient_norm_before_clip"] = before_norm
        report["gradients_after_clip"] = validate_gradients(adapters, originals, torch)
        total = torch.nn.utils.get_total_norm([p.grad for p in adapters.values()], foreach=False, error_if_nonfinite=True)
        after_norm = float(total.item())
        if not math.isfinite(after_norm) or not 0 < after_norm <= 1.00001:
            raise RuntimeError("LoRA clipped norm outside (0,1]")
        report["gradient_norm_after_clip"] = after_norm
        monitor.boundary("after_clipping")
        optimizer = torch.optim.SGD(list(adapters.values()), lr=0.0001, momentum=0.0, dampening=0.0,
                                    weight_decay=0.0, nesterov=False, maximize=False,
                                    foreach=False, fused=False, differentiable=False)
        report["optimizer_created"] = True
        validate_optimizer(optimizer, adapters, torch)
        report["optimizer_configuration"] = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
        pre_step()  # repeat source/baseline/base-byte/version checks immediately before update
        validate_layout(model, originals, wrappers, torch, training=True)
        validate_gradients(adapters, originals, torch)
        validate_optimizer(optimizer, adapters, torch)
        require_authorization(enabled, authorization)
        monitor.boundary("before_optimizer_step")
        acquisition.seal_json(run_path / "STEP_ATTEMPTED.json", {
            "status": "step may have happened; require final report and COMPLETE",
            "max_optimizer_steps": 1, "adapter_sha256_before": report["adapter_sha256_before"],
            "base_parameter_sha256_before": report["base_parameter_sha256_before"],
            "spec_sha256": SPEC_SHA256})
        monitor.check()
        report["step_attempted"] = True
        optimizer.step()
        report["optimizer_steps"] = 1
        monitor.boundary("after_optimizer_step")
        validate_optimizer(optimizer, adapters, torch)
        report["optimizer_state_bytes"] = 0
        if any(not bool(torch.isfinite(p).all().item()) for p in adapters.values()):
            raise FloatingPointError("nonfinite adapter after step")
    finally:
        model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def compare_rows(rows, reference):
    if len(rows) != len(reference):
        raise ValueError("baseline evaluation coverage changed")
    for row, old in zip(rows, reference):
        nll, old_nll = row["answer_nll"], old["answer_nll"]
        if (type(nll) not in (int, float) or not math.isfinite(nll)
                or not math.isclose(nll, old_nll, rel_tol=1e-4, abs_tol=1e-4)
                or {k: v for k, v in row.items() if k != "answer_nll"}
                != {k: v for k, v in old.items() if k != "answer_nll"}):
            raise ValueError("zero-effect/no-op baseline drift (including greedy generation)")


def validate_native_baseline(report, manifest, historical, expected_provenance):
    expected = {"status": "lora_baseline_complete_no_updates", "mode": "baseline",
                "model_id": MODEL_ID, "revision": REVISION, "format": FORMAT,
                "spec_sha256": SPEC_SHA256, "historical_baseline_sha256": HISTORICAL_SHA256,
                "provenance": expected_provenance, "adapter_configuration": json.loads(SPEC.read_text())["adapter"],
                "model_loaded": True, "actual_device": "mps", "precision": "float32",
                "training_authorized": False, "optimizer_created": False, "optimizer_steps": 0,
                "step_attempted": False, "backward_attempted": False, "backward_count": 0,
                "checkpoint_writes": 0, "checkpoint_write_attempted": False, "base_checkpoint_unchanged": True,
                "base_parameter_sha256_before": BASE_SHA256, "base_parameter_sha256_after": BASE_SHA256,
                "checkpoint": historical["checkpoint"],
                "base_versions_unchanged": True, "adapter_versions_unchanged": True, "initial_adapter_zero_effect": True,
                "adapter_parameter_count": PARAMETERS, "trainable_parameter_count": 0, "adapter_tensor_count": 120,
                "encoding_sha256": historical["encoding_sha256"], "downloads_enabled": False,
                "seed": 2026, "deterministic_algorithms": True}
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("incomplete/incompatible native LoRA baseline")
    if (not re.fullmatch(r"[0-9a-f]{64}", report.get("adapter_sha256_before", ""))
            or report["adapter_sha256_after"] != report["adapter_sha256_before"]):
        raise ValueError("baseline adapter integrity missing")
    acquisition.validate_baseline({**historical, "evaluation": report["evaluation"]}, manifest)
    compare_rows(report["evaluation"], historical["evaluation"])
    samples = report["measurements"]
    preloads = [s for s in samples if s["stage"] in ("preload", "before_load")]
    required = {"after_cpu_load", "after_adapter_injection", "after_load", "complete"}
    required.update(prefix + "evaluation:" + row["id"] for row in manifest["evaluation"]
                    for prefix in ("before_", "after_"))
    if ([s["stage"] for s in preloads] != ["preload", "before_load"]
            or not required <= {s["stage"] for s in samples}
            or report["memory_policy"] != memory_policy(preloads[-1]["process_peak_rss_bytes"], mode="baseline")
            or report["peaks"] != telemetry.summarize_samples(samples)):
        raise ValueError("missing baseline memory policy/boundaries/peaks")
    previous = None
    for sample in samples:
        counter_checks(sample, previous=previous, first=samples[0], mode="baseline",
                       before_load=sample["stage"] in ("preload", "before_load"),
                       active=previous is not None and previous["timestamp_ns"] >= preloads[-1]["timestamp_ns"])
        previous = sample


def read_native_baseline(path, expected_sha256, manifest, historical, expected_provenance):
    path = telemetry._safe_path(path, RUNS)
    if (path.name != "report.json" or path.parent.parent != RUNS or not path.parent.name.startswith("baseline-")
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "")
            or not path.is_file() or path.stat().st_mode & 0o222 or path.stat().st_nlink != 1
            or sha256_file(path) != expected_sha256):
        raise ValueError("requires sealed native baseline and its explicit verified SHA-256")
    acquisition.require_ignored(path)
    marker_path = telemetry._safe_path(path.parent / "COMPLETE.json", RUNS)
    if not marker_path.is_file() or marker_path.stat().st_mode & 0o222:
        raise ValueError("baseline COMPLETE missing or unsealed")
    marker = json.loads(marker_path.read_text())
    if marker.get("report_sha256") != expected_sha256 or marker.get("status") != "lora_baseline_complete_no_updates":
        raise ValueError("baseline COMPLETE/report binding mismatch")
    report = json.loads(path.read_text())
    validate_native_baseline(report, manifest, historical, expected_provenance)
    return report


def save_adapter(directory, adapters, report, torch):
    """Native adapter-only writer and exact small-tensor disk round trip; no base save."""
    from safetensors.torch import save_file
    from safetensors import safe_open
    if (report["optimizer_steps"] != 1 or report.get("weight_update_verified") is not True
            or set(adapters) != set(expected_adapter_shapes())
            or any(tuple(p.shape) != expected_adapter_shapes()[k] or p.requires_grad or p.grad is not None
                   or p.dtype != torch.float32 for k, p in adapters.items())):
        raise RuntimeError("adapter checkpoint prerequisites failed")
    telemetry._safe_path(directory, RUNS)
    directory.mkdir(exist_ok=False)
    config = {"format": FORMAT, "model_id": MODEL_ID, "revision": REVISION,
              "base_weight_file_sha256": report["checkpoint"]["weight_sha256"],
              "spec_sha256": SPEC_SHA256, "adapter": report["adapter_configuration"],
              "adapter_parameter_sha256": report["adapter_sha256_after"], "provenance": report["provenance"]}
    acquisition.seal_json(directory / "adapter_config.json", config)
    weights = directory / "adapter_model.safetensors"
    cpu = {name: p.detach().cpu().contiguous() for name, p in adapters.items()}
    if fingerprint(cpu) != report["adapter_sha256_after"]:
        raise RuntimeError("adapter CPU staging mismatch")
    save_file(cpu, weights, metadata={"format": FORMAT})
    with weights.open("rb") as stream:
        os.fsync(stream.fileno())
    if weights.stat().st_size > 2 * 1024**2:
        raise RuntimeError("adapter file unexpectedly large")
    with safe_open(weights, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(cpu) or handle.metadata() != {"format": FORMAT}:
            raise RuntimeError("adapter key/format coverage mismatch")
        for name, p in cpu.items():
            tensor = handle.get_tensor(name)
            if (tensor.dtype != torch.float32 or tensor.shape != p.shape or not torch.equal(tensor, p)
                    or fingerprint({name: tensor}) != fingerprint({name: p})):
                raise RuntimeError("adapter serialization byte/value mismatch")
    if set(p.name for p in directory.iterdir()) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise RuntimeError("unexpected checkpoint content")
    weights.chmod(0o444)
    return {p.name: sha256_file(p) for p in directory.iterdir()}


def environment_before_import():
    if not sys.stdin.isatty():
        raise RuntimeError("real execution requires an owner-run normal Mac Terminal")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback must be disabled before Torch import")
    for name in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                 "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
        if name in os.environ:
            raise RuntimeError(f"remove allocator/math override: {name}")
    if any(os.environ.get(name, "1") != "1" for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise RuntimeError("single-process execution required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false",
                      PYTORCH_ENABLE_MPS_FALLBACK="0", HF_HOME=str(ROOT / ".cache/smollm2_hub"),
                      HF_HUB_CACHE=str(ROOT / ".cache/smollm2_hub/hub"), HF_XET_CACHE=str(ROOT / ".cache/smollm2_hub/xet"))
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        telemetry._safe_path(Path(os.environ[name]), ROOT / ".cache")


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(*, mode, enabled, authorization, run_path, baseline_path=None, baseline_sha256=None):
    if mode == "pilot":
        require_authorization(enabled, authorization)  # before paths/imports/writes
        if baseline_path is None or not re.fullmatch(r"[0-9a-f]{64}", baseline_sha256 or ""):
            raise ValueError("one-step pilot requires --lora-baseline and its verified SHA-256")
    elif mode != "baseline" or enabled is not True or authorization is not None or baseline_path is not None or baseline_sha256 is not None:
        raise ValueError("explicit baseline mode cannot accept update/baseline-input arguments")
    destination = new_run_path(run_path, mode)
    RUNS.mkdir(exist_ok=True)
    destination.mkdir(exist_ok=False)
    acquisition.seal_json(destination / "RUNNING.json", {"mode": mode, "status": "attempt reserved; no success implied"})
    fsync_directory(RUNS)
    report = {"schema_version": "1.0", "status": "STOP", "mode": mode, "format": FORMAT,
              "model_id": MODEL_ID, "revision": REVISION, "spec_sha256": SPEC_SHA256,
              "historical_baseline_sha256": HISTORICAL_SHA256, "lora_baseline_sha256": baseline_sha256,
              "training_authorized": mode == "pilot", "model_loaded": False, "actual_device": None,
              "precision": "float32", "optimizer_created": False, "step_attempted": False,
              "optimizer_steps": 0, "backward_attempted": False, "backward_count": 0,
              "checkpoint_writes": 0, "checkpoint_write_attempted": False,
              "base_checkpoint_unchanged": False, "downloads_enabled": False,
              "seed": 2026, "deterministic_algorithms": True, "measurements": [],
              "base_parameter_sha256_before": None, "base_parameter_sha256_after": None,
              "adapter_sha256_before": None, "adapter_sha256_after": None,
              "adapter_parameter_count": None, "trainable_parameter_count": 0, "adapter_tensor_count": None,
              "memory_policy": memory_policy(mode=mode), "scientific_use_cleared": False,
              "saved_model_reload_performed": False}
    monitor = model = None
    try:
        spec, manifest, historical = frozen_inputs()
        report["adapter_configuration"] = spec["adapter"]
        report["provenance"] = provenance()
        report["architecture_metadata"] = metadata_targets()
        baseline = None
        if mode == "pilot":
            baseline = read_native_baseline(baseline_path, baseline_sha256, manifest, historical, report["provenance"])
        environment_before_import()
        import torch
        import transformers
        from src.lora_adapter import inject_qv, adapter_parameters
        device = telemetry.validate_environment(torch)
        if telemetry._command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip() != "Apple M4 Pro":
            raise RuntimeError("LoRA policy requires Apple M4 Pro")
        if torch.__version__ != "2.9.1" or transformers.__version__ != "4.57.1":
            raise RuntimeError("imported runtime version mismatch")
        report["environment"] = {"os": platform.mac_ver()[0], "machine": platform.machine(), "processor": "Apple M4 Pro"}
        profile = json.loads(PROFILE.read_text())
        monitor = LoRAMonitor(torch, profile=profile, device=device.type, mode=mode)
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        torch.mps.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        monitor.capture("preload", before_load=True)
        report["checkpoint"] = telemetry.ensure_checkpoint(profile, allow_download=False)
        config = transformers.AutoConfig.from_pretrained(telemetry.CHECKPOINT, local_files_only=True, trust_remote_code=False)
        telemetry.validate_config(config.to_dict(), profile)
        config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(telemetry.CHECKPOINT, local_files_only=True,
                                                              trust_remote_code=False, use_fast=True)
        examples = acquisition.prepare_examples(tokenizer, manifest)
        report["encoding_sha256"] = digest({k: [e.input_ids, e.labels] for k, e in examples.items()})
        if report["encoding_sha256"] != historical["encoding_sha256"]:
            raise ValueError("tokenizer encoding drift")
        reading = monitor.capture("before_load", before_load=True)
        report["memory_policy"] = memory_policy(reading["process_peak_rss_bytes"], mode=mode)
        if shutil.disk_usage(ROOT).free < 4 * GIB:
            raise RuntimeError("requires 4 GiB free disk")
        monitor.stage = "loading_cpu"
        monitor.start()
        model, info = transformers.AutoModelForCausalLM.from_pretrained(
            telemetry.CHECKPOINT, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        report["model_loaded"] = True
        if type(info) is not dict or any(info.get(k) != [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("incompatible base checkpoint load")
        model.requires_grad_(False)
        model.eval()
        telemetry.validate_model(model, torch, profile, device="cpu")
        monitor.boundary("after_cpu_load")
        monitor.boundary("before_base_fingerprint")
        originals = dict(model.named_parameters())  # canonical pre-injection names; references, no copies
        report["base_parameter_sha256_before"] = fingerprint(originals)
        if report["base_parameter_sha256_before"] != BASE_SHA256:
            raise ValueError("loaded base differs from verified historical fingerprint")
        monitor.boundary("after_base_fingerprint")
        torch.manual_seed(2026)  # initialization independent of loader RNG consumption
        wrappers = inject_qv(model)
        adapters = adapter_parameters(wrappers)
        report["adapter_parameter_count"] = sum(p.numel() for p in adapters.values())
        report["adapter_tensor_count"] = len(adapters)
        if report["adapter_parameter_count"] != PARAMETERS or len(adapters) != 120:
            raise RuntimeError("unexpected LoRA size")
        report["initial_adapter_zero_effect"] = all(bool(p.eq(0).all().item()) for name, p in adapters.items() if name.endswith(".lora_B"))
        if not report["initial_adapter_zero_effect"]:
            raise RuntimeError("initial B must be exactly zero")
        report["adapter_sha256_before"] = fingerprint(adapters)
        monitor.boundary("after_adapter_injection")
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        report["actual_device"] = device.type
        # _apply may replace Parameters: refresh references by canonical wrapper paths after transfer.
        originals = {name: model.get_parameter(re.sub(r"(\.(?:q_proj|v_proj))\.weight$", r"\1.base_layer.weight", name))
                     for name in originals}
        adapters = adapter_parameters(wrappers)
        versions = tuple(p._version for p in originals.values())
        adapter_versions = tuple(p._version for p in adapters.values())
        validate_layout(model, originals, wrappers, torch, training=False)
        if fingerprint(originals) != BASE_SHA256 or fingerprint(adapters) != report["adapter_sha256_before"]:
            raise RuntimeError("device transfer changed parameter bytes")
        monitor.boundary("after_load")

        def base_integrity():
            monitor.boundary("before_base_integrity")
            if fingerprint(originals) != BASE_SHA256 or tuple(p._version for p in originals.values()) != versions:
                raise RuntimeError("pretrained bytes/versions changed")
            monitor.boundary("after_base_integrity")

        def prerequisites():
            base_integrity()
            frozen_inputs()
            if provenance() != report["provenance"] or telemetry.ensure_checkpoint(profile, allow_download=False) != report["checkpoint"]:
                raise RuntimeError("code/dependencies/original checkpoint changed during run")
            if mode == "pilot" and sha256_file(baseline_path) != baseline_sha256:
                raise RuntimeError("LoRA baseline changed during run")

        # Fresh no-op evaluation must pass BEFORE any backward or optimizer construction.
        report["evaluation_before"] = acquisition.evaluate(model, tokenizer, examples, manifest, torch, device, monitor)
        acquisition.validate_baseline({**historical, "evaluation": report["evaluation_before"]}, manifest)
        compare_rows(report["evaluation_before"], historical["evaluation"])
        if mode == "pilot":
            if report["adapter_sha256_before"] != baseline["adapter_sha256_before"]:
                raise ValueError("adapter initialization differs from sealed LoRA baseline")
            compare_rows(report["evaluation_before"], baseline["evaluation"])
            prerequisites()
            for p in adapters.values():
                p.requires_grad_(True)
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
            from src.graddiff_integration import collate_causal_examples
            case_id = spec["pilot_case_id"]
            report["training_case_id"] = case_id
            batch = collate_causal_examples([examples[case_id]], tokenizer, device=device, max_sequence_length=128)
            one_step(model, batch, originals, wrappers, torch, monitor, report,
                     enabled=enabled, authorization=authorization, pre_step=prerequisites, run_path=destination)
            del batch
            report["adapter_sha256_after"] = fingerprint(adapters)
            report["weight_update_verified"] = report["adapter_sha256_after"] != report["adapter_sha256_before"]
            if not report["weight_update_verified"] or not any(bool(p.ne(0).any().item()) for n, p in adapters.items() if n.endswith(".lora_B")):
                raise RuntimeError("adapter/effective update not verified")
            model.requires_grad_(False)
            model.eval()
            model.gradient_checkpointing_disable()
            validate_layout(model, originals, wrappers, torch, training=False)
            base_integrity()
            report["evaluation"] = acquisition.evaluate(model, tokenizer, examples, manifest, torch, device, monitor)
        else:
            validate_layout(model, originals, wrappers, torch, training=False)
            report["evaluation"] = report["evaluation_before"]
            report["adapter_sha256_after"] = fingerprint(adapters)
            if report["adapter_sha256_after"] != report["adapter_sha256_before"]:
                raise RuntimeError("read-only baseline changed adapter bytes")
        acquisition.validate_baseline({**historical, "evaluation": report["evaluation"]}, manifest)
        prerequisites()
        report["base_parameter_sha256_after"] = fingerprint(originals)
        report["base_versions_unchanged"] = tuple(p._version for p in originals.values()) == versions
        if mode == "baseline":
            # Check before cleanup can erase evidence of unexpected gradients.
            validate_layout(model, originals, wrappers, torch, training=False)
            report["adapter_versions_unchanged"] = tuple(p._version for p in adapters.values()) == adapter_versions
            if not report["adapter_versions_unchanged"]:
                raise RuntimeError("read-only baseline changed adapter versions")
        report["base_checkpoint_unchanged"] = True
        if fingerprint(adapters) != report["adapter_sha256_after"]:
            raise RuntimeError("evaluation changed adapter bytes")
        monitor.boundary("before_serialization")
        if mode == "pilot":
            require_authorization(enabled, authorization)
            report["checkpoint_write_attempted"] = True
            report["saved_files_sha256"] = save_adapter(destination / "checkpoint", adapters, report, torch)
            report["checkpoint_writes"] = 1
            monitor.boundary("after_serialization")
            prerequisites()
        monitor.boundary("complete")
        report["status"] = "lora_baseline_complete_no_updates" if mode == "baseline" else "lora_one_step_complete_mechanics_only"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "STOP"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if model is not None:
                model.zero_grad(set_to_none=True)
            if monitor is not None:
                monitor.close()
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "STOP"
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if monitor is not None:
            report["measurements"] = monitor.samples
        report["peaks"] = telemetry.summarize_samples(report["measurements"])
        report["partial_in_memory_update_possible"] = report["status"] == "STOP" and report["step_attempted"]
    if report["status"] == "lora_baseline_complete_no_updates":
        try:
            validate_native_baseline(report, manifest, historical, report["provenance"])
        except (Exception, KeyboardInterrupt) as exc:
            report["status"], report["error"] = "STOP", f"baseline sealing validation: {exc}"
    report_sha = acquisition.seal_json(destination / "report.json", report)
    success = report["status"] != "STOP"
    acquisition.seal_json(destination / ("COMPLETE.json" if success else "FAILED.json"), {
        "status": report["status"], "report_sha256": report_sha,
        "checkpoint_usable_for_mechanics": success and mode == "pilot",
        "scientific_use_cleared": False, "saved_files_sha256": report.get("saved_files_sha256", {})})
    if success:
        if mode == "pilot":
            (destination / "checkpoint").chmod(0o555)
        fsync_directory(destination)
        destination.chmod(0o555)
    return report


def validation_plan():
    spec, manifest, _ = frozen_inputs()
    return {"status": "validation_only", "model_loaded": False, "training_authorized": False,
            "optimizer_steps": 0, "backward_count": 0, "checkpoint_writes": 0, "downloads_enabled": False,
            "spec": spec, "spec_sha256": SPEC_SHA256, "provenance": provenance(),
            "historical_baseline_sha256": HISTORICAL_SHA256, "architecture": metadata_targets(),
            "training_cases": len(manifest["training"]), "evaluation_cases": len(manifest["evaluation"]),
            "memory_policy": memory_policy(), "baseline_memory_policy": memory_policy(mode="baseline"),
            "blockers": ["new native LoRA baseline/no-op physical verification not yet performed",
                         "native LoRA MPS autograd/update/save memory unmeasured",
                         "fresh physical admission and explicit new-attempt authorization required",
                         "saved model reload required before scientific use"]}


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run-baseline", action="store_true")
    modes.add_argument("--run-one-step", action="store_true")
    parser.add_argument("--authorize-weight-update")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--lora-baseline", type=Path)
    parser.add_argument("--lora-baseline-sha256")
    args = parser.parse_args(argv)
    try:
        if args.run_baseline or args.run_one_step:
            if args.run_dir is None:
                raise ValueError("explicit execution requires a fresh --run-dir")
            report = run(mode="pilot" if args.run_one_step else "baseline", enabled=True,
                         authorization=args.authorize_weight_update, run_path=args.run_dir,
                         baseline_path=args.lora_baseline, baseline_sha256=args.lora_baseline_sha256)
        else:
            if any(v is not None for v in (args.run_dir, args.authorize_weight_update, args.lora_baseline, args.lora_baseline_sha256)):
                raise ValueError("execution arguments require an explicit run mode")
            report = validation_plan()
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if report["status"] == "STOP" else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(2, f"STOP: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
