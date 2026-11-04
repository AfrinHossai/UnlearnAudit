"""Opt-in, frozen SmolLM2 MPS forward measurement. Default is an offline plan.

This module has no training or model-save path. Only --run-preflight can load
weights; network acquisition additionally requires --allow-download. Run that
command in a normal Mac Terminal, never as an automated/sandbox measurement.
"""

from __future__ import annotations

import _thread
import argparse
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.graddiff_memory import parse_vm_stat
from src.inspect_tofu import sha256_file
from src.small_model_experiment import MODEL_ID, PROFILE, REVISION, ROOT, digest, encode_plain_qa
from src.small_model_memory import GIB, select_memory_guard

CHECKPOINT = ROOT / ".cache/smollm2_readonly" / REVISION
OUTPUTS = ROOT / "outputs"
VERSIONS = {"torch": "2.9.1", "transformers": "4.57.1", "huggingface_hub": "0.36.2"}
QUESTION = "What color is a clear daytime sky?"
ANSWER = "Blue."
MAX_TOKENS = 128
SAMPLE_INTERVAL_SECONDS = 0.25
MAX_SAMPLE_SECONDS = 2.0
MAX_RUN_SECONDS = 180
READONLY_GUARD_ID = "smollm2_135m_fp32_mps_single_forward_v1"
INTEGER_FIELDS = (
    "timestamp_ns", "sample_duration_ns", "total_physical_memory_bytes",
    "available_physical_memory_lower_bound_bytes", "system_memory_free_percentage",
    "macos_pressure_level", "process_current_rss_bytes", "process_peak_rss_bytes",
    "mps_tensor_allocated_bytes", "mps_driver_allocated_bytes", "swap_used_bytes", "pageouts",
)


def readonly_memory_policy(profile: Mapping[str, Any], *, device: str,
                           baseline_peak_rss_bytes: int | None = None) -> dict[str, Any]:
    """Estimated admission budget for this frozen probe only; never training.

    Existing process RSS is already excluded from available pages. Its peak is
    used for *additional* process headroom, not added as measured model memory.
    See docs/smollm2_readonly_admission_task27.md for every allowance.
    """
    select_memory_guard(profile)  # exact model/revision and frozen profile digest
    if device != "mps":
        raise RuntimeError("unknown/unsupported read-only device; requires mps")
    if baseline_peak_rss_bytes is not None and (
            type(baseline_peak_rss_bytes) is not int or baseline_peak_rss_bytes < 0):
        raise RuntimeError("invalid baseline process memory")
    terms = {
        "three_fp32_weight_copies_estimate": 3 * 4 * profile["model"]["parameter_count"],
        "source_file_residency_allowance_estimate": profile["model"]["weight_file_metadata_only"]["size_bytes"],
        "forward_workspace_allowance_estimate": 2 * GIB,
        "tokenizer_and_host_cache_allowance_estimate": GIB,
        "metal_runtime_cache_allowance_estimate": GIB,
        "additional_process_headroom_estimate": max(GIB, baseline_peak_rss_bytes or 0),
        "uncertainty_reserve_estimate": 3 * GIB,
    }
    incremental = sum(terms.values())
    reserve = 6 * GIB
    required = reserve + ((incremental + GIB - 1) // GIB) * GIB
    return {"guard_id": READONLY_GUARD_ID, "scope": "readonly_single_forward",
            "model_id": MODEL_ID, "revision": REVISION, "device": device,
            "precision": "float32", "batch_size": 1, "max_input_tokens": MAX_TOKENS,
            "use_cache": False, "baseline_process_peak_rss_bytes": baseline_peak_rss_bytes,
            "estimated_incremental_terms_bytes": terms,
            "estimated_incremental_total_bytes": incremental,
            "preload_available_required_bytes": required,
            "continuing_available_required_bytes": reserve,
            "minimum_total_host_bytes": 48 * GIB, "minimum_system_memory_free_percentage": 20,
            "measured_model_requirement": False, "training_authorized": False,
            "approved_small_model_training_threshold_bytes": None}


def plan() -> dict[str, Any]:
    profile = json.loads(PROFILE.read_text())
    policy = readonly_memory_policy(profile, device="mps")
    return {"mode": "plan_only", "model_id": MODEL_ID, "revision": REVISION,
            "license": "Apache-2.0", "weight_file_bytes": 269060552,
            "recommended_free_disk_bytes_estimate": 2 * GIB,
            "device": "mps", "precision": "float32", "max_input_tokens": MAX_TOKENS,
            "downloads_enabled": False, "model_loaded": False, "training_authorized": False,
            "approved_small_model_training_threshold_bytes": None,
            "preload_safety_floor_bytes": policy["preload_available_required_bytes"],
            "readonly_memory_policy": policy,
            "safety_floor_is_not_measured_model_requirement": True}


def validate_environment(torch: Any) -> Any:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("requires a normal Apple Silicon macOS process")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback must be disabled before importing Torch")
    for name in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                 "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
        if name in os.environ:
            raise RuntimeError(f"remove allocator/math override {name}; use backend defaults")
    if any(os.environ.get(name, "1") != "1" for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise RuntimeError("single-process preflight required")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no device fallback")
    device = torch.device("mps")
    if device.type != "mps":
        raise RuntimeError("device mismatch")
    return device


def _safe_path(path: Path, root: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    if path.is_symlink() or path.absolute() != path.resolve() or not path.resolve().is_relative_to(root):
        raise ValueError("path must stay inside its repository root without symlinks")
    if root.is_symlink():
        raise ValueError("symlink root forbidden")
    return path


def validate_report_path(path: Path) -> Path:
    path = _safe_path(path, OUTPUTS)
    if path.suffix != ".json" or path.exists() or not path.parent.is_dir():
        raise ValueError("report must be a new JSON file in an existing outputs directory")
    result = subprocess.run(["git", "check-ignore", "--quiet", str(path)], cwd=ROOT, timeout=2)
    if result.returncode != 0:
        raise ValueError("report must be Git-ignored")
    return path


def _download(filename: str) -> None:
    # Called only after explicit runtime/download opt-ins and device/memory checks.
    os.environ["HF_HOME"] = str(ROOT / ".cache/smollm2_hub")
    os.environ["HF_XET_CACHE"] = str(ROOT / ".cache/smollm2_hub/xet")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=MODEL_ID, revision=REVISION, filename=filename,
                    local_dir=str(CHECKPOINT), cache_dir=str(ROOT / ".cache/smollm2_hub/hub"),
                    token=False, endpoint="https://huggingface.co",
                    local_files_only=False)


def ensure_checkpoint(profile: Mapping[str, Any], *, allow_download: bool) -> dict[str, Any]:
    select_memory_guard(profile)
    _safe_path(CHECKPOINT, ROOT / ".cache")
    if CHECKPOINT.exists() and {p.name for p in CHECKPOINT.iterdir()} - {
        *profile["model"]["metadata_sha256"], "model.safetensors", ".cache"
    }:
        raise ValueError("unexpected checkpoint files; adapters/alternate weights are forbidden")
    if allow_download and shutil.disk_usage(ROOT).free < 2 * GIB:
        raise RuntimeError("explicit download requires at least 2 GiB free disk")
    hashes = profile["model"]["metadata_sha256"]
    specs = {"README.md": hashes["README.md"], **hashes}
    for name, expected in specs.items():
        path = _safe_path(CHECKPOINT / name, ROOT / ".cache")
        if not path.is_file():
            if not allow_download:
                raise FileNotFoundError("pinned checkpoint unavailable; downloads disabled")
            _download(name)
        _safe_path(path, ROOT / ".cache")
        if sha256_file(path) != expected:
            raise ValueError(f"checkpoint metadata checksum mismatch: {name}")
        if name == "README.md" and not re.search(r"^license: apache-2\.0\s*$", path.read_text(), re.MULTILINE):
            raise ValueError("pinned official model card license mismatch")
    config = json.loads((CHECKPOINT / "config.json").read_text())
    validate_config(config, profile)
    weight_spec = profile["model"]["weight_file_metadata_only"]
    weights = _safe_path(CHECKPOINT / "model.safetensors", ROOT / ".cache")
    if not weights.is_file():
        if not allow_download:
            raise FileNotFoundError("pinned weights unavailable; downloads disabled")
        _download("model.safetensors")
    _safe_path(weights, ROOT / ".cache")
    if weights.stat().st_size != weight_spec["size_bytes"] or sha256_file(weights) != weight_spec["hub_lfs_sha256"]:
        raise ValueError("pinned weight size/SHA-256 mismatch")
    return {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0",
            "weight_sha256": weight_spec["hub_lfs_sha256"], "weight_bytes": weights.stat().st_size,
            "metadata_sha256": dict(hashes), "local_files_only_loading": True}


def validate_config(config: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "tie_word_embeddings",
                "vocab_size", "max_position_embeddings", "eos_token_id"):
        if type(config.get(key)) is not type(profile["model"][key]) or config[key] != profile["model"][key]:
            raise ValueError(f"incompatible checkpoint config: {key}")
    if config.get("architectures") != ["LlamaForCausalLM"] or config.get("auto_map"):
        raise ValueError("incompatible architecture or remote-code mapping")


def _command(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=2,
                          env={**os.environ, "LC_ALL": "C"}).stdout


def read_sample(torch: Any, stage: str) -> dict[str, Any]:
    """Observe counters without synchronization; phase-boundary caller synchronizes."""
    start = time.monotonic_ns()
    vm = _command(["/usr/bin/vm_stat"])
    pressure = _command(["/usr/bin/memory_pressure", "-Q"])
    sysctl = _command(["/usr/sbin/sysctl", "-n", "hw.memsize", "kern.memorystatus_vm_pressure_level", "vm.swapusage"])
    fields = sysctl.strip().splitlines()
    percent = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    pageouts = re.search(r"^Pageouts:\s*(\d+)\.?$", vm, re.MULTILINE)
    used = re.search(r"used\s*=\s*([0-9.]+)M", fields[2] if len(fields) == 3 else "")
    if len(fields) != 3 or not percent or not pageouts or not used:
        raise RuntimeError("required physical memory telemetry unavailable")
    rss = int(_command(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) * 1024
    sample = {"stage": stage, "timestamp_ns": start,
            "total_physical_memory_bytes": int(fields[0]), "macos_pressure_level": int(fields[1]),
            "available_physical_memory_lower_bound_bytes": parse_vm_stat(vm),
            "system_memory_free_percentage": int(percent.group(1)), "pageouts": int(pageouts.group(1)),
            "swap_used_bytes": int(float(used.group(1)) * 1024**2),
            "process_current_rss_bytes": rss,
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "mps_tensor_allocated_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    sample["sample_duration_ns"] = time.monotonic_ns() - start
    return sample


def _validate_reading(sample: Mapping[str, Any]) -> None:
    if not isinstance(sample, Mapping):
        raise RuntimeError("invalid memory measurement object")
    for key in INTEGER_FIELDS:
        if type(sample.get(key)) is not int or sample[key] < 0:
            raise RuntimeError(f"invalid or missing memory reading: {key}")
    if not isinstance(sample.get("stage"), str) or not sample["stage"]:
        raise RuntimeError("missing measurement stage")
    total, available = sample["total_physical_memory_bytes"], sample["available_physical_memory_lower_bound_bytes"]
    if total < 48 * GIB or not 6 * GIB <= available <= total:
        raise RuntimeError("memory safety floor failed (48 GiB capacity / 6 GiB continuing)")
    if not 20 <= sample["system_memory_free_percentage"] <= 100 or sample["macos_pressure_level"] != 1:
        raise RuntimeError("memory pressure is not normal")
    if (not 0 < sample["process_current_rss_bytes"] <= sample["process_peak_rss_bytes"] <= total
            or not sample["mps_tensor_allocated_bytes"] <= sample["mps_driver_allocated_bytes"] <= total
            or not 0 < sample["sample_duration_ns"] <= MAX_SAMPLE_SECONDS * 1e9
            or sample["timestamp_ns"] == 0):
        raise RuntimeError("inconsistent or stale memory telemetry")


def validate_sample(sample: Mapping[str, Any], *, profile: Mapping[str, Any], device: str,
                    before_load: bool = False, baseline: Mapping[str, Any] | None = None) -> None:
    """Read-only admission; baseline is the immediately preceding observation."""
    _validate_reading(sample)
    policy = readonly_memory_policy(profile, device=device,
                                    baseline_peak_rss_bytes=sample["process_peak_rss_bytes"])
    now = time.monotonic_ns()
    end = sample["timestamp_ns"] + sample["sample_duration_ns"]
    if not end <= now <= sample["timestamp_ns"] + int(MAX_SAMPLE_SECONDS * 1e9):
        raise RuntimeError("inconsistent or stale memory telemetry")
    required = policy["preload_available_required_bytes"]
    if before_load and sample["available_physical_memory_lower_bound_bytes"] < required:
        raise RuntimeError(f"read-only preload memory floor failed ({required // GIB} GiB available required)")
    if baseline is not None:
        _validate_reading(baseline)
        if sample["timestamp_ns"] <= baseline["timestamp_ns"]:
            raise RuntimeError("memory timestamps must increase")
        if (sample["total_physical_memory_bytes"] != baseline["total_physical_memory_bytes"]
                or sample["swap_used_bytes"] > baseline["swap_used_bytes"]
                or sample["pageouts"] != baseline["pageouts"]):
            raise RuntimeError("capacity changed or swap/page-outs grew; stop measurement")


class Monitor:
    """Bounded sampling watchdog. SIGINT requests exit at a Python safe point."""

    def __init__(self, torch: Any, *, profile: Mapping[str, Any], device: str):
        readonly_memory_policy(profile, device=device)
        self.torch = torch
        self.profile, self.device = profile, device
        self.samples: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.stage = "before_load"
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def capture(self, stage: str, *, before_load: bool = False) -> dict[str, Any]:
        with self.lock:
            self.check()
            sample = read_sample(self.torch, stage)
            self.samples.append(sample)  # retain the failing reading for the STOP report
            validate_sample(sample, profile=self.profile, device=self.device, before_load=before_load,
                            baseline=self.samples[-2] if len(self.samples) > 1 else None)
            return sample

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"memory watchdog stopped: {self.error}") from self.error

    def _loop(self) -> None:
        started = time.monotonic()
        try:
            while not self.stop_event.wait(SAMPLE_INTERVAL_SECONDS):
                if time.monotonic() - started > MAX_RUN_SECONDS:
                    raise RuntimeError("read-only measurement time budget exceeded")
                self.capture(self.stage)
        except Exception as exc:
            self.error = exc
            _thread.interrupt_main()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True, name="smollm2-memory-watchdog")
        self.thread.start()

    def boundary(self, stage: str) -> None:
        self.check()
        self.torch.mps.synchronize()
        self.capture(stage)

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise RuntimeError("memory watchdog could not stop")
        self.check()


def validate_model(model: Any, torch: Any, profile: Mapping[str, Any], *, device: str) -> None:
    if (type(model).__name__ != "LlamaForCausalLM" or model.training or model.config.use_cache
            or getattr(model.config, "_attn_implementation", None) != "sdpa"):
        raise RuntimeError("model class/eval/cache configuration mismatch")
    validate_config(model.config.to_dict(), profile)
    params = list(model.parameters())
    if not params or sum(p.numel() for p in params) != profile["model"]["parameter_count"]:
        raise RuntimeError("parameter count mismatch")
    if any(p.requires_grad or p.grad is not None or p.dtype != torch.float32 or p.device.type != device for p in params):
        raise RuntimeError("parameters must be frozen FP32 on the requested device")
    if any(buffer.device.type != device for buffer in model.buffers()):
        raise RuntimeError("model buffer device mismatch")


def forward_once(model: Any, tokenizer: Any, torch: Any, device: Any,
                 *, observe_output: Any = None) -> dict[str, Any]:
    from src.graddiff import causal_lm_nll
    if (type(tokenizer).__name__ != "GPT2TokenizerFast" or not tokenizer.is_fast
            or len(tokenizer) != 49152 or tokenizer.eos_token_id != 0 or tokenizer.chat_template
            or tokenizer.pad_token_id is not None):
        raise ValueError("incompatible tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    example = encode_plain_qa(tokenizer, QUESTION, ANSWER, max_sequence_length=MAX_TOKENS)
    versions = [p._version for p in model.parameters()]
    with torch.inference_mode():
        ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        labels = torch.tensor([example.labels], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        if any(t.device.type != "mps" for t in (ids, labels, mask)):
            raise RuntimeError("input device mismatch")
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False,
                       return_dict=True, output_hidden_states=False, output_attentions=False)
        logits, loss = output.logits, output.loss
        if (logits.device.type != "mps" or loss.device.type != "mps"
                or logits.dtype != torch.float32 or loss.dtype != torch.float32
                or tuple(logits.shape) != (1, len(example.input_ids), 49152)
                or tuple(loss.shape) != () or logits.requires_grad or loss.requires_grad
                or getattr(output, "past_key_values", None) is not None):
            raise RuntimeError("invalid output shape/dtype/device/cache/gradients")
        local_loss = causal_lm_nll(logits, labels)
        nll, local_nll = float(loss.item()), float(local_loss.item())
        if (not math.isfinite(nll) or nll < 0 or not math.isfinite(local_nll)
                or not math.isclose(nll, local_nll, rel_tol=1e-4, abs_tol=1e-4)):
            raise RuntimeError("nonfinite output or causal loss parity failure")
        next_id = int(logits[0, -1].argmax().item())
        if not 0 <= next_id < 49152:
            raise RuntimeError("invalid next-token output")
        if observe_output is not None:
            observe_output()  # capture with logits/loss still resident
    if versions != [p._version for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("parameter versions or gradients changed")
    return {"input_source": "project_authored_synthetic_CC0_1.0", "input_sha256": digest([QUESTION, ANSWER]),
            "input_tokens": len(example.input_ids), "batch_size": 1, "forward_count": 1,
            "generation_performed": False, "next_token_id": next_id, "answer_nll": nll,
            "local_answer_nll": local_nll, "parameter_versions_unchanged": True}


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("process_current_rss_bytes", "mps_tensor_allocated_bytes", "mps_driver_allocated_bytes")
    def observed(key: str) -> list[int]:
        return [s[key] for s in samples if type(s.get(key)) is int and s[key] >= 0]

    times = observed("timestamp_ns")
    return {"sample_count": len(samples), "sample_interval_target_seconds": SAMPLE_INTERVAL_SECONDS,
            "max_sample_gap_ns_observed": max((b - a for a, b in zip(times, times[1:])), default=0),
            "sampled_maxima_bytes": {k: max(observed(k), default=None) for k in keys},
            "os_process_lifetime_peak_rss_bytes_observed": max(observed("process_peak_rss_bytes"), default=None),
            "native_mps_peak_bytes": None,
            "mps_peak_method": "maximum of observed samples; lower bound on true peak; no native peak API in Torch 2.9.1",
            "rss_and_mps_are_overlapping_do_not_sum": True}


def run_preflight(*, enabled: bool, allow_download: bool, report_path: Path) -> dict[str, Any]:
    if not enabled:
        raise ValueError("model loading requires explicit --run-preflight")
    path = validate_report_path(report_path)
    profile = json.loads(PROFILE.read_text())
    select_memory_guard(profile)
    # Reject unsafe environment values before importing/initializing the backend.
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback must be disabled")
    # Set cache/network behavior before importing Hub/Transformers constants.
    # HTTP-only acquisition avoids a secondary Xet cache outside the repository.
    os.environ.update(HF_HOME=str(ROOT / ".cache/smollm2_hub"),
                      HF_HUB_CACHE=str(ROOT / ".cache/smollm2_hub/hub"),
                      HF_XET_CACHE=str(ROOT / ".cache/smollm2_hub/xet"),
                      HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_TELEMETRY="1")
    for cache_name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        _safe_path(Path(os.environ[cache_name]), ROOT / ".cache")
    if not allow_download:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import transformers
    import huggingface_hub
    for name, module in (("torch", torch), ("transformers", transformers), ("huggingface_hub", huggingface_hub)):
        if module.__version__ != VERSIONS[name]:
            raise RuntimeError(f"requires pinned {name}=={VERSIONS[name]}")
    device = validate_environment(torch)
    monitor = Monitor(torch, profile=profile, device=device.type)
    report: dict[str, Any] = {"schema_version": "1.0", "status": "STOP", "scope": "readonly_single_forward",
                              "model_id": MODEL_ID, "revision": REVISION, "profile_sha256": digest(profile),
                              "requested_device": "mps", "telemetry_device": str(device), "actual_device": None,
                              "model_loaded": False, "precision": "float32", "training_authorized": False,
                              "approved_small_model_training_threshold_bytes": None,
                              "backward_performed": False, "optimizer_created": False, "weight_updates": 0,
                              "environment": {**VERSIONS, "os": platform.mac_ver()[0], "machine": platform.machine()},
                              "measurement_kind": "direct_counters_and_sampled_maxima_not_training_estimates",
                              "measurement_sources": {
                                  "rss": "ps RSS KiB converted to bytes; getrusage lifetime peak bytes on Darwin",
                                  "mps": "Torch current/driver counters; sampled maxima are lower bounds on peaks",
                                  "availability": "derived vm_stat free+speculative pages times page size; excludes reclaimable caches",
                                  "pressure": "memory_pressure -Q free indicator plus read-only kernel pressure level",
                                  "swap": "sysctl vm.swapusage rounded MiB converted to bytes; sub-resolution growth can be missed",
                                  "sample_atomic": False},
                              "preload_safety_floor_is_not_measured_model_requirement": True}
    report["readonly_memory_policy"] = readonly_memory_policy(profile, device=device.type)
    try:
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        torch.mps.synchronize()
        monitor.capture("baseline", before_load=True)
        # Two observations are required to detect active swap/page-out growth
        # before any checkpoint acquisition. Later readings compare adjacent
        # samples, so a fall followed by a rise cannot hide behind the baseline.
        time.sleep(SAMPLE_INTERVAL_SECONDS)
        before_checkpoint = monitor.capture("before_checkpoint", before_load=True)
        report["readonly_memory_policy"] = readonly_memory_policy(
            profile, device=device.type, baseline_peak_rss_bytes=before_checkpoint["process_peak_rss_bytes"])
        provenance = ensure_checkpoint(profile, allow_download=allow_download)
        report["checkpoint"] = provenance
        # Deserialization and all Transformers accesses are local-only regardless
        # of whether the explicit acquisition above was enabled.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        config = transformers.AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=False)
        validate_config(config.to_dict(), profile)
        config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True,
                                                              trust_remote_code=False, use_fast=True)
        before_load = monitor.capture("before_load", before_load=True)
        report["readonly_memory_policy"] = readonly_memory_policy(
            profile, device=device.type, baseline_peak_rss_bytes=before_load["process_peak_rss_bytes"])
        monitor.stage = "loading_cpu"
        monitor.start()
        model, loading_info = transformers.AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        report["model_loaded"] = True
        monitor.check()
        if not isinstance(loading_info, dict) or any(loading_info.get(k) != [] for k in
                                                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("incomplete or incompatible checkpoint loading")
        model.requires_grad_(False)
        model.eval()
        validate_model(model, torch, profile, device="cpu")
        report["actual_device"] = "cpu"
        monitor.boundary("after_cpu_load")
        monitor.stage = "transfer_to_mps"
        model.to(device)
        validate_model(model, torch, profile, device="mps")
        report["actual_device"] = "mps"
        monitor.boundary("after_load")
        monitor.stage = "inference"
        result = forward_once(model, tokenizer, torch, device,
                              observe_output=lambda: monitor.boundary("after_inference_outputs_live"))
        monitor.boundary("after_inference")
        validate_model(model, torch, profile, device="mps")
        report.update(status="readonly_measurement_passed_not_training_clearance", checkpoint=provenance,
                      inference=result, parameters_frozen=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "reason": str(monitor.error or exc) or "interrupted"}
    finally:
        try:
            monitor.close()
        except Exception as exc:
            report.update(status="STOP", error={"type": type(exc).__name__, "reason": str(exc)})
        report["measurements"] = monitor.samples
        report["peaks"] = summarize_samples(monitor.samples)
    # Exclusive creation and a second path check prevent overwrites/symlink escapes.
    validate_report_path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="explicit pinned official checkpoint acquisition")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)
    if (args.allow_download or args.report_path) and not args.run_preflight:
        parser.error("download/report options require --run-preflight")
    if args.run_preflight and args.report_path is None:
        parser.error("--run-preflight requires a new --report-path under outputs/")
    try:
        report = run_preflight(enabled=True, allow_download=args.allow_download, report_path=args.report_path) if args.run_preflight else plan()
        print(json.dumps(report, indent=2, allow_nan=False))
        if args.run_preflight:
            return 0 if report["status"] == "readonly_measurement_passed_not_training_clearance" else 2
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOP: read-only preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
