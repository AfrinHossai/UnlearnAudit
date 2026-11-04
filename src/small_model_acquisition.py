"""Task 38: offline validation by default; owner-run frozen baseline / ONE SGD step.

Separate SmolLM2 development policy. No Llama admission path, network option,
resume, scheduler, multi-step option, or caller-selected checkpoint destination.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import small_model_preflight as telemetry
from src.small_model_backward import (parameter_fingerprint, validate_backward_model,
                                      validate_gradients)
from src.small_model_experiment import (MODEL_ID, PROFILE, REVISION, ROOT,
                                        acquisition_order, digest, encode_plain_qa)
from src.small_model_memory import GIB, select_memory_guard
from src.inspect_tofu import sha256_file
from src.evaluate_tofu_baseline import normalize_answer

MANIFEST = ROOT / "experiments/smollm2_135m_acquisition_task38.json"
# Filled from the reviewed, generated-once manifest, never from runtime arguments.
MANIFEST_SHA256 = "60e9872e86e8bd98b1a281866f686c7ff3ed41ca3c8b5709e5b4c4b1daa239de"
DEVELOPMENT = ROOT / "experiments/smollm2_135m_development_facts.json"
UTILITY = ROOT / "experiments/smollm2_135m_utility_probes.json"
SOURCES = {
    str(DEVELOPMENT.relative_to(ROOT)): "a1aa5af725724e738a9d0d47fde3011f8d063a6443041cffb25198aaa9032f76",
    str(UTILITY.relative_to(ROOT)): "dd11adbbb2a489c7b111d31a31f35666dbd22c54af80430b3d13ed875426c350",
    "experiments/smollm2_135m_tofu_manifest.json": "0ba5d80715dd9fb0c542d16f5167c23a0464da89a2f413d6a6bbd18235f2ea21",
}
RUNS = ROOT / "outputs/smollm2_acquisition_task38"
GUARD_ID = "smollm2_135m_fp32_mps_dev_one_sgd_step_v1"
AUTHORIZATION = "SmolLM2-135M:development:one-SGD-step"
LR = 0.0001
MAX_TOKENS = 128
MAX_NEW_TOKENS = 64
MAX_SECONDS = 1800  # includes sequential frozen evaluation; no extra updates


def build_manifest() -> dict[str, Any]:
    """Pure offline derivation; CLI cannot regenerate/overwrite the frozen file."""
    profile = json.loads(PROFILE.read_text())
    select_memory_guard(profile)
    for name, expected in SOURCES.items():
        telemetry._safe_path(ROOT / name, ROOT / "experiments")
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    facts = json.loads(DEVELOPMENT.read_text())["facts"]
    utility = json.loads(UTILITY.read_text())["cases"]
    training = [{k: row[k] for k in ("id", "role", "question", "answer")} for row in facts]
    evaluation = []
    for row in facts:
        for kind, field in (("direct", "question"), ("paraphrase", "paraphrased_question")):
            evaluation.append({"id": row["id"] + ":" + kind, "kind": kind,
                               "role": row["role"], "question": row[field],
                               "answer": row["answer"], "target_aliases": row["target_aliases"]})
    evaluation.extend({**row, "kind": "utility", "role": "utility", "target_aliases": []}
                      for row in utility)
    direct = {normalize_answer(r["question"]) for r in training}
    tests = {normalize_answer(r["question"]) for r in evaluation if r["kind"] != "direct"}
    if (len(training) != 32 or len(evaluation) != 76 or len(direct) != 32
            or len(tests) != 44 or direct & tests):
        raise ValueError("development/test wording overlap or count mismatch")
    return {
        "schema_version": "1.0", "scope": "development_fact_acquisition_one_step_only",
        "model_id": MODEL_ID, "revision": REVISION, "profile_sha256": digest(profile),
        "source_files_sha256": SOURCES, "seed": 2026,
        "training": training, "evaluation": evaluation,
        "epoch_zero_order": acquisition_order([r["id"] for r in training], 0),
        "pilot_steps": 1, "learning_rate": LR, "max_sequence_length": MAX_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS, "batch_size": 1,
        "objective": "positive answer-only mean causal NLL; acquire both development roles",
        "metrics": ["answer_nll", "normalized_exact_match", "alias_recovery"],
        "generation": "greedy question-only; EOS stop; no cache; 64 new tokens; no target in prompt",
        "overlap": "direct evaluation reuses all 32 training QA; paraphrases share entities/answers but not wording; utility never trained",
        "limitations": "32 artificial public facts and 12 utility probes; no held-out facts; development paraphrases used for later calibration; not TOFU scores or evidence of unlearning",
    }


def frozen_manifest() -> dict[str, Any]:
    telemetry._safe_path(MANIFEST, ROOT / "experiments")
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("frozen evaluation manifest checksum mismatch")
    manifest = json.loads(MANIFEST.read_text())
    if manifest != build_manifest():
        raise ValueError("frozen evaluation manifest/source mismatch")
    return manifest


def memory_policy(profile: Mapping[str, Any], baseline_peak: int = 0) -> dict[str, Any]:
    """New narrow admission policy, estimated conservatively, not calibrated fit."""
    select_memory_guard(profile)
    if type(baseline_peak) is not int or baseline_peak < 0:
        raise ValueError("invalid baseline RSS")
    w = 4 * profile["model"]["parameter_count"]
    terms = {
        "loading_three_fp32_copies": 3 * w, "fp32_gradients": w,
        "source_file_residency": 269060552, "forward_backward_workspace": 4 * GIB,
        "tokenizer_integrity": GIB, "metal_cache": GIB,
        "additional_process_headroom": max(GIB, baseline_peak),
        "clipping_step_serialization_allowance": 4 * GIB, "uncertainty": 4 * GIB,
    }
    return {"guard_id": GUARD_ID, "model_id": MODEL_ID, "revision": REVISION,
            "device": "mps", "precision": "float32", "max_sequence_length": MAX_TOKENS,
            "batch_size": 1, "max_optimizer_steps": 1, "optimizer_state_bytes": 0,
            "estimated_incremental_terms_bytes": terms,
            "preload_available_required_bytes": 6 * GIB + math.ceil(sum(terms.values()) / GIB) * GIB,
            "continuing_available_required_bytes": 6 * GIB, "minimum_total_host_bytes": 48 * GIB,
            "max_process_peak_rss_bytes": 8 * GIB, "max_mps_driver_bytes": 8 * GIB,
            "calibrated_training_requirement": False,
            "scope": "Task38 baseline and development one-step pilot only; no longer-run admission"}


def validate_sample(sample: Mapping[str, Any], *, profile: Mapping[str, Any],
                    before_load: bool = False, previous: Mapping[str, Any] | None = None,
                    first: Mapping[str, Any] | None = None, active: bool = False) -> None:
    telemetry._validate_reading(sample)  # direct counter validity; no Llama guard
    now = time.monotonic_ns()
    if not sample["timestamp_ns"] + sample["sample_duration_ns"] <= now <= sample["timestamp_ns"] + 2_000_000_000:
        raise RuntimeError("stale/future telemetry")
    policy = memory_policy(profile, sample["process_peak_rss_bytes"])
    if before_load and sample["available_physical_memory_lower_bound_bytes"] < policy["preload_available_required_bytes"]:
        raise RuntimeError("SmolLM2 pilot preload floor failed (minimum 24 GiB)")
    if (sample["process_peak_rss_bytes"] > policy["max_process_peak_rss_bytes"]
            or sample["mps_driver_allocated_bytes"] > policy["max_mps_driver_bytes"]):
        raise RuntimeError("pilot RSS/driver 8 GiB stop cap exceeded")
    if previous is not None:
        gap = sample["timestamp_ns"] - previous["timestamp_ns"]
        if gap <= 0 or (active and gap > 2_000_000_000):
            raise RuntimeError("unordered samples or watchdog gap exceeds 2 seconds")
    if first is not None and (
            sample["total_physical_memory_bytes"] != first["total_physical_memory_bytes"]
            or sample["swap_used_bytes"] > first["swap_used_bytes"]
            or sample["pageouts"] != first["pageouts"]):
        raise RuntimeError("capacity changed or cumulative swap/page-out growth")


class PilotMonitor(telemetry.Monitor):
    """Reuse collection only. Capture and time budget use this policy exclusively."""

    def __init__(self, torch: Any, *, profile: Mapping[str, Any], device: str):
        if device != "mps":
            raise RuntimeError("pilot requires MPS")
        # Parent initializes collection primitives; its read-only policy is not admission.
        super().__init__(torch, profile=profile, device=device)
        memory_policy(profile)
        self.started = time.monotonic()

    def capture(self, stage: str, *, before_load: bool = False) -> dict[str, Any]:
        with self.lock:
            self.check()
            if time.monotonic() - self.started > MAX_SECONDS:
                raise RuntimeError("pilot/evaluation time budget exceeded")
            sample = telemetry.read_sample(self.torch, stage)
            previous = self.samples[-1] if self.samples else None
            first = self.samples[0] if self.samples else None
            self.samples.append(sample)
            validate_sample(sample, profile=self.profile, before_load=before_load,
                            previous=previous, first=first, active=self.thread is not None)
            return sample

    def _loop(self) -> None:
        import _thread
        try:
            while not self.stop_event.wait(telemetry.SAMPLE_INTERVAL_SECONDS):
                self.capture(self.stage)
        except Exception as exc:
            self.error = exc
            _thread.interrupt_main()


def artifact_path(path: Path, *, new: bool) -> Path:
    """Dedicated namespace; reject aliases, symlinks, historical or tracked output."""
    path = telemetry._safe_path(path, RUNS)
    if path == RUNS or path.suffix != ".json" or path.parent != RUNS:
        raise ValueError("artifact must be a direct JSON child of Task38 outputs")
    require_ignored(path)
    if new:
        if path.exists():
            raise FileExistsError("historical artifact cannot be replaced")
    elif not path.is_file() or path.stat().st_mode & 0o222 or path.stat().st_nlink != 1:
        raise ValueError("baseline must exist and be sealed read-only")
    return path


def new_run_path(path: Path) -> Path:
    path = telemetry._safe_path(path, RUNS)
    if (path.parent != RUNS or not re.fullmatch(r"pilot-[A-Za-z0-9_-]+", path.name)
            or path.exists()):
        raise ValueError("checkpoint run must be a new pilot-* directory in Task38 outputs")
    require_ignored(path)
    return path


def require_ignored(path: Path) -> None:
    check = subprocess.run(["git", "check-ignore", "--quiet", str(path)], cwd=ROOT, timeout=2)
    if check.returncode != 0:
        raise ValueError("runtime artifacts must be Git-ignored and untracked")


def seal_json(path: Path, report: Mapping[str, Any]) -> str:
    """Exclusive creation, flushed to disk and sealed; never overwrites a file."""
    data = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return sha256_file(path)


def checkpoint_hashes(directory: Path) -> dict[str, str]:
    files = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    if not {"model.safetensors", "config.json"} <= set(files):
        raise RuntimeError("checkpoint serialization missing weights/config")
    return files


def require_authorization(enabled: bool, authorization: str | None) -> None:
    if enabled is not True or authorization != AUTHORIZATION:
        raise ValueError("requires --run-one-step AND exact --authorize-weight-update token")


def validate_optimizer(optimizer: Any, model: Any, torch: Any) -> None:
    if type(optimizer) is not torch.optim.SGD or optimizer.state:
        raise RuntimeError("requires exact stateless torch.optim.SGD")
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise RuntimeError("one full-parameter optimizer group required")
    group = groups[0]
    expected = {"lr": LR, "momentum": 0.0, "dampening": 0.0, "weight_decay": 0.0,
                "nesterov": False, "maximize": False, "foreach": False,
                "fused": False, "differentiable": False}
    if any(type(group.get(k)) is not type(v) or group[k] != v for k, v in expected.items()):
        raise RuntimeError("optimizer configuration drift")
    if [id(p) for p in group["params"]] != [id(p) for p in model.parameters()]:
        raise RuntimeError("optimizer must cover every parameter exactly once")


def checked_loss(model: Any, batch: Mapping[str, Any], torch: Any) -> Any:
    from src.graddiff import causal_lm_nll
    if (set(batch) != {"input_ids", "attention_mask", "labels"}
            or any(t.device.type != "mps" or t.dtype != torch.long for t in batch.values())
            or batch["input_ids"].shape[0] != 1
            or not 2 <= batch["input_ids"].shape[1] <= MAX_TOKENS):
        raise RuntimeError("batch identity/device/size mismatch")
    output = model(**batch, use_cache=False, return_dict=True,
                   output_attentions=False, output_hidden_states=False)
    loss, logits = output.loss, output.logits
    if (loss.ndim != 0 or loss.dtype != torch.float32 or loss.device.type != "mps"
            or logits.dtype != torch.float32 or logits.device.type != "mps"
            or tuple(logits.shape) != (*batch["input_ids"].shape, 49152)
            or getattr(output, "past_key_values", None) is not None):
        raise RuntimeError("loss/logits shape/device/precision/cache mismatch")
    with torch.no_grad():
        local = causal_lm_nll(logits.detach(), batch["labels"])
    value = float(loss.item())
    if not math.isfinite(value) or value < 0 or not math.isclose(value, float(local.item()), rel_tol=1e-4, abs_tol=1e-4):
        raise FloatingPointError("nonfinite loss or loss parity failure")
    return loss


def one_step(model: Any, batch: Mapping[str, Any], torch: Any, monitor: Any,
             profile: Mapping[str, Any], report: dict[str, Any], *,
             enabled: bool, authorization: str | None) -> None:
    """Single update boundary; all preconditions repeat immediately before step."""
    require_authorization(enabled, authorization)
    if report.get("optimizer_steps") != 0 or report.get("step_attempted") is not False:
        raise RuntimeError("one-step budget already consumed")
    validate_backward_model(model, torch, profile)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("preexisting gradients forbidden")
    optimizer = None
    try:
        monitor.boundary("before_training_forward")
        loss = checked_loss(model, batch, torch)
        if not loss.requires_grad:
            raise RuntimeError("training loss lacks graph")
        report["training_nll_before"] = float(loss.item())
        monitor.boundary("before_backward")
        loss.backward()
        del loss
        monitor.boundary("after_backward")
        report["gradients_before_clip"] = validate_gradients(model, torch, device="mps")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        report["gradient_norm_before_clip"] = float(norm.item())
        if not math.isfinite(report["gradient_norm_before_clip"]):
            raise FloatingPointError("nonfinite clipping norm")
        report["gradients_after_clip"] = validate_gradients(model, torch, device="mps")
        clipped_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()],
                                                     foreach=False, error_if_nonfinite=True)
        report["gradient_norm_after_clip"] = float(clipped_norm.item())
        if not 0 < report["gradient_norm_after_clip"] <= 1.00001:
            raise RuntimeError("gradient clipping cap not satisfied")
        monitor.boundary("after_clipping")
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, dampening=0.0,
                                    weight_decay=0.0, nesterov=False, maximize=False,
                                    foreach=False, fused=False, differentiable=False)
        report["optimizer_created"] = True
        validate_optimizer(optimizer, model, torch)
        validate_backward_model(model, torch, profile)
        require_authorization(enabled, authorization)
        monitor.boundary("before_optimizer_step")
        report["step_attempted"] = True  # failures from here may have partial RAM updates
        optimizer.step()
        report["optimizer_steps"] = 1
        monitor.boundary("after_optimizer_step")
        validate_optimizer(optimizer, model, torch)
        report["optimizer_state_bytes"] = 0
        for parameter in model.parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError("nonfinite parameter after step")
    finally:
        model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def prepare_examples(tokenizer: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if (type(tokenizer).__name__ != "GPT2TokenizerFast" or not tokenizer.is_fast
            or len(tokenizer) != 49152 or tokenizer.eos_token_id != 0
            or tokenizer.chat_template or tokenizer.pad_token_id is not None):
        raise ValueError("pinned tokenizer mismatch")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    examples = {}
    for row in [*manifest["training"], *manifest["evaluation"]]:
        example = encode_plain_qa(tokenizer, row["question"], row["answer"], max_sequence_length=MAX_TOKENS)
        if example.assistant_target_start + MAX_NEW_TOKENS > MAX_TOKENS:
            raise ValueError("generation context exceeds frozen token cap")
        examples[row["id"]] = example
    return examples


def evaluate(model: Any, tokenizer: Any, examples: Mapping[str, Any], manifest: Mapping[str, Any],
             torch: Any, device: Any, monitor: Any) -> list[dict[str, Any]]:
    """Sequential answer NLL + greedy generation from question ONLY, no KV cache."""
    from src.graddiff_integration import collate_causal_examples
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("evaluation requires frozen eval model and no gradients")
    rows = []
    with torch.inference_mode():
        for row in manifest["evaluation"]:
            monitor.stage = "evaluation:" + row["id"]
            monitor.boundary("before_" + monitor.stage)
            example = examples[row["id"]]
            batch = collate_causal_examples([example], tokenizer, device=device, max_sequence_length=MAX_TOKENS)
            nll = float(checked_loss(model, batch, torch).item())
            ids = torch.tensor([example.input_ids[:example.assistant_target_start]], dtype=torch.long, device=device)
            generated = []
            for _ in range(MAX_NEW_TOKENS):
                monitor.check()
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               return_dict=True, output_attentions=False, output_hidden_states=False)
                logits = output.logits
                if (logits.device.type != "mps" or logits.dtype != torch.float32
                        or not bool(torch.isfinite(logits).all().item())
                        or getattr(output, "past_key_values", None) is not None):
                    raise RuntimeError("generation device/precision/finiteness/cache mismatch")
                token = int(logits[0, -1].argmax().item())
                del output, logits
                if token == tokenizer.eos_token_id:
                    break
                generated.append(token)
                ids = torch.cat([ids, torch.tensor([[token]], dtype=torch.long, device=device)], dim=1)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            normalized = normalize_answer(text)
            aliases = row["target_aliases"]
            recovery = (any(re.search(r"\b" + re.escape(normalize_answer(a)) + r"\b", normalized)
                            for a in aliases) if aliases else None)
            rows.append({"id": row["id"], "answer_nll": nll, "generation": text,
                         "normalized_exact_match": normalized == normalize_answer(row["answer"]),
                         "alias_recovery": recovery, "generated_tokens": len(generated)})
            monitor.boundary("after_" + monitor.stage)
    return rows


def validate_baseline(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected = {"status": "baseline_complete_no_updates", "mode": "baseline", "model_id": MODEL_ID,
                "revision": REVISION, "manifest_file_sha256": MANIFEST_SHA256,
                "downloads_enabled": False, "optimizer_created": False, "optimizer_steps": 0,
                "step_attempted": False, "checkpoint_writes": 0, "base_checkpoint_unchanged": True,
                "parameters_unchanged": True, "parameter_versions_unchanged": True,
                "actual_device": "mps", "precision": "float32", "model_loaded": True,
                "training_authorized": False, "seed": 2026, "deterministic_algorithms": True,
                "runner_sha256": sha256_file(Path(__file__))}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"incomplete/mismatched baseline: {key}")
    if (report["base_parameter_sha256"] != report["parameter_sha256_after"]
            or not re.fullmatch(r"[0-9a-f]{64}", report["base_parameter_sha256"])):
        raise ValueError("baseline parameter integrity missing")
    if report["checkpoint"] != telemetry.ensure_checkpoint(json.loads(PROFILE.read_text()), allow_download=False):
        raise ValueError("baseline checkpoint provenance mismatch")
    rows = report["evaluation"]
    if [r["id"] for r in rows] != [r["id"] for r in manifest["evaluation"]]:
        raise ValueError("baseline must cover every frozen evaluation case in order")
    for row, case in zip(rows, manifest["evaluation"]):
        if (type(row["answer_nll"]) not in (float, int) or not math.isfinite(row["answer_nll"])
                or row["answer_nll"] < 0 or not isinstance(row["generation"], str)
                or type(row["normalized_exact_match"]) is not bool
                or row["normalized_exact_match"] != (normalize_answer(row["generation"]) == normalize_answer(case["answer"]))
                or type(row["generated_tokens"]) is not int or not 0 <= row["generated_tokens"] <= MAX_NEW_TOKENS):
            raise ValueError("invalid baseline metric")
        alias = (any(re.search(r"\b" + re.escape(normalize_answer(a)) + r"\b", normalize_answer(row["generation"]))
                     for a in case["target_aliases"]) if case["target_aliases"] else None)
        if type(row["alias_recovery"]) is not type(alias) or row["alias_recovery"] != alias:
            raise ValueError("baseline recovery metric mismatch")
    if not report["measurements"] or report["peaks"]["sample_count"] != len(report["measurements"]):
        raise ValueError("missing baseline memory evidence")
    if report["memory_policy"]["guard_id"] != GUARD_ID:
        raise ValueError("baseline used another memory guard")
    if not re.fullmatch(r"[0-9a-f]{64}", report["encoding_sha256"]):
        raise ValueError("missing baseline encoding provenance")
    previous = None
    for sample in report["measurements"]:
        telemetry._validate_reading(sample)
        if (sample["swap_used_bytes"] > report["measurements"][0]["swap_used_bytes"]
                or sample["pageouts"] != report["measurements"][0]["pageouts"]
                or sample["process_peak_rss_bytes"] > 8 * GIB
                or sample["mps_driver_allocated_bytes"] > 8 * GIB
                or (previous is not None and sample["timestamp_ns"] <= previous["timestamp_ns"])):
            raise ValueError("unsafe baseline memory evidence")
        previous = sample
    if not {"baseline", "before_load", "after_load", "complete"} <= {s["stage"] for s in report["measurements"]}:
        raise ValueError("incomplete baseline memory boundaries")
    json.dumps(report, allow_nan=False)


def validation_plan() -> dict[str, Any]:
    manifest = frozen_manifest()
    checkpoint = telemetry.ensure_checkpoint(json.loads(PROFILE.read_text()), allow_download=False)
    return {"status": "validation_only", "model_loaded": False, "optimizer_created": False,
            "optimizer_steps": 0, "checkpoint_writes": 0, "downloads_enabled": False,
            "training_authorized": False, "manifest_file_sha256": MANIFEST_SHA256,
            "training_cases": len(manifest["training"]), "evaluation_cases": len(manifest["evaluation"]),
            "pilot_case_id": manifest["epoch_zero_order"][0], "checkpoint": checkpoint,
            "memory_policy": memory_policy(json.loads(PROFILE.read_text())),
            "blockers": ["owner-run immutable baseline evaluation not performed in Task38",
                         "fresh M4 Pro MPS admission required", "explicit one-step authorization required"]}


def run(*, mode: str, report_path: Path, run_path: Path | None = None,
        baseline_path: Path | None = None, baseline_sha256: str | None = None,
        enabled: bool = False, authorization: str | None = None) -> dict[str, Any]:
    """Real execution ONLY from an explicitly invoked normal Terminal command."""
    if mode not in ("baseline", "pilot"):
        raise ValueError("unsupported mode")
    if mode == "pilot":
        require_authorization(enabled, authorization)  # before imports, files, load
    elif enabled or authorization is not None or run_path is not None or baseline_path is not None or baseline_sha256 is not None:
        raise ValueError("baseline cannot authorize updates or checkpoints")
    path = artifact_path(report_path, new=True)
    manifest = frozen_manifest()
    profile = json.loads(PROFILE.read_text())
    baseline = None
    destination = None
    if mode == "pilot":
        if baseline_path is None or not re.fullmatch(r"[0-9a-f]{64}", baseline_sha256 or "") or run_path is None:
            raise ValueError("pilot requires sealed baseline, its explicit SHA-256, and fresh run path")
        baseline_path = artifact_path(baseline_path, new=False)
        if sha256_file(baseline_path) != baseline_sha256:
            raise ValueError("baseline SHA-256 mismatch")
        baseline = json.loads(baseline_path.read_text())
        validate_baseline(baseline, manifest)
        destination = new_run_path(run_path)
    if not sys.stdin.isatty():
        raise RuntimeError("real execution requires an owner-run normal Terminal")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback must be disabled before Torch import")
    for name in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                 "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL"):
        if name in os.environ:
            raise RuntimeError(f"remove allocator/math override {name} before Torch import")
    if any(os.environ.get(name, "1") != "1" for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise RuntimeError("single-process pilot required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1",
                      HF_HOME=str(ROOT / ".cache/smollm2_hub"),
                      HF_HUB_CACHE=str(ROOT / ".cache/smollm2_hub/hub"),
                      HF_XET_CACHE=str(ROOT / ".cache/smollm2_hub/xet"))
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        telemetry._safe_path(Path(os.environ[name]), ROOT / ".cache")
    import torch
    import transformers
    import huggingface_hub
    for name, module in (("torch", torch), ("transformers", transformers), ("huggingface_hub", huggingface_hub)):
        if module.__version__ != telemetry.VERSIONS[name]:
            raise RuntimeError(f"requires pinned {name}=={telemetry.VERSIONS[name]}")
    device = telemetry.validate_environment(torch)
    if telemetry._command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip() != "Apple M4 Pro":
        raise RuntimeError("Task38 policy is limited to Apple M4 Pro")
    monitor = PilotMonitor(torch, profile=profile, device=device.type)
    report: dict[str, Any] = {
        "schema_version": "1.0", "status": "STOP", "mode": mode, "model_id": MODEL_ID,
        "revision": REVISION, "manifest_file_sha256": MANIFEST_SHA256, "precision": "float32",
        "downloads_enabled": False, "optimizer_created": False, "optimizer_steps": 0,
        "step_attempted": False, "checkpoint_writes": 0, "base_checkpoint_unchanged": False,
        "model_loaded": False, "actual_device": None, "training_authorized": mode == "pilot",
        "memory_policy": memory_policy(profile), "baseline_file_sha256": baseline_sha256,
        "environment": {**telemetry.VERSIONS, "os": platform.mac_ver()[0], "machine": platform.machine()},
        "runner_sha256": sha256_file(Path(__file__)), "seed": 2026, "deterministic_algorithms": True,
    }
    model = None
    checkpoint_dir = None
    report_stream = None
    reserved_destination = False
    try:
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        torch.mps.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        monitor.capture("baseline", before_load=True)
        report["checkpoint"] = telemetry.ensure_checkpoint(profile, allow_download=False)
        config = transformers.AutoConfig.from_pretrained(telemetry.CHECKPOINT, local_files_only=True, trust_remote_code=False)
        telemetry.validate_config(config.to_dict(), profile)
        config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(telemetry.CHECKPOINT, local_files_only=True,
                                                              trust_remote_code=False, use_fast=True)
        examples = prepare_examples(tokenizer, manifest)
        report["encoding_sha256"] = digest({k: [e.input_ids, e.labels] for k, e in examples.items()})
        if baseline is not None and report["encoding_sha256"] != baseline["encoding_sha256"]:
            raise ValueError("baseline encoding drift")
        sample = monitor.capture("before_load", before_load=True)
        report["memory_policy"] = memory_policy(profile, sample["process_peak_rss_bytes"])
        if shutil.disk_usage(ROOT).free < 4 * GIB:
            raise RuntimeError("requires 4 GiB free disk before model load")
        # Reserve names exclusively before load. Failed directories remain tombstones.
        RUNS.mkdir(exist_ok=True)
        if destination is not None:
            destination.mkdir(exist_ok=False)
            reserved_destination = True
            checkpoint_dir = destination / "checkpoint"
        report_stream = path.open("x", encoding="utf-8")
        monitor.stage = "loading_cpu"
        monitor.start()
        model, info = transformers.AutoModelForCausalLM.from_pretrained(
            telemetry.CHECKPOINT, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        report["model_loaded"] = True
        if not isinstance(info, dict) or any(info.get(k) != [] for k in
                                           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("incompatible checkpoint load")
        model.requires_grad_(False)
        model.eval()
        telemetry.validate_model(model, torch, profile, device="cpu")
        monitor.boundary("after_cpu_load")
        model.to(device)
        report["actual_device"] = device.type
        telemetry.validate_model(model, torch, profile, device="mps")
        monitor.boundary("after_load")
        versions = tuple(p._version for p in model.parameters())
        before = parameter_fingerprint(model)
        report["base_parameter_sha256"] = before
        if baseline is not None:
            if before != baseline["base_parameter_sha256"]:
                raise ValueError("loaded model does not match immutable baseline")
            if sha256_file(baseline_path) != baseline_sha256:
                raise ValueError("baseline changed before training")
            if telemetry.ensure_checkpoint(profile, allow_download=False) != report["checkpoint"]:
                raise ValueError("base checkpoint changed before training")
            model.requires_grad_(True)
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
            from src.graddiff_integration import collate_causal_examples
            case_id = manifest["epoch_zero_order"][0]
            report["training_case_id"] = case_id
            batch = collate_causal_examples([examples[case_id]], tokenizer, device=device, max_sequence_length=MAX_TOKENS)
            one_step(model, batch, torch, monitor, profile, report, enabled=enabled, authorization=authorization)
            del batch
            report["parameter_sha256_after"] = parameter_fingerprint(model)
            if before == report["parameter_sha256_after"] or versions == tuple(p._version for p in model.parameters()):
                raise RuntimeError("authorized step did not change weights and versions")
            report["weight_update_verified"] = True
            model.requires_grad_(False)
            model.eval()
            model.gradient_checkpointing_disable()
        report["evaluation"] = evaluate(model, tokenizer, examples, manifest, torch, device, monitor)
        after_eval = parameter_fingerprint(model)
        if baseline is None:
            report["parameter_sha256_after"] = after_eval
            report["parameters_unchanged"] = before == after_eval
            report["parameter_versions_unchanged"] = versions == tuple(p._version for p in model.parameters())
            if not report["parameters_unchanged"] or not report["parameter_versions_unchanged"]:
                raise RuntimeError("baseline evaluation changed weights")
        elif after_eval != report["parameter_sha256_after"]:
            raise RuntimeError("post-step evaluation mutated weights")
        if telemetry.ensure_checkpoint(profile, allow_download=False) != report["checkpoint"]:
            raise RuntimeError("original checkpoint changed")
        report["base_checkpoint_unchanged"] = True
        monitor.boundary("before_serialization")
        if checkpoint_dir is not None:
            require_authorization(enabled, authorization)
            if (report["optimizer_steps"] != 1 or not report["weight_update_verified"]
                    or sha256_file(baseline_path) != baseline_sha256 or frozen_manifest() != manifest):
                raise RuntimeError("checkpoint prerequisites failed")
            # No arbitrary destination and no overwrite/resume. Never save to base cache.
            telemetry._safe_path(checkpoint_dir, RUNS)
            checkpoint_dir.mkdir(exist_ok=False)
            report["checkpoint_write_attempted"] = True
            model.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="1GB")
            report["checkpoint_writes"] = 1
            monitor.boundary("after_serialization")
            report["saved_files_sha256"] = checkpoint_hashes(checkpoint_dir)
            if telemetry.ensure_checkpoint(profile, allow_download=False) != report["checkpoint"]:
                raise RuntimeError("original checkpoint changed after serialization")
        monitor.boundary("complete")
        report["status"] = "baseline_complete_no_updates" if baseline is None else "one_step_pilot_complete_not_longer_run_clearance"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "STOP"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["partial_in_memory_update_possible"] = report["step_attempted"]
    finally:
        try:
            if model is not None:
                model.zero_grad(set_to_none=True)
            monitor.close()
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "STOP"
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        report["measurements"] = monitor.samples
        report["peaks"] = telemetry.summarize_samples(monitor.samples)
    if mode == "baseline" and report["status"] == "baseline_complete_no_updates":
        try:
            validate_baseline(report, manifest)
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "STOP"
            report["error"] = f"baseline validation: {type(exc).__name__}: {exc}"
    # Write only our exclusively reserved report, never reopen a caller's old result.
    if report_stream is not None:
        with report_stream as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
    if reserved_destination:
        seal_json(destination / "COMPLETE.json" if report["status"].startswith("one_step") else destination / "FAILED.json",
                  {"status": report["status"], "report_sha256": sha256_file(path) if report_stream is not None else None,
                   "checkpoint_usable": report["status"].startswith("one_step")})
        if report["status"].startswith("one_step"):
            for saved in checkpoint_dir.iterdir():
                if saved.is_file():
                    saved.chmod(0o444)
            checkpoint_dir.chmod(0o555)
            destination.chmod(0o555)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run-baseline", action="store_true")
    modes.add_argument("--run-one-step", action="store_true")
    parser.add_argument("--authorize-weight-update")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.run_baseline or args.run_one_step:
            if args.report is None:
                raise ValueError("real execution requires a fresh --report path")
            report = run(mode="pilot" if args.run_one_step else "baseline", report_path=args.report,
                         run_path=args.run_dir, baseline_path=args.baseline, baseline_sha256=args.baseline_sha256,
                         enabled=args.run_one_step, authorization=args.authorize_weight_update)
        else:
            if any(v is not None for v in (args.report, args.baseline, args.baseline_sha256, args.run_dir, args.authorize_weight_update)):
                raise ValueError("execution arguments require explicit baseline or one-step mode")
            report = validation_plan()
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if report["status"] == "STOP" else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f"STOP: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
