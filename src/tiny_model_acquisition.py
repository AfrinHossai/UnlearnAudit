"""Frozen CPU synthetic acquisition. Default: weight-free validation, never training.

--dry-run uses random four-token fixtures, backward only, no optimizer or files.
Real acquisition requires two explicit opt-ins and a fresh normal Mac Terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

from src import small_model_preflight as telemetry
from src.small_model_acquisition import require_ignored, seal_json
from src.small_model_backward import parameter_fingerprint, validate_gradients
from src.small_model_experiment import ROOT, acquisition_order, digest, encode_plain_qa
from src.inspect_tofu import sha256_file

SPEC = ROOT / "experiments/tiny_cpu_acquisition.json"
SPEC_SHA256 = "6f15549257be0a59105d9a69e3f3ac92a5a8cb0cf8bfc6015388cef765252ad2"
RUNS = ROOT / "outputs/local_synthetic_unlearning"
AUTHORIZATION = "tiny-90496:cpu:acquire-eight:max-2000"
GIB = 2**30
PARAMETERS = 90496
MAX_STEPS = 2000


class ByteTokenizer:
    eos_token_id = pad_token_id = 256
    all_special_ids = [256]
    padding_side = "right"

    def __len__(self):
        return 257

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("no implicit BOS or special tokens")
        return list(text.encode("utf-8"))


def frozen_inputs():
    telemetry._safe_path(SPEC, ROOT / "experiments")
    if sha256_file(SPEC) != SPEC_SHA256:
        raise ValueError("frozen tiny profile changed")
    spec = json.loads(SPEC.read_text())
    source = telemetry._safe_path(ROOT / spec["source"], ROOT / "experiments")
    if sha256_file(source) != spec["source_sha256"]:
        raise ValueError("frozen fact source changed")
    facts = {r["id"]: r for r in json.loads(source.read_text())["facts"]}
    train = [facts[k] for k in spec["training_ids"]]
    unseen = [facts[k] for k in spec["untrained_ids"]]
    if (len(train) != 8 or sum(r["role"] == "forget" for r in train) != 2
            or len(unseen) != 2 or set(spec["training_ids"]) & set(spec["untrained_ids"])):
        raise ValueError("fact roles/counts/overlap changed")
    rows = []
    for fact in train + unseen:
        answer = fact["target_aliases"][0]
        if not re.fullmatch("[a-z]{12}", answer):
            raise ValueError("expected twelve-byte public synthetic target")
        for kind, field in (("direct", "question"), ("paraphrase", "paraphrased_question")):
            question = fact[field]
            if any(f["target_aliases"][0] in question for f in train + unseen):
                raise ValueError("target leaked into evaluation prompt")
            example = encode_plain_qa(ByteTokenizer(), question, answer, max_sequence_length=128)
            if example.assistant_target_start + 16 > 128:
                raise ValueError("generation exceeds context cap")
            rows.append({"id": fact["id"] + ":" + kind, "fact_id": fact["id"],
                         "kind": kind if fact in train else "untrained_" + kind,
                         "role": fact["role"], "question": question, "answer": answer})
    if len({r["question"] for r in rows}) != 20:
        raise ValueError("evaluation wording overlap")
    return spec, rows


def memory_policy(peak=0):
    if type(peak) is not int or peak < 0:
        raise ValueError("invalid process RSS")
    incremental = 10 * PARAMETERS * 4 + GIB + max(2 * GIB, peak) + 2 * GIB
    return {"guard_id": "tiny_random_90496_fp32_cpu_acquisition_v1", "device": "cpu",
            "estimated_incremental_bytes": incremental,
            "preload_available_required_bytes": ((incremental + GIB - 1) // GIB + 6) * GIB,
            "continuing_available_required_bytes": 6 * GIB, "minimum_host_bytes": 48 * GIB,
            "max_process_peak_rss_bytes": 8 * GIB, "measured_training_fit": False,
            "mps_counters": "inapplicable; no Metal execution"}


def read_sample(stage):
    # Reuse existing host commands/parsers; never initialize or query MPS on CPU.
    start = time.monotonic_ns()
    vm = telemetry._command(["/usr/bin/vm_stat"])
    pressure = telemetry._command(["/usr/bin/memory_pressure", "-Q"])
    fields = telemetry._command(["/usr/sbin/sysctl", "-n", "hw.memsize",
                                 "kern.memorystatus_vm_pressure_level", "vm.swapusage"]).strip().splitlines()
    percent = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    pageouts = re.search(r"^Pageouts:\s*(\d+)\.?$", vm, re.MULTILINE)
    swap = re.search(r"used\s*=\s*([0-9.]+)M", fields[2] if len(fields) == 3 else "")
    if len(fields) != 3 or not percent or not pageouts or not swap:
        raise RuntimeError("missing host telemetry")
    sample = {"stage": stage, "timestamp_ns": start,
              "total_physical_memory_bytes": int(fields[0]), "macos_pressure_level": int(fields[1]),
              "available_physical_memory_lower_bound_bytes": telemetry.parse_vm_stat(vm),
              "system_memory_free_percentage": int(percent.group(1)),
              "pageouts": int(pageouts.group(1)), "swap_used_bytes": int(float(swap.group(1)) * 1024**2),
              "process_current_rss_bytes": int(telemetry._command(
                  ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) * 1024,
              "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              "mps_tensor_allocated_bytes": None, "mps_driver_allocated_bytes": None}
    sample["sample_duration_ns"] = time.monotonic_ns() - start
    return sample


def validate_sample(s, *, previous=None, first=None, active=False, before_load=False):
    fields = [k for k in telemetry.INTEGER_FIELDS if not k.startswith("mps_")]
    if any(type(s.get(k)) is not int or s[k] < 0 for k in fields):
        raise RuntimeError("missing/invalid CPU telemetry")
    now = time.monotonic_ns()
    if (not s.get("stage") or not 0 < s["sample_duration_ns"] <= 2_000_000_000
            or not 0 < s["timestamp_ns"] < s["timestamp_ns"] + s["sample_duration_ns"] <= now
            or now - s["timestamp_ns"] > 2_000_000_000):
        raise RuntimeError("stale/future telemetry")
    policy = memory_policy(s["process_peak_rss_bytes"])
    available, total = s["available_physical_memory_lower_bound_bytes"], s["total_physical_memory_bytes"]
    if (total < 48 * GIB or not 6 * GIB <= available <= total
            or s["macos_pressure_level"] != 1 or not 20 <= s["system_memory_free_percentage"] <= 100
            or not 0 < s["process_current_rss_bytes"] <= s["process_peak_rss_bytes"] <= 8 * GIB
            or s.get("mps_tensor_allocated_bytes") is not None
            or s.get("mps_driver_allocated_bytes") is not None):
        raise RuntimeError("CPU physical memory/pressure/RSS guard failed")
    if before_load and available < policy["preload_available_required_bytes"]:
        raise RuntimeError("tiny CPU dynamic preload guard failed (minimum 12 GiB)")
    if previous is not None:
        gap = s["timestamp_ns"] - previous["timestamp_ns"]
        if (gap <= 0 or (active and gap > 2_000_000_000)
                or s["process_peak_rss_bytes"] < previous["process_peak_rss_bytes"]):
            raise RuntimeError("unordered telemetry/RSS or sampling gap")
    if first is not None and (total != first["total_physical_memory_bytes"]
            or s["swap_used_bytes"] > first["swap_used_bytes"] or s["pageouts"] != first["pageouts"]):
        raise RuntimeError("capacity changed or cumulative swap/page-out growth")


class CPUMonitor(telemetry.Monitor):
    """Reuse watchdog start/close, with independent CPU collection and admission."""
    def __init__(self):
        self.samples, self.error, self.thread = [], None, None
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.stage, self.started = "preload", time.monotonic()

    def check(self):
        super().check()
        if time.monotonic() - self.started > 1800:
            raise RuntimeError("30-minute time budget exceeded")
        if self.thread is not None and self.samples and time.monotonic_ns() - self.samples[-1]["timestamp_ns"] > 2_000_000_000:
            raise RuntimeError("watchdog observation stale")

    def capture(self, stage, *, before_load=False):
        with self.lock:
            self.check()
            s = read_sample(stage)
            previous = self.samples[-1] if self.samples else None
            self.samples.append(s)  # retain failing observations
            validate_sample(s, previous=previous, first=self.samples[0],
                            active=self.thread is not None, before_load=before_load)
            return s

    def boundary(self, stage):
        self.stage = stage
        self.capture(stage)

    def _loop(self):
        import _thread
        try:
            while not self.stop_event.wait(0.25):
                self.capture(self.stage)
        except Exception as exc:
            self.error = exc
            _thread.interrupt_main()


def environment():
    if not sys.stdin.isatty() or sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("real execution requires a normal Apple Silicon Mac Terminal")
    if "Apple M4 Pro" not in telemetry._command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]):
        raise RuntimeError("requires reviewed Apple M4 Pro host")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback override forbidden")
    if any(k in os.environ for k in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                                     "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")):
        raise RuntimeError("allocator/math overrides forbidden")
    if any(os.environ.get(k, "1") != "1" for k in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise RuntimeError("single process required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="0")


def new_run_path(path):
    path = telemetry._safe_path(Path(path), RUNS)
    if path.parent != RUNS or not re.fullmatch(r"acquisition-[A-Za-z0-9_-]+", path.name) or path.exists():
        raise ValueError("requires a unique new acquisition-* directory in tiny outputs; no resume")
    require_ignored(path)
    return path


def authorize(enabled, authorization):
    if enabled is not True or authorization != AUTHORIZATION:
        raise ValueError("requires --run-acquisition AND exact --authorize-weight-update token")


def initialize(spec):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    # torch.manual_seed also seeds MPS/CUDA; this experiment touches CPU RNG only.
    torch.random.default_generator.manual_seed(spec["seed"])
    # Explicit CPU context also overrides caller-selected default device/dtype.
    with torch.device("cpu"):
        model = LlamaForCausalLM(LlamaConfig(**spec["model"])).to(dtype=torch.float32)
    validate_model(model, spec)
    return model


def validate_model(model, spec):
    import torch
    if (type(model).__name__ != "LlamaForCausalLM" or sum(p.numel() for p in model.parameters()) != PARAMETERS
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or any(not bool(torch.isfinite(p).all()) for p in model.parameters())
            or model.get_input_embeddings().weight is not model.get_output_embeddings().weight
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("tiny model count/ties/CPU/FP32/determinism mismatch")
    for key, expected in spec["model"].items():
        actual = getattr(model.config, "_attn_implementation" if key == "attn_implementation" else key)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError("model configuration mismatch: " + key)


def examples(rows):
    return {r["id"]: encode_plain_qa(ByteTokenizer(), r["question"], r["answer"], max_sequence_length=128)
            for r in rows}


def batch_for(example):
    from src.graddiff_integration import collate_causal_examples
    return collate_causal_examples([example], ByteTokenizer(), device="cpu", max_sequence_length=128)


def checked_forward(model, batch):
    import torch
    from src.graddiff import causal_lm_nll
    if (set(batch) != {"input_ids", "attention_mask", "labels"}
            or any(t.device.type != "cpu" or t.dtype != torch.long for t in batch.values())
            or batch["input_ids"].ndim != 2 or batch["input_ids"].shape[0] != 1
            or not 2 <= batch["input_ids"].shape[1] <= 128):
        raise RuntimeError("CPU batch/sequence mismatch")
    output = model(**batch, use_cache=False, return_dict=True)
    if (output.loss.ndim != 0 or output.loss.device.type != "cpu" or output.loss.dtype != torch.float32
            or output.logits.device.type != "cpu" or output.logits.dtype != torch.float32
            or tuple(output.logits.shape) != (*batch["input_ids"].shape, 257)
            or getattr(output, "past_key_values", None) is not None):
        raise RuntimeError("logits/cache mismatch")
    local = causal_lm_nll(output.logits, batch["labels"])
    if (not bool(torch.isfinite(output.loss)) or not math.isclose(float(local.detach()),
            float(output.loss.detach()), rel_tol=1e-5, abs_tol=1e-5)):
        raise FloatingPointError("loss nonfinite/parity failure")
    return output


def recovery_metrics(generated, finish, target):
    prediction = bytes(generated).decode("utf-8", errors="replace")
    return {"generated_ids": generated + ([256] if finish == "eos" else []),
            "prediction": prediction, "finish_reason": finish,
            "valid_answer": bool(re.fullmatch("[a-z]{12}", prediction)) and finish == "eos",
            "exact_recovery": generated == list(target.encode()) and finish == "eos"}


def evaluate(model, rows, encoded, monitor):
    import torch
    from src.graddiff import causal_lm_nll
    model.eval().requires_grad_(False)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("evaluation has preexisting gradients")
    before = parameter_fingerprint(model)
    versions = tuple(p._version for p in model.parameters())
    results = []
    with torch.inference_mode():
        for row in rows:
            monitor.boundary("evaluation:" + row["id"])
            example = encoded[row["id"]]
            batch = batch_for(example)
            output = checked_forward(model, batch)
            byte_labels = batch["labels"].clone()
            byte_labels[:, -1] = -100
            byte_nll = float(causal_lm_nll(output.logits, byte_labels))
            ids = torch.tensor([example.input_ids[:example.assistant_target_start]], dtype=torch.long)
            generated, finish = [], "length"
            for _ in range(16):
                monitor.check()
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                if not bool(torch.isfinite(out.logits).all()) or getattr(out, "past_key_values", None) is not None:
                    raise FloatingPointError("nonfinite generation/cache")
                token = int(out.logits[0, -1].argmax())
                if token == 256:
                    finish = "eos"
                    break
                generated.append(token)
                ids = torch.cat((ids, torch.tensor([[token]], dtype=torch.long)), dim=1)
            result = {"id": row["id"], "kind": row["kind"], "role": row["role"],
                      "answer_nll_including_eos": float(output.loss), "answer_byte_nll": byte_nll,
                      "eos_nll": float(torch.nn.functional.cross_entropy(
                          output.logits[:, -2, :], batch["labels"][:, -1])),
                      **recovery_metrics(generated, finish, row["answer"])}
            # Fixed wrong-binding control, same attribute and a different trained entity.
            if row["kind"] == "direct":
                directs = [r for r in rows if r["kind"] == "direct"]
                wrong = directs[(directs.index(row) + 2) % 8]["answer"]
                wrong_example = encode_plain_qa(ByteTokenizer(), row["question"], wrong, max_sequence_length=128)
                wrong_batch = batch_for(wrong_example)
                wrong_output = checked_forward(model, wrong_batch)
                wrong_batch["labels"][:, -1] = -100
                result["wrong_binding_answer_byte_nll"] = float(causal_lm_nll(wrong_output.logits, wrong_batch["labels"]))
                result["correct_over_wrong_nll_margin"] = result["wrong_binding_answer_byte_nll"] - byte_nll
            results.append(result)
    if before != parameter_fingerprint(model) or versions != tuple(p._version for p in model.parameters()):
        raise RuntimeError("evaluation mutated parameters")
    monitor.boundary("evaluation_complete")
    return results


def scores(rows):
    result = {kind: {"recovered": sum(r["exact_recovery"] for r in rows if r["kind"] == kind),
                   "total": sum(r["kind"] == kind for r in rows)}
            for kind in ("direct", "paraphrase", "untrained_direct", "untrained_paraphrase")}
    for role in ("forget", "retain"):
        group = [r for r in rows if r["kind"] == "direct" and r["role"] == role]
        result["direct_" + role] = {"recovered": sum(r["exact_recovery"] for r in group), "total": len(group)}
    margins = [r["correct_over_wrong_nll_margin"] for r in rows if "correct_over_wrong_nll_margin" in r]
    result["correct_over_wrong_binding"] = {"positive_margins": sum(m > 0 for m in margins), "total": len(margins)}
    return result


def acceptance(rows, baseline):
    direct = [r for r in rows if r["kind"] == "direct"]
    old = {r["id"]: r for r in baseline}
    recovered = len(direct) == 8 and all(r["exact_recovery"] for r in direct)
    gain = len(direct) == 8 and all(old[r["id"]]["answer_byte_nll"] - r["answer_byte_nll"] >= 1 for r in direct)
    return {"all_eight_recovered": recovered, "per_fact_nll_gate": gain, "acquisition_accepted": recovered and gain}


def parameter_hashes(model):
    return {name: hashlib.sha256(p.detach().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def checkpoint(model, run_path, step, evaluation, report, monitor):
    from safetensors.torch import save_model
    from src.small_model_lora import fsync_directory
    monitor.boundary("before_checkpoint")
    if shutil.disk_usage(ROOT).free < 4 * GIB:
        raise RuntimeError("checkpoint requires 4 GiB free disk")
    directory = run_path / f"checkpoint-{step:06d}"
    directory.mkdir()  # no exist_ok, no overwrite, no resume
    before = parameter_fingerprint(model)
    target = directory / "model.safetensors"
    save_model(model, str(target))
    with target.open("rb") as stream:
        os.fsync(stream.fileno())
    target.chmod(0o444)
    seal_json(directory / "config.json", model.config.to_dict())
    seal_json(directory / "metrics.json", {"step": step, "evaluation": evaluation,
                                         "parameter_sha256": before})
    if before != parameter_fingerprint(model):
        raise RuntimeError("checkpoint mutated parameters")
    record = {"directory": directory.name, "step": step, "parameter_sha256": before,
              "files_sha256": {p.name: sha256_file(p) for p in directory.iterdir()}}
    marker_hash = seal_json(directory / "COMPLETE.json", record)
    record["files_sha256"]["COMPLETE.json"] = marker_hash
    fsync_directory(directory)
    fsync_directory(run_path)
    report["checkpoints"].append(record)
    monitor.boundary("after_checkpoint")
    return target


def projection(monitor, report, step_seconds, evaluation_seconds, checkpoint_seconds):
    remaining = MAX_STEPS - report["optimizer_steps"]
    future_checks = math.ceil(remaining / 100)
    estimate = (time.monotonic() - monitor.started + 2 * max(step_seconds) * remaining
                + 2 * future_checks * (evaluation_seconds + checkpoint_seconds) + 60)
    report["projected_total_seconds"] = estimate
    if estimate >= 1500:
        raise RuntimeError("conservative projected run exceeds 25-minute admission budget")


def validate_optimizer(optimizer, model):
    import torch
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise RuntimeError("requires single-group AdamW")
    group = optimizer.param_groups[0]
    expected = {"lr": 0.003, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
                "foreach": False, "fused": False, "maximize": False, "capturable": False,
                "differentiable": False, "amsgrad": False}
    if (any(type(group.get(k)) is not type(v) or group[k] != v for k, v in expected.items())
            or [id(p) for p in group["params"]] != [id(p) for p in model.parameters()]):
        raise RuntimeError("optimizer configuration/membership drift")


def train_step(model, batch, optimizer, monitor, report, *, enabled, authorization):
    import torch
    authorize(enabled, authorization)
    steps, attempts = report["optimizer_steps"], report["optimizer_step_attempts"]
    if type(steps) is not int or type(attempts) is not int or not 0 <= steps == attempts < MAX_STEPS:
        raise RuntimeError("optimizer budget exhausted or partial attempt; cannot retry")
    if monitor.mode != "acquisition":
        raise RuntimeError("optimizer requires acquisition admission")
    validate_optimizer(optimizer, model)
    model.train().requires_grad_(True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("preexisting training gradients")
    try:
        monitor.boundary("before_training_forward")
        loss = checked_forward(model, batch).loss
        value = float(loss.detach())
        loss.backward()
        validate_gradients(model, torch, device="cpu")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False, error_if_nonfinite=True))
        validate_gradients(model, torch, device="cpu")
        clipped = float(torch.nn.utils.get_total_norm([p.grad for p in model.parameters()], foreach=False, error_if_nonfinite=True))
        if not math.isfinite(norm) or norm <= 0 or not 0 < clipped <= 1.00001:
            raise FloatingPointError("invalid gradient norm")
        authorize(enabled, authorization)
        validate_optimizer(optimizer, model)
        monitor.boundary("before_optimizer_step")
        report["optimizer_step_attempts"] += 1
        report["possible_partial_update"] = True
        optimizer.step()
        report["optimizer_steps"] += 1
        report["possible_partial_update"] = False
        monitor.boundary("after_optimizer_step")
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite parameter after update")
        for state in optimizer.state.values():
            if any(not bool(torch.isfinite(v).all()) for v in state.values() if isinstance(v, torch.Tensor)):
                raise FloatingPointError("nonfinite AdamW state")
        return {"step": report["optimizer_steps"], "training_answer_nll": value, "gradient_norm_before_clip": norm}
    finally:
        model.zero_grad(set_to_none=True)


def rehearse(model, monitor):
    import torch
    before = parameter_fingerprint(model)
    versions = tuple(p._version for p in model.parameters())
    # Max context fixture, never a target/paraphrase; gradients only, no optimizer.
    ids = torch.tensor([[1, 3, 4, 5] * 32], dtype=torch.long)
    labels = ids.clone()
    labels[:, :115] = -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    times = []
    model.train().requires_grad_(True)
    for _ in range(3):
        start = time.monotonic()
        try:
            monitor.boundary("rehearsal_before_forward")
            checked_forward(model, batch).loss.backward()
            validate_gradients(model, torch, device="cpu")
            monitor.boundary("rehearsal_after_backward")
            times.append(time.monotonic() - start)
        finally:
            model.zero_grad(set_to_none=True)
    if before != parameter_fingerprint(model) or versions != tuple(p._version for p in model.parameters()):
        raise RuntimeError("backward rehearsal changed weights")
    return times


def dry_run():
    import torch
    spec, _ = frozen_inputs()
    model = initialize(spec)
    before = parameter_fingerprint(model)
    ids = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    output = checked_forward(model, batch)
    output.loss.backward()
    gradients = validate_gradients(model, torch, device="cpu")
    model.zero_grad(set_to_none=True)
    if before != parameter_fingerprint(model):
        raise RuntimeError("dry-run mutated weights")
    return {"status": "synthetic_fixture_dry_run_not_acquisition", "parameter_count": PARAMETERS,
            "loss": float(output.loss.detach()), "gradients": gradients, "weights_unchanged": True,
            "optimizer_steps": 0, "checkpoint_writes": 0, "target_facts_executed": False,
            "memory_admission_measured": False, "training_authorized": False}


def run(*, enabled, authorization, run_path):
    authorize(enabled, authorization)  # before imports, paths, measurements or files
    spec, rows = frozen_inputs()
    run_path = new_run_path(run_path)
    environment()
    run_path.mkdir(parents=True)
    report = {"status": "RUNNING", "scope": spec["scope"], "device": "cpu", "dtype": "float32",
              "training_authorized": True, "optimizer_created": False, "optimizer_steps": 0,
              "optimizer_step_attempts": 0, "possible_partial_update": False, "parameter_count": PARAMETERS,
              "training": [], "checkpoints": [], "evaluations": [], "stop_reason": None,
              "all_eight_recovered": False, "acquisition_accepted": False, "unlearning_performed": False}
    seal_json(run_path / "RUNNING.json", {"spec_sha256": SPEC_SHA256, "max_steps": MAX_STEPS})
    monitor, model, initial_hashes = CPUMonitor(), None, None
    monitor.mode = "acquisition"
    try:
        from src.small_model_lora import dependency_versions
        report["versions"] = dependency_versions()
        report["python"] = platform.python_version()
        report["source_files_sha256"] = {str(p.relative_to(ROOT)): sha256_file(p)
                                           for p in [*sorted((ROOT / "src").glob("*.py")), SPEC,
                                                     ROOT / spec["source"], ROOT / "requirements-ml.lock"]}
        report["data_splits"] = {"training_ids": spec["training_ids"], "untrained_ids": spec["untrained_ids"],
                                 "paraphrases": "evaluation only", "direct": "training resubstitution"}
        report["spec"] = spec
        monitor.capture("preload", before_load=True)
        if shutil.disk_usage(ROOT).free < 4 * GIB:
            raise RuntimeError("requires 4 GiB free disk")
        import torch
        import transformers
        report["installed_llama_source_sha256"] = sha256_file(Path(transformers.__file__).parent / "models/llama/modeling_llama.py")
        monitor.capture("immediately_before_initialization", before_load=True)
        monitor.start()
        model = initialize(spec)
        monitor.boundary("initialized")
        report["memory_policy"] = memory_policy(monitor.samples[1]["process_peak_rss_bytes"])
        report["initial_parameter_sha256"] = parameter_fingerprint(model)
        initial_hashes = parameter_hashes(model)
        encoded = examples(rows)
        report["encoding_sha256"] = digest({k: [v.input_ids, v.labels] for k, v in encoded.items()})
        start = time.monotonic()
        baseline = evaluate(model, rows, encoded, monitor)
        evaluation_seconds = time.monotonic() - start
        report["baseline_file_sha256"] = seal_json(run_path / "baseline.json", {
            "parameter_sha256": report["initial_parameter_sha256"], "optimizer_created": False,
            "optimizer_steps": 0, "rows": baseline, "scores": scores(baseline)})
        start = time.monotonic()
        initial_checkpoint = checkpoint(model, run_path, 0, baseline, report, monitor)
        checkpoint_seconds = time.monotonic() - start
        from safetensors.torch import load_model
        load_model(model, str(initial_checkpoint), strict=True, device="cpu")
        if parameter_fingerprint(model) != report["initial_parameter_sha256"]:
            raise RuntimeError("M0 serialization/reload mismatch")
        start = time.monotonic()
        noop = evaluate(model, rows, encoded, monitor)
        evaluation_seconds = max(evaluation_seconds, time.monotonic() - start)
        if (any(r["exact_recovery"] for r in baseline if r["kind"] == "direct")
                or any(a["generated_ids"] != b["generated_ids"] or abs(a["answer_byte_nll"] - b["answer_byte_nll"]) > 1e-5
                       for a, b in zip(baseline, noop))):
            raise RuntimeError("initial target recovery or M0/no-op drift")
        report["noop_file_sha256"] = seal_json(run_path / "noop.json", {
            "rows": noop, "parameter_sha256": parameter_fingerprint(model)})
        times = rehearse(model, monitor)
        report["rehearsal_seconds"] = {"all": times, "median": statistics.median(times), "maximum": max(times)}
        projection(monitor, report, times, evaluation_seconds, checkpoint_seconds)
        optimizer = torch.optim.AdamW(model.parameters(), **{**spec["optimizer"], "betas": tuple(spec["optimizer"]["betas"])})
        report["optimizer_created"] = True
        training = {r["fact_id"]: r for r in rows if r["kind"] == "direct"}
        for epoch in range(250):
            for fact_id in acquisition_order(spec["training_ids"], epoch):
                validate_model(model, spec)
                start = time.monotonic()
                record = train_step(model, batch_for(encoded[training[fact_id]["id"]]), optimizer,
                                    monitor, report, enabled=enabled, authorization=authorization)
                times.append(time.monotonic() - start)
                report["training"].append({**record, "fact_id": fact_id, "epoch": epoch, "seconds": times[-1]})
                step = report["optimizer_steps"]
                if step == 1:
                    report["first_step_seconds"] = times[-1]
                    report["first_step_rss_high_water_bytes"] = monitor.samples[-1]["process_peak_rss_bytes"]
                    report["optimizer_state_tensor_bytes"] = sum(v.numel() * v.element_size() for state in optimizer.state.values()
                                                                 for v in state.values() if isinstance(v, torch.Tensor))
                    projection(monitor, report, times, evaluation_seconds, checkpoint_seconds)
                if step % 100 == 0:
                    start = time.monotonic()
                    result = evaluate(model, rows, encoded, monitor)
                    evaluation_seconds = max(evaluation_seconds, time.monotonic() - start)
                    report.update(acceptance(result, baseline))
                    report["evaluations"].append({"step": step, "scores": scores(result), "rows": result})
                    changed = [name for name, value in parameter_hashes(model).items() if initial_hashes[name] != value]
                    if not changed:
                        raise RuntimeError("no parameter bytes changed")
                    report["changed_parameter_names"] = changed
                    report["changed_parameter_tensor_count"] = len(changed)
                    start = time.monotonic()
                    checkpoint(model, run_path, step, result, report, monitor)
                    checkpoint_seconds = max(checkpoint_seconds, time.monotonic() - start)
                    print(json.dumps({"step": step, "scores": scores(result), "acquisition_accepted": report["acquisition_accepted"]}), flush=True)
                    if report["acquisition_accepted"]:
                        break
                    projection(monitor, report, times, evaluation_seconds, checkpoint_seconds)
            if report["acquisition_accepted"]:
                break
        report["stop_reason"] = "direct_acquisition_gate" if report["acquisition_accepted"] else "maximum_2000_steps"
        report["status"] = "acquisition_complete" if report["acquisition_accepted"] else "acquisition_not_achieved"
        report["final_parameter_sha256"] = parameter_fingerprint(model)
        frozen_inputs()
        for name in ("baseline", "noop"):
            if sha256_file(run_path / (name + ".json")) != report[name + "_file_sha256"]:
                raise RuntimeError("sealed baseline/no-op artifact changed")
        for name, expected in report["source_files_sha256"].items():
            if sha256_file(ROOT / name) != expected:
                raise RuntimeError("input/source changed during run")
        for cp in report["checkpoints"]:
            for name, expected in cp["files_sha256"].items():
                if sha256_file(run_path / cp["directory"] / name) != expected:
                    raise RuntimeError("checkpoint changed during run")
    except BaseException as exc:
        report.update(status="STOP", stop_reason=f"{type(exc).__name__}: {exc}", acquisition_accepted=False)
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
            report["final_parameter_sha256"] = parameter_fingerprint(model)
            if initial_hashes is not None:
                report["changed_parameter_names"] = [k for k, v in parameter_hashes(model).items() if v != initial_hashes[k]]
                report["changed_parameter_tensor_count"] = len(report["changed_parameter_names"])
        try:
            monitor.close()
        except BaseException as exc:
            report.update(status="STOP", stop_reason=f"cleanup/watchdog: {exc}", acquisition_accepted=False)
        report["measurements"] = monitor.samples
        report["checkpoint_writes"] = len(report["checkpoints"])
        report["elapsed_seconds"] = time.monotonic() - monitor.started
        report["observed_rss_high_water_bytes"] = max((s.get("process_peak_rss_bytes", 0) for s in monitor.samples), default=0)
        report_hash = seal_json(run_path / "report.json", report)
        seal_json(run_path / ("FAILED.json" if report["status"] == "STOP" else "COMPLETE.json"),
                  {"report_sha256": report_hash, "status": report["status"], "optimizer_steps": report["optimizer_steps"]})
        from src.small_model_lora import fsync_directory
        fsync_directory(run_path)
        fsync_directory(run_path.parent)
    return report


def plan():
    spec, rows = frozen_inputs()
    return {"status": "validation_only", "spec_sha256": SPEC_SHA256, "parameter_count": PARAMETERS,
            "training_ids": spec["training_ids"], "evaluation_cases": len(rows),
            "max_optimizer_steps": MAX_STEPS, "memory_policy": memory_policy(), "training_authorized": False,
            "model_loaded": False, "optimizer_steps": 0, "checkpoint_writes": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run-acquisition", action="store_true")
    parser.add_argument("--authorize-weight-update")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.run_acquisition:
        if args.run_dir is None:
            parser.error("--run-dir is required")
        result = run(enabled=True, authorization=args.authorize_weight_update, run_path=args.run_dir)
    else:
        if args.authorize_weight_update is not None or args.run_dir is not None:
            parser.error("update authorization/output path requires --run-acquisition")
        result = dry_run() if args.dry_run else plan()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["status"] in ("STOP", "acquisition_not_achieved") else 0


if __name__ == "__main__":
    raise SystemExit(main())
