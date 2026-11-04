"""Separate, opt-in CUDA stages. Importing this module is offline and ML-free.

No stage dispatches another stage. The local Mac/Llama admission paths are not
used or changed. See docs/smollm2_colab_workflow.md before any runtime action.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid

from src.evaluate_tofu_baseline import answer_metrics, normalize_answer
from src.inspect_tofu import sha256_file
from src.small_model_experiment import (
    MODEL_ID, REVISION, acquisition_order, build_manifest, digest, encode_plain_qa,
    load_sources, validate_profile,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/smollm2_135m_colab.json"
STAGES = ("environment", "data", "freeze", "load", "baseline", "acquire", "reference",
          "pre", "reference-eval", "noop", "retain-control", "retain-control-eval", "unlearn", "post")
EVAL_SETS = ("forget_direct", "retain_direct", "forget_paraphrase", "retain_paraphrase",
             "retain_unseen_records", "utility")
PARENTS = {"acquire": "load", "reference": "load", "noop": "acquire",
           "retain-control": "acquire", "unlearn": "acquire"}


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configuration():
    config = read_json(CONFIG)
    # Do not silently repurpose this prepared profile for another device/model.
    expected = {"schema_version": 1, "device": "cuda:0", "precision": "float32",
                "batch_size": 1, "gradient_accumulation_steps": 1, "seed": 2026,
                "max_sequence_length": 512, "optimizer": "SGD", "momentum": 0.0,
                "weight_decay": 0.0, "gamma": 1.0, "alpha": 1.0,
                "no_cpu_fallback": True, "requires_frozen_reviewed_protocol": True,
                "acquisition_learning_rate": None, "unlearning_learning_rate": None}
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"unsupported cloud config: {key}")
    profile = read_json(ROOT / config["source_profile"])
    validate_profile(profile)
    manifest = read_json(ROOT / config["manifest"])
    if manifest["profile_sha256"] != digest(profile):
        raise ValueError("source profile/manifest mismatch")
    return config, profile, manifest


def outside_git(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("persistent run directory must be outside the repository")
    # A different Git checkout is not a checkpoint destination either.
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("output destination is inside a Git checkout")
    return path


def inside_run(run, path):
    resolved = outside_git(path)
    if run.resolve() not in resolved.parents:
        raise ValueError("STOP: artifact/cache path escaped persistent run directory")
    return path


def environment(require_cuda=False):
    """No model, downloads, installation or weights. CUDA check uses tiny tensors."""
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {}, "cuda_ready": False, "billing_status": "not_detectable_by_python"}
    for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                 "numpy", "psutil", "packaging"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if not require_cuda:
        return report
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    config, _, _ = configuration()
    if sys.platform != "linux" or sys.version_info.major != 3 or sys.version_info.minor not in config["python_minor_versions"]:
        raise RuntimeError("STOP: separate CUDA profile supports Linux Python 3.11/3.12 only")
    expected = {"torch": "2.9.1+cu126"}
    for line in (ROOT / "requirements-colab.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            expected[name] = version
    for name, version in expected.items():
        if report["packages"].get(name) != version:
            raise RuntimeError(f"STOP: {name} must be {version}; use isolated Colab environment")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    import torch
    import transformers
    import psutil
    if torch.version.cuda != "12.6" or not torch.cuda.is_available():
        raise RuntimeError("STOP: CUDA 12.6 wheel and a visible compatible GPU are required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("STOP: this profile requires exactly one visible GPU")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    if int(driver.split(".")[0]) < 525:
        raise RuntimeError("STOP: driver is below CUDA 12.x minor-compatibility minimum")
    # nvidia-smi's CUDA label is a driver capability, not the installed Torch runtime.
    torch.ones((2, 2), device="cuda").matmul(torch.ones((2, 2), device="cuda"))
    torch.cuda.synchronize()
    # Import the actual architecture; no constructor or pretrained loader.
    from transformers import LlamaForCausalLM
    report.update(cuda_ready=True, torch_cuda=torch.version.cuda, driver=driver,
                  gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                  host_total_gib=psutil.virtual_memory().total / 2**30,
                  architecture=LlamaForCausalLM.__name__, transformers=transformers.__version__)
    return report


def memory_check(config, run, before_load=False):
    import psutil
    import torch
    free, total = torch.cuda.mem_get_info(0)
    host = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage(run).free / 2**30
    phase = "before_load" if before_load else "continuing"
    if (host < config[f"{phase}_host_available_gib"]
            or free / 2**30 < config[f"{phase}_gpu_free_gib"]
            or disk < config["minimum_disk_free_gib"]):
        raise RuntimeError("STOP: host/GPU/disk reserve violated; do not lower guards to continue")
    return {"host_available_gib": host, "gpu_free_gib": free / 2**30,
            "gpu_total_gib": total / 2**30, "disk_free_gib": disk,
            "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30}


def source_identity(config, profile, manifest):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True):
        raise RuntimeError("STOP: tracked workflow files must match the recorded Git commit")
    return {"git_commit": commit, "config_sha256": digest(config), "profile_sha256": digest(profile),
            "manifest_sha256": digest(manifest), "model_revision": REVISION}


def prepare_data(run, profile, manifest, allow_download):
    folder = inside_run(run, run / "data")
    inside_run(run, run / "hub")
    folder.mkdir(exist_ok=True)
    for spec in profile["dataset"]["files"].values():
        destination = inside_run(run, folder / spec["filename"])
        if not destination.exists():
            if not allow_download:
                raise FileNotFoundError("STOP: missing data; only the explicit data stage can download")
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(profile["dataset"]["repo_id"], spec["filename"], repo_type="dataset",
                                     revision=profile["dataset"]["revision"], cache_dir=run / "hub")
            shutil.copyfile(cached, destination)
    records = load_sources(profile, folder)
    probes = read_json(ROOT / profile["evaluation"]["utility_probes"])
    development = read_json(ROOT / profile["development"]["path"])
    if build_manifest(profile, records, probes, development) != manifest:
        raise ValueError("STOP: data did not reproduce the entire frozen manifest")
    sets = {}
    for name in EVAL_SETS[:-1]:
        sets[name] = [{"case_id": case["case_id"],
                       "question": records[case["source_split"]][case["record_index"]][case["question_field"]],
                       "answer": records[case["source_split"]][case["record_index"]][case["answer_field"]]}
                      for case in manifest["sets"][name]]
    sets["utility"] = [{"case_id": row["id"], "question": row["question"], "answer": row["answer"]}
                       for row in probes["cases"]]
    # Sealed holdout is checksum/overlap validated above, never scored by this pilot.
    return sets


def validate_protocol(protocol, sets, profile, manifest, evidence):
    if (protocol.get("status") != "frozen_reviewed" or not protocol.get("reviewed_by")
            or not protocol.get("review_notes") or protocol.get("manifest_sha256") != digest(manifest)
            or protocol.get("target_scores_used_for_rate_selection") is not False):
        raise ValueError("STOP: independently reviewed frozen protocol required before any outputs")
    for phase in ("acquisition", "unlearning"):
        rate = protocol.get(f"{phase}_learning_rate")
        if type(rate) not in (int, float) or rate not in profile[phase]["candidate_learning_rates"]:
            raise ValueError("STOP: learning rate is unset or outside preregistered candidates")
    if (evidence.get("status") != "completed_reviewed_development_only"
            or evidence.get("target_scores_used_for_selection") is not False
            or evidence.get("model_revision") != REVISION
            or evidence.get("development_sha256") != manifest["development_sha256"]
            or evidence.get("acquisition_learning_rate") != protocol["acquisition_learning_rate"]
            or evidence.get("unlearning_learning_rate") != protocol["unlearning_learning_rate"]
            or not evidence.get("candidate_reports") or not evidence.get("reviewed_by")):
        raise ValueError("STOP: missing reviewed development calibration evidence; never invent passing results")
    # Candidate reports are reviewed experimental artifacts, not generated here.
    aliases = protocol.get("aliases_by_case_id", {})
    rows = {row["case_id"]: row for group in sets.values() for row in group}
    if set(aliases) != set(rows):
        raise ValueError("STOP: complete aliases for all evaluation and utility cases required")
    for case, values in aliases.items():
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not normalize_answer(value)
            or f" {normalize_answer(value)} " not in f" {normalize_answer(rows[case]['answer'])} " for value in values
        ):
            raise ValueError(f"STOP: invalid source-grounded key fact for {case}")
    if (not isinstance(protocol.get("refusal_phrases"), list) or not protocol["refusal_phrases"]
            or any(not isinstance(p, str) or not normalize_answer(p) for p in protocol["refusal_phrases"])
            or protocol.get("repetition_ngram") != 3 or protocol.get("repetition_count") != 3):
        raise ValueError("STOP: frozen refusal/repetition diagnostic rules required")


def freeze_protocol(run, path, sets, profile, manifest):
    protocol = read_json(path)
    evidence_path = Path(protocol.get("calibration_evidence_file") or "MISSING")
    if not evidence_path.is_file() or sha256_file(evidence_path) != protocol.get("calibration_evidence_sha256"):
        raise ValueError("STOP: missing or changed calibration evidence file")
    evidence = read_json(evidence_path)
    validate_protocol(protocol, sets, profile, manifest, evidence)
    if (run / "protocol.json").exists():
        if read_json(run / "protocol.json") != protocol:
            raise FileExistsError("STOP: run protocol is immutable; use a new run for a different protocol")
        if (run / "calibration.json").exists() and sha256_file(run / "calibration.json") != protocol["calibration_evidence_sha256"]:
            raise ValueError("STOP: conflicting frozen evidence")
    # Preserve bytes so the supplied SHA-256 remains independently auditable.
    shutil.copyfile(evidence_path, run / "calibration.json")
    atomic_json(run / "protocol.json", protocol)
    binding = {"protocol_sha256": digest(protocol), "calibration_sha256": digest(evidence)}
    atomic_json(run / "freeze.json", binding)
    return binding


def complete(run, stage, identity):
    path = run / f"{stage}.complete.json"
    if not path.is_file():
        raise RuntimeError(f"STOP: {stage} is not durably complete")
    value = read_json(path)
    if value.get("identity") != identity or value.get("stage") != stage or value.get("status") != "completed":
        raise ValueError("STOP: stage identity/status mismatch")
    artifact = run / value["artifact"]
    if run.resolve() not in artifact.resolve().parents:
        raise ValueError("STOP: stage artifact escaped run directory")
    if not artifact.is_file() or sha256_file(artifact) != value["artifact_sha256"]:
        raise ValueError("STOP: missing/corrupted stage artifact")
    if artifact.name == "checkpoint.json":
        info = checkpoint_info(artifact.parent, identity, stage)
        if info["step"] != info["budget"]:
            raise ValueError("STOP: partial checkpoint cannot complete a stage")
    for name, expected in value.get("related_artifacts", {}).items():
        if Path(name).name != name or sha256_file(run / name) != expected:
            raise ValueError("STOP: missing/corrupted related artifact")
    return artifact


def checkpoint_info(folder, identity, stage=None):
    info = read_json(folder / "checkpoint.json")
    if not isinstance(info, dict) or not isinstance(info.get("files"), dict):
        raise ValueError("STOP: malformed checkpoint manifest")
    if info["identity"] != identity or (stage is not None and info["stage"] != stage):
        raise ValueError("STOP: checkpoint belongs to a different run/stage")
    if not 0 <= info["step"] <= info["budget"]:
        raise ValueError("STOP: invalid checkpoint progress")
    for name, expected in info["files"].items():
        if Path(name).name != name or sha256_file(folder / name) != expected:
            raise ValueError("STOP: checkpoint is incomplete/corrupt")
    if (not info["files"] or "resume.pt" not in info["files"]
            or "model.safetensors" not in info["files"] or "config.json" not in info["files"]):
        raise ValueError("STOP: missing required checkpoint files")
    return info


def latest_checkpoint(run, stage, identity):
    folder = run / stage
    candidates = sorted(folder.glob("step-[0-9]*"), reverse=True) if folder.exists() else []
    for candidate in candidates:
        try:
            checkpoint_info(candidate, identity, stage)
            return candidate
        except (OSError, KeyError, ValueError):
            continue  # interrupted writes are never considered committed
    if candidates:
        raise RuntimeError("STOP: no intact checkpoint; inspect corruption before restarting")
    return None


def recovery_checkpoint(run, stage, identity, schedule_hash, parent_hash, budget, resume):
    previous = latest_checkpoint(run, stage, identity)
    if previous and not resume:
        raise RuntimeError("STOP: checkpoint exists; explicitly select resume")
    if resume and not previous:
        raise RuntimeError("STOP: resume requested but no intact checkpoint exists")
    start = 0
    if previous:
        info = checkpoint_info(previous, identity, stage)
        if (info["schedule_sha256"] != schedule_hash or info["parent_sha256"] != parent_hash
                or info["budget"] != budget):
            raise ValueError("STOP: resume schedule/parent/budget mismatch")
        start = info["step"]
    return previous, start


def seed_runtime(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_model(run, config, checkpoint=None, download=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    memory_check(config, run, before_load=True)
    inside_run(run, run / "hub")
    from huggingface_hub import hf_hub_download
    _, profile, _ = configuration()
    metadata = profile["model"]["metadata_sha256"]
    for name in metadata:
        local = Path(hf_hub_download(MODEL_ID, name, revision=REVISION, cache_dir=run / "hub", local_files_only=not download))
        if sha256_file(local) != metadata[name]:
            raise ValueError("STOP: original model/tokenizer metadata checksum mismatch")
    snapshot = local.parent
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if checkpoint:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                                                    torch_dtype=torch.float32, attn_implementation="sdpa")
    else:
        weights = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION,
                                      cache_dir=run / "hub", local_files_only=not download))
        if sha256_file(weights) != profile["model"]["weight_file_metadata_only"]["hub_lfs_sha256"]:
            raise ValueError("STOP: original pinned model weight checksum mismatch")
        model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=False,
                                                    local_files_only=True,
                                                    torch_dtype=torch.float32, attn_implementation="sdpa",
                                                    use_safetensors=True)
    if sum(p.numel() for p in model.parameters()) != 134515008:
        raise ValueError("STOP: wrong model parameter count")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    model.to("cuda:0")
    memory_check(config, run)
    return model, tokenizer


def save_checkpoint(run, stage, step, budget, model, tokenizer, optimizer, identity, schedule_hash, parent_hash, config):
    import torch
    base = inside_run(run, run / stage)
    base.mkdir(exist_ok=True)
    final = base / f"step-{step:06d}"
    if final.exists():
        raise FileExistsError("STOP: checkpoint already exists; resume rather than overwrite")
    temporary = base / (".partial-" + uuid.uuid4().hex)
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    # Only tensor/builtin states; resume uses weights_only=True, never remote pickle.
    torch.save({"optimizer": optimizer.state_dict() if optimizer else None,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, temporary / "resume.pt")
    files = {p.name: sha256_file(p) for p in temporary.iterdir() if p.is_file()}
    atomic_json(temporary / "checkpoint.json", {
        "stage": stage, "step": step, "budget": budget, "identity": identity,
        "schedule_sha256": schedule_hash, "parent_sha256": parent_hash, "files": files})
    os.replace(temporary, final)
    checkpoint_info(final, identity, stage)  # read back before pruning any old generation
    generations = sorted(base.glob("step-[0-9]*"), reverse=True)
    for old in generations[config["checkpoints_to_keep"]:]:
        # Only our validated generations in this stage; keep current + previous.
        checkpoint_info(old, identity, stage)
        shutil.rmtree(old)
    return final


def batch(row, tokenizer):
    from src.graddiff_integration import collate_causal_examples
    encoded = encode_plain_qa(tokenizer, row["question"], row["answer"], max_sequence_length=512)
    return collate_causal_examples([encoded], tokenizer, device="cuda:0", max_sequence_length=512)


def train_schedule(stage, manifest):
    if stage in ("acquire", "reference"):
        groups = ("retain_direct",) if stage == "reference" else ("forget_direct", "retain_direct")
        ids = [case["case_id"] for group in groups for case in manifest["sets"][group]]
        return [(case, None) for epoch in range(20) for case in acquisition_order(ids, epoch)]
    return [(p["forget"], p["retain"]) for p in manifest["unlearning_and_retain_control_pairs"]]


def acquisition_gate(before, after, gates):
    f, r, p = (after[name] for name in ("forget_direct", "retain_direct", "forget_paraphrase"))
    return (f["key_fact_recovery"] >= gates["minimum_direct_forget_recovery"]
            and r["key_fact_recovery"] >= gates["minimum_direct_retain_recovery"]
            and f["key_fact_recovery"] - before["forget_direct"]["key_fact_recovery"] >= gates["minimum_forget_gain_over_M0"]
            and p["key_fact_recovery"] >= gates["minimum_forget_paraphrase_recovery"]
            and all(after[s]["answer_nll"] < before[s]["answer_nll"] for s in ("forget_direct", "retain_direct")))


def require_acquisition(run, identity, profile):
    before = read_json(complete(run, "baseline", identity))["summary"]
    after = read_json(complete(run, "pre", identity))["summary"]
    if not acquisition_gate(before, after, profile["evaluation"]["acquisition_gates"]):
        raise RuntimeError("STOP: fixed-budget acquisition gates failed; no target tuning/unlearning")
    complete(run, "reference-eval", identity)


def train(run, stage, config, profile, manifest, sets, protocol, identity, resume):
    import torch
    from src.graddiff import causal_lm_nll, graddiff_objective
    parent = complete(run, PARENTS[stage], identity).parent
    checkpoint_info(parent, identity, PARENTS[stage])
    if stage in ("acquire", "reference"):
        complete(run, "baseline", identity)
    else:
        require_acquisition(run, identity, profile)
    if stage == "unlearn":
        complete(run, "noop", identity)
        complete(run, "retain-control-eval", identity)
    schedule = train_schedule(stage, manifest)
    schedule_hash = digest(schedule)
    parent_hash = sha256_file(parent / "checkpoint.json")
    previous, start = recovery_checkpoint(run, stage, identity, schedule_hash, parent_hash, len(schedule), resume)
    model, tokenizer = load_model(run, config, previous or parent)
    # Validate every fixed diagnostic/training encoding before constructing optimizer.
    for rows in sets.values():
        for row in rows:
            encode_plain_qa(tokenizer, row["question"], row["answer"], max_sequence_length=512)
    lr = protocol["acquisition_learning_rate" if stage in ("acquire", "reference") else "unlearning_learning_rate"]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0, foreach=False, fused=False)
    if previous:
        state = torch.load(previous / "resume.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    rows = {r["case_id"]: r for group in sets.values() for r in group}
    model.train()
    began = time.monotonic()
    attempt_id = uuid.uuid4().hex
    checkpoint = previous
    for index in range(start, len(schedule)):
        if (run / "STOP").exists() or time.monotonic() - began >= config["max_stage_wall_seconds"]:
            raise RuntimeError("STOP: owner stop/time budget; resume from last committed checkpoint")
        telemetry = memory_check(config, run)
        optimizer.zero_grad(set_to_none=True)
        forget, retain = schedule[index]
        if stage == "unlearn":
            # Sequential backwards yield exactly -gamma*NLL(F)+alpha*NLL(R),
            # with separate mean reductions and one clipped optimizer step.
            f_batch = batch(rows[forget], tokenizer)
            f_loss = causal_lm_nll(model(**f_batch).logits, f_batch["labels"])
            (-config["gamma"] * f_loss).backward()
            r_batch = batch(rows[retain], tokenizer)
            r_loss = causal_lm_nll(model(**r_batch).logits, r_batch["labels"])
            (config["alpha"] * r_loss).backward()
            objective = graddiff_objective(f_loss.detach(), r_loss.detach(), config["gamma"], config["alpha"]).item()
            losses = {"forget_nll": f_loss.item(), "retain_nll": r_loss.item()}
            del f_loss, r_loss, f_batch, r_batch
        else:
            selected = retain if stage == "retain-control" else forget
            inputs = batch(rows[selected], tokenizer)
            loss = causal_lm_nll(model(**inputs).logits, inputs["labels"])
            loss.backward()
            objective = loss.item()
            losses = {"nll": objective}
            del loss, inputs
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if (len(gradients) != sum(p.requires_grad for p in model.parameters())
                or not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients)):
            raise FloatingPointError("STOP: missing or nonfinite gradients; no optimizer step")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        if float(norm) <= 0:
            raise FloatingPointError("STOP: zero gradient norm; no optimizer step")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("STOP: nonfinite updated parameters; discard unsaved state")
        step = index + 1
        with (run / f"{stage}.attempts.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "objective": objective, **losses, "attempt_id": attempt_id,
                                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                     "grad_norm": float(norm), "elapsed_seconds": time.monotonic() - began,
                                     "memory": telemetry, "committed": False}) + "\n")
        if step % config["checkpoint_every_steps"] == 0 or step == len(schedule):
            checkpoint = save_checkpoint(run, stage, step, len(schedule), model, tokenizer, optimizer,
                                         identity, schedule_hash, parent_hash, config)
    if checkpoint is None:
        raise RuntimeError("STOP: no committed checkpoint")
    return checkpoint / "checkpoint.json"


def score_response(response, target, aliases, protocol):
    normalized = normalize_answer(response)
    tokens = normalized.split()
    grams = Counter(tuple(tokens[i:i+3]) for i in range(max(0, len(tokens)-2)))
    return {"exact_match": float(answer_metrics(response, target)["normalized_exact_match"]),
            "f1": answer_metrics(response, target)["normalized_token_overlap_f1"],
            "key_fact_recovery": float(any(f" {normalize_answer(a)} " in f" {normalized} " for a in aliases)),
            "empty": float(not normalized),
            "refusal": float(any(normalize_answer(p) in normalized for p in protocol["refusal_phrases"])),
            "repetition": float(any(count >= 3 for count in grams.values()))}


def evaluate(run, stage, source, config, sets, protocol, identity):
    import torch
    from src.graddiff import causal_lm_nll
    checkpoint = complete(run, source, identity).parent
    checkpoint_info(checkpoint, identity, source)
    model, tokenizer = load_model(run, config, checkpoint)
    model.eval()
    rows, summary = [], {}
    began = time.monotonic()
    with torch.inference_mode():
        for name in EVAL_SETS:
            group = []
            for row in sets[name]:
                if (run / "STOP").exists() or time.monotonic() - began > config["max_stage_wall_seconds"]:
                    raise RuntimeError("STOP: evaluation incomplete; rerun the same fixed stage")
                memory_check(config, run)
                inputs = batch(row, tokenizer)
                nll = causal_lm_nll(model(**inputs).logits, inputs["labels"]).item()
                count = int(inputs["labels"][:, 1:].ne(-100).sum())
                prompt = tokenizer(f"Question: {row['question']}\nAnswer:\n", add_special_tokens=False, return_tensors="pt").to("cuda:0")
                output = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=128,
                                        use_cache=False, pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(output[0, prompt.input_ids.shape[1]:], skip_special_tokens=True)
                metrics = score_response(response, row["answer"], protocol["aliases_by_case_id"][row["case_id"]], protocol)
                result = {"set": name, "case_id": row["case_id"], "response": response,
                          "answer_nll": nll, "target_tokens": count, **metrics}
                rows.append(result)
                group.append(result)
            summary[name] = {key: sum(r[key] for r in group) / len(group)
                             for key in ("exact_match", "f1", "key_fact_recovery", "empty", "refusal", "repetition")}
            summary[name].update(count=len(group), answer_nll=sum(r["answer_nll"] * r["target_tokens"] for r in group) / sum(r["target_tokens"] for r in group))
    path = run / f"{stage}.evaluation.json"
    atomic_json(path, {"stage": stage, "source_checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
                       "identity": identity, "summary": summary, "rows": rows,
                       "official_TOFU_scores": "not_computed", "elapsed_seconds": time.monotonic() - began})
    return path


def compare_summaries(before, after, control, gates):
    deltas = {name: {key: after[name][key] - before[name][key] for key in before[name] if key != "count"}
              for name in EVAL_SETS}
    forget_pass = all(
        -deltas[name]["key_fact_recovery"] >= gates["minimum_forget_direct_and_paraphrase_recovery_drop"]
        and after[name]["key_fact_recovery"] < control[name]["key_fact_recovery"]
        and deltas[name]["answer_nll"] > 0 for name in ("forget_direct", "forget_paraphrase"))
    utility_pass = all(
        deltas[name]["key_fact_recovery"] >= -gates["maximum_retain_recovery_drop"]
        and deltas[name]["f1"] >= -gates["maximum_retain_F1_drop"]
        and deltas[name]["answer_nll"] <= gates["maximum_retain_NLL_increase"]
        and all(deltas[name][key] <= gates["maximum_empty_refusal_repetition_rate_increase"] for key in ("empty", "refusal", "repetition"))
        for name in ("retain_direct", "retain_paraphrase", "retain_unseen_records", "utility"))
    return {"after_minus_before": deltas, "proposed_diagnostic_gates_pass": forget_pass and utility_pass,
            "semantic_review_required": True, "proof_of_erasure": False, "official_TOFU_scores": "not_computed"}


def execute(args):
    config, profile, manifest = configuration()
    if args.stage is None:
        return {"status": "plan_only", "stages": STAGES, "weight_updates_performed": False}
    if args.confirm != args.stage:
        raise ValueError("STOP: explicit per-stage --confirm required")
    if args.stage == "environment":
        return environment(require_cuda=True)
    if not args.run_dir:
        raise ValueError("STOP: persistent run directory required")
    run = outside_git(args.run_dir)
    # Prevent accidental use of Colab's ephemeral /content disk as recovery storage.
    drive = Path("/content/drive/MyDrive")
    if not drive.is_dir() or drive.resolve() not in run.parents:
        raise ValueError("STOP: mount Drive and use a run directory below /content/drive/MyDrive")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "STOP").exists():
        raise RuntimeError("STOP: owner stop file is present; inspect before starting any stage")
    base_identity = source_identity(config, profile, manifest)
    if (run / "run.json").exists():
        if read_json(run / "run.json") != base_identity:
            raise ValueError("STOP: run configuration/commit changed; restore original or use new run")
    else:
        atomic_json(run / "run.json", base_identity)
    if args.stage == "data":
        prepare_data(run, profile, manifest, allow_download=True)
        return {"status": "verified_data_only", "manifest_sha256": digest(manifest)}
    sets = prepare_data(run, profile, manifest, allow_download=False)
    if args.stage == "freeze":
        if not args.protocol:
            raise ValueError("STOP: reviewed protocol path required")
        return freeze_protocol(run, args.protocol, sets, profile, manifest)
    protocol = read_json(run / "protocol.json")
    evidence = read_json(run / "calibration.json")
    validate_protocol(protocol, sets, profile, manifest, evidence)
    if (sha256_file(run / "calibration.json") != protocol["calibration_evidence_sha256"]
            or read_json(run / "freeze.json") != {"protocol_sha256": digest(protocol), "calibration_sha256": digest(evidence)}):
        raise ValueError("STOP: frozen protocol/calibration evidence changed")
    runtime = environment(require_cuda=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    runtime_identity = {key: runtime[key] for key in ("python", "packages", "torch_cuda", "gpu", "capability", "driver")}
    runtime_identity["pip_freeze"] = freeze
    identity = {**base_identity, "protocol_sha256": digest(protocol), "calibration_sha256": digest(evidence),
                "runtime_sha256": digest(runtime_identity)}
    if (run / "runtime.json").exists() and read_json(run / "runtime.json") != runtime_identity:
        raise ValueError("STOP: runtime changed; exact resume unsupported, restore environment or use new run")
    atomic_json(run / "runtime.json", runtime_identity)
    if (run / f"{args.stage}.complete.json").exists():
        artifact = complete(run, args.stage, identity)
        return {"status": "already_completed_verified", "artifact": str(artifact)}
    seed_runtime(config["seed"])
    started = time.monotonic()
    attempt = run / f"{args.stage}.{uuid.uuid4().hex}.attempt.json"
    atomic_json(attempt, {"status": "running_not_completed", "stage": args.stage, "identity": identity,
                          "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        if args.stage == "load":
            previous = latest_checkpoint(run, "load", identity)
            if previous:
                artifact = previous / "checkpoint.json"
                if checkpoint_info(previous, identity, "load")["step"] != 0:
                    raise ValueError("STOP: invalid base checkpoint")
                value = {"status": "completed", "stage": "load", "identity": identity,
                         "artifact": str(artifact.relative_to(run)), "artifact_sha256": sha256_file(artifact)}
                atomic_json(run / "load.complete.json", value)
                atomic_json(attempt, value)
                return value
            model, tokenizer = load_model(run, config, download=True)
            # Read-only one-batch loss parity and memory; no optimizer/backward.
            import torch
            from src.graddiff import causal_lm_nll
            inputs = batch(sets["forget_direct"][0], tokenizer)
            model.eval()
            with torch.inference_mode():
                output = model(**inputs)
                expected = causal_lm_nll(output.logits, inputs["labels"])
                if not torch.allclose(output.loss, expected, atol=1e-5, rtol=1e-5):
                    raise RuntimeError("STOP: model loss parity failed")
            artifact = save_checkpoint(run, "load", 0, 0, model, tokenizer, None, identity, digest([]), REVISION, config) / "checkpoint.json"
        elif args.stage in ("acquire", "reference", "retain-control", "unlearn"):
            artifact = train(run, args.stage, config, profile, manifest, sets, protocol, identity, args.resume)
        elif args.stage == "noop":
            require_acquisition(run, identity, profile)
            artifact = evaluate(run, "noop", "acquire", config, sets, protocol, identity)
            if read_json(artifact)["summary"] != read_json(complete(run, "pre", identity))["summary"]:
                raise RuntimeError("STOP: no-op evaluation differs from pre; inspect reproducibility")
        elif args.stage == "post":
            # Every control is mandatory, independently initialized from M_all.
            complete(run, "noop", identity)
            complete(run, "retain-control", identity)
            complete(run, "reference-eval", identity)
            control_artifact = read_json(complete(run, "retain-control-eval", identity))
            control = control_artifact["summary"]
            artifact = evaluate(run, "post", "unlearn", config, sets, protocol, identity)
            before = read_json(complete(run, "pre", identity))["summary"]
            after = read_json(artifact)["summary"]
            expected_control = sha256_file(complete(run, "retain-control", identity))
            if control_artifact["identity"] != identity or control_artifact["source_checkpoint_sha256"] != expected_control:
                raise ValueError("STOP: retain-control evaluation does not match its checkpoint")
            comparison = compare_summaries(before, after, control, profile["evaluation"]["unlearning_gates"])
            reference = read_json(complete(run, "reference-eval", identity))["summary"]
            comparison["after_minus_retain_reference"] = {
                name: {key: after[name][key] - reference[name][key] for key in reference[name] if key != "count"}
                for name in EVAL_SETS}
            atomic_json(run / "comparison.json", comparison)
        else:
            source = {"baseline": "load", "pre": "acquire", "reference-eval": "reference",
                      "retain-control-eval": "retain-control"}[args.stage]
            artifact = evaluate(run, args.stage, source, config, sets, protocol, identity)
        # Only a read-back, hashed final artifact can create a completion marker.
        value = {"status": "completed", "stage": args.stage, "identity": identity,
                 "artifact": str(artifact.relative_to(run)), "artifact_sha256": sha256_file(artifact),
                 "elapsed_seconds": time.monotonic() - started,
                 "finished_utc": datetime.now(timezone.utc).isoformat()}
        if args.stage == "post":
            value["related_artifacts"] = {"comparison.json": sha256_file(run / "comparison.json")}
        atomic_json(run / f"{args.stage}.complete.json", value)
        atomic_json(attempt, value)
        return value
    except BaseException as exc:
        atomic_json(attempt, {"status": "interrupted_or_failed_not_completed", "stage": args.stage,
                              "error_type": type(exc).__name__, "elapsed_seconds": time.monotonic() - started})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--confirm")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(execute(args), indent=2))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
