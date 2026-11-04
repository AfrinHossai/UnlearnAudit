"""Task 70 preparation. Default audits bytes only; --dry-run uses fixtures only.

The real three-arm CPU pilot requires NEW, independent human authorization.
Neither an acquisition authorization nor running the plan authorizes updates.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from src import tiny_model_acquisition as a
from src.small_model_acquisition import seal_json, require_ignored
from src.small_model_experiment import ROOT, acquisition_order
from src.small_model_backward import parameter_fingerprint, validate_gradients
from src.inspect_tofu import sha256_file

PROFILE = ROOT / "experiments/tiny_cpu_unlearning_task70.json"
PROFILE_SHA256 = "c1e128665c97279a017a9e762d06e8ca464a774265986c967a4b05d2b671e18b"
AUTHORIZATION = "task70:tiny-90496:cpu:three-arms:20-per-arm:40-total"


def frozen_inputs():
    a.telemetry._safe_path(PROFILE, ROOT / "experiments")
    if sha256_file(PROFILE) != PROFILE_SHA256:
        raise ValueError("Task 70 profile drift")
    profile = json.loads(PROFILE.read_text())
    path = a.telemetry._safe_path(ROOT / profile["frozen_manifest"], ROOT / "experiments")
    if sha256_file(path) != profile["frozen_manifest_sha256"]:
        raise ValueError("Task 70 manifest drift")
    frozen = json.loads(path.read_text())
    spec, rows = a.frozen_inputs()
    s = frozen["selection"]
    if (rows != frozen["evaluation_rows"] or len(s["forget_ids"]) != 2 or len(s["retain_ids"]) != 6
            or set(s["forget_ids"]) & set(s["retain_ids"])
            or sorted(s["forget_ids"] + s["retain_ids"]) != sorted(spec["training_ids"])):
        raise ValueError("frozen selection/encoding coverage mismatch")
    return profile, frozen, spec, rows


def verify_acquisition(frozen):
    """Verify original bytes/markers, without importing Torch or evaluating a model."""
    directory = a.telemetry._safe_path(ROOT / frozen["acquisition_directory"], a.RUNS)
    inventory = frozen["acquisition_files_sha256"]
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(inventory):
        raise ValueError("original acquisition file inventory drift")
    for name, expected in inventory.items():
        path = a.telemetry._safe_path(directory / name, directory)
        if sha256_file(path) != expected:
            raise ValueError("original acquisition artifact drift: " + name)
    for name, expected in frozen["acquisition_source_files_sha256"].items():
        if sha256_file(a.telemetry._safe_path(ROOT / name, ROOT)) != expected:
            raise ValueError("original acquisition source drift: " + name)
    report = json.loads((directory / "report.json").read_text())
    marker = json.loads((directory / "COMPLETE.json").read_text())
    if (report["status"] != "acquisition_complete" or report["optimizer_steps"] != 700
            or report["optimizer_step_attempts"] != 700 or report["parameter_count"] != 90496
            or report["all_eight_recovered"] is not True or report["acquisition_accepted"] is not True
            or report["possible_partial_update"] or report["unlearning_performed"]
            or marker != {"report_sha256": inventory["report.json"], "status": "acquisition_complete", "optimizer_steps": 700}):
        raise ValueError("acquisition acceptance/COMPLETE mismatch")
    for cp in report["checkpoints"]:
        d = directory / cp["directory"]
        expected = {**cp, "files_sha256": {k: v for k, v in cp["files_sha256"].items() if k != "COMPLETE.json"}}
        if json.loads((d / "COMPLETE.json").read_text()) != expected:
            raise ValueError("checkpoint COMPLETE mismatch")
        for name, value in cp["files_sha256"].items():
            if sha256_file(d / name) != value:
                raise ValueError("checkpoint report/hash mismatch")
    cp = report["checkpoints"][-1]
    metrics = json.loads((directory / cp["directory"] / "metrics.json").read_text())
    baseline = json.loads((directory / "baseline.json").read_text())
    if (cp != frozen["accepted_checkpoint"] or cp["step"] != 700
            or report["final_parameter_sha256"] != frozen["accepted_parameter_sha256"]
            or metrics["parameter_sha256"] != frozen["accepted_parameter_sha256"]
            or metrics["evaluation"] != frozen["acquired_evaluation"]["rows"]
            or report["evaluations"][-1] != frozen["acquired_evaluation"]
            or baseline != frozen["initial_random_baseline"]
            or not a.acceptance(metrics["evaluation"], baseline["rows"])["acquisition_accepted"]):
        raise ValueError("accepted checkpoint/baseline mismatch")
    return directory / cp["directory"] / "model.safetensors"


def authorize(enabled, authorization):
    if enabled is not True or authorization != AUTHORIZATION:
        raise ValueError("NEW explicit authorization required: --run-controlled-experiment AND exact Task 70 token")


def new_run_path(path):
    path = a.telemetry._safe_path(Path(path), a.RUNS)
    if path.parent != a.RUNS or not re.fullmatch(r"unlearning-task70-[A-Za-z0-9_-]+", path.name) or path.exists():
        raise ValueError("requires unique new unlearning-task70-* output; no overwrite/resume")
    require_ignored(path)
    return path


class Monitor(a.CPUMonitor):
    """Same conservative CPU physical policy; shorter entire-suite deadline."""
    def check(self):
        super().check()
        if time.monotonic() - self.started > 600:
            raise RuntimeError("10-minute entire-suite deadline")


def assert_same(before, after, tolerance=1e-5):
    if len(before) != len(after):
        raise RuntimeError("evaluation coverage drift")
    for x, y in zip(before, after):
        if any(x[k] != y[k] for k in ("id", "kind", "role", "generated_ids", "exact_recovery", "valid_answer", "finish_reason")):
            raise RuntimeError("baseline/no-op generated output drift")
        for k in ("answer_byte_nll", "answer_nll_including_eos", "eos_nll", "correct_over_wrong_nll_margin"):
            if k in x and (k not in y or not math.isfinite(y[k]) or abs(x[k] - y[k]) > tolerance):
                raise RuntimeError("baseline/no-op NLL drift")


def summarize(rows):
    """Keep numerator/denominator and mean loss distinct; never call format validity accuracy."""
    groups = {}
    for kind in ("direct", "paraphrase", "untrained_direct", "untrained_paraphrase"):
        for role in ("forget", "retain", "all"):
            group = [r for r in rows if r["kind"] == kind and (role == "all" or r["role"] == role)]
            if group:
                groups[kind + ":" + role] = {
                    "recovered": sum(r["exact_recovery"] for r in group), "total": len(group),
                    "valid_answers": sum(r["valid_answer"] for r in group),
                    "mean_answer_byte_nll": sum(r["answer_byte_nll"] for r in group) / len(group),
                    "mean_eos_nll": sum(r["eos_nll"] for r in group) / len(group)}
    return groups


def sentinel_evaluation(model, texts, monitor, reference=None):
    """Full-vocabulary KL(M_all || arm) on fixed teacher-forced ASCII contexts."""
    import torch
    model.eval().requires_grad_(False)
    before = parameter_fingerprint(model)
    refs, stats = [], []
    with torch.inference_mode():
        for i, text in enumerate(texts):
            monitor.boundary("sentinel:" + str(i))
            ids = torch.tensor([list(text.encode()) + [256]], dtype=torch.long)
            output = a.checked_forward(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
            logp = output.logits[:, :-1].log_softmax(-1)
            ref = logp if reference is None else reference[i]
            kl = float((ref.exp() * (ref - logp)).sum(-1).mean())
            stats.append({"id": i, "nll_including_eos": float(output.loss), "kl_from_acquired_nats": max(0.0, kl),
                          "entropy_nats": float(-(logp.exp() * logp).sum(-1).mean())})
            refs.append(logp.clone())
    if parameter_fingerprint(model) != before:
        raise RuntimeError("sentinel evaluation mutated weights")
    return {"rows": stats, "mean_nll": sum(s["nll_including_eos"] for s in stats) / len(stats),
            "mean_kl_nats": sum(s["kl_from_acquired_nats"] for s in stats) / len(stats)}, refs


def measure(model, rows, encoded, profile, monitor, reference=None):
    results = a.evaluate(model, rows, encoded, monitor)
    for row in results:
        row["target_byte_sequence_probability_teacher_forced"] = math.exp(-12 * row["answer_byte_nll"])
    sentinels, refs = sentinel_evaluation(model, profile["sentinels"], monitor, reference)
    return {"rows": results, "summary": summarize(results), "sentinels": sentinels}, refs


def safety_metrics(evaluation, baseline, profile):
    old = {r["id"]: r for r in baseline["rows"]}
    limits = profile["limits"]
    for row in evaluation["rows"]:
        for key in ("answer_byte_nll", "answer_nll_including_eos", "eos_nll"):
            if not math.isfinite(row[key]):
                raise FloatingPointError("nonfinite evaluation metric")
        if row["kind"] == "direct":
            if row["role"] == "forget" and row["answer_byte_nll"] > limits["max_forget_byte_nll"]:
                raise RuntimeError("forget loss runaway STOP")
            if row["role"] == "retain" and (not row["exact_recovery"] or
                    row["answer_byte_nll"] - old[row["id"]]["answer_byte_nll"] > limits["max_retain_byte_nll_increase"]):
                raise RuntimeError("retain degradation STOP")
    s, b = evaluation["sentinels"], baseline["sentinels"]
    if (not all(math.isfinite(s[k]) for k in ("mean_nll", "mean_kl_nats"))
            or s["mean_nll"] - b["mean_nll"] > limits["max_sentinel_nll_increase"]
            or s["mean_kl_nats"] > limits["max_sentinel_kl_nats"]):
        raise RuntimeError("model-wide sentinel degradation STOP")


def validate_optimizer(optimizer, model, profile):
    import torch
    expected = {k: v for k, v in profile["optimizer"].items() if k != "name"}
    if type(optimizer) is not torch.optim.SGD or len(optimizer.param_groups) != 1 or optimizer.state:
        raise RuntimeError("requires stateless single-group SGD")
    group = optimizer.param_groups[0]
    if (any(type(group.get(k)) is not type(v) or group[k] != v for k, v in expected.items())
            or [id(p) for p in group["params"]] != [id(p) for p in model.parameters()]):
        raise RuntimeError("SGD configuration/membership drift")


def backward_objective(model, groups, arm, profile, monitor):
    """Sequential batch-one backward equals -mean_F NLL + mean_R NLL.

    Each example has 12 answer bytes + EOS supervised, prompts masked. Summing
    scaled gradients avoids retaining eight forward graphs. No optimizer here.
    """
    import torch
    if arm not in ("retain_only", "gradient_difference"):
        raise ValueError("no backward path for no-op")
    model.train().requires_grad_(True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("preexisting gradients")
    losses = {}
    for role in ("forget", "retain"):
        if role == "forget" and arm == "retain_only":
            continue
        batches = groups[role]
        coefficient = -profile["gamma"] if role == "forget" else profile["alpha"]
        values = []
        for batch in batches:
            monitor.boundary("objective:" + role)
            loss = a.checked_forward(model, batch).loss
            values.append(float(loss.detach()))
            (coefficient * loss / len(batches)).backward()
        losses[role + "_nll"] = sum(values) / len(values)
    validate_gradients(model, torch, device="cpu")
    losses["objective"] = profile["alpha"] * losses["retain_nll"] - profile["gamma"] * losses.get("forget_nll", 0.0)
    return losses


def update(model, groups, arm, profile, monitor, report, arm_report, *, enabled, authorization):
    """Authorization is repeated immediately before every optimizer invocation."""
    import torch
    authorize(enabled, authorization)
    if monitor.mode != "task70_controlled":
        raise RuntimeError("missing controlled CPU admission")
    for r, limit in ((arm_report, profile["steps_per_update_arm"]), (report, profile["total_optimizer_step_limit"])):
        if (type(r["optimizer_steps"]) is not int or type(r["optimizer_step_attempts"]) is not int
                or not 0 <= r["optimizer_steps"] == r["optimizer_step_attempts"] < limit or r["possible_partial_update"]):
            raise RuntimeError("step limit/partial attempt; no retry")
    optimizer = arm_report["_optimizer"]
    validate_optimizer(optimizer, model, profile)
    try:
        losses = backward_objective(model, groups, arm, profile, monitor)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), profile["gradient_clip"], foreach=False, error_if_nonfinite=True))
        if not 0 < norm <= profile["limits"]["max_gradient_norm_before_clip"]:
            raise FloatingPointError("zero/excessive gradient norm")
        validate_gradients(model, torch, device="cpu")
        authorize(enabled, authorization)
        validate_optimizer(optimizer, model, profile)
        monitor.boundary("immediately_before_optimizer")
        for r in (report, arm_report):
            r["optimizer_step_attempts"] += 1
            r["possible_partial_update"] = True
        optimizer.step()
        for r in (report, arm_report):
            r["optimizer_steps"] += 1
            r["possible_partial_update"] = False
        monitor.boundary("after_optimizer")
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite post-update parameter")
        return {**losses, "gradient_norm_before_clip": norm, "step": arm_report["optimizer_steps"]}
    finally:
        model.zero_grad(set_to_none=True)


def projected_budget(monitor, profile, report, step_times, evaluation_time):
    remaining = profile["total_optimizer_step_limit"] - report["optimizer_steps"]
    # Ten full evaluation passes (five per update arm), plus serialization/load reserve.
    estimate = time.monotonic() - monitor.started + 2 * max(step_times) * remaining + 20 * evaluation_time + 60
    report["projected_total_seconds"] = estimate
    if estimate >= profile["limits"]["projection_seconds"]:
        raise RuntimeError("projected entire-suite time exceeds 8-minute admission")


def decide(arms, profile):
    """A complete negative result is valid evidence; only fixed final step is tested."""
    c = profile["acceptance"]
    if set(arms) != set(profile["arms"]):
        raise RuntimeError("incomplete control arms")
    for name, ar in arms.items():
        expected = 0 if name == "noop" else profile["steps_per_update_arm"]
        if (ar["optimizer_steps"] != expected or ar["optimizer_step_attempts"] != expected
                or ar["possible_partial_update"] or ar["evaluations"][-1]["step"] != expected):
            raise RuntimeError("incomplete control budget/final evaluation")
    gd, retain = arms["gradient_difference"], arms["retain_only"]
    base = {r["id"]: r for r in gd["before"]["rows"]}
    control = {r["id"]: r for r in retain["evaluations"][-1]["rows"]}
    final = gd["evaluations"][-1]
    checks = []
    for row in final["rows"]:
        if row["kind"] != "direct":
            continue
        delta = row["answer_byte_nll"] - base[row["id"]]["answer_byte_nll"]
        if row["role"] == "forget":
            checks.extend([not row["exact_recovery"], delta >= c["minimum_each_forget_byte_nll_increase"],
                           row["correct_over_wrong_nll_margin"] <= c["maximum_each_forget_correct_over_wrong_margin"],
                           row["answer_byte_nll"] - control[row["id"]]["answer_byte_nll"] >= c["minimum_forget_nll_increase_beyond_retain_only"]])
        else:
            checks.extend([row["exact_recovery"], delta <= c["maximum_each_retain_byte_nll_increase"]])
    checks.extend([final["sentinels"]["mean_nll"] - gd["before"]["sentinels"]["mean_nll"] <= c["maximum_sentinel_nll_increase"],
                   final["sentinels"]["mean_kl_nats"] <= c["maximum_sentinel_kl_nats"]])
    return {"selective_direct_forgetting_gate": all(checks), "checks": checks,
            "paraphrase_effect_identifiable": False, "privacy_protection_established": False,
            "comparison": paired_deltas(arms)}


def paired_deltas(arms):
    result = {}
    for arm, ar in arms.items():
        before = {r["id"]: r for r in ar["before"]["rows"]}
        result[arm] = [{"id": r["id"], "exact_before": before[r["id"]]["exact_recovery"],
                        "exact_after": r["exact_recovery"],
                        "answer_byte_nll_delta": r["answer_byte_nll"] - before[r["id"]]["answer_byte_nll"],
                        "eos_nll_delta": r["eos_nll"] - before[r["id"]]["eos_nll"],
                        "margin_delta": r["correct_over_wrong_nll_margin"] - before[r["id"]]["correct_over_wrong_nll_margin"]
                        if "correct_over_wrong_nll_margin" in r else None} for r in ar["evaluations"][-1]["rows"]]
    return result


def run(*, enabled, authorization, run_path):
    authorize(enabled, authorization)  # before model imports, files, or telemetry
    profile, frozen, spec, rows = frozen_inputs()
    source_checkpoint = verify_acquisition(frozen)
    run_path = new_run_path(run_path)
    a.environment()
    run_path.mkdir()
    report = {"status": "RUNNING", "scope": "synthetic_controlled_pilot_not_privacy_evidence",
              "optimizer_steps": 0, "optimizer_step_attempts": 0, "possible_partial_update": False,
              "training_authorized": True, "arms": {}, "profile_sha256": PROFILE_SHA256,
              "manifest_sha256": profile["frozen_manifest_sha256"], "stop_reason": None}
    seal_json(run_path / "RUNNING.json", {"authorization_scope": AUTHORIZATION, "max_optimizer_steps": 40})
    monitor, model = Monitor(), None
    monitor.mode = "task70_controlled"
    try:
        from src.small_model_lora import provenance
        report["provenance"] = provenance()
        if report["provenance"]["versions"] != frozen["versions"] or platform.python_version() != frozen["python"]:
            raise RuntimeError("acquisition runtime version drift")
        if report["provenance"]["installed_llama_source_sha256"] != frozen["installed_llama_source_sha256"]:
            raise RuntimeError("installed Llama implementation drift")
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["memory_policy"] = a.memory_policy()
        report["encoding_sha256"] = a.digest({k: [v.input_ids, v.labels] for k, v in a.examples(rows).items()})
        monitor.capture("preload", before_load=True)
        if shutil.disk_usage(ROOT).free < profile["limits"]["minimum_disk_free_bytes"]:
            raise RuntimeError("requires 4 GiB free disk")
        import torch
        from safetensors.torch import load_model
        monitor.capture("before_initialization", before_load=True)
        monitor.start()
        encoded = a.examples(rows)
        references, baseline = None, None
        max_eval_time, times = 0.0, []
        for arm in profile["arms"]:
            monitor.boundary("load:" + arm)
            model = a.initialize(spec)
            load_model(model, str(source_checkpoint), strict=True, device="cpu")
            a.validate_model(model, spec)
            if parameter_fingerprint(model) != frozen["accepted_parameter_sha256"]:
                raise RuntimeError("loaded accepted parameter fingerprint mismatch")
            ar = {"optimizer_steps": 0, "optimizer_step_attempts": 0, "possible_partial_update": False,
                  "initial_parameter_sha256": parameter_fingerprint(model), "evaluations": [], "training": [], "checkpoints": []}
            report["arms"][arm] = ar
            start = time.monotonic()
            before, refs = measure(model, rows, encoded, profile, monitor, references)
            max_eval_time = max(max_eval_time, time.monotonic() - start)
            assert_same(frozen["acquired_evaluation"]["rows"], before["rows"])
            if baseline is None:
                baseline, references = before, refs
            else:
                if (abs(before["sentinels"]["mean_nll"] - baseline["sentinels"]["mean_nll"]) > 1e-5
                        or before["sentinels"]["mean_kl_nats"] > 1e-5):
                    raise RuntimeError("arm initial sentinel mismatch")
            ar["before"] = before
            seal_json(run_path / (arm + "-before.json"), before)
            if arm == "noop":
                after, _ = measure(model, rows, encoded, profile, monitor, references)
                assert_same(before["rows"], after["rows"])
                if after["sentinels"] != before["sentinels"] or parameter_fingerprint(model) != frozen["accepted_parameter_sha256"]:
                    raise RuntimeError("no-op weight/sentinel drift")
                ar["evaluations"].append({"step": 0, **after})
            else:
                # Backward-only objective rehearsal on this fresh copy, before optimizer creation.
                directs = {r["fact_id"]: r for r in rows if r["kind"] == "direct"}
                def groups_for(step):
                    return {role: [a.batch_for(encoded[directs[k]["id"]]) for k in acquisition_order(
                        frozen["selection"][role + "_ids"], step, seed=profile["seed"])] for role in ("forget", "retain")}
                start = time.monotonic()
                try:
                    backward_objective(model, groups_for(0), arm, profile, monitor)
                finally:
                    model.zero_grad(set_to_none=True)
                times.append(time.monotonic() - start)
                if parameter_fingerprint(model) != frozen["accepted_parameter_sha256"]:
                    raise RuntimeError("backward rehearsal mutated weights")
                projected_budget(monitor, profile, report, times, max_eval_time)
                authorize(enabled, authorization)
                ar["_optimizer"] = torch.optim.SGD(model.parameters(), **{k: v for k, v in profile["optimizer"].items() if k != "name"})
                for step in range(1, profile["steps_per_update_arm"] + 1):
                    a.validate_model(model, spec)
                    start = time.monotonic()
                    record = update(model, groups_for(step), arm, profile, monitor, report, ar,
                                    enabled=enabled, authorization=authorization)
                    times.append(time.monotonic() - start)
                    ar["training"].append({**record, "seconds": times[-1], "order": {
                        role: acquisition_order(frozen["selection"][role + "_ids"], step, seed=profile["seed"])
                        for role in (("retain",) if arm == "retain_only" else ("forget", "retain"))}})
                    projected_budget(monitor, profile, report, times, max_eval_time)
                    if step in profile["evaluation_steps"]:
                        start = time.monotonic()
                        evaluation, _ = measure(model, rows, encoded, profile, monitor, references)
                        max_eval_time = max(max_eval_time, time.monotonic() - start)
                        ar["evaluations"].append({"step": step, **evaluation})
                        seal_json(run_path / f"{arm}-step-{step:03d}.json", ar["evaluations"][-1])
                        safety_metrics(evaluation, before, profile)
                        print(json.dumps({"arm": arm, "step": step, "summary": evaluation["summary"]}), flush=True)
                ar.pop("_optimizer")
            ar["final_parameter_sha256"] = parameter_fingerprint(model)
            branch = run_path / arm
            branch.mkdir()
            a.checkpoint(model, branch, ar["optimizer_steps"], ar["evaluations"][-1]["rows"], ar, monitor)
            # Verify round-trip of NEW branch output only.
            load_model(model, str(branch / f"checkpoint-{ar['optimizer_steps']:06d}" / "model.safetensors"), strict=True, device="cpu")
            if parameter_fingerprint(model) != ar["final_parameter_sha256"]:
                raise RuntimeError("branch checkpoint round-trip mismatch")
            model.zero_grad(set_to_none=True)
            del model
            model = None
            gc.collect()
        report["result"] = decide(report["arms"], profile)
        report["status"] = "controlled_pilot_complete"
    except BaseException as exc:
        report.update(status="STOP", stop_reason=f"{type(exc).__name__}: {exc}")
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
        for ar in report["arms"].values():
            ar.pop("_optimizer", None)
        try:
            monitor.close()
            verify_acquisition(frozen)
            frozen_inputs()
            for name, value in report.get("provenance", {}).get("files_sha256", {}).items():
                if sha256_file(ROOT / name) != value:
                    raise RuntimeError("runner/source changed during experiment")
        except BaseException as exc:
            report.update(status="STOP", stop_reason=f"cleanup/integrity: {exc}")
        report["measurements"] = monitor.samples
        report["elapsed_seconds"] = time.monotonic() - monitor.started
        h = seal_json(run_path / "report.json", report)
        seal_json(run_path / ("COMPLETE.json" if report["status"] == "controlled_pilot_complete" else "FAILED.json"),
                  {"report_sha256": h, "status": report["status"], "optimizer_steps": report["optimizer_steps"]})
        from src.small_model_lora import fsync_directory
        fsync_directory(run_path)
        fsync_directory(run_path.parent)
    return report


def dry_run():
    """Only a random tiny model and two four-token fixture batches; no optimizer."""
    import torch
    profile, _, spec, _ = frozen_inputs()
    model = a.initialize(spec)
    before = parameter_fingerprint(model)
    class FixtureMonitor:
        def boundary(self, stage):
            pass
    def batch(values):
        ids = torch.tensor([values], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    groups = {"forget": [batch([1, 3, 4, 5])], "retain": [batch([6, 7, 8, 9])]}
    results = {}
    for arm in ("retain_only", "gradient_difference"):
        try:
            results[arm] = backward_objective(model, groups, arm, profile, FixtureMonitor())
        finally:
            model.zero_grad(set_to_none=True)
    if parameter_fingerprint(model) != before:
        raise RuntimeError("fixture weights changed")
    return {"status": "synthetic_fixture_dry_run_not_unlearning", "objectives": results,
            "optimizer_steps": 0, "checkpoint_writes": 0, "weights_unchanged": True,
            "acquired_checkpoint_loaded": False, "target_facts_executed": False,
            "training_authorized": False, "memory_admission_measured": False}


def plan():
    profile, frozen, _, rows = frozen_inputs()
    checkpoint = verify_acquisition(frozen)
    return {"status": "preparation_only", "acquisition_verified": True,
            "source_checkpoint": str(checkpoint.relative_to(ROOT)), "selection": frozen["selection"],
            "baseline": summarize(frozen["acquired_evaluation"]["rows"]), "evaluation_cases": len(rows),
            "arms": profile["arms"], "total_optimizer_step_limit": 40,
            "optimizer_steps": 0, "training_authorized": False, "model_loaded": False,
            "privacy_protection_established": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run-controlled-experiment", action="store_true")
    parser.add_argument("--authorize-weight-update")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.run_controlled_experiment:
        if args.run_dir is None:
            parser.error("--run-dir required")
        result = run(enabled=True, authorization=args.authorize_weight_update, run_path=args.run_dir)
    else:
        if args.authorize_weight_update is not None or args.run_dir is not None:
            parser.error("update token/output path requires --run-controlled-experiment")
        result = dry_run() if args.dry_run else plan()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["status"] == "STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
