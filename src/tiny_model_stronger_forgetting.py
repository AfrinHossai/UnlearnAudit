"""Task 79: one gamma=10 follow-up, with sealed Task 70 controls reused.

Default: byte/JSON audit only. --dry-run: random fixture backward only.
Real updates require separate explicit human authorization in normal Terminal.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

from src import tiny_model_unlearning as u

a, ROOT = u.a, u.ROOT
PROFILE = ROOT / "experiments/tiny_cpu_stronger_forgetting_task79.json"
PROFILE_SHA256 = "153c34b551a4c7fd38b4026a6dd509befcd176594d297ce581c3edfffe88fbc3"
AUTHORIZATION = "task79:tiny-90496:cpu:gamma10:alpha1:20-total:reuse-task70"
OUTPUT = a.RUNS / "stronger-forgetting-task79-001"


def frozen_inputs():
    a.telemetry._safe_path(PROFILE, ROOT / "experiments")
    if u.sha256_file(PROFILE) != PROFILE_SHA256:
        raise ValueError("Task 79 profile drift")
    profile = json.loads(PROFILE.read_text())
    old, frozen, spec, rows = u.frozen_inputs()
    expected = {**old, "gamma": 10.0, "total_optimizer_step_limit": 20}
    if (set(profile) != set(old) | {"followup"}
            or {k: profile[k] for k in old} != expected):
        raise ValueError("only gamma and fresh execution budget may differ from Task 70")
    return profile, frozen, spec, rows


def verify_history(profile, frozen):
    """Hash every historical artifact as opaque bytes; never load old arm weights."""
    f = profile["followup"]
    manifest = a.telemetry._safe_path(ROOT / f["historical_inventory"], ROOT / "experiments")
    if u.sha256_file(manifest) != f["historical_inventory_sha256"]:
        raise ValueError("historical inventory drift")
    inventory = json.loads(manifest.read_text())
    directory = a.telemetry._safe_path(ROOT / f["historical_directory"], a.RUNS)
    hashes = inventory["files_sha256"]
    if (inventory["directory"] != f["historical_directory"]
            or {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()} != set(hashes)):
        raise ValueError("historical file inventory mismatch")
    for name, expected in hashes.items():
        if u.sha256_file(a.telemetry._safe_path(directory / name, directory)) != expected:
            raise ValueError("historical artifact drift: " + name)
    report = json.loads((directory / "report.json").read_text())
    if (hashes["report.json"] != f["historical_report_sha256"]
            or report["status"] != "controlled_pilot_complete" or report["stop_reason"] is not None
            or report["optimizer_steps"] != 40 or report["optimizer_step_attempts"] != 40
            or report["possible_partial_update"] or "FAILED.json" in hashes
            or report["profile_sha256"] != u.PROFILE_SHA256
            or report["manifest_sha256"] != profile["frozen_manifest_sha256"]
            or json.loads((directory / "COMPLETE.json").read_text()) != {
                "report_sha256": hashes["report.json"], "status": report["status"], "optimizer_steps": 40}):
        raise ValueError("historical completion/provenance mismatch")
    for name, expected in report["provenance"]["files_sha256"].items():
        if u.sha256_file(a.telemetry._safe_path(ROOT / name, ROOT)) != expected:
            raise ValueError("historical source drift: " + name)
    if report["encoding_sha256"] != a.digest({k: [v.input_ids, v.labels] for k, v in a.examples(frozen["evaluation_rows"]).items()}):
        raise ValueError("historical encoding drift")
    for arm, ar in report["arms"].items():
        if ar["initial_parameter_sha256"] != frozen["accepted_parameter_sha256"]:
            raise ValueError("historical initialization mismatch")
        u.assert_same(frozen["acquired_evaluation"]["rows"], ar["before"]["rows"])
        if json.loads((directory / (arm + "-before.json")).read_text()) != ar["before"]:
            raise ValueError("historical baseline mismatch")
        expected_steps = [0] if arm == "noop" else [5, 10, 15, 20]
        if [ev["step"] for ev in ar["evaluations"]] != expected_steps:
            raise ValueError("historical evaluation schedule mismatch")
        for ev in ar["evaluations"]:
            if arm != "noop" and json.loads((directory / f"{arm}-step-{ev['step']:03d}.json").read_text()) != ev:
                raise ValueError("historical evaluation mismatch")
        for cp in ar["checkpoints"]:
            branch = directory / arm / cp["directory"]
            marker = {**cp, "files_sha256": {k: v for k, v in cp["files_sha256"].items() if k != "COMPLETE.json"}}
            if json.loads((branch / "COMPLETE.json").read_text()) != marker:
                raise ValueError("historical checkpoint marker mismatch")
            for name, expected in cp["files_sha256"].items():
                if hashes[str((branch / name).relative_to(directory))] != expected:
                    raise ValueError("historical checkpoint hash mismatch")
            metrics = json.loads((branch / "metrics.json").read_text())
            if (metrics["parameter_sha256"] != ar["final_parameter_sha256"]
                    or metrics["step"] != ar["optimizer_steps"]
                    or metrics["evaluation"] != ar["evaluations"][-1]["rows"]):
                raise ValueError("historical checkpoint metrics mismatch")
    noop = report["arms"]["noop"]
    if (noop["final_parameter_sha256"] != frozen["accepted_parameter_sha256"]
            or noop["before"] != {k: v for k, v in noop["evaluations"][-1].items() if k != "step"}):
        raise ValueError("historical no-op drift")
    old, _, _, _ = u.frozen_inputs()
    if u.decide(report["arms"], old) != report["result"]:
        raise ValueError("historical decision mismatch")
    return report


def authorize(enabled, authorization):
    if enabled is not True or authorization != AUTHORIZATION:
        raise ValueError("NEW explicit Task 79 authorization required; old tokens do not authorize gamma=10")


def new_run_path(path):
    path = a.telemetry._safe_path(Path(path), a.RUNS)
    if path != OUTPUT or path.exists():
        raise ValueError("requires unused stronger-forgetting-task79-001; no overwrite/resume/retry")
    u.require_ignored(path)
    return path


def update(model, groups, profile, monitor, report, ar, *, enabled, authorization):
    """Task 79 authorization and fresh 20-attempt budget; same Task 70 SGD math."""
    authorize(enabled, authorization)
    if monitor.mode != "task79_stronger_forgetting":
        raise RuntimeError("missing controlled CPU admission")
    for r in (report, ar):
        if (type(r["optimizer_steps"]) is not int or type(r["optimizer_step_attempts"]) is not int
                or not 0 <= r["optimizer_steps"] == r["optimizer_step_attempts"] < 20
                or r["possible_partial_update"]):
            raise RuntimeError("step limit/partial attempt; no retry")
    import torch
    optimizer = ar["_optimizer"]
    u.validate_optimizer(optimizer, model, profile)
    try:
        losses = u.backward_objective(model, groups, "gradient_difference", profile, monitor)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), profile["gradient_clip"], foreach=False, error_if_nonfinite=True))
        if not 0 < norm <= profile["limits"]["max_gradient_norm_before_clip"]:
            raise FloatingPointError("zero/excessive gradient norm")
        u.validate_gradients(model, torch, device="cpu")
        authorize(enabled, authorization)
        u.validate_optimizer(optimizer, model, profile)
        monitor.boundary("immediately_before_optimizer")
        for r in (report, ar):
            r["optimizer_step_attempts"] += 1
            r["possible_partial_update"] = True
        optimizer.step()
        for r in (report, ar):
            r["optimizer_steps"] += 1
            r["possible_partial_update"] = False
        monitor.boundary("after_optimizer")
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite post-update parameter")
        return {**losses, "gradient_norm_before_clip": norm, "step": ar["optimizer_steps"]}
    finally:
        model.zero_grad(set_to_none=True)


def comparison(history, fresh, profile):
    arms = {**history["arms"], "gradient_difference": fresh}
    result = u.decide(arms, profile)  # Identical preregistered gate, with reused no-op/retain-only.
    all_arms = {"task70_" + k: v for k, v in history["arms"].items()}
    all_arms["task79_gamma10"] = fresh
    result["comparison"] = u.paired_deltas(all_arms)
    old = {r["id"]: r for r in history["arms"]["gradient_difference"]["evaluations"][-1]["rows"]}
    result["gamma10_minus_gamma1"] = [{
        "id": r["id"], "exact_gamma1": old[r["id"]]["exact_recovery"], "exact_gamma10": r["exact_recovery"],
        **{k + "_difference": r[k] - old[r["id"]][k] for k in (
            "answer_byte_nll", "answer_nll_including_eos", "eos_nll",
            "target_byte_sequence_probability_teacher_forced")},
        "margin_difference": r["correct_over_wrong_nll_margin"] - old[r["id"]]["correct_over_wrong_nll_margin"]
        if "correct_over_wrong_nll_margin" in r else None} for r in fresh["evaluations"][-1]["rows"]]
    result["sentinels_by_arm"] = {k: {"before": v["before"]["sentinels"], "final": v["evaluations"][-1]["sentinels"]}
                                   for k, v in all_arms.items()}
    return result


def verify_new_checkpoint(run_path, ar, frozen):
    if len(ar["checkpoints"]) != 1:
        raise RuntimeError("requires exactly one final checkpoint")
    cp = ar["checkpoints"][0]
    branch = run_path / "gradient_difference" / "checkpoint-000020"
    if (cp["directory"] != branch.name or cp["step"] != 20
            or cp["parameter_sha256"] != ar["final_parameter_sha256"]):
        raise RuntimeError("new checkpoint metadata mismatch")
    marker = {**cp, "files_sha256": {k: v for k, v in cp["files_sha256"].items() if k != "COMPLETE.json"}}
    if json.loads((branch / "COMPLETE.json").read_text()) != marker:
        raise RuntimeError("new checkpoint COMPLETE mismatch")
    if {p.name for p in branch.iterdir()} != set(cp["files_sha256"]):
        raise RuntimeError("new checkpoint inventory mismatch")
    for name, expected in cp["files_sha256"].items():
        if u.sha256_file(a.telemetry._safe_path(branch / name, branch)) != expected:
            raise RuntimeError("new checkpoint file hash mismatch")
    if cp["files_sha256"]["config.json"] != frozen["accepted_checkpoint"]["files_sha256"]["config.json"]:
        raise RuntimeError("new checkpoint model configuration drift")
    metrics = json.loads((branch / "metrics.json").read_text())
    if metrics != {"step": 20, "evaluation": ar["evaluations"][-1]["rows"],
                   "parameter_sha256": ar["final_parameter_sha256"]}:
        raise RuntimeError("new checkpoint evaluation mismatch")


def run(*, enabled, authorization, run_path):
    authorize(enabled, authorization)  # before files, model imports or telemetry
    profile, frozen, spec, rows = frozen_inputs()
    source = u.verify_acquisition(frozen)
    history = verify_history(profile, frozen)
    run_path = new_run_path(run_path)
    a.environment()
    run_path.mkdir()  # fixed independent destination, exclusive reservation
    report = {"status": "RUNNING", "scope": "single_gamma10_comparison_not_privacy_evidence",
              "training_authorized": True, "optimizer_steps": 0, "optimizer_step_attempts": 0,
              "possible_partial_update": False, "stop_reason": None, "arms": {},
              "profile_sha256": PROFILE_SHA256, "manifest_sha256": profile["frozen_manifest_sha256"],
              "arm_origins": {"task70_" + arm: {"measurement": "reused", "directory": profile["followup"]["historical_directory"],
                  "report_sha256": profile["followup"]["historical_report_sha256"]} for arm in history["arms"]}}
    report["arm_origins"]["task79_gamma10"] = {"measurement": "fresh", "directory": str(run_path.relative_to(ROOT))}
    u.seal_json(run_path / "RUNNING.json", {"authorization_scope": AUTHORIZATION, "max_optimizer_steps": 20})
    monitor, model = u.Monitor(), None
    monitor.mode = "task79_stronger_forgetting"
    ar = {"optimizer_steps": 0, "optimizer_step_attempts": 0, "possible_partial_update": False,
          "evaluations": [], "training": [], "checkpoints": []}
    report["arms"]["gradient_difference"] = ar
    try:
        from src.small_model_lora import provenance
        report["provenance"] = provenance()
        if (report["provenance"]["versions"] != frozen["versions"] or platform.python_version() != frozen["python"]
                or report["provenance"]["installed_llama_source_sha256"] != frozen["installed_llama_source_sha256"]):
            raise RuntimeError("acquisition runtime/Llama source drift")
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["memory_policy"] = a.memory_policy()
        encoded = a.examples(rows)
        report["encoding_sha256"] = a.digest({k: [v.input_ids, v.labels] for k, v in encoded.items()})
        monitor.capture("preload", before_load=True)
        if shutil.disk_usage(ROOT).free < profile["limits"]["minimum_disk_free_bytes"]:
            raise RuntimeError("requires 4 GiB free disk")
        import torch
        from safetensors.torch import load_model
        monitor.capture("before_initialization", before_load=True)
        monitor.start()
        model = a.initialize(spec)
        load_model(model, str(source), strict=True, device="cpu")
        a.validate_model(model, spec)
        ar["initial_parameter_sha256"] = u.parameter_fingerprint(model)
        if ar["initial_parameter_sha256"] != frozen["accepted_parameter_sha256"]:
            raise RuntimeError("loaded accepted parameter fingerprint mismatch")
        start = time.monotonic()
        before, references = u.measure(model, rows, encoded, profile, monitor)
        # Fresh baseline distributions from identical M_all; no gamma=1 final weights loaded.
        u.assert_same(frozen["acquired_evaluation"]["rows"], before["rows"])
        for old in history["arms"].values():
            u.assert_same(old["before"]["rows"], before["rows"])
            if (abs(before["sentinels"]["mean_nll"] - old["before"]["sentinels"]["mean_nll"]) > 1e-5
                    or before["sentinels"]["mean_kl_nats"] > 1e-5):
                raise RuntimeError("historical sentinel baseline mismatch")
        ar["before"] = before
        u.seal_json(run_path / "gradient_difference-before.json", before)
        after, _ = u.measure(model, rows, encoded, profile, monitor, references)
        u.assert_same(before["rows"], after["rows"])
        if before["sentinels"] != after["sentinels"] or u.parameter_fingerprint(model) != ar["initial_parameter_sha256"]:
            raise RuntimeError("fresh baseline repeat mutated weights or measurements")
        u.seal_json(run_path / "baseline-repeat.json", after)
        max_eval_time = time.monotonic() - start  # conservative two-pass allowance
        directs = {r["fact_id"]: r for r in rows if r["kind"] == "direct"}

        def groups_for(step):
            return {role: [a.batch_for(encoded[directs[k]["id"]]) for k in u.acquisition_order(
                frozen["selection"][role + "_ids"], step, seed=profile["seed"])] for role in ("forget", "retain")}

        start = time.monotonic()
        try:
            u.backward_objective(model, groups_for(0), "gradient_difference", profile, monitor)
        finally:
            model.zero_grad(set_to_none=True)
        times = [time.monotonic() - start]
        if u.parameter_fingerprint(model) != ar["initial_parameter_sha256"]:
            raise RuntimeError("backward rehearsal mutated weights")
        u.projected_budget(monitor, profile, report, times, max_eval_time)
        authorize(enabled, authorization)
        ar["_optimizer"] = torch.optim.SGD(model.parameters(), **{k: v for k, v in profile["optimizer"].items() if k != "name"})
        for step in range(1, 21):
            a.validate_model(model, spec)
            start = time.monotonic()
            record = update(model, groups_for(step), profile, monitor, report, ar, enabled=enabled, authorization=authorization)
            times.append(time.monotonic() - start)
            ar["training"].append({**record, "seconds": times[-1], "order": {
                role: u.acquisition_order(frozen["selection"][role + "_ids"], step, seed=profile["seed"])
                for role in ("forget", "retain")}})
            u.projected_budget(monitor, profile, report, times, max_eval_time)
            if step in profile["evaluation_steps"]:
                start = time.monotonic()
                evaluation, _ = u.measure(model, rows, encoded, profile, monitor, references)
                max_eval_time = max(max_eval_time, time.monotonic() - start)
                ar["evaluations"].append({"step": step, **evaluation})
                u.seal_json(run_path / f"gradient_difference-step-{step:03d}.json", ar["evaluations"][-1])
                u.safety_metrics(evaluation, before, profile)
                print(json.dumps({"arm": "task79_gamma10", "step": step, "summary": evaluation["summary"]}), flush=True)
        ar.pop("_optimizer")
        ar["final_parameter_sha256"] = u.parameter_fingerprint(model)
        if ar["final_parameter_sha256"] == ar["initial_parameter_sha256"]:
            raise RuntimeError("updates did not change parameter fingerprint")
        branch = run_path / "gradient_difference"
        branch.mkdir()
        target = a.checkpoint(model, branch, 20, ar["evaluations"][-1]["rows"], ar, monitor)
        load_model(model, str(target), strict=True, device="cpu")
        if u.parameter_fingerprint(model) != ar["final_parameter_sha256"]:
            raise RuntimeError("new checkpoint round-trip mismatch")
        verify_new_checkpoint(run_path, ar, frozen)
        report["result"] = comparison(history, ar, profile)
        report["status"] = "stronger_forgetting_comparison_complete"
    except BaseException as exc:
        report.update(status="STOP", stop_reason=f"{type(exc).__name__}: {exc}")
    finally:
        ar.pop("_optimizer", None)
        if model is not None:
            model.zero_grad(set_to_none=True)
        del model
        gc.collect()
        try:
            monitor.close()
            u.verify_acquisition(frozen)
            verify_history(profile, frozen)
            frozen_inputs()
            for name, expected in report.get("provenance", {}).get("files_sha256", {}).items():
                if u.sha256_file(ROOT / name) != expected:
                    raise RuntimeError("runner/source changed during experiment")
            if report["status"] == "stronger_forgetting_comparison_complete":
                verify_new_checkpoint(run_path, ar, frozen)
                report["integrity_verified"] = True
        except BaseException as exc:
            report.update(status="STOP", stop_reason=f"cleanup/integrity: {exc}")
        report["result_valid"] = report["status"] == "stronger_forgetting_comparison_complete"
        report["measurements"] = monitor.samples
        report["elapsed_seconds"] = time.monotonic() - monitor.started
        h = u.seal_json(run_path / "report.json", report)
        u.seal_json(run_path / ("COMPLETE.json" if report["status"] == "stronger_forgetting_comparison_complete" else "FAILED.json"),
                    {"report_sha256": h, "status": report["status"], "optimizer_steps": report["optimizer_steps"]})
        from src.small_model_lora import fsync_directory
        fsync_directory(run_path)
        fsync_directory(run_path.parent)
    return report


def dry_run():
    """Four-token random fixture only: no target checkpoint, facts, optimizer or files."""
    import torch
    profile, _, spec, _ = frozen_inputs()
    model = a.initialize(spec)
    before = u.parameter_fingerprint(model)

    class FixtureMonitor:
        def boundary(self, stage):
            pass

    def batch(values):
        ids = torch.tensor([values], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    groups = {"forget": [batch([1, 3, 4, 5])], "retain": [batch([6, 7, 8, 9])]}
    try:
        losses = u.backward_objective(model, groups, "gradient_difference", profile, FixtureMonitor())
    finally:
        model.zero_grad(set_to_none=True)
    if u.parameter_fingerprint(model) != before:
        raise RuntimeError("fixture weights changed")
    return {"status": "synthetic_fixture_dry_run_not_unlearning", "gamma": profile["gamma"], "alpha": profile["alpha"],
            "objective": losses, "optimizer_steps": 0, "checkpoint_writes": 0, "weights_unchanged": True,
            "acquired_checkpoint_loaded": False, "target_facts_executed": False,
            "training_authorized": False, "memory_admission_measured": False}


def plan():
    profile, frozen, _, rows = frozen_inputs()
    checkpoint = u.verify_acquisition(frozen)
    history = verify_history(profile, frozen)
    return {"status": "preparation_only", "acquisition_verified": True, "history_verified": True,
            "source_checkpoint": str(checkpoint.relative_to(ROOT)), "selection": frozen["selection"],
            "evaluation_cases": len(rows), "gamma": profile["gamma"], "alpha": profile["alpha"],
            "fresh_arms": profile["followup"]["fresh_arms"], "reused_arms": profile["followup"]["reused_arms"],
            "historical_report_sha256": profile["followup"]["historical_report_sha256"],
            "historical_gate": history["result"]["selective_direct_forgetting_gate"],
            "total_fresh_optimizer_step_limit": 20, "optimizer_steps": 0,
            "training_authorized": False, "model_loaded": False, "privacy_protection_established": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run-stronger-forgetting", action="store_true")
    parser.add_argument("--authorize-weight-update")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.run_stronger_forgetting:
        if args.run_dir is None:
            parser.error("--run-dir required")
        result = run(enabled=True, authorization=args.authorize_weight_update, run_path=args.run_dir)
    else:
        if args.authorize_weight_update is not None or args.run_dir is not None:
            parser.error("token/output requires --run-stronger-forgetting")
        result = dry_run() if args.dry_run else plan()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["status"] == "STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
