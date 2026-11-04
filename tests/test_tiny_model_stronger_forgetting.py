"""Offline fixtures: all optimizer steps trapped or mocked; no acquired-model execution."""
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from src import tiny_model_stronger_forgetting as s

u = s.u


class FixtureMonitor:
    mode = "task79_stronger_forgetting"

    def __init__(self):
        self.started, self.samples = time.monotonic(), []

    def boundary(self, *args):
        pass

    def capture(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def close(self):
        pass


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for cls in (torch.optim.SGD, torch.optim.AdamW):
            self.stack.enter_context(mock.patch.object(cls, "step", side_effect=AssertionError("real step forbidden")))
        self.profile, self.frozen, self.spec, self.rows = s.frozen_inputs()

    def test_only_gamma_and_fresh_execution_budget_change(self):
        old, _, _, _ = u.frozen_inputs()
        for key in old:
            self.assertEqual(self.profile[key], {"gamma": 10.0, "total_optimizer_step_limit": 20}.get(key, old[key]), key)
        self.assertEqual(self.profile["followup"]["fresh_arms"], ["gradient_difference"])
        self.assertEqual(self.profile["followup"]["reused_arms"], old["arms"])
        self.assertEqual(self.profile["steps_per_update_arm"], 20)

    def test_plan_hashes_evidence_without_importing_ml_or_loading_weights(self):
        with mock.patch.object(s.a, "initialize", side_effect=AssertionError("no model")):
            result = s.plan()
        self.assertTrue(result["history_verified"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["historical_gate"])
        code = "import sys; from src.tiny_model_stronger_forgetting import plan; plan(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=s.ROOT, check=True, capture_output=True, timeout=30)

    def test_profile_and_history_drift_fail_closed(self):
        with mock.patch.object(u, "sha256_file", return_value="bad"):
            with self.assertRaisesRegex(ValueError, "profile drift"):
                s.frozen_inputs()
            with self.assertRaisesRegex(ValueError, "inventory drift"):
                s.verify_history(self.profile, self.frozen)
        original = u.sha256_file
        def altered(path):
            if str(path).endswith("unlearning-task70-001/retain_only/checkpoint-000020/model.safetensors"):
                return "bad"
            return original(path)
        with mock.patch.object(u, "sha256_file", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "historical artifact drift"):
                s.verify_history(self.profile, self.frozen)

    def test_parameter_drift_rejected_even_if_profile_hash_check_bypassed(self):
        original = Path.read_text
        for field, value in [("alpha", 2.0), ("steps_per_update_arm", 21), ("seed", 2071),
                             ("gradient_clip", 2.0), ("optimizer", {"name": "AdamW"})]:
            altered = {**self.profile, field: value}
            with self.subTest(field=field), mock.patch.object(u, "sha256_file", side_effect=lambda p: s.PROFILE_SHA256 if p == s.PROFILE else u_sha(p)):
                with mock.patch.object(Path, "read_text", lambda path, *a, **kw: json.dumps(altered) if path == s.PROFILE else original(path, *a, **kw)):
                    with self.assertRaisesRegex(ValueError, "only gamma"):
                        s.frozen_inputs()

    def test_new_authorization_precedes_all_input_or_model_work(self):
        for enabled, token in [(False, s.AUTHORIZATION), (True, None), (True, u.AUTHORIZATION), (True, s.a.AUTHORIZATION)]:
            with self.subTest(enabled=enabled, token=token), mock.patch.object(s, "frozen_inputs", side_effect=AssertionError("too late")) as inputs:
                with self.assertRaisesRegex(ValueError, "NEW explicit Task 79"):
                    s.run(enabled=enabled, authorization=token, run_path=s.OUTPUT)
                inputs.assert_not_called()

    def test_cli_disallows_overrides_mixed_scope_and_missing_path(self):
        cases = [["--dry-run", "--authorize-weight-update", s.AUTHORIZATION], ["--run-dir", str(s.OUTPUT)],
                 ["--gamma", "100"], ["--run-stronger-forgetting"], ["--dry-run", "--run-stronger-forgetting"]]
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                s.main(args)

    def test_only_one_unused_independent_output_is_allowed(self):
        self.assertEqual(s.new_run_path(s.OUTPUT), s.OUTPUT)
        for path in [s.a.RUNS / "unlearning-task70-001", s.a.RUNS / "acquisition-task64-001", s.a.RUNS / "stronger-forgetting-task79-002",
                     s.a.RUNS / "../escape", s.a.RUNS / "stronger-forgetting-task79-001/child"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                s.new_run_path(path)
        with mock.patch.object(Path, "exists", return_value=True), self.assertRaises(ValueError):
            s.new_run_path(s.OUTPUT)
        with mock.patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            s.new_run_path(s.OUTPUT)

    def test_dry_run_never_constructs_optimizer_loads_checkpoint_or_executes_facts(self):
        import safetensors.torch
        with mock.patch.object(torch.optim, "SGD", side_effect=AssertionError("no optimizer")), \
                mock.patch.object(safetensors.torch, "load_model", side_effect=AssertionError("no checkpoint")), \
                mock.patch.object(s.a, "evaluate", side_effect=AssertionError("no target facts")), \
                mock.patch.object(u, "seal_json", side_effect=AssertionError("no artifact writes")):
            result = s.dry_run()
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(result["weights_unchanged"])
        self.assertFalse(result["target_facts_executed"])
        self.assertFalse(result["memory_admission_measured"])

    def test_gamma10_gradient_matches_unequal_group_mean_algebra(self):
        model = torch.nn.Linear(2, 1, bias=False)
        groups = {"forget": [torch.tensor([[1., 2.]]), torch.tensor([[3., 1.]])],
                  "retain": [torch.tensor([[2., 4.]]), torch.tensor([[4., 2.]]), torch.tensor([[1., 3.]])]}
        forward = lambda m, x: type("O", (), {"loss": m(x).square().mean()})()
        before = model.weight.detach().clone()
        with mock.patch.object(s.a, "checked_forward", side_effect=forward):
            losses = u.backward_objective(model, groups, "gradient_difference", self.profile, FixtureMonitor())
        actual = model.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        objective = -10 * sum(model(x).square().mean() for x in groups["forget"]) / 2 + sum(model(x).square().mean() for x in groups["retain"]) / 3
        objective.backward()
        self.assertTrue(torch.allclose(actual, model.weight.grad, atol=1e-5))
        self.assertAlmostEqual(losses["objective"], float(objective.detach()), places=4)
        self.assertTrue(torch.equal(before, model.weight))

    def fixture_update(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), **{k: v for k, v in self.profile["optimizer"].items() if k != "name"})
        report = {"optimizer_steps": 0, "optimizer_step_attempts": 0, "possible_partial_update": False}
        ar = {**report, "_optimizer": optimizer}
        def backward(*args):
            model.weight.grad = torch.ones_like(model.weight)
            return {"objective": 1.0}
        return model, optimizer, report, ar, backward

    def invoke(self, model, report, ar, monitor=None, token=s.AUTHORIZATION):
        return s.update(model, {}, self.profile, monitor or FixtureMonitor(), report, ar, enabled=True, authorization=token)

    def test_exactly20_mocked_steps_and_21st_refused(self):
        model, optimizer, report, ar, backward = self.fixture_update()
        before = model.weight.detach().clone()
        with mock.patch.object(u, "backward_objective", side_effect=backward), mock.patch.object(optimizer, "step") as step:
            for _ in range(20):
                self.invoke(model, report, ar)
            with self.assertRaisesRegex(RuntimeError, "step limit"):
                self.invoke(model, report, ar)
        self.assertEqual(step.call_count, 20)
        self.assertEqual(report["optimizer_step_attempts"], 20)
        self.assertEqual(ar["optimizer_steps"], 20)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertIsNone(model.weight.grad)

    def test_failed_step_preserves_partial_attempt_and_cannot_retry(self):
        model, optimizer, report, ar, backward = self.fixture_update()
        with mock.patch.object(u, "backward_objective", side_effect=backward), \
                mock.patch.object(optimizer, "step", side_effect=RuntimeError("fixture partial")) as step:
            with self.assertRaisesRegex(RuntimeError, "fixture partial"):
                self.invoke(model, report, ar)
            with self.assertRaisesRegex(RuntimeError, "partial attempt"):
                self.invoke(model, report, ar)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["optimizer_step_attempts"], 1)
        self.assertTrue(report["possible_partial_update"])
        self.assertIsNone(model.weight.grad)

    def test_stops_before_step_for_drift_norm_telemetry_and_authorization(self):
        for problem in ["lr", "nan", "zero", "large", "telemetry", "token", "mode", "reauthorize"]:
            model, optimizer, report, ar, backward = self.fixture_update()
            monitor = FixtureMonitor()
            token = u.AUTHORIZATION if problem == "token" else s.AUTHORIZATION
            if problem == "lr": optimizer.param_groups[0]["lr"] = 0.5
            if problem == "mode": monitor.mode = "task70_controlled"
            def gradient(*args):
                value = {"nan": float("nan"), "zero": 0., "large": 1000.}.get(problem, 1.)
                model.weight.grad = torch.full_like(model.weight, value)
                return {}
            with self.subTest(problem=problem), contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(u, "backward_objective", side_effect=gradient))
                step = stack.enter_context(mock.patch.object(optimizer, "step"))
                if problem == "telemetry": stack.enter_context(mock.patch.object(monitor, "boundary", side_effect=RuntimeError("memory STOP")))
                if problem == "reauthorize": stack.enter_context(mock.patch.object(s, "authorize", side_effect=[None, ValueError("revoked")]))
                with self.assertRaises((RuntimeError, ValueError, FloatingPointError)):
                    self.invoke(model, report, ar, monitor=monitor, token=token)
                step.assert_not_called()
                self.assertEqual(report["optimizer_step_attempts"], 0)

    def test_unchanged_projection_and_deadline_guards(self):
        with self.assertRaisesRegex(RuntimeError, "projected"):
            u.projected_budget(FixtureMonitor(), self.profile, {"optimizer_steps": 0}, [100], 10)
        monitor = u.Monitor()
        monitor.started -= 601
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            monitor.check()

    def test_new_checkpoint_hash_marker_and_metrics_integrity(self):
        with tempfile.TemporaryDirectory(dir=s.a.RUNS) as directory:
            root = Path(directory)
            branch = root / "gradient_difference" / "checkpoint-000020"
            branch.mkdir(parents=True)
            (branch / "model.safetensors").write_bytes(b"opaque fixture; no tensors")
            (branch / "config.json").write_text('{}\n')
            metrics = {"step": 20, "evaluation": [], "parameter_sha256": "fixture-final"}
            u.seal_json(branch / "metrics.json", metrics)
            cp = {"directory": branch.name, "step": 20, "parameter_sha256": "fixture-final",
                  "files_sha256": {p.name: u.sha256_file(p) for p in branch.iterdir()}}
            cp["files_sha256"]["COMPLETE.json"] = u.seal_json(branch / "COMPLETE.json", cp)
            ar = {"checkpoints": [cp], "final_parameter_sha256": "fixture-final", "evaluations": [{"rows": []}]}
            frozen = {"accepted_checkpoint": {"files_sha256": {"config.json": cp["files_sha256"]["config.json"]}}}
            s.verify_new_checkpoint(root, ar, frozen)
            with self.assertRaisesRegex(RuntimeError, "configuration drift"):
                s.verify_new_checkpoint(root, ar, {"accepted_checkpoint": {"files_sha256": {"config.json": "bad"}}})
            (branch / "model.safetensors").write_bytes(b"corrupted fixture")
            with self.assertRaisesRegex(RuntimeError, "file hash"):
                s.verify_new_checkpoint(root, ar, frozen)
            ar["checkpoints"] = []
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                s.verify_new_checkpoint(root, ar, frozen)

    def test_confidence_change_is_not_forgetting_and_gate_is_fixed(self):
        history = s.verify_history(self.profile, self.frozen)
        fresh = copy.deepcopy(history["arms"]["gradient_difference"])
        final = fresh["evaluations"][-1]
        for row in final["rows"]:
            if row["kind"] == "direct" and row["role"] == "forget":
                row["answer_byte_nll"] += 0.01
        self.assertFalse(s.comparison(history, fresh, self.profile)["selective_direct_forgetting_gate"])
        for row in final["rows"]:
            if row["kind"] == "direct" and row["role"] == "forget":
                row.update(exact_recovery=False, answer_byte_nll=1.2, correct_over_wrong_nll_margin=-0.1)
        result = s.comparison(history, fresh, self.profile)
        self.assertTrue(result["selective_direct_forgetting_gate"])
        self.assertEqual(len(result["comparison"]), 4)
        self.assertEqual(len(result["gamma10_minus_gamma1"]), 20)
        self.assertFalse(result["privacy_protection_established"])
        self.assertFalse(result["paraphrase_effect_identifiable"])
        for change in ["retain", "sentinel_nll", "sentinel_kl", "one_forget", "margin", "small_nll", "incomplete"]:
            bad = copy.deepcopy(fresh)
            ev = bad["evaluations"][-1]
            if change == "retain": ev["rows"][4]["exact_recovery"] = False
            if change == "sentinel_nll": ev["sentinels"]["mean_nll"] += 0.2
            if change == "sentinel_kl": ev["sentinels"]["mean_kl_nats"] = 0.2
            if change == "one_forget": ev["rows"][0]["exact_recovery"] = True
            if change == "margin": ev["rows"][0]["correct_over_wrong_nll_margin"] = 0.1
            if change == "small_nll": ev["rows"][0]["answer_byte_nll"] = 0.02
            if change == "incomplete": bad["optimizer_steps"] = 19
            with self.subTest(change=change):
                if change == "incomplete":
                    with self.assertRaisesRegex(RuntimeError, "incomplete"):
                        s.comparison(history, bad, self.profile)
                else:
                    self.assertFalse(s.comparison(history, bad, self.profile)["selective_direct_forgetting_gate"])


u_sha = u.sha256_file


class OrchestrationTests(unittest.TestCase):
    def test_only_one_fresh_arm_with_synthetic_measurements_and_mocked_steps(self):
        self.run_fixture()

    def test_safety_stop_seals_failure_and_never_continues(self):
        self.run_fixture(failure="safety")

    def test_history_integrity_drift_invalidates_complete_result(self):
        self.run_fixture(failure="integrity")

    def test_reload_mismatch_seals_failure(self):
        self.run_fixture(failure="reload")

    def test_initial_sentinel_drift_prevents_optimizer_creation(self):
        self.run_fixture(failure="baseline")

    def run_fixture(self, failure=None):
        profile, frozen, spec, rows = s.frozen_inputs()
        frozen = copy.deepcopy(frozen)
        model = torch.nn.Linear(2, 1, bias=False)
        original = u.parameter_fingerprint(model)
        frozen["accepted_parameter_sha256"] = original
        # Independent fake facts: never feed project targets through a model.
        evaluation = [{"id": f"fixture-{i}", "kind": "direct", "role": "forget" if i < 2 else "retain",
                       "exact_recovery": True, "generated_ids": [1, 256], "valid_answer": True, "finish_reason": "eos",
                       "answer_byte_nll": 1.0, "answer_nll_including_eos": 1.0, "eos_nll": 1.0,
                       "correct_over_wrong_nll_margin": 1.0, "target_byte_sequence_probability_teacher_forced": 0.001} for i in range(8)]
        frozen["acquired_evaluation"]["rows"] = evaluation
        measured = {"rows": evaluation, "summary": {}, "sentinels": {"mean_nll": 8.0, "mean_kl_nats": 0.0}}
        history = {"arms": {arm: {"before": copy.deepcopy(measured), "evaluations": [{"step": 0 if arm == "noop" else 20, **copy.deepcopy(measured)}],
                   "optimizer_steps": 0 if arm == "noop" else 20, "optimizer_step_attempts": 0 if arm == "noop" else 20,
                   "possible_partial_update": False} for arm in profile["arms"]}}
        report_fixture = {"versions": frozen["versions"], "installed_llama_source_sha256": frozen["installed_llama_source_sha256"]}
        calls = []
        def measure(*args):
            result = copy.deepcopy(measured)
            if failure == "baseline": result["sentinels"]["mean_nll"] += 0.01
            return result, []
        def update(model, groups, profile, monitor, report, ar, **kw):
            calls.append(ar["optimizer_steps"] + 1)
            for r in (report, ar):
                r["optimizer_steps"] += 1
                r["optimizer_step_attempts"] += 1
            # This is a counter-only step mock. No fixture or acquired weights change.
            return {"step": ar["optimizer_steps"]}
        fingerprints = [original, original, original, "fixture-final", "bad-reload" if failure == "reload" else "fixture-final"]
        import safetensors.torch
        from src import small_model_lora
        with tempfile.TemporaryDirectory(dir=s.a.RUNS) as directory, contextlib.ExitStack() as stack:
            path = Path(directory) / "fixture-task79"
            patches = [(s, "frozen_inputs", lambda: (profile, frozen, spec, rows)),
                       (u, "verify_acquisition", lambda f: Path("fixture-checkpoint")),
                       (s, "verify_history", mock.Mock(side_effect=[history, ValueError("history drift")] if failure == "integrity" else [history, history])),
                       (s, "new_run_path", lambda p: p), (s.a, "environment", lambda: None), (u, "Monitor", FixtureMonitor),
                       (s.a, "initialize", lambda *a: model), (s.a, "validate_model", lambda *a: None),
                       (u, "parameter_fingerprint", mock.Mock(side_effect=fingerprints)),
                       (u, "measure", measure), (u, "backward_objective", lambda *a: {}),
                       (u, "projected_budget", lambda *a: None), (s, "update", update),
                       (s.a, "checkpoint", lambda *a: Path("fixture-final")), (s, "verify_new_checkpoint", lambda *a: None),
                       (safetensors.torch, "load_model", lambda *a, **kw: None), (small_model_lora, "provenance", lambda: report_fixture)]
            for obj, name, value in patches:
                stack.enter_context(mock.patch.object(obj, name, value))
            if failure == "safety": stack.enter_context(mock.patch.object(u, "safety_metrics", side_effect=RuntimeError("retain STOP")))
            optimizer = stack.enter_context(mock.patch.object(torch.optim.SGD, "step", side_effect=AssertionError("no real step")))
            if failure == "baseline": stack.enter_context(mock.patch.object(torch.optim, "SGD", side_effect=AssertionError("no optimizer construction")))
            with contextlib.redirect_stdout(io.StringIO()):
                report = s.run(enabled=True, authorization=s.AUTHORIZATION, run_path=path)
            optimizer.assert_not_called()
            self.assertEqual(report["status"], "STOP" if failure else "stronger_forgetting_comparison_complete", report["stop_reason"])
            self.assertEqual(report["optimizer_steps"], {"baseline": 0, "safety": 5}.get(failure, 20))
            self.assertEqual(calls, list(range(1, report["optimizer_steps"] + 1)))
            self.assertEqual(set(report["arms"]), {"gradient_difference"})
            self.assertEqual(len(report["arm_origins"]), 4)
            self.assertNotIn("_optimizer", report["arms"]["gradient_difference"])
            self.assertEqual(report["result_valid"], failure is None)
            self.assertTrue((path / ("FAILED.json" if failure else "COMPLETE.json")).exists())
            if failure: self.assertFalse((path / "COMPLETE.json").exists())
            else:
                self.assertTrue(report["integrity_verified"])
                self.assertFalse(report["result"]["selective_direct_forgetting_gate"])
                self.assertEqual([e["step"] for e in report["arms"]["gradient_difference"]["evaluations"]], [5, 10, 15, 20])


if __name__ == "__main__":
    unittest.main()
