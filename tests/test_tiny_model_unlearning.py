"""Offline fixtures only. Optimizer calls are trapped/mocked; no acquired-model updates."""
import contextlib
import copy
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src import tiny_model_unlearning as u


class FixtureMonitor:
    mode = "task70_controlled"
    def __init__(self):
        import time
        self.started, self.samples = time.monotonic(), []
    def capture(self, *a, **kw):
        pass
    def boundary(self, *a):
        pass
    def check(self):
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
            self.stack.enter_context(mock.patch.object(cls, "step", side_effect=AssertionError("real optimizer step forbidden")))
        self.profile, self.frozen, self.spec, self.rows = u.frozen_inputs()

    def test_frozen_selection_and_acquisition_integrity(self):
        self.assertEqual(self.frozen["selection"]["forget_ids"], ["dev:00:0", "dev:00:1"])
        self.assertEqual(len(self.frozen["selection"]["retain_ids"]), 6)
        self.assertEqual(u.verify_acquisition(self.frozen).name, "model.safetensors")
        self.assertEqual(len(self.rows), 20)

    def test_plan_does_not_load_or_import_a_model(self):
        with mock.patch.object(u.a, "initialize", side_effect=AssertionError("no model")):
            result = u.plan()
        self.assertFalse(result["model_loaded"])
        self.assertEqual(result["baseline"]["direct:forget"]["recovered"], 2)
        self.assertEqual(result["baseline"]["paraphrase:all"]["recovered"], 0)
        command = "import sys; from src.tiny_model_unlearning import plan; plan(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", command], cwd=u.ROOT, check=True, capture_output=True, timeout=30)

    def test_profile_hash_and_manifest_hash_fail_closed(self):
        with mock.patch.object(u, "sha256_file", return_value="bad"):
            with self.assertRaisesRegex(ValueError, "profile drift"):
                u.frozen_inputs()
        original = u.sha256_file
        with mock.patch.object(u, "sha256_file", side_effect=lambda p: original(p) if p == u.PROFILE else "bad"):
            with self.assertRaisesRegex(ValueError, "manifest drift"):
                u.frozen_inputs()

    def test_original_hash_drift_refused(self):
        with mock.patch.object(u, "sha256_file", return_value="bad"):
            with self.assertRaisesRegex(ValueError, "artifact drift"):
                u.verify_acquisition(self.frozen)

    def test_acquisition_token_cannot_authorize_unlearning(self):
        for enabled, token in [(False, u.AUTHORIZATION), (True, None), (True, u.a.AUTHORIZATION)]:
            with self.subTest(enabled=enabled, token=token):
                with mock.patch.object(u, "frozen_inputs", side_effect=AssertionError("too late")) as inputs:
                    with self.assertRaises(ValueError):
                        u.run(enabled=enabled, authorization=token, run_path=Path("unused"))
                    inputs.assert_not_called()

    def test_cli_rejects_mixed_scope(self):
        for args in [["--dry-run", "--authorize-weight-update", u.AUTHORIZATION], ["--run-dir", "unused"], ["--run-controlled-experiment"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                u.main(args)

    def test_paths_cannot_touch_acquisition_or_escape(self):
        for name in ["acquisition-task64-001", "../outside", "unlearning-task70-../bad"]:
            with self.assertRaises(ValueError):
                u.new_run_path(u.a.RUNS / name)
        with tempfile.TemporaryDirectory(prefix="unlearning-task70-", dir=u.a.RUNS) as directory:
            with self.assertRaises(ValueError):
                u.new_run_path(Path(directory))
        with tempfile.TemporaryDirectory(dir=u.a.RUNS) as directory:
            link = Path(directory) / "unlearning-task70-symlink"
            link.symlink_to(u.a.RUNS, target_is_directory=True)
            with self.assertRaises(ValueError):
                u.new_run_path(link)

    def test_exact_eos_metric_and_format_are_distinct(self):
        self.assertFalse(u.a.recovery_metrics(list(b"abcdefghijkl"), "length", "abcdefghijkl")["exact_recovery"])
        metric = u.a.recovery_metrics(list(b"abcdefghijkl"), "eos", "zzzzzzzzzzzz")
        self.assertTrue(metric["valid_answer"])
        self.assertFalse(metric["exact_recovery"])

    def test_noop_drift_and_coverage_refused(self):
        rows = self.frozen["acquired_evaluation"]["rows"]
        u.assert_same(rows, copy.deepcopy(rows))
        for key, value in [("generated_ids", [256]), ("answer_byte_nll", float("nan")), ("eos_nll", 1.0)]:
            other = copy.deepcopy(rows)
            other[0][key] = value
            with self.assertRaises(RuntimeError):
                u.assert_same(rows, other)
        with self.assertRaises(RuntimeError):
            u.assert_same(rows, rows[:-1])

    def test_random_fixture_dry_run_never_creates_optimizer_or_loads_checkpoint(self):
        import safetensors.torch
        with mock.patch.object(torch.optim, "SGD", side_effect=AssertionError("forbidden")), \
                mock.patch.object(safetensors.torch, "load_model", side_effect=AssertionError("forbidden")), \
                mock.patch.object(u, "verify_acquisition", side_effect=AssertionError("no target checkpoint")):
            result = u.dry_run()
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(result["weights_unchanged"])
        self.assertFalse(result["target_facts_executed"])

    def test_sequential_gradient_is_exact_difference_of_group_means(self):
        model = torch.nn.Linear(2, 1, bias=False)
        groups = {"forget": [torch.tensor([[1., 2.]]), torch.tensor([[3., 1.]])],
                  "retain": [torch.tensor([[2., 4.]]), torch.tensor([[4., 2.]]), torch.tensor([[1., 3.]])]}
        forward = lambda m, x: type("Output", (), {"loss": m(x).square().mean()})()
        with mock.patch.object(u.a, "checked_forward", side_effect=forward):
            u.backward_objective(model, groups, "gradient_difference", self.profile, FixtureMonitor())
        actual = model.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        objective = -sum(model(x).square().mean() for x in groups["forget"]) / 2 + sum(model(x).square().mean() for x in groups["retain"]) / 3
        objective.backward()
        self.assertTrue(torch.allclose(actual, model.weight.grad, atol=1e-6))

    def test_retain_only_has_no_forget_forward(self):
        model = torch.nn.Linear(2, 1, bias=False)
        retain = torch.tensor([[1., 3.]])
        with mock.patch.object(u.a, "checked_forward", side_effect=lambda m, x: type("O", (), {"loss": m(x).square().mean()})()) as f:
            losses = u.backward_objective(model, {"forget": ["forbidden"], "retain": [retain]}, "retain_only", self.profile, FixtureMonitor())
        self.assertEqual(f.call_count, 1)
        self.assertNotIn("forget_nll", losses)
        with self.assertRaises(ValueError):
            u.backward_objective(model, {}, "noop", self.profile, FixtureMonitor())

    def test_sentinel_fixture_is_readonly_and_detects_full_vocabulary_drift(self):
        model = u.a.initialize(self.spec)
        before = u.parameter_fingerprint(model)
        baseline, reference = u.sentinel_evaluation(model, ["Fixture only."], FixtureMonitor())
        same, _ = u.sentinel_evaluation(model, ["Fixture only."], FixtureMonitor(), reference)
        self.assertEqual(same, baseline)
        self.assertEqual(before, u.parameter_fingerprint(model))
        with torch.no_grad():
            model.get_input_embeddings().weight[0].add_(1.0)
        changed, _ = u.sentinel_evaluation(model, ["Fixture only."], FixtureMonitor(), reference)
        self.assertGreater(changed["mean_kl_nats"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(changed["mean_nll"])))

    def fixture_update(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), **{k: v for k, v in self.profile["optimizer"].items() if k != "name"})
        report = {"optimizer_steps": 0, "optimizer_step_attempts": 0, "possible_partial_update": False}
        arm = {**report, "_optimizer": optimizer}
        def backward(*args):
            model.weight.grad = torch.ones_like(model.weight)
            return {"objective": 1.0}
        return model, optimizer, report, arm, backward

    def invoke_update(self, model, report, arm, **kwargs):
        return u.update(model, {}, "gradient_difference", self.profile, FixtureMonitor(), report, arm,
                        enabled=True, authorization=u.AUTHORIZATION, **kwargs)

    def test_fixture_step_is_mocked_and_counters_are_bounded(self):
        model, optimizer, report, arm, backward = self.fixture_update()
        before = model.weight.detach().clone()
        with mock.patch.object(u, "backward_objective", side_effect=backward), mock.patch.object(optimizer, "step") as step:
            self.invoke_update(model, report, arm)
            step.assert_called_once()
        self.assertTrue(torch.equal(before, model.weight))
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertIsNone(model.weight.grad)
        for field, value in [("optimizer_steps", 40), ("optimizer_step_attempts", 2), ("possible_partial_update", True)]:
            bad = {**report, field: value}
            with mock.patch.object(optimizer, "step") as step, self.assertRaises(RuntimeError):
                self.invoke_update(model, bad, arm)
            step.assert_not_called()

    def test_partial_step_failure_is_never_retried(self):
        model, optimizer, report, arm, backward = self.fixture_update()
        with mock.patch.object(u, "backward_objective", side_effect=backward), \
                mock.patch.object(optimizer, "step", side_effect=RuntimeError("fixture partial attempt")):
            with self.assertRaisesRegex(RuntimeError, "partial attempt"):
                self.invoke_update(model, report, arm)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["optimizer_step_attempts"], 1)
        self.assertTrue(report["possible_partial_update"])
        self.assertIsNone(model.weight.grad)
        with self.assertRaises(RuntimeError):
            self.invoke_update(model, report, arm)

    def test_optimizer_drift_and_nonfinite_gradient_block_step(self):
        model, optimizer, report, arm, backward = self.fixture_update()
        optimizer.param_groups[0]["lr"] = 1.0
        with mock.patch.object(optimizer, "step") as step, self.assertRaises(RuntimeError):
            self.invoke_update(model, report, arm)
        step.assert_not_called()
        optimizer.param_groups[0]["lr"] = 0.05
        def nonfinite(*args):
            model.weight.grad = torch.full_like(model.weight, float("nan"))
            return {}
        with mock.patch.object(u, "backward_objective", side_effect=nonfinite), mock.patch.object(optimizer, "step") as step:
            with self.assertRaises(RuntimeError):
                self.invoke_update(model, report, arm)
        step.assert_not_called()
        self.assertEqual(report["optimizer_step_attempts"], 0)
        self.assertIsNone(model.weight.grad)

    def test_monitor_failure_prevents_optimizer(self):
        model, optimizer, report, arm, backward = self.fixture_update()
        with mock.patch.object(u, "backward_objective", side_effect=backward), \
                mock.patch.object(FixtureMonitor, "boundary", side_effect=RuntimeError("memory STOP")), \
                mock.patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(RuntimeError, "memory STOP"):
                self.invoke_update(model, report, arm)
        step.assert_not_called()
        self.assertEqual(report["optimizer_step_attempts"], 0)

    def test_projection_and_deadline_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "projected"):
            u.projected_budget(FixtureMonitor(), self.profile, {"optimizer_steps": 0}, [100], 10)
        monitor = u.Monitor()
        monitor.started -= 601
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            monitor.check()

    def metric_fixture(self):
        return {"rows": copy.deepcopy(self.frozen["acquired_evaluation"]["rows"]),
                "sentinels": {"mean_nll": 8.0, "mean_kl_nats": 0.0}}

    def test_stop_on_retain_collapse_runaway_and_global_degradation(self):
        base = self.metric_fixture()
        u.safety_metrics(base, base, self.profile)
        for change in ("retain", "forget", "sentinel", "nan"):
            after = copy.deepcopy(base)
            if change == "retain": after["rows"][4]["exact_recovery"] = False
            if change == "forget": after["rows"][0]["answer_byte_nll"] = 21.0
            if change == "sentinel": after["sentinels"]["mean_kl_nats"] = 0.6
            if change == "nan": after["rows"][0]["eos_nll"] = float("nan")
            with self.subTest(change=change), self.assertRaises((RuntimeError, FloatingPointError)):
                u.safety_metrics(after, base, self.profile)

    def test_acceptance_requires_controls_and_both_facts_not_only_generation(self):
        base = self.metric_fixture()
        arms = {arm: {"before": copy.deepcopy(base), "evaluations": [{"step": 0 if arm == "noop" else 20, **copy.deepcopy(base)}],
                      "optimizer_steps": 0 if arm == "noop" else 20, "optimizer_step_attempts": 0 if arm == "noop" else 20,
                      "possible_partial_update": False} for arm in self.profile["arms"]}
        self.assertFalse(u.decide(arms, self.profile)["selective_direct_forgetting_gate"])
        final = arms["gradient_difference"]["evaluations"][-1]
        for r in final["rows"]:
            if r["kind"] == "direct" and r["role"] == "forget":
                r.update(exact_recovery=False, answer_byte_nll=r["answer_byte_nll"] + 1.1, correct_over_wrong_nll_margin=-0.1)
        result = u.decide(arms, self.profile)
        self.assertTrue(result["selective_direct_forgetting_gate"])
        self.assertFalse(result["paraphrase_effect_identifiable"])
        self.assertFalse(result["privacy_protection_established"])
        arms["retain_only"]["evaluations"][-1] = copy.deepcopy(final)
        self.assertFalse(u.decide(arms, self.profile)["selective_direct_forgetting_gate"])
        arms["retain_only"]["optimizer_steps"] = 19
        with self.assertRaisesRegex(RuntimeError, "incomplete control"):
            u.decide(arms, self.profile)


class OrchestrationTests(unittest.TestCase):
    def test_entire_suite_with_synthetic_evaluations_and_mocked_steps(self):
        self.run_fixture()

    def test_failed_fixture_seals_failure_never_complete(self):
        self.run_fixture(fail=True)

    def run_fixture(self, fail=False):
        profile, frozen, spec, rows = u.frozen_inputs()
        model = torch.nn.Linear(2, 1, bias=False)
        frozen = copy.deepcopy(frozen)
        frozen["accepted_parameter_sha256"] = u.parameter_fingerprint(model)
        # All strings/IDs below are synthetic fixture evaluations, no target forwards.
        evaluation = [{"id": f"fixture-{i}", "kind": "direct", "role": "forget" if i < 2 else "retain",
                       "generated_ids": [256], "exact_recovery": False if i < 2 else True,
                       "valid_answer": True, "finish_reason": "eos", "answer_byte_nll": 1.0,
                       "answer_nll_including_eos": 1.0, "eos_nll": 1.0, "correct_over_wrong_nll_margin": 1.0} for i in range(8)]
        frozen["acquired_evaluation"]["rows"] = evaluation
        measured = {"rows": evaluation, "summary": {}, "sentinels": {"mean_nll": 8.0, "mean_kl_nats": 0.0}}
        def checkpoint(model, path, step, evaluation, report, monitor):
            report["checkpoints"].append({"fixture": True})
        def update(model, groups, arm, profile, monitor, report, ar, **kw):
            if fail: raise RuntimeError("synthetic orchestration failure")
            for r in (report, ar):
                r["optimizer_steps"] += 1
                r["optimizer_step_attempts"] += 1
            return {"step": ar["optimizer_steps"]}
        import safetensors.torch
        from src import small_model_lora
        with tempfile.TemporaryDirectory(dir=u.a.RUNS) as directory, contextlib.ExitStack() as stack:
            path = Path(directory) / "unlearning-task70-fixture"
            patches = [(u, "frozen_inputs", lambda: (profile, frozen, spec, rows)),
                       (u, "verify_acquisition", lambda f: Path("fixture-only")), (u, "Monitor", FixtureMonitor),
                       (u.a, "environment", lambda: None), (u.a, "initialize", lambda s: model),
                       (u.a, "validate_model", lambda *a: None), (u.a, "checkpoint", checkpoint),
                       (u, "measure", lambda *a: (copy.deepcopy(measured), [])),
                       (u, "backward_objective", lambda *a: {}), (u, "update", update),
                       (u, "projected_budget", lambda *a: None), (u, "new_run_path", lambda p: p),
                       (safetensors.torch, "load_model", lambda *a, **kw: None),
                       (small_model_lora, "provenance", lambda: {"versions": frozen["versions"],
                        "installed_llama_source_sha256": frozen["installed_llama_source_sha256"]})]
            for obj, key, value in patches: stack.enter_context(mock.patch.object(obj, key, value))
            stack.enter_context(mock.patch.object(torch.optim.SGD, "step", side_effect=AssertionError("no real steps")))
            with contextlib.redirect_stdout(io.StringIO()):
                report = u.run(enabled=True, authorization=u.AUTHORIZATION, run_path=path)
            self.assertEqual(report["status"], "STOP" if fail else "controlled_pilot_complete", report["stop_reason"])
            self.assertEqual(report["optimizer_steps"], 0 if fail else 40)
            self.assertTrue((path / ("FAILED.json" if fail else "COMPLETE.json")).exists())
            self.assertNotIn("_optimizer", report["arms"]["retain_only"])
            self.assertEqual(report["arms"]["noop"]["optimizer_steps"], 0)
            if fail: self.assertFalse((path / "COMPLETE.json").exists())
            else:
                self.assertFalse(report["result"]["selective_direct_forgetting_gate"])
                self.assertEqual(len(report["arms"]["gradient_difference"]["evaluations"]), 4)


if __name__ == "__main__":
    unittest.main()
