"""Offline gates, CPU tensor diagnostics and mocked updates; no real SGD.step/save."""

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src import small_model_acquisition as a

REAL_MONITOR = a.PilotMonitor


def sample(stage="baseline"):
    return {"stage": stage, "timestamp_ns": time.monotonic_ns() - 1_000_000,
            "sample_duration_ns": 100_000, "total_physical_memory_bytes": 48 * a.GIB,
            "available_physical_memory_lower_bound_bytes": 26 * a.GIB,
            "system_memory_free_percentage": 90, "macos_pressure_level": 1,
            "process_current_rss_bytes": a.GIB, "process_peak_rss_bytes": a.GIB,
            "mps_tensor_allocated_bytes": 100, "mps_driver_allocated_bytes": 200,
            "swap_used_bytes": 0, "pageouts": 0}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(a.PROFILE.read_text())

    def test_manifest_frozen_and_wording_separated(self):
        m = a.frozen_manifest()
        self.assertEqual(len(m["training"]), 32)
        self.assertEqual(len(m["evaluation"]), 76)
        self.assertEqual(m["epoch_zero_order"][0], "dev:09:0")
        self.assertFalse(any("paraphrased_question" in r for r in m["training"]))
        self.assertEqual(sum(r["role"] == "forget" for r in m["training"]), 8)
        self.assertEqual(len(set(m["epoch_zero_order"])), 32)

    def test_default_never_imports_torch_or_constructs_model(self):
        # Fresh interpreter: reject even a transitive Torch/Transformers import.
        code = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('default imported ML runtime')
sys.meta_path.insert(0, Block())
from src.small_model_acquisition import main
raise SystemExit(main([]))
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=a.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "validation_only")
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["checkpoint_writes"], 0)

    def test_authorization_fails_before_any_execution(self):
        for enabled, token in ((False, None), (True, None), (True, "yes"), (1, a.AUTHORIZATION)):
            with self.subTest(enabled=enabled, token=token), mock.patch.object(a, "frozen_manifest") as frozen:
                with self.assertRaises(ValueError):
                    a.run(mode="pilot", enabled=enabled, authorization=token, report_path=a.RUNS / "test.json")
                frozen.assert_not_called()

    def test_cli_cannot_enable_by_token_alone_or_change_budget(self):
        for argv in (["--authorize-weight-update", a.AUTHORIZATION], ["--max-steps", "2"],
                     ["--device", "cpu"], ["--allow-download"], ["--learning-rate", "1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                a.main(argv)
            self.assertEqual(exc.exception.code, 2)

    def test_baseline_rejects_all_update_arguments_before_any_work(self):
        for kwargs in ({"enabled": True}, {"authorization": a.AUTHORIZATION},
                       {"run_path": a.RUNS / "pilot-new"},
                       {"baseline_path": a.RUNS / "baseline.json"}, {"baseline_sha256": "a" * 64}):
            with self.subTest(kwargs=kwargs), mock.patch.object(a, "frozen_manifest") as frozen:
                with self.assertRaises(ValueError):
                    a.run(mode="baseline", report_path=a.RUNS / "never.json", **kwargs)
                frozen.assert_not_called()

    def test_model_specific_policy_no_llama_admission(self):
        p = a.memory_policy(self.profile)
        self.assertEqual(p["preload_available_required_bytes"], 24 * a.GIB)
        self.assertEqual(p["optimizer_state_bytes"], 0)
        self.assertEqual(a.memory_policy(self.profile, 3 * a.GIB)["preload_available_required_bytes"], 26 * a.GIB)
        self.profile["model"]["repo_id"] = "meta-llama/Llama-3.2-1B-Instruct"
        with self.assertRaises(ValueError):
            a.memory_policy(self.profile)

    def test_training_policy_matches_entire_frozen_task38_audit(self):
        audit = json.loads((a.ROOT / "experiments/smollm2_task38_evidence_audit.json").read_text())
        self.assertEqual(a.memory_policy(self.profile), audit["new_policy"])
        for peak in (0, a.GIB, 3 * a.GIB):
            self.assertEqual(a.admission_policy(self.profile, peak, mode="pilot"),
                             a.memory_policy(self.profile, peak))

    def test_baseline_budget_is_estimated_bounded_and_not_training_admission(self):
        p = a.baseline_memory_policy(self.profile)
        self.assertEqual(p["guard_id"], a.BASELINE_GUARD_ID)
        self.assertEqual(p["preload_available_required_bytes"], 18 * a.GIB)
        self.assertEqual(p["estimated_incremental_total_bytes"], 12620658888)
        self.assertEqual(p["max_optimizer_steps"], 0)
        self.assertIsNone(p["baseline_process_peak_rss_bytes"])
        for key in ("backward_permitted", "training_authorized", "measured_model_requirement", "use_cache"):
            self.assertIs(p[key], False)
        self.assertNotIn("fp32_gradients", p["estimated_incremental_terms_bytes"])
        self.assertNotIn("clipping_step_serialization_allowance", p["estimated_incremental_terms_bytes"])
        self.assertEqual(a.baseline_memory_policy(self.profile, 3 * a.GIB)["preload_available_required_bytes"], 20 * a.GIB)
        for peak in (-1, True, 1.0):
            with self.subTest(peak=peak), self.assertRaises(ValueError):
                a.baseline_memory_policy(self.profile, peak)
        with self.assertRaises(ValueError):
            a.admission_policy(self.profile, mode="unknown")
        self.profile["model"]["repo_id"] = "meta-llama/Llama-3.2-1B-Instruct"
        with self.assertRaises(ValueError):
            a.baseline_memory_policy(self.profile)

    def test_exact_preload_continuing_and_cap_boundaries_in_both_modes(self):
        for mode, minimum in (("baseline", 18), ("pilot", 24)):
            for peak, extra in ((a.GIB, 0), (3 * a.GIB, 2)):
                reading = {**sample(), "process_peak_rss_bytes": peak,
                           "available_physical_memory_lower_bound_bytes": (minimum + extra) * a.GIB}
                a.validate_sample(reading, profile=self.profile, mode=mode, before_load=True)
                reading["available_physical_memory_lower_bound_bytes"] -= 1
                with self.subTest(mode=mode, peak=peak), self.assertRaises(RuntimeError):
                    a.validate_sample(reading, profile=self.profile, mode=mode, before_load=True)
            reading = {**sample(), "available_physical_memory_lower_bound_bytes": 6 * a.GIB,
                       "process_peak_rss_bytes": 8 * a.GIB, "mps_driver_allocated_bytes": 8 * a.GIB}
            a.validate_sample(reading, profile=self.profile, mode=mode)
            reading["available_physical_memory_lower_bound_bytes"] -= 1
            with self.assertRaises(RuntimeError):
                a.validate_sample(reading, profile=self.profile, mode=mode)
        reading = {**sample(), "available_physical_memory_lower_bound_bytes": 18 * a.GIB}
        a.validate_sample(reading, profile=self.profile, mode="baseline", before_load=True)
        # Omitting mode must still select the original, stricter training policy.
        with self.assertRaises(RuntimeError):
            a.validate_sample(reading, profile=self.profile, before_load=True)

    def test_baseline_retains_every_common_telemetry_stop(self):
        bad = {"available_physical_memory_lower_bound_bytes": 6 * a.GIB - 1,
               "total_physical_memory_bytes": 48 * a.GIB - 1,
               "macos_pressure_level": 2, "system_memory_free_percentage": 19,
               "process_peak_rss_bytes": 8 * a.GIB + 1,
               "mps_driver_allocated_bytes": 8 * a.GIB + 1,
               "timestamp_ns": 1, "sample_duration_ns": 0,
               "mps_tensor_allocated_bytes": 201, "pageouts": None,
               "swap_used_bytes": 1}
        for mode in ("baseline", "pilot"):
            for key, value in bad.items():
                with self.subTest(mode=mode, key=key), self.assertRaises(RuntimeError):
                    a.validate_sample({**sample(), key: value}, profile=self.profile, mode=mode, first=sample())
            for key in ("pageouts", "swap_used_bytes"):
                with self.assertRaises(RuntimeError):
                    a.validate_sample({**sample(), key: 1}, profile=self.profile, mode=mode, first=sample())
            reading = sample()
            for timestamp in (reading["timestamp_ns"], reading["timestamp_ns"] - 2_000_000_001):
                with self.assertRaises(RuntimeError):
                    a.validate_sample(reading, profile=self.profile, mode=mode, active=True,
                                      previous={**reading, "timestamp_ns": timestamp})
            with self.assertRaises(RuntimeError):
                a.validate_sample({**sample(), "timestamp_ns": time.monotonic_ns() + 1_000_000_000},
                                  profile=self.profile, mode=mode)

    def test_monitor_dispatch_is_explicit_and_unknown_devices_or_modes_stop(self):
        reading = {**sample(), "available_physical_memory_lower_bound_bytes": 18 * a.GIB}
        for mode in ("baseline", "pilot"):
            monitor = a.PilotMonitor(None, profile=self.profile, device="mps", mode=mode)
            with mock.patch.object(a.telemetry, "read_sample", return_value=reading):
                if mode == "baseline":
                    monitor.capture("baseline", before_load=True)
                else:
                    with self.assertRaises(RuntimeError):
                        monitor.capture("baseline", before_load=True)
            self.assertEqual(monitor.samples, [reading])  # retain failing evidence too
            monitor.started -= a.MAX_SECONDS + 1
            with self.assertRaises(RuntimeError):
                monitor.capture("timeout")
            monitor.error = RuntimeError("watchdog failed")
            with self.assertRaises(RuntimeError):
                monitor.check()
        for device, mode in (("cpu", "baseline"), ("mps", "unknown")):
            with self.assertRaises((ValueError, RuntimeError)):
                a.PilotMonitor(None, profile=self.profile, device=device, mode=mode)

    def test_memory_failures(self):
        a.validate_sample(sample(), profile=self.profile, before_load=True)
        bad = {
            "available_physical_memory_lower_bound_bytes": 23 * a.GIB,
            "macos_pressure_level": 2, "system_memory_free_percentage": 19,
            "process_peak_rss_bytes": 8 * a.GIB + 1,
            "mps_driver_allocated_bytes": 8 * a.GIB + 1,
            "sample_duration_ns": 0, "timestamp_ns": 1,
            "total_physical_memory_bytes": 32 * a.GIB,
        }
        for key, value in bad.items():
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                a.validate_sample({**sample(), key: value}, profile=self.profile, before_load=True)
        for key in ("swap_used_bytes", "pageouts"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                a.validate_sample({**sample(), key: 1}, profile=self.profile, first=sample())
        reading = sample()
        old = {**reading, "timestamp_ns": reading["timestamp_ns"] - 3_000_000_000}
        with self.assertRaises(RuntimeError):
            a.validate_sample(reading, profile=self.profile, previous=old, active=True)
        with self.assertRaises(RuntimeError):
            a.validate_sample(reading, profile=self.profile, previous=reading)
        with self.assertRaises(RuntimeError):
            a.validate_sample({k: v for k, v in reading.items() if k != "pageouts"}, profile=self.profile)

    def test_source_and_manifest_drift(self):
        with mock.patch.object(a, "sha256_file", return_value="0" * 64), self.assertRaises(ValueError):
            a.frozen_manifest()
        with mock.patch.object(a, "MANIFEST_SHA256", a.sha256_file(a.MANIFEST)), \
                mock.patch.object(a, "build_manifest", return_value={}), self.assertRaises(ValueError):
            a.frozen_manifest()


class PathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=a.ROOT / "outputs")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patch = mock.patch.object(a, "RUNS", self.root)
        patch.start()
        self.addCleanup(patch.stop)

    def test_paths_cannot_overwrite_original_or_history(self):
        old = self.root / "pilot-old"
        old.mkdir()
        for path in (old, a.telemetry.CHECKPOINT, a.ROOT / "results", self.root / "x" / "pilot-new",
                     self.root / ".." / "pilot-escape"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                a.new_run_path(path)
        self.assertEqual(a.new_run_path(self.root / "pilot-new"), self.root / "pilot-new")
        history = self.root / "baseline.json"
        a.seal_json(history, {"immutable": True})
        with self.assertRaises(FileExistsError):
            a.artifact_path(history, new=True)
        with self.assertRaises(FileExistsError):
            a.seal_json(history, {"immutable": False})
        self.assertEqual(json.loads(history.read_text()), {"immutable": True})
        self.assertEqual(history.stat().st_mode & 0o222, 0)
        self.assertEqual(a.artifact_path(history, new=False), history)
        history.chmod(0o644)
        with self.assertRaises(ValueError):
            a.artifact_path(history, new=False)

    def test_symlink_and_hardlink_aliases_rejected(self):
        target = self.root / "target.json"
        a.seal_json(target, {})
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            a.artifact_path(link, new=False)
        hard = self.root / "hard.json"
        os.link(target, hard)
        with self.assertRaises(ValueError):
            a.artifact_path(hard, new=False)
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            a.new_run_path(alias / "pilot-escape")

    def test_pilot_requires_real_baseline_and_hash_before_model_loading(self):
        for kwargs in ({}, {"baseline_path": self.root / "absent.json", "baseline_sha256": "f" * 64},
                       {"baseline_path": self.root / "absent.json", "baseline_sha256": "bad"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, OSError)):
                a.run(mode="pilot", enabled=True, authorization=a.AUTHORIZATION,
                      report_path=self.root / "report.json", run_path=self.root / "pilot-new", **kwargs)
            self.assertFalse((self.root / "report.json").exists())
            self.assertFalse((self.root / "pilot-new").exists())


class FakeModel:
    def __init__(self):
        self.weight = SimpleNamespace(value=1.0, grad=None)

    def parameters(self):
        return iter([self.weight])

    def zero_grad(self, **kwargs):
        self.weight.grad = None


class SimulatedSGD:
    """Python fake only; never invokes Torch's optimizer or modifies model tensors."""
    calls = 0

    def __init__(self, params, **kwargs):
        self.param_groups = [{"params": list(params), **kwargs}]
        self.state = {}

    def step(self):
        type(self).calls += 1
        for p in self.param_groups[0]["params"]:
            p.value -= a.LR  # simulated weight mutation to verify control flow

    def zero_grad(self, **kwargs):
        for p in self.param_groups[0]["params"]:
            p.grad = None


class StepTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.monitor = SimpleNamespace(boundary=mock.Mock(), mode="pilot")
        self.report = {"optimizer_steps": 0, "step_attempted": False}
        SimulatedSGD.calls = 0
        self.loss = SimpleNamespace(requires_grad=True, item=lambda: 2.0,
                                    backward=lambda: setattr(self.model.weight, "grad", 3.0))
        self.torch = SimpleNamespace(
            optim=SimpleNamespace(SGD=SimulatedSGD),
            isfinite=lambda p: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
            nn=SimpleNamespace(utils=SimpleNamespace(
                clip_grad_norm_=mock.Mock(return_value=SimpleNamespace(item=lambda: 3.0)),
                get_total_norm=mock.Mock(return_value=SimpleNamespace(item=lambda: 0.999999)))))
        for name, value in (("validate_backward_model", None), ("checked_loss", self.loss),
                            ("validate_gradients", {"finite": True, "nonzero": True})):
            patch = mock.patch.object(a, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def step(self, **kwargs):
        return a.one_step(self.model, {}, self.torch, self.monitor, {}, self.report,
                          enabled=kwargs.get("enabled", True), authorization=kwargs.get("authorization", a.AUTHORIZATION))

    def test_only_explicit_authorization_permits_simulated_update_once(self):
        for enabled, token in ((False, a.AUTHORIZATION), (True, None), (True, "yes")):
            with self.assertRaises(ValueError):
                self.step(enabled=enabled, authorization=token)
        self.assertEqual(self.model.weight.value, 1.0)
        self.assertEqual(SimulatedSGD.calls, 0)
        self.step()
        self.assertEqual(self.report["optimizer_steps"], 1)
        self.assertEqual(SimulatedSGD.calls, 1)
        self.assertLess(self.model.weight.value, 1.0)
        self.assertIsNone(self.model.weight.grad)
        with self.assertRaises(RuntimeError):
            self.step()
        self.assertEqual(SimulatedSGD.calls, 1)

    def test_baseline_monitor_cannot_reach_backward_or_optimizer_even_with_authorization(self):
        self.monitor.mode = "baseline"
        with mock.patch.object(a, "checked_loss") as loss, self.assertRaises(RuntimeError):
            self.step()
        loss.assert_not_called()
        self.monitor.boundary.assert_not_called()
        self.assertEqual(SimulatedSGD.calls, 0)
        self.assertEqual(self.model.weight.value, 1.0)

    def test_failures_before_step_never_update(self):
        for stage in ("before_training_forward", "before_backward", "after_backward", "after_clipping", "before_optimizer_step"):
            with self.subTest(stage=stage):
                self.monitor.boundary.side_effect = lambda s: (_ for _ in ()).throw(RuntimeError("memory STOP")) if s == stage else None
                with self.assertRaises(RuntimeError):
                    self.step()
                self.assertEqual(SimulatedSGD.calls, 0)
                self.assertEqual(self.model.weight.value, 1.0)
                self.assertIsNone(self.model.weight.grad)

    def test_nonfinite_gradients_clip_and_optimizer_state_stop(self):
        for name, error in (("validate_gradients", FloatingPointError("nan")),
                            ("validate_optimizer", RuntimeError("unexpected optimizer state")),
                            ("checked_loss", FloatingPointError("loss parity"))):
            with self.subTest(name=name), mock.patch.object(a, name, side_effect=error):
                with self.assertRaises((FloatingPointError, RuntimeError)):
                    self.step()
                self.assertEqual(SimulatedSGD.calls, 0)
                self.assertIsNone(self.model.weight.grad)
        self.torch.nn.utils.get_total_norm.return_value = SimpleNamespace(item=lambda: 2.0)
        with self.assertRaises(RuntimeError):
            self.step()
        self.assertEqual(SimulatedSGD.calls, 0)

    def test_failure_during_step_marks_possible_partial_update_and_clears_gradients(self):
        with mock.patch.object(SimulatedSGD, "step", side_effect=RuntimeError("device failure")):
            with self.assertRaises(RuntimeError):
                self.step()
        self.assertTrue(self.report["step_attempted"])
        self.assertEqual(self.report["optimizer_steps"], 0)
        self.assertIsNone(self.model.weight.grad)

    def test_real_sgd_configuration_validation_without_step(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=a.LR, momentum=0.0, dampening=0.0,
                              weight_decay=0.0, nesterov=False, maximize=False,
                              foreach=False, fused=False, differentiable=False)
        a.validate_optimizer(opt, model, torch)
        for key, value in (("momentum", 0.1), ("lr", 0.1), ("foreach", True), ("weight_decay", 0.1)):
            previous = opt.param_groups[0][key]
            opt.param_groups[0][key] = value
            with self.assertRaises(RuntimeError):
                a.validate_optimizer(opt, model, torch)
            opt.param_groups[0][key] = previous
        opt.state[next(model.parameters())] = {"momentum_buffer": torch.zeros(1)}
        with self.assertRaises(RuntimeError):
            a.validate_optimizer(opt, model, torch)

    def test_real_cpu_gradient_diagnostics_and_clipping_without_step(self):
        from src.small_model_backward import validate_gradients
        model = torch.nn.Linear(2, 2)
        for p in model.parameters():
            p.grad = torch.full_like(p, 3.0)
        self.assertTrue(validate_gradients(model, torch, device="cpu")["finite"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False, error_if_nonfinite=True)
        self.assertLessEqual(torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]).item(), 1.00001)
        next(model.parameters()).grad[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            validate_gradients(model, torch, device="cpu")


class OrchestrationTests(unittest.TestCase):
    """All model/backend/optimizer/save operations replaced; only JSON reports persist."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=a.ROOT / "outputs")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = a.frozen_manifest()
        self.model = mock.Mock()
        self.model.parameters.return_value = []
        self.model.buffers.return_value = []
        self.rows = [{"id": r["id"], "answer_nll": 2.0, "generation": "",
                      "normalized_exact_match": False,
                      "alias_recovery": False if r["target_aliases"] else None,
                      "generated_tokens": 0} for r in self.manifest["evaluation"]]
        self.examples = {r["id"]: SimpleNamespace(input_ids=(1, 2), labels=(-100, 2))
                         for r in [*self.manifest["training"], *self.manifest["evaluation"]]}
        self.checkpoint = {"weight_sha256": "c" * 64}
        self.monitor = SimpleNamespace(samples=[], stage="", start=mock.Mock(), close=mock.Mock(), check=mock.Mock())
        def capture(stage, **kwargs):
            reading = sample(stage)
            self.monitor.samples.append(reading)
            return reading
        self.monitor.capture = capture
        self.monitor.boundary = capture
        runtime = SimpleNamespace(__version__=a.telemetry.VERSIONS["torch"], float32="float32",
                                  set_num_threads=mock.Mock(), manual_seed=mock.Mock(),
                                  use_deterministic_algorithms=mock.Mock(), mps=SimpleNamespace(manual_seed=mock.Mock()))
        transformers = SimpleNamespace(__version__=a.telemetry.VERSIONS["transformers"],
            AutoConfig=SimpleNamespace(from_pretrained=mock.Mock(return_value=SimpleNamespace(use_cache=False, to_dict=lambda: {}))),
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=object())),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=mock.Mock(return_value=(self.model,
                {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}))))
        patches = [mock.patch.object(a, "RUNS", self.root),
                   mock.patch.object(a.sys.stdin, "isatty", return_value=True),
                   mock.patch.dict(sys.modules, {"torch": runtime, "transformers": transformers,
                       "huggingface_hub": SimpleNamespace(__version__=a.telemetry.VERSIONS["huggingface_hub"])}),
                   mock.patch.object(a.telemetry, "validate_environment", return_value=SimpleNamespace(type="mps")),
                   mock.patch.object(a.telemetry, "_command", return_value="Apple M4 Pro"),
                   mock.patch.object(a.telemetry, "ensure_checkpoint", return_value=self.checkpoint),
                   mock.patch.object(a.telemetry, "validate_config"),
                   mock.patch.object(a.telemetry, "validate_model"),
                   mock.patch.object(a, "PilotMonitor", return_value=self.monitor),
                   mock.patch.object(a, "prepare_examples", return_value=self.examples),
                   mock.patch.object(a, "evaluate", return_value=self.rows),
                   mock.patch.object(a, "parameter_fingerprint", return_value="a" * 64),
                   mock.patch.object(a.shutil, "disk_usage", return_value=SimpleNamespace(free=8 * a.GIB)),
                   mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"})]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def baseline(self):
        self.monitor.samples.clear()
        path = self.root / "baseline.json"
        report = a.run(mode="baseline", report_path=path)
        self.assertEqual(report["status"], "baseline_complete_no_updates", report)
        return path, copy.deepcopy(report)

    def pilot(self, path):
        return a.run(mode="pilot", report_path=self.root / "pilot-report.json", run_path=self.root / "pilot-new",
                     baseline_path=path, baseline_sha256=a.sha256_file(path), enabled=True, authorization=a.AUTHORIZATION)

    def test_baseline_sealed_before_updates_and_no_checkpoint_call(self):
        with mock.patch.object(a, "one_step", side_effect=AssertionError("baseline reached training")) as step:
            path, report = self.baseline()
        step.assert_not_called()
        a.PilotMonitor.assert_called_once_with(sys.modules["torch"], profile=json.loads(a.PROFILE.read_text()),
                                               device="mps", mode="baseline")
        a.validate_baseline(report, self.manifest)
        self.assertEqual(path.stat().st_mode & 0o222, 0)
        self.model.save_pretrained.assert_not_called()
        self.assertEqual(report["optimizer_steps"], 0)
        self.model.requires_grad_.assert_called_once_with(False)
        self.model.train.assert_not_called()
        self.model.gradient_checkpointing_enable.assert_not_called()
        self.assertEqual(report["memory_policy"]["guard_id"], a.BASELINE_GUARD_ID)
        for key, value in (("optimizer_steps", 1), ("parameters_unchanged", False),
                           ("manifest_file_sha256", "0" * 64), ("actual_device", "cpu")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_baseline({**report, key: value}, self.manifest)
        broken = copy.deepcopy(report)
        broken["evaluation"].pop()
        with self.assertRaises(ValueError):
            a.validate_baseline(broken, self.manifest)

    def test_baseline_rejects_training_policy_or_unsafe_recorded_admission(self):
        _, report = self.baseline()
        profile = json.loads(a.PROFILE.read_text())
        changed = copy.deepcopy(report)
        changed["memory_policy"] = a.memory_policy(profile)
        with self.assertRaises(ValueError):
            a.validate_baseline(changed, self.manifest)
        for stage in ("baseline", "before_load"):
            changed = copy.deepcopy(report)
            next(s for s in changed["measurements"] if s["stage"] == stage)["available_physical_memory_lower_bound_bytes"] = 18 * a.GIB - 1
            with self.assertRaises(ValueError):
                a.validate_baseline(changed, self.manifest)
        for key, value in (("max_optimizer_steps", 1), ("backward_permitted", True),
                           ("preload_available_required_bytes", 0)):
            changed = copy.deepcopy(report)
            changed["memory_policy"][key] = value
            with self.assertRaises(ValueError):
                a.validate_baseline(changed, self.manifest)
        for kind in ("capacity", "gap", "preload"):
            changed = copy.deepcopy(report)
            if kind == "capacity":
                changed["measurements"][-1]["total_physical_memory_bytes"] += a.GIB
            elif kind == "gap":
                changed["measurements"][-1]["timestamp_ns"] += 3_000_000_000
            else:
                changed["measurements"][0]["stage"] = "missing_preload"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                a.validate_baseline(changed, self.manifest)

    def test_each_real_preload_gate_stops_before_model_load(self):
        # Exercise the actual monitor/dispatcher, without starting its thread.
        for fail_at in (0, 1):
            a.telemetry.ensure_checkpoint.reset_mock()
            readings = [sample("baseline"), sample("before_load")]
            readings[fail_at]["available_physical_memory_lower_bound_bytes"] = 18 * a.GIB - 1
            with self.subTest(fail_at=fail_at), mock.patch.object(a, "PilotMonitor", REAL_MONITOR), \
                    mock.patch.object(a.telemetry, "read_sample", side_effect=readings), \
                    mock.patch.object(a, "one_step") as step:
                report = a.run(mode="baseline", report_path=self.root / f"preload-{fail_at}.json")
            self.assertEqual(report["status"], "STOP")
            self.assertIn("18 GiB", report["error"])
            self.assertFalse(report["model_loaded"])
            self.assertFalse(report["optimizer_created"])
            self.assertFalse((self.root / f"preload-{fail_at}.json").exists())
            self.assertEqual(a.telemetry.ensure_checkpoint.call_count, fail_at)
            step.assert_not_called()
        sys.modules["transformers"].AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_pilot_still_stops_at_18_gib_with_valid_baseline_and_authorization(self):
        path, _ = self.baseline()
        sys.modules["transformers"].AutoModelForCausalLM.from_pretrained.reset_mock()
        reading = {**sample(), "available_physical_memory_lower_bound_bytes": 18 * a.GIB}
        with mock.patch.object(a, "PilotMonitor", REAL_MONITOR), \
                mock.patch.object(a.telemetry, "read_sample", return_value=reading), \
                mock.patch.object(a, "one_step") as step:
            report = self.pilot(path)
        self.assertEqual(report["status"], "STOP")
        self.assertIn("24 GiB", report["error"])
        self.assertEqual(report["memory_policy"]["guard_id"], a.GUARD_ID)
        sys.modules["transformers"].AutoModelForCausalLM.from_pretrained.assert_not_called()
        step.assert_not_called()

    def test_baseline_detects_parameter_hash_or_version_drift_and_never_saves(self):
        for kind in ("hash", "version"):
            self.monitor.samples.clear()
            parameter = SimpleNamespace(_version=0)
            self.model.parameters.return_value = [parameter]
            def evaluation(*args):
                if kind == "version":
                    parameter._version += 1  # mock counter only; no tensor/weight mutation
                return self.rows
            with mock.patch.object(a, "evaluate", side_effect=evaluation), \
                    mock.patch.object(a, "parameter_fingerprint", side_effect=["a" * 64, ("b" if kind == "hash" else "a") * 64]), \
                    mock.patch.object(a, "one_step") as step:
                report = a.run(mode="baseline", report_path=self.root / f"drift-{kind}.json")
            self.assertEqual(report["status"], "STOP")
            self.assertIn("changed weights", report["error"])
            self.assertEqual(report["optimizer_steps"], 0)
            step.assert_not_called()
            self.model.save_pretrained.assert_not_called()

    def test_load_failure_stops_and_never_updates_or_saves(self):
        with mock.patch.object(sys.modules["transformers"].AutoModelForCausalLM, "from_pretrained", side_effect=RuntimeError("OOM")), \
                mock.patch.object(a, "one_step") as step:
            report = a.run(mode="baseline", report_path=self.root / "failed.json")
        self.assertEqual(report["status"], "STOP")
        self.assertEqual(report["optimizer_steps"], 0)
        step.assert_not_called()
        self.model.save_pretrained.assert_not_called()

    def test_baseline_checksum_or_loaded_weight_drift_stops_before_update(self):
        path, report = self.baseline()
        with mock.patch.object(a, "one_step") as step:
            with self.assertRaises(ValueError):
                a.run(mode="pilot", report_path=self.root / "pilot-report.json", run_path=self.root / "pilot-new",
                      baseline_path=path, baseline_sha256="0" * 64, enabled=True, authorization=a.AUTHORIZATION)
            self.monitor.samples.clear()
            with mock.patch.object(a, "parameter_fingerprint", return_value="b" * 64):
                result = self.pilot(path)
            self.assertEqual(result["status"], "STOP")
            step.assert_not_called()
        self.model.save_pretrained.assert_not_called()
        self.assertTrue((self.root / "pilot-new/FAILED.json").is_file())
        self.assertEqual(json.loads(path.read_text()), report)

    def test_step_failure_cannot_reach_checkpoint_writer(self):
        path, _ = self.baseline()
        self.monitor.samples.clear()
        with mock.patch.object(a, "one_step", side_effect=FloatingPointError("gradient failure")), \
                mock.patch("src.graddiff_integration.collate_causal_examples", return_value={}):
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["optimizer_steps"], 0)
        self.model.save_pretrained.assert_not_called()
        self.assertFalse((self.root / "pilot-new/checkpoint").exists())


    def test_racing_report_creation_cannot_overwrite_history(self):
        path = self.root / "race.json"
        original_capture = self.monitor.capture
        def capture(stage, **kwargs):
            if stage == "before_load":
                path.write_text("historical")
            return original_capture(stage, **kwargs)
        self.monitor.capture = capture
        result = a.run(mode="baseline", report_path=path)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(path.read_text(), "historical")
        self.model.save_pretrained.assert_not_called()

    def simulated_pilot_context(self):
        parameter = SimpleNamespace(_version=0)
        self.model.parameters.return_value = [parameter]
        def update(*args, **kwargs):
            parameter._version += 1
            args[5].update(optimizer_steps=1, step_attempted=True, optimizer_created=True)
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(a, "one_step", side_effect=update))
        stack.enter_context(mock.patch("src.graddiff_integration.collate_causal_examples", return_value={}))
        stack.enter_context(mock.patch.object(a, "parameter_fingerprint", side_effect=["a" * 64, "b" * 64, "b" * 64]))
        stack.enter_context(mock.patch.object(a, "checkpoint_hashes", return_value={"model.safetensors": "d" * 64, "config.json": "e" * 64}))
        return stack

    def test_mocked_success_saves_only_fresh_checkpoint_after_all_gates(self):
        path, _ = self.baseline()
        self.monitor.samples.clear()
        with self.simulated_pilot_context():
            result = self.pilot(path)
        self.assertEqual(result["status"], "one_step_pilot_complete_not_longer_run_clearance", result)
        self.assertEqual(result["optimizer_steps"], 1)
        self.model.save_pretrained.assert_called_once_with(self.root / "pilot-new/checkpoint", safe_serialization=True, max_shard_size="1GB")
        self.assertTrue((self.root / "pilot-new/COMPLETE.json").exists())
        self.assertFalse((self.root / "pilot-new/checkpoint/model.safetensors").exists())  # mock save only

    def test_post_update_memory_failure_cannot_save_checkpoint(self):
        path, _ = self.baseline()
        self.monitor.samples.clear()
        capture = self.monitor.boundary
        def boundary(stage):
            if stage == "before_serialization":
                raise RuntimeError("memory STOP")
            return capture(stage)
        self.monitor.boundary = boundary
        with self.simulated_pilot_context():
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["optimizer_steps"], 1)  # accurately reports mock step already happened
        self.assertTrue(result["partial_in_memory_update_possible"])
        self.model.save_pretrained.assert_not_called()
        self.assertFalse((self.root / "pilot-new/checkpoint").exists())

    def test_unsafe_environment_rejected_before_load_or_output(self):
        for key, value in (("PYTORCH_ENABLE_MPS_FALLBACK", "1"), ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0"),
                           ("PYTORCH_MPS_FAST_MATH", "1"), ("WORLD_SIZE", "2")):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: value}), self.assertRaises(RuntimeError):
                a.run(mode="baseline", report_path=self.root / "unsafe.json")
        self.assertFalse((self.root / "unsafe.json").exists())
        sys.modules["transformers"].AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_save_failure_leaves_failed_tombstone_without_success_marker(self):
        path, _ = self.baseline()
        self.monitor.samples.clear()
        self.model.save_pretrained.side_effect = RuntimeError("serialization failure")
        with self.simulated_pilot_context():
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["checkpoint_writes"], 0)
        self.assertTrue(result["checkpoint_write_attempted"])
        self.assertTrue((self.root / "pilot-new/FAILED.json").exists())
        self.assertFalse((self.root / "pilot-new/COMPLETE.json").exists())


class EvaluationTests(unittest.TestCase):
    def test_greedy_prompt_excludes_target_answer_tokens(self):
        class Ids:
            def __init__(self, values):
                self.values = [list(row) for row in values]
        prompts = []
        next_ids = iter([7, 0])
        class Logits:
            device = SimpleNamespace(type="mps")
            dtype = "float32"
            def __getitem__(self, key):
                return self
            def argmax(self):
                return SimpleNamespace(item=lambda: next(next_ids))
        def forward(**kwargs):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
            prompts.append(copy.deepcopy(kwargs["input_ids"].values))
            return SimpleNamespace(logits=Logits(), past_key_values=None)
        model = mock.Mock(side_effect=forward)
        model.training = False
        model.parameters.return_value = []
        runtime = SimpleNamespace(inference_mode=torch.inference_mode, long="long", float32="float32",
            tensor=lambda values, **kwargs: Ids(values), ones_like=lambda ids: None,
            cat=lambda ids, **kwargs: Ids([ids[0].values[0] + ids[1].values[0]]),
            isfinite=lambda logits: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
        case = {"id": "dev:00:0:paraphrase", "question": "q", "answer": "Target dkqqudhttrnz.",
                "target_aliases": ["dkqqudhttrnz"]}
        example = SimpleNamespace(input_ids=(11, 22, 99, 100), assistant_target_start=2)
        tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda tokens, **kwargs: "dkqqudhttrnz")
        monitor = SimpleNamespace(boundary=mock.Mock(), check=mock.Mock())
        loss = SimpleNamespace(item=lambda: 2.0, requires_grad=False,
                               backward=mock.Mock(side_effect=AssertionError("baseline called backward")))
        with mock.patch.object(a, "checked_loss", return_value=loss), \
                mock.patch("src.graddiff_integration.collate_causal_examples", return_value={}), \
                mock.patch.object(torch.optim, "SGD", side_effect=AssertionError("baseline created SGD")) as sgd, \
                mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("baseline created AdamW")) as adam, \
                mock.patch.object(torch.Tensor, "backward", side_effect=AssertionError("baseline called backward")) as backward:
            rows = a.evaluate(model, tokenizer, {case["id"]: example}, {"evaluation": [case]},
                              runtime, SimpleNamespace(type="mps"), monitor)
            sgd.assert_not_called()
            adam.assert_not_called()
            backward.assert_not_called()
        loss.backward.assert_not_called()
        self.assertEqual(tuple(prompts[0][0]), (11, 22))
        self.assertEqual(tuple(prompts[1][0]), (11, 22, 7))
        self.assertTrue(rows[0]["alias_recovery"])
        self.assertEqual(rows[0]["generated_tokens"], 1)

        for training, requires_grad, grad, device in ((True, False, None, "mps"),
                                                     (False, True, None, "mps"),
                                                     (False, False, object(), "mps"),
                                                     (False, False, None, "cpu")):
            model.training = training
            model.parameters.return_value = [SimpleNamespace(requires_grad=requires_grad, grad=grad)]
            with self.subTest(training=training, requires_grad=requires_grad, device=device), self.assertRaises(RuntimeError):
                a.evaluate(model, tokenizer, {}, {}, runtime, SimpleNamespace(type=device), monitor)

        model.training = False
        model.parameters.return_value = []
        with mock.patch.object(a, "checked_loss", return_value=SimpleNamespace(requires_grad=True)), \
                mock.patch("src.graddiff_integration.collate_causal_examples", return_value={}), self.assertRaises(RuntimeError):
            a.evaluate(model, tokenizer, {case["id"]: example}, {"evaluation": [case]},
                       runtime, SimpleNamespace(type="mps"), monitor)


if __name__ == "__main__":
    unittest.main()
