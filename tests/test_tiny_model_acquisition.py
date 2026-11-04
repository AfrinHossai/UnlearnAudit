"""Offline preparation tests: fixtures/mocked loops, never train the eight facts."""
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
from src import tiny_model_acquisition as t


def sample(**changes):
    result = {"stage": "fixture", "timestamp_ns": time.monotonic_ns() - 1_000_000,
              "sample_duration_ns": 100_000, "total_physical_memory_bytes": 48*t.GIB,
              "macos_pressure_level": 1, "available_physical_memory_lower_bound_bytes": 16*t.GIB,
              "system_memory_free_percentage": 90, "pageouts": 0, "swap_used_bytes": 0,
              "process_current_rss_bytes": t.GIB, "process_peak_rss_bytes": t.GIB,
              "mps_tensor_allocated_bytes": None, "mps_driver_allocated_bytes": None}
    return {**result, **changes}


class FixtureMonitor:
    mode = "acquisition"
    def __init__(self):
        self.samples = []
        self.started = time.monotonic()
    def capture(self, stage, **kwargs):
        self.samples.append(sample(stage=stage))
        return self.samples[-1]
    def boundary(self, stage):
        self.capture(stage)
    def check(self):
        pass
    def start(self):
        pass
    def close(self):
        pass


class OfflineCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # Real fixture autograd is allowed; optimizer execution is always trapped.
        for cls in (torch.optim.AdamW, torch.optim.SGD):
            self.stack.enter_context(mock.patch.object(cls, "step", side_effect=AssertionError("real optimizer step forbidden")))

    def temp_root(self):
        tmp = tempfile.TemporaryDirectory(dir=t.ROOT / "outputs", prefix="task64-test-")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.stack.enter_context(mock.patch.object(t, "RUNS", root))
        return root


class ProtocolTests(OfflineCase):
    def test_frozen_data_roles_and_encoding(self):
        spec, rows = t.frozen_inputs()
        self.assertEqual(spec["max_steps"], 2000)
        self.assertEqual(spec["parameter_count"], 90496)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(r["kind"] == "direct" and r["role"] == "forget" for r in rows), 2)
        self.assertEqual(sum(r["kind"] == "direct" and r["role"] == "retain" for r in rows), 6)
        encoded = t.examples(rows)
        self.assertEqual(max(len(e.input_ids) for e in encoded.values()), 100)
        for r in rows:
            e = encoded[r["id"]]
            self.assertEqual(e.labels[:e.assistant_target_start], (-100,)*e.assistant_target_start)
            self.assertEqual(e.input_ids[e.assistant_target_start:], tuple(r["answer"].encode())+(256,))
            self.assertLessEqual(e.assistant_target_start+16, 128)
        self.assertEqual(t.ByteTokenizer().encode("é"), [195, 169])
        with self.assertRaises(ValueError):
            t.ByteTokenizer().encode("x", add_special_tokens=True)
        with self.assertRaisesRegex(ValueError, "truncation"):
            t.encode_plain_qa(t.ByteTokenizer(), "q"*128, "answer", max_sequence_length=128)

    def test_profile_and_data_hash_drift_fail(self):
        with mock.patch.object(t, "sha256_file", return_value="bad"):
            with self.assertRaisesRegex(ValueError, "profile"):
                t.frozen_inputs()
        original = t.sha256_file
        with mock.patch.object(t, "sha256_file", side_effect=lambda p: original(p) if p == t.SPEC else "bad"):
            with self.assertRaisesRegex(ValueError, "fact source"):
                t.frozen_inputs()

    def test_default_cli_does_not_import_frameworks(self):
        code = '''
import sys
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch', 'transformers'):
   raise AssertionError('unexpected ML runtime')
sys.meta_path.insert(0, Block())
from src.tiny_model_acquisition import main
raise SystemExit(main([]))
'''
        p = subprocess.run([sys.executable, "-B", "-c", code], cwd=t.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["optimizer_steps"], 0)

    def test_cli_rejects_scope_changes_and_mixed_modes(self):
        for argv in (["--max-steps", "2001"], ["--resume", "x"], ["--device", "mps"],
                     ["--allow-download"], ["--authorize-weight-update", t.AUTHORIZATION],
                     ["--dry-run", "--run-acquisition"], ["--dry-run", "--run-dir", "x"],
                     ["--run-acquisition"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                t.main(argv)

    def test_authorization_precedes_all_work(self):
        for enabled, token in ((False, t.AUTHORIZATION), (True, None), (1, t.AUTHORIZATION), (True, "yes")):
            with mock.patch.object(t, "frozen_inputs") as inputs:
                with self.assertRaises(ValueError):
                    t.run(enabled=enabled, authorization=token, run_path=Path("unused"))
                inputs.assert_not_called()

    def test_unique_paths_and_symlinks(self):
        root = self.temp_root()
        self.assertEqual(t.new_run_path(root / "acquisition-new"), root / "acquisition-new")
        (root / "acquisition-old").mkdir()
        (root / "acquisition-alias").symlink_to(root / "acquisition-old", target_is_directory=True)
        for path in (root, root/"acquisition-old", root/"acquisition-alias", root/"pilot-new", root/".."/"acquisition-out"):
            with self.assertRaises(ValueError):
                t.new_run_path(path)

    def test_exact_scoring_gate_is_not_loss_only_or_paraphrase_selected(self):
        _, data = t.frozen_inputs()
        old = [{**r, "exact_recovery": False, "answer_byte_nll": 5.0} for r in data]
        new = [{**r, "exact_recovery": r["kind"] == "direct", "answer_byte_nll": 3.0} for r in data]
        self.assertTrue(t.acceptance(new, old)["acquisition_accepted"])
        new[0]["exact_recovery"] = False
        self.assertFalse(t.acceptance(new, old)["acquisition_accepted"])
        new[0]["exact_recovery"] = True
        new[0]["answer_byte_nll"] = 4.1
        gate = t.acceptance(new, old)
        self.assertTrue(gate["all_eight_recovered"])
        self.assertFalse(gate["acquisition_accepted"])
        self.assertEqual(t.scores(new)["direct_retain"], {"recovered": 6, "total": 6})

    def test_generation_requires_complete_exact_answer_and_eos(self):
        target = "abcdefghijkl"
        self.assertTrue(t.recovery_metrics(list(target.encode()), "eos", target)["exact_recovery"])
        for text, finish in ((target, "length"), (target+" ", "eos"), (target[:-1], "eos"),
                             (target.upper(), "eos"), ("prefix "+target, "eos")):
            self.assertFalse(t.recovery_metrics(list(text.encode()), finish, target)["exact_recovery"])


class SafetyTests(OfflineCase):
    def test_policy_and_exact_boundaries(self):
        self.assertEqual(t.memory_policy()["preload_available_required_bytes"], 12*t.GIB)
        self.assertEqual(t.memory_policy(3*t.GIB)["preload_available_required_bytes"], 13*t.GIB)
        for bad in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                t.memory_policy(bad)
        t.validate_sample(sample(available_physical_memory_lower_bound_bytes=12*t.GIB), before_load=True)
        with self.assertRaisesRegex(RuntimeError, "preload"):
            t.validate_sample(sample(available_physical_memory_lower_bound_bytes=12*t.GIB-1), before_load=True)
        t.validate_sample(sample(available_physical_memory_lower_bound_bytes=6*t.GIB))

    def test_unsafe_telemetry_fails_closed(self):
        changes = {"available_physical_memory_lower_bound_bytes": 6*t.GIB-1,
                   "total_physical_memory_bytes": 48*t.GIB-1, "macos_pressure_level": 2,
                   "system_memory_free_percentage": 19, "process_peak_rss_bytes": 8*t.GIB+1,
                   "pageouts": None, "timestamp_ns": 1, "sample_duration_ns": 0,
                   "mps_tensor_allocated_bytes": 0}
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                t.validate_sample(sample(**{field: value}))

    def test_cumulative_growth_order_and_gap(self):
        first = sample(timestamp_ns=time.monotonic_ns()-1_500_000)
        previous = sample(timestamp_ns=time.monotonic_ns()-1_200_000)
        t.validate_sample(sample(), first=first, previous=previous, active=True)
        for field in ("swap_used_bytes", "pageouts"):
            with self.assertRaisesRegex(RuntimeError, "growth"):
                t.validate_sample(sample(**{field: 1}), first=first)
        with self.assertRaises(RuntimeError):
            t.validate_sample(sample(), previous=sample(timestamp_ns=time.monotonic_ns()+10000))
        with self.assertRaisesRegex(RuntimeError, "gap"):
            t.validate_sample(sample(), previous=sample(timestamp_ns=time.monotonic_ns()-3_000_000_000), active=True)

    def test_watchdog_time_and_failing_sample_retention(self):
        monitor = t.CPUMonitor()
        with mock.patch.object(t, "read_sample", return_value=sample(macos_pressure_level=2)):
            with self.assertRaises(RuntimeError):
                monitor.capture("test")
        self.assertEqual(len(monitor.samples), 1)
        monitor.started -= 1801
        with self.assertRaisesRegex(RuntimeError, "time budget"):
            monitor.check()
        monitor = t.CPUMonitor()
        monitor.thread = object()
        monitor.samples = [sample(timestamp_ns=1)]
        with self.assertRaisesRegex(RuntimeError, "stale"):
            monitor.check()

    def test_cpu_collection_uses_host_only_no_mps(self):
        vm = "mach virtual memory statistics: (page size of 16384 bytes)\nPages free: 800000.\nPages speculative: 100000.\nPageouts: 0.\n"
        def command(args):
            if args[0].endswith("vm_stat"):
                return vm
            if args[0].endswith("memory_pressure"):
                return "System-wide memory free percentage: 90%"
            if args[0].endswith("sysctl"):
                return f"{48*t.GIB}\n1\ntotal = 0.00M used = 0.00M free = 0.00M\n"
            return "1024\n"
        with mock.patch.object(t.telemetry, "_command", side_effect=command), \
             mock.patch.object(t.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=t.GIB)):
            s = t.read_sample("fixture")
        self.assertIsNone(s["mps_tensor_allocated_bytes"])
        t.validate_sample(s, before_load=True)

    def test_projection_rejects_slow_plan(self):
        report = {"optimizer_steps": 0}
        t.projection(FixtureMonitor(), report, [0.01], 1, 0.1)
        with self.assertRaisesRegex(RuntimeError, "25-minute"):
            t.projection(FixtureMonitor(), report, [1.0], 1, 0.1)

    def test_environment_rejects_nonterminal_and_overrides(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False), self.assertRaises(RuntimeError):
            t.environment()
        with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch.object(t.sys, "platform", "darwin"), \
             mock.patch.object(t.platform, "machine", return_value="arm64"), \
             mock.patch.object(t.telemetry, "_command", return_value="Apple M4 Pro"), \
             mock.patch.dict(os.environ, {"PYTORCH_MPS_FAST_MATH": "1"}):
            with self.assertRaisesRegex(RuntimeError, "overrides"):
                t.environment()

    def test_budget_partial_attempt_and_authorization_block_optimizer(self):
        monitor = FixtureMonitor()
        for steps, attempts in ((2000, 2000), (0, 1), (-1, -1), (True, True)):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                t.train_step(None, {}, None, monitor, {"optimizer_steps": steps, "optimizer_step_attempts": attempts},
                             enabled=True, authorization=t.AUTHORIZATION)
        with self.assertRaises(ValueError):
            t.train_step(None, {}, None, monitor, {}, enabled=False, authorization=t.AUTHORIZATION)
        monitor.mode = "baseline"
        with self.assertRaisesRegex(RuntimeError, "admission"):
            t.train_step(None, {}, None, monitor, {"optimizer_steps": 0, "optimizer_step_attempts": 0},
                         enabled=True, authorization=t.AUTHORIZATION)


class ModelFixtureTests(OfflineCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, _ = t.frozen_inputs()
        cls.model = t.initialize(cls.spec)

    def test_deterministic_count_config_and_ties(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 90496)
        with mock.patch.object(torch.mps, "manual_seed", side_effect=AssertionError("CPU must not seed MPS")), \
             mock.patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CPU must not seed CUDA")):
            other = t.initialize(self.spec)
        self.assertEqual(t.parameter_fingerprint(self.model), t.parameter_fingerprint(other))
        self.assertIs(other.get_input_embeddings().weight, other.get_output_embeddings().weight)
        other.config.use_cache = True
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            t.validate_model(other, self.spec)

    def test_fixture_dry_run_backward_without_optimizer(self):
        result = t.dry_run()
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(result["weights_unchanged"])
        self.assertFalse(result["target_facts_executed"])

    def test_exact_generation_masking_and_wrong_binding_controls(self):
        _, data = t.frozen_inputs()
        encoded = t.examples(data)
        model = t.initialize(self.spec)
        original_forward = model.forward
        # Real fixture forwards, controlled generation tokens; no target training.
        def forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if "labels" not in kwargs:
                # Forced EOS makes all outputs empty, never credited as acquisition.
                out.logits[..., 256] = out.logits.max() + 1
            return out
        with mock.patch.object(model, "forward", side_effect=forward):
            result = t.evaluate(model, data, encoded, FixtureMonitor())
        self.assertEqual(len(result), 20)
        self.assertTrue(all(r["prediction"] == "" and r["finish_reason"] == "eos" for r in result))
        self.assertTrue(all(not r["exact_recovery"] for r in result))
        self.assertEqual(sum("correct_over_wrong_nll_margin" in r for r in result), 8)

    def test_readonly_checkpoint_roundtrip_and_collision(self):
        from safetensors.torch import load_model
        root = self.temp_root()
        model = t.initialize(self.spec)
        before = t.parameter_fingerprint(model)
        report = {"checkpoints": []}
        with mock.patch.object(t.shutil, "disk_usage", return_value=SimpleNamespace(free=5*t.GIB)):
            path = t.checkpoint(model, root, 0, [], report, FixtureMonitor())
            load_model(model, str(path), strict=True, device="cpu")
            self.assertEqual(t.parameter_fingerprint(model), before)
            with self.assertRaises(FileExistsError):
                t.checkpoint(model, root, 0, [], report, FixtureMonitor())
        self.assertFalse(path.stat().st_mode & 0o222)

    def test_optimizer_membership_and_nonfinite_loss_stop_without_update(self):
        model = t.initialize(self.spec)
        optimizer = torch.optim.AdamW(model.parameters(), **{**self.spec["optimizer"], "betas": (0.9, 0.999)})
        t.validate_optimizer(optimizer, model)
        optimizer.param_groups[0]["lr"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "drift"):
            t.validate_optimizer(optimizer, model)
        optimizer.param_groups[0]["lr"] = 0.003
        monitor = FixtureMonitor()
        report = {"optimizer_steps": 0, "optimizer_step_attempts": 0}
        ids = torch.tensor([[1, 3, 4, 5]])
        batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        with mock.patch.object(model, "forward", return_value=SimpleNamespace(
                loss=torch.tensor(float("nan")), logits=torch.zeros(1, 4, 257), past_key_values=None)):
            with self.assertRaises(FloatingPointError):
                t.train_step(model, batch, optimizer, monitor, report, enabled=True, authorization=t.AUTHORIZATION)
        self.assertEqual(report["optimizer_step_attempts"], 0)

    def test_fixture_backward_mocked_step_and_last_boundary_stop(self):
        model = t.initialize(self.spec)
        optimizer = torch.optim.AdamW(model.parameters(), **{**self.spec["optimizer"], "betas": (0.9, 0.999)})
        ids = torch.tensor([[1, 3, 4, 5]])
        batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        report = {"optimizer_steps": 0, "optimizer_step_attempts": 0}
        before = t.parameter_fingerprint(model)
        # Real four-token gradient fixture; optimizer execution remains mocked.
        with mock.patch.object(optimizer, "step", return_value=None) as step:
            result = t.train_step(model, batch, optimizer, FixtureMonitor(), report, enabled=True, authorization=t.AUTHORIZATION)
        step.assert_called_once()
        self.assertEqual(result["step"], 1)
        self.assertEqual(t.parameter_fingerprint(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        monitor = FixtureMonitor()
        monitor.boundary = lambda stage: (_ for _ in ()).throw(RuntimeError("memory STOP")) if stage == "before_optimizer_step" else None
        report = {"optimizer_steps": 0, "optimizer_step_attempts": 0}
        with mock.patch.object(optimizer, "step") as step, self.assertRaisesRegex(RuntimeError, "memory STOP"):
            t.train_step(model, batch, optimizer, monitor, report, enabled=True, authorization=t.AUTHORIZATION)
        step.assert_not_called()
        self.assertEqual(report["optimizer_step_attempts"], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class OrchestrationTests(OfflineCase):
    def setup_run(self, *, succeed_at=None, fail_step=False):
        root = self.temp_root()
        model = torch.nn.Linear(1, 1)
        model.config = SimpleNamespace(to_dict=lambda: {})
        self.report_ref = None
        spec, data = t.frozen_inputs()
        baseline = [{**r, "exact_recovery": False, "answer_byte_nll": 5.0, "generated_ids": [256]} for r in data]
        self.eval_calls = 0
        def evaluate(*args):
            self.eval_calls += 1
            if self.eval_calls <= 2:
                return copy.deepcopy(baseline)
            passed = succeed_at is not None and self.report_ref["optimizer_steps"] >= succeed_at
            return [{**r, "exact_recovery": passed and r["kind"] == "direct", "answer_byte_nll": 3.0} for r in data]
        def step(model, batch, optimizer, monitor, report, **kwargs):
            self.report_ref = report
            self.assertLess(report["optimizer_steps"], 2000)
            report["optimizer_step_attempts"] += 1
            report["possible_partial_update"] = True
            if fail_step:
                raise RuntimeError("synthetic partial step failure")
            with torch.no_grad():
                model.weight.add_(0.001)  # fixture mutation, no optimizer execution
            report["optimizer_steps"] += 1
            report["possible_partial_update"] = False
            return {"step": report["optimizer_steps"], "training_answer_nll": 3.0}
        def checkpoint(model, run_path, step, evaluation, report, monitor):
            directory = run_path / f"checkpoint-{step:06d}"
            directory.mkdir()
            target = directory / "fixture.txt"
            target.write_text("synthetic checkpoint")
            report["checkpoints"].append({"directory": directory.name, "step": step,
                                          "files_sha256": {target.name: t.sha256_file(target)}})
            return target
        import safetensors.torch
        patches = ((t, "environment", lambda: None), (t, "CPUMonitor", FixtureMonitor),
                   (t, "initialize", lambda s: model), (t, "validate_model", lambda *a: None),
                   (t, "evaluate", evaluate), (t, "checkpoint", checkpoint),
                   (t, "rehearse", lambda *a: [0.01]*3), (t, "train_step", step),
                   (t, "projection", lambda *a: None), (safetensors.torch, "load_model", lambda *a, **kw: None))
        for obj, name, value in patches:
            self.stack.enter_context(mock.patch.object(obj, name, value))
        self.stack.enter_context(mock.patch.object(torch.optim, "AdamW", return_value=SimpleNamespace(state={})))
        self.stack.enter_context(mock.patch.object(t.shutil, "disk_usage", return_value=SimpleNamespace(free=5*t.GIB)))
        return root / "acquisition-fixture"

    def invoke(self, path):
        with contextlib.redirect_stdout(io.StringIO()):
            return t.run(enabled=True, authorization=t.AUTHORIZATION, run_path=path)

    def test_baseline_precedes_optimizer_early_stop_at_100(self):
        path = self.setup_run(succeed_at=100)
        report = self.invoke(path)
        self.assertEqual(report["status"], "acquisition_complete", report["stop_reason"])
        self.assertEqual(report["optimizer_steps"], 100)
        self.assertEqual(report["optimizer_step_attempts"], 100)
        self.assertEqual(report["checkpoint_writes"], 2)
        self.assertGreater(report["changed_parameter_tensor_count"], 0)
        self.assertEqual(json.loads((path/"baseline.json").read_text())["optimizer_steps"], 0)
        self.assertTrue((path/"COMPLETE.json").exists())
        with self.assertRaises(ValueError):
            self.invoke(path)

    def test_no_acquisition_is_truthful_bounded_negative_result(self):
        path = self.setup_run()
        report = self.invoke(path)
        self.assertEqual(report["optimizer_steps"], 2000)
        self.assertEqual(report["checkpoint_writes"], 21)
        self.assertEqual(report["status"], "acquisition_not_achieved")
        self.assertFalse(report["acquisition_accepted"])
        self.assertTrue((path/"COMPLETE.json").exists())

    def test_partial_attempt_is_sealed_stop_never_success(self):
        path = self.setup_run(fail_step=True)
        report = self.invoke(path)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["optimizer_step_attempts"], 1)
        self.assertTrue(report["possible_partial_update"])
        self.assertEqual(report["status"], "STOP")
        self.assertTrue((path/"FAILED.json").exists())
        self.assertFalse((path/"COMPLETE.json").exists())

    def test_preload_failure_seals_zero_step_report(self):
        path = self.setup_run()
        with mock.patch.object(FixtureMonitor, "capture", side_effect=RuntimeError("preload failed")):
            report = self.invoke(path)
        self.assertEqual(report["status"], "STOP")
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["optimizer_created"])
        self.assertTrue((path/"FAILED.json").exists())

    def test_noop_drift_stops_before_any_update(self):
        path = self.setup_run(succeed_at=100)
        original = t.evaluate
        calls = 0
        def drift(*args):
            nonlocal calls
            calls += 1
            rows = original(*args)
            if calls == 2:
                rows[0]["generated_ids"] = [1, 256]
            return rows
        with mock.patch.object(t, "evaluate", side_effect=drift):
            report = self.invoke(path)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertIn("no-op drift", report["stop_reason"])
        self.assertFalse(report["optimizer_created"])

    def test_checkpoint_failure_after_updates_cannot_complete(self):
        path = self.setup_run(succeed_at=100)
        original = t.checkpoint
        def save(*args):
            if args[2] > 0:
                raise RuntimeError("synthetic serialization failure")
            return original(*args)
        with mock.patch.object(t, "checkpoint", side_effect=save):
            report = self.invoke(path)
        self.assertEqual(report["status"], "STOP")
        self.assertEqual(report["optimizer_steps"], 100)
        self.assertFalse(report["acquisition_accepted"])
        self.assertFalse((path/"COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
