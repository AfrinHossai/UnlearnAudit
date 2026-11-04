"""Offline gates and mocked full-model control flow; no pretrained/MPS execution."""

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

from src import graddiff
from src import small_model_backward as rehearsal
from src import small_model_preflight as readonly
from test_small_model_preflight import FakeTorch, FakeTensor, GPT2TokenizerFast


class GraphTensor(FakeTensor):
    def __init__(self, shape=(), **kwargs):
        super().__init__(shape, **kwargs)
        self.ndim = len(shape)
        self.requires_grad = True
        self.is_sparse = False
        self.on_backward = lambda: None

    def detach(self):
        return self

    def backward(self):
        self.on_backward()


class LlamaForCausalLM:
    def __init__(self, config):
        self.config, self.training, self.is_gradient_checkpointing = config, True, False
        self.parameter = SimpleNamespace(requires_grad=True, dtype="float32", device=SimpleNamespace(type="cpu"),
                                         grad=None, _version=0, numel=lambda: 134515008)
        self.output_change = lambda x: x
        self.checkpoint_kwargs = None
        self.calls = []

    def parameters(self):
        return iter([self.parameter])

    def buffers(self):
        return iter([])

    def requires_grad_(self, enabled):
        self.parameter.requires_grad = enabled

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def to(self, device):
        self.parameter.device = device

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True

    def zero_grad(self, *, set_to_none):
        assert set_to_none
        self.parameter.grad = None

    def __call__(self, **kwargs):
        assert self.training and self.parameter.requires_grad and self.is_gradient_checkpointing
        assert kwargs["use_cache"] is False
        self.calls.append(kwargs)
        return self.output_change(SimpleNamespace(logits=GraphTensor((1, 4, 49152)), loss=GraphTensor(), past_key_values=None))


class SampleFixture:
    def setUp(self):
        self.profile = json.loads(rehearsal.PROFILE.read_text())
        self.counter = 0

    def sample(self, stage="baseline"):
        self.counter += 1
        return {"stage": stage, "timestamp_ns": time.monotonic_ns() - 1_000_000, "sample_duration_ns": 100_000,
                "total_physical_memory_bytes": 48 * rehearsal.GIB, "macos_pressure_level": 1,
                "available_physical_memory_lower_bound_bytes": 24 * rehearsal.GIB,
                "system_memory_free_percentage": 90, "pageouts": 0, "swap_used_bytes": 0,
                "process_current_rss_bytes": rehearsal.GIB, "process_peak_rss_bytes": rehearsal.GIB,
                "mps_tensor_allocated_bytes": 100, "mps_driver_allocated_bytes": 200}

    def validate(self, sample, **kwargs):
        rehearsal.validate_sample(sample, profile=self.profile, device="mps", **kwargs)


class BackwardSafetyTests(SampleFixture, unittest.TestCase):
    def test_budget_is_separate_estimate_with_gradient_and_uncertainty_reserves(self):
        policy = rehearsal.rehearsal_memory_policy(self.profile, device="mps")
        self.assertEqual(policy["preload_available_required_bytes"], 20 * rehearsal.GIB)
        self.assertFalse(policy["measured_backward_requirement"])
        self.assertFalse(policy["training_authorized"])
        terms = policy["estimated_incremental_terms_bytes"]
        self.assertEqual(terms["fp32_gradients_estimate"], 538060032)
        self.assertEqual(sum(terms.values()), policy["estimated_incremental_total_bytes"])
        larger = rehearsal.rehearsal_memory_policy(self.profile, device="mps", baseline_peak_rss_bytes=3 * rehearsal.GIB)
        self.assertEqual(larger["preload_available_required_bytes"], 22 * rehearsal.GIB)
        for peak in (True, -1, float("nan")):
            with self.assertRaises(RuntimeError):
                rehearsal.rehearsal_memory_policy(self.profile, device="mps", baseline_peak_rss_bytes=peak)

    def test_exact_preload_and_continuing_boundaries(self):
        for preload, minimum in ((True, 20), (False, 6)):
            sample = self.sample()
            sample["available_physical_memory_lower_bound_bytes"] = minimum * rehearsal.GIB
            self.validate(sample, before_load=preload)
            sample["available_physical_memory_lower_bound_bytes"] -= 1
            with self.assertRaises(RuntimeError):
                self.validate(sample, before_load=preload)

    def test_profile_device_memory_pressure_and_invalid_telemetry_fail_closed(self):
        for device in (None, "cpu", "cuda", "auto"):
            with self.assertRaises(RuntimeError):
                rehearsal.rehearsal_memory_policy(self.profile, device=device)
        changed = copy.deepcopy(self.profile)
        changed["training"]["max_sequence_length"] = 128
        with self.assertRaises(ValueError):
            rehearsal.rehearsal_memory_policy(changed, device="mps")
        for key, value in (("macos_pressure_level", 2), ("macos_pressure_level", 4),
                           ("system_memory_free_percentage", 19), ("mps_driver_allocated_bytes", None),
                           ("process_current_rss_bytes", True), ("swap_used_bytes", float("nan")),
                           ("total_physical_memory_bytes", 47 * rehearsal.GIB),
                           ("timestamp_ns", time.monotonic_ns() - 3_000_000_000)):
            sample = self.sample()
            sample[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.validate(sample)

    def test_adjacent_swap_growth_pageouts_and_monitor_gap_stop(self):
        previous, current = self.sample(), self.sample()
        for key in ("swap_used_bytes", "pageouts"):
            changed = {**current, key: 1}
            with self.assertRaisesRegex(RuntimeError, "swap/page-outs"):
                self.validate(changed, baseline=previous)
        # A decrease does not permit a subsequent rise below original occupancy.
        previous["swap_used_bytes"], current["swap_used_bytes"] = 50, 75
        with self.assertRaises(RuntimeError):
            self.validate(current, baseline=previous)
        previous = self.sample()
        previous["timestamp_ns"] -= 3_000_000_000
        with self.assertRaisesRegex(RuntimeError, "sampling gap"):
            self.validate(self.sample(), baseline=previous, active_monitor=True)

    def test_default_cli_imports_no_framework_and_rejects_execution_options(self):
        code = ("import sys; from src import small_model_backward as b; "
                "b.select_pair=lambda p: ({}, {}); assert b.main([])==0; "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=rehearsal.ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "validation_only")
        self.assertFalse(report["model_loaded"])
        for args in (["--report-path", "outputs/test.json"], ["--run-backward-rehearsal"],
                     ["--allow-download"], ["--apply-updates"], ["--device", "cpu"]):
            with mock.patch.object(rehearsal, "run_rehearsal") as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rehearsal.main(args)
                run.assert_not_called()
        for enabled in (False, 1, "yes"):
            with self.assertRaises(ValueError):
                rehearsal.run_rehearsal(enabled=enabled, report_path=Path("unused.json"))

    def test_source_selection_checks_hashes_and_does_not_read_holdout(self):
        manifest = json.loads(rehearsal.MANIFEST.read_text())
        source = {}
        for split, size in (("forget01", 40), ("retain99", 3960)):
            source[split] = [{"question": "Question?", "answer": "Answer."}] * size
        # Test hash verification without requiring private local dataset cache.
        with mock.patch.object(rehearsal, "load_pinned_split", side_effect=lambda split, *a, **k: (source[split], {})) as load:
            with self.assertRaisesRegex(ValueError, "QA checksum"):
                rehearsal.select_pair(self.profile)
            self.assertEqual(load.call_args.args[0], "forget01")
        with mock.patch.object(rehearsal, "sha256_file", return_value="corrupt"):
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                rehearsal.select_pair(self.profile)
        self.assertEqual(manifest["unlearning_and_retain_control_pairs"][0]["retain"], "retain99:0501")

    def test_encoder_masks_prompt_and_eos_and_refuses_truncation(self):
        rows = {role: {"question": "Q?", "answer": "A."} for role in ("forget", "retain")}
        batches, enc = rehearsal.prepare_batches(GPT2TokenizerFast(), rows, torch.device("cpu"))
        for role, batch in batches.items():
            start = enc[role]["assistant_target_start"]
            self.assertTrue(torch.all(batch["labels"][0, :start] == -100))
            self.assertTrue(torch.equal(batch["input_ids"][0, start:], batch["labels"][0, start:]))
            self.assertEqual(batch["labels"][0, -1].item(), 0)
            self.assertEqual(batch["attention_mask"].sum().item(), enc[role]["tokens"])
        rows["forget"]["answer"] = "a" * 200
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            rehearsal.prepare_batches(GPT2TokenizerFast(), rows, torch.device("cpu"))

    def test_real_tiny_cpu_gradient_checks_nonfinite_missing_zero_and_mismatch(self):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        model = SimpleNamespace(named_parameters=lambda: iter([("tiny", parameter)]))
        for value in (float("nan"), float("inf")):
            parameter.grad = torch.tensor([value, 1.])
            with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                rehearsal.validate_gradients(model, torch, device="cpu")
        parameter.grad = None
        with self.assertRaisesRegex(RuntimeError, "missing"):
            rehearsal.validate_gradients(model, torch, device="cpu")
        parameter.grad = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            rehearsal.validate_gradients(model, torch, device="cpu")
        parameter.grad = torch.ones(2)
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            rehearsal.validate_gradients(model, torch, device="mps")
        self.assertTrue(rehearsal.validate_gradients(model, torch, device="cpu")["finite"])
        self.assertEqual(rehearsal.parameter_fingerprint(model), rehearsal.parameter_fingerprint(model))
        before = rehearsal.parameter_fingerprint(model)
        version = parameter._version
        with torch.no_grad():
            parameter.data.add_(1)  # A byte change that bypasses the version counter.
        self.assertEqual(version, parameter._version)
        self.assertNotEqual(before, rehearsal.parameter_fingerprint(model))


class BackwardOrchestrationTests(SampleFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        readonly.OUTPUTS.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=readonly.OUTPUTS)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "backward.json"
        config_dict = {**self.profile["model"], "architectures": ["LlamaForCausalLM"]}
        config = SimpleNamespace(use_cache=False, _attn_implementation="sdpa", to_dict=lambda: config_dict)
        self.model = LlamaForCausalLM(config)
        self.fake_torch = FakeTorch()
        self.fake_torch.no_grad = contextlib.nullcontext
        self.fake_torch.optim = SimpleNamespace(SGD=mock.Mock(side_effect=AssertionError("optimizer forbidden")),
                                               AdamW=mock.Mock(side_effect=AssertionError("optimizer forbidden")))
        self.info = {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        self.loader = mock.Mock(return_value=(self.model, self.info))
        self.transformers = SimpleNamespace(__version__="4.57.1",
            AutoConfig=SimpleNamespace(from_pretrained=mock.Mock(return_value=config)),
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=GPT2TokenizerFast())),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self.loader))
        self.batches = {role: {k: GraphTensor((1, 4)) for k in ("input_ids", "attention_mask", "labels")}
                        for role in ("forget", "retain")}
        self.objective = GraphTensor(value=0.)
        self.gradient_result = {"finite": True, "nonzero": True, "parameter_elements_with_gradients": 134515008}

    @contextlib.contextmanager
    def runtime(self):
        # Each mocked pretrained load represents a fresh CPU model.
        self.model.parameter.device = SimpleNamespace(type="cpu")
        self.model.parameter.grad = None
        self.model.parameter._version = 0
        with contextlib.ExitStack() as stack:
            environment = {k: v for k, v in os.environ.items() if not k.startswith("PYTORCH_MPS_")
                           and k not in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")}
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            patches = [mock.patch.dict(os.environ, environment, clear=True),
                       mock.patch.dict(sys.modules, {"torch": self.fake_torch, "transformers": self.transformers,
                                                     "huggingface_hub": SimpleNamespace(__version__="0.36.2")}),
                       mock.patch.object(readonly.sys, "platform", "darwin"),
                       mock.patch.object(readonly.platform, "machine", return_value="arm64"),
                       mock.patch.object(readonly, "ensure_checkpoint", return_value={"weight_sha256": "fixture"}),
                       mock.patch.object(readonly, "_download", side_effect=AssertionError("download forbidden")),
                       mock.patch.object(readonly, "read_sample", side_effect=lambda t, stage: self.sample(stage)),
                       mock.patch.object(readonly.Monitor, "start"), mock.patch.object(readonly.time, "sleep"),
                       mock.patch.object(rehearsal, "select_pair", return_value=({}, {})),
                       mock.patch.object(rehearsal, "prepare_batches", return_value=(self.batches, {})),
                       mock.patch.object(rehearsal, "parameter_fingerprint", return_value="fixture-digest"),
                       mock.patch.object(rehearsal, "validate_gradients", return_value=self.gradient_result),
                       mock.patch.object(graddiff, "causal_lm_nll", return_value=GraphTensor()),
                       mock.patch.object(graddiff, "graddiff_objective", return_value=self.objective)]
            for patch in patches:
                stack.enter_context(patch)
            yield

    def run_mock(self):
        return rehearsal.run_rehearsal(enabled=True, report_path=self.path)

    def test_full_mock_control_flow_no_optimizer_local_only_and_report_integrity(self):
        with self.runtime(), mock.patch.object(self.objective, "backward", wraps=self.objective.backward) as backward:
            report = self.run_mock()
        self.assertEqual(report["status"], rehearsal.PASS)
        backward.assert_called_once_with()
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(self.model.checkpoint_kwargs, {"use_reentrant": False, "preserve_rng_state": True})
        self.fake_torch.optim.SGD.assert_not_called()
        self.fake_torch.optim.AdamW.assert_not_called()
        self.assertTrue(self.loader.call_args.kwargs["local_files_only"])
        self.assertFalse(self.loader.call_args.kwargs["trust_remote_code"])
        self.assertEqual(json.loads(self.path.read_text()), report)
        self.assertEqual(report["computation"]["parameter_sha256_before"], report["computation"]["parameter_sha256_after"])
        for key, value in (("training_authorized", True), ("weight_updates", 1), ("optimizer_created", True),
                           ("checkpoint_unchanged", False), ("actual_device", "cpu"), ("model_loaded", False),
                           ("telemetry_device", "cpu"), ("revision", "main")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                rehearsal.validate_report({**report, key: value})
        changed = copy.deepcopy(report)
        changed["measurements"] = []
        with self.assertRaisesRegex(ValueError, "measurement boundaries"):
            rehearsal.validate_report(changed)
        report["computation"]["objective"] = float("nan")
        with self.assertRaises(ValueError):
            rehearsal.validate_report(report)

    def test_preload_pressure_and_swap_fail_before_weight_loader(self):
        for key, value in (("available_physical_memory_lower_bound_bytes", 20 * rehearsal.GIB - 1),
                           ("macos_pressure_level", 2), ("system_memory_free_percentage", 19)):
            with self.runtime(), mock.patch.object(readonly, "read_sample", return_value={**self.sample(), key: value}):
                report = self.run_mock()
            self.assertEqual(report["status"], "STOP")
            self.loader.assert_not_called()
            self.path.unlink()
        samples = [self.sample(), {**self.sample(), "swap_used_bytes": 1}]
        with self.runtime(), mock.patch.object(readonly, "read_sample", side_effect=samples), \
             mock.patch.object(readonly, "ensure_checkpoint") as checkpoint:
            self.assertEqual(self.run_mock()["status"], "STOP")
            checkpoint.assert_not_called()
            self.loader.assert_not_called()

    def test_runtime_device_mismatch_and_unsafe_environment_stop(self):
        with self.runtime(), mock.patch.object(self.model, "to"):
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.assertEqual(len(self.model.calls), 0)
        self.path.unlink()
        with self.runtime(), mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "fallback"):
                self.run_mock()
        with self.runtime(), mock.patch.dict(os.environ, {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0"}):
            with self.assertRaisesRegex(RuntimeError, "override"):
                self.run_mock()

    def test_nonfinite_loss_output_device_or_parameter_mutation_cannot_pass(self):
        for field, value in (("loss", GraphTensor(value=float("nan"))), ("loss", GraphTensor(value=3.)),
                             ("loss", GraphTensor(device="cpu")), ("past_key_values", ("cache",))):
            self.model.output_change = lambda x, field=field, value=value: (setattr(x, field, value) or x)
            with self.runtime():
                report = self.run_mock()
            self.assertEqual(report["status"], "STOP")
            self.assertFalse(report["computation"]["backward_performed"])
            self.path.unlink()
        self.model.output_change = lambda x: x
        self.objective.on_backward = lambda: setattr(self.model.parameter, "_version", 1)
        with self.runtime():
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.assertIn("versions changed", report["error"]["reason"])

    def test_nonfinite_gradient_oom_watchdog_and_checkpoint_change_are_stops(self):
        for failure in (FloatingPointError("nonfinite gradient"), RuntimeError("MPS OOM")):
            with self.runtime(), mock.patch.object(rehearsal, "validate_gradients", side_effect=failure):
                report = self.run_mock()
            self.assertEqual(report["status"], "STOP")
            self.path.unlink()
        with self.runtime(), mock.patch.object(rehearsal.RehearsalMonitor, "close", side_effect=RuntimeError("watchdog")):
            self.assertEqual(self.run_mock()["status"], "STOP")
        self.path.unlink()
        with self.runtime(), mock.patch.object(readonly, "ensure_checkpoint", side_effect=[{"weight_sha256": "a"}, {"weight_sha256": "b"}]):
            self.assertEqual(self.run_mock()["status"], "STOP")

    def test_pressure_during_backward_retains_failing_reading_and_stops(self):
        original = self.sample

        def pressure_after_backward(stage):
            sample = original(stage)
            if stage == "after_backward_gradients_live":
                sample["macos_pressure_level"] = 2
            return sample

        with self.runtime(), mock.patch.object(readonly, "read_sample", side_effect=lambda t, stage: pressure_after_backward(stage)):
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.assertTrue(report["computation"]["backward_performed"])
        self.assertEqual(report["measurements"][-1]["macos_pressure_level"], 2)
        self.assertFalse(report["checkpoint_unchanged"])

    def test_background_watchdog_failure_interrupts_and_cannot_be_ignored(self):
        monitor = rehearsal.RehearsalMonitor(self.fake_torch, profile=self.profile, device="mps")
        with mock.patch.object(monitor.stop_event, "wait", return_value=False), \
             mock.patch.object(monitor, "capture", side_effect=RuntimeError("lost memory telemetry")), \
             mock.patch.object(readonly._thread, "interrupt_main") as interrupt:
            monitor._loop()
        interrupt.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "lost memory telemetry"):
            monitor.check()

    def test_changed_parameter_bytes_with_unchanged_versions_still_stop(self):
        with self.runtime(), mock.patch.object(rehearsal, "parameter_fingerprint", side_effect=["before", "after"]):
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.assertEqual(self.model.parameter._version, 0)
        self.assertIn("values or versions changed", report["error"]["reason"])

    def test_report_overwrite_symlink_escape_rejected_before_load(self):
        self.path.write_text("keep")
        with self.runtime(), self.assertRaises(ValueError):
            self.run_mock()
        self.assertEqual(self.path.read_text(), "keep")
        link = self.path.parent / "link.json"
        link.symlink_to(self.path)
        for path in (link, readonly.ROOT / "report.json", readonly.CHECKPOINT / "report.json"):
            with self.runtime(), self.assertRaises(ValueError):
                rehearsal.run_rehearsal(enabled=True, report_path=path)
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
