"""Mock-only physical preflight tests: no network, pretrained load or MPS work."""

import contextlib
import copy
import hashlib
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

from src import small_model_preflight as preflight
from src import graddiff


class FakeTensor:
    def __init__(self, shape=(), *, device="mps", dtype="float32", value=2.0):
        self.shape, self.device, self.dtype = shape, SimpleNamespace(type=device), dtype
        self.value, self.requires_grad = value, False

    def item(self):
        return self.value

    def __getitem__(self, key):
        return FakeTensor(value=1)

    def argmax(self):
        return self


class FakeTorch:
    __version__ = "2.9.1"
    float32, long = "float32", "int64"

    def __init__(self):
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: True))
        self.mps = SimpleNamespace(synchronize=mock.Mock(), current_allocated_memory=lambda: 100,
                                   driver_allocated_memory=lambda: 200)
        self.manual_seed = mock.Mock()
        self.use_deterministic_algorithms = mock.Mock()
        self.in_inference_mode = False

    def device(self, name):
        class Device:
            type = name

            def __str__(self):
                return name
        return Device()

    @contextlib.contextmanager
    def inference_mode(self):
        self.in_inference_mode = True
        try:
            yield
        finally:
            self.in_inference_mode = False

    def tensor(self, data, *, dtype, device):
        return FakeTensor((len(data), len(data[0])), device=device.type, dtype=dtype)

    def ones_like(self, tensor):
        return FakeTensor(tensor.shape, device=tensor.device.type, dtype=tensor.dtype)


class GPT2TokenizerFast:
    is_fast, chat_template, padding_side = True, None, "right"
    eos_token_id, eos_token, all_special_ids = 0, "<|endoftext|>", [0]

    def __init__(self):
        self.pad_token_id = None

    def __len__(self):
        return 49152

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("ascii"))

    @property
    def pad_token(self):
        return self.eos_token if self.pad_token_id == 0 else None

    @pad_token.setter
    def pad_token(self, value):
        assert value == self.eos_token
        self.pad_token_id = 0


class LlamaForCausalLM:
    def __init__(self, config, torch):
        self.config, self.torch, self.training = config, torch, True
        self.parameter = SimpleNamespace(requires_grad=True, grad=None, dtype="float32",
                                         device=SimpleNamespace(type="cpu"), _version=0, numel=lambda: 134515008)
        self.output_change = lambda output: output

    def parameters(self):
        return iter([self.parameter])

    def buffers(self):
        return iter([])

    def requires_grad_(self, value):
        self.parameter.requires_grad = value
        return self

    def eval(self):
        self.training = False
        return self

    def to(self, device):
        self.parameter.device = device
        return self

    def __call__(self, **kwargs):
        assert self.torch.in_inference_mode and not self.parameter.requires_grad
        assert kwargs["use_cache"] is False and kwargs["return_dict"] is True
        assert kwargs["output_hidden_states"] is False and kwargs["output_attentions"] is False
        return self.output_change(SimpleNamespace(
            logits=FakeTensor((1, kwargs["input_ids"].shape[1], 49152)), loss=FakeTensor(), past_key_values=None))


class SmallModelPhysicalPreflightTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(preflight.PROFILE.read_text())
        config_dict = copy.deepcopy(self.profile["model"])
        config_dict["architectures"] = ["LlamaForCausalLM"]
        self.config = SimpleNamespace(use_cache=False, _attn_implementation="sdpa", to_dict=lambda: config_dict)
        self.torch = FakeTorch()
        self.model = LlamaForCausalLM(self.config, self.torch)
        self.tokenizer = GPT2TokenizerFast()
        self.loading_info = {key: [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        self.transformers = SimpleNamespace(__version__="4.57.1",
            AutoConfig=SimpleNamespace(from_pretrained=mock.Mock(return_value=self.config)),
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=self.tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=mock.Mock(return_value=(self.model, self.loading_info))))
        preflight.OUTPUTS.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=preflight.OUTPUTS)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "report.json"
        self.network = mock.patch.object(preflight, "_download", side_effect=AssertionError("network forbidden in tests")).start()
        self.addCleanup(mock.patch.stopall)
        self.counter = 0

    def sample(self, stage="before_load"):
        self.counter += 1
        return {"stage": stage, "timestamp_ns": time.monotonic_ns() - 1000000, "sample_duration_ns": 100000,
                "total_physical_memory_bytes": 48 * preflight.GIB,
                "available_physical_memory_lower_bound_bytes": 36 * preflight.GIB,
                "system_memory_free_percentage": 90, "macos_pressure_level": 1,
                "process_current_rss_bytes": preflight.GIB, "process_peak_rss_bytes": preflight.GIB,
                "mps_tensor_allocated_bytes": 100, "mps_driver_allocated_bytes": 200,
                "swap_used_bytes": 0, "pageouts": 0}

    def validate_sample(self, sample, **kwargs):
        return preflight.validate_sample(sample, profile=self.profile, device="mps", **kwargs)

    @contextlib.contextmanager
    def runtime(self):
        with contextlib.ExitStack() as stack:
            environment = {k: v for k, v in os.environ.items()
                           if not k.startswith("PYTORCH_MPS_") and k not in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")}
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
            stack.enter_context(mock.patch.object(preflight.sys, "platform", "darwin"))
            stack.enter_context(mock.patch.object(preflight.platform, "machine", return_value="arm64"))
            stack.enter_context(mock.patch.dict(sys.modules, {"torch": self.torch, "transformers": self.transformers,
                                       "huggingface_hub": SimpleNamespace(__version__="0.36.2")}))
            stack.enter_context(mock.patch.object(preflight, "ensure_checkpoint", return_value={"license": "Apache-2.0"}))
            stack.enter_context(mock.patch.object(preflight, "read_sample", side_effect=lambda torch, stage: self.sample(stage)))
            stack.enter_context(mock.patch.object(preflight.Monitor, "start"))
            stack.enter_context(mock.patch.object(preflight.time, "sleep"))
            stack.enter_context(mock.patch.object(graddiff, "causal_lm_nll", return_value=FakeTensor()))
            yield

    def run_mock(self):
        return preflight.run_preflight(enabled=True, allow_download=False, report_path=self.path)

    def test_plan_import_is_offline_and_has_no_model_or_framework_import(self):
        code = ("import sys; from src import small_model_preflight as p; assert p.main([])==0; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=preflight.ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["downloads_enabled"])
        self.assertFalse(report["training_authorized"])
        self.assertIsNone(report["approved_small_model_training_threshold_bytes"])
        self.assertEqual(report["preload_safety_floor_bytes"], 16 * preflight.GIB)
        self.assertIsNone(report["readonly_memory_policy"]["baseline_process_peak_rss_bytes"])

    def test_cli_requires_explicit_runtime_and_report_before_any_work(self):
        for args in (["--allow-download"], ["--run-preflight"], ["--report-path", str(self.path)],
                     ["--apply-updates"], ["--backward-only"]):
            with mock.patch.object(preflight, "run_preflight") as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    preflight.main(args)
                run.assert_not_called()
        with self.assertRaises(ValueError):
            preflight.run_preflight(enabled=False, allow_download=False, report_path=self.path)
        for status, expected in (("STOP", 2), ("readonly_measurement_passed_not_training_clearance", 0)):
            with mock.patch.object(preflight, "run_preflight", return_value={"status": status}) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(preflight.main(["--run-preflight", "--report-path", str(self.path)]), expected)
                self.assertFalse(run.call_args.kwargs["allow_download"])

    def test_mock_forward_is_frozen_inference_local_only_and_output_schema(self):
        with self.runtime():
            report = self.run_mock()
        self.assertEqual(report["status"], "readonly_measurement_passed_not_training_clearance")
        self.assertEqual(json.loads(self.path.read_text()), report)
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["actual_device"], "mps")
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["backward_performed"])
        self.assertFalse(report["optimizer_created"])
        self.assertEqual(report["weight_updates"], 0)
        self.assertTrue(report["parameters_frozen"])
        self.assertTrue(report["inference"]["parameter_versions_unchanged"])
        self.assertEqual(report["inference"]["forward_count"], 1)
        stages = [s["stage"] for s in report["measurements"]]
        for stage in ("before_load", "after_load", "after_inference_outputs_live", "after_inference"):
            self.assertIn(stage, stages)
        self.assertIsNone(report["peaks"]["native_mps_peak_bytes"])
        self.assertEqual(report["peaks"]["sampled_maxima_bytes"]["mps_driver_allocated_bytes"], 200)
        kwargs = self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(kwargs["use_safetensors"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.network.assert_not_called()

    def test_measurement_readings_fail_closed_on_invalid_types_pressure_and_reserves(self):
        for key in preflight.INTEGER_FIELDS:
            for value in (None, True, -1, float("nan"), "0"):
                sample = self.sample()
                sample[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    self.validate_sample(sample)
        for key, value in (("macos_pressure_level", 2), ("macos_pressure_level", 4),
                           ("system_memory_free_percentage", 19), ("system_memory_free_percentage", 101),
                           ("available_physical_memory_lower_bound_bytes", 6 * preflight.GIB - 1),
                           ("total_physical_memory_bytes", 47 * preflight.GIB),
                           ("mps_tensor_allocated_bytes", 201), ("process_peak_rss_bytes", 0),
                           ("process_current_rss_bytes", 0),
                           ("sample_duration_ns", 3000000000)):
            sample = self.sample()
            sample[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate_sample(sample)
        sample = self.sample()
        sample["available_physical_memory_lower_bound_bytes"] = 16 * preflight.GIB
        self.validate_sample(sample, before_load=True)
        sample["available_physical_memory_lower_bound_bytes"] -= 1
        with self.assertRaises(RuntimeError):
            self.validate_sample(sample, before_load=True)
        self.validate_sample(sample, before_load=False)

    def test_swap_growth_and_pageout_changes_stop(self):
        baseline = self.sample()
        for key in ("swap_used_bytes", "pageouts"):
            sample = self.sample()
            sample[key] += 1
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.validate_sample(sample, baseline=baseline)

    def test_readonly_budget_arithmetic_and_process_headroom(self):
        policy = preflight.readonly_memory_policy(self.profile, device="mps", baseline_peak_rss_bytes=226738176)
        terms = policy["estimated_incremental_terms_bytes"]
        self.assertEqual(terms["three_fp32_weight_copies_estimate"], 3 * 538060032)
        self.assertEqual(terms["source_file_residency_allowance_estimate"], 269060552)
        self.assertEqual(sum(terms.values()), policy["estimated_incremental_total_bytes"])
        self.assertEqual(policy["preload_available_required_bytes"], 16 * preflight.GIB)
        self.assertFalse(policy["training_authorized"])
        self.assertFalse(policy["measured_model_requirement"])
        sample = self.sample()
        sample["process_current_rss_bytes"] = sample["process_peak_rss_bytes"] = 3 * preflight.GIB
        sample["available_physical_memory_lower_bound_bytes"] = 18 * preflight.GIB
        self.validate_sample(sample, before_load=True)
        sample["available_physical_memory_lower_bound_bytes"] -= 1
        with self.assertRaisesRegex(RuntimeError, "18 GiB"):
            self.validate_sample(sample, before_load=True)

    def test_continuing_and_preload_exact_boundaries(self):
        for before_load, floor in ((True, 16), (False, 6)):
            sample = self.sample()
            sample["system_memory_free_percentage"] = 20
            sample["available_physical_memory_lower_bound_bytes"] = floor * preflight.GIB
            self.validate_sample(sample, before_load=before_load)
            sample["available_physical_memory_lower_bound_bytes"] -= 1
            with self.subTest(before_load=before_load), self.assertRaises(RuntimeError):
                self.validate_sample(sample, before_load=before_load)

    def test_readonly_policy_rejects_device_profile_and_baseline_drift(self):
        for device in (None, "", "cpu", "cuda", "unknown"):
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                preflight.validate_sample(self.sample(), profile=self.profile, device=device, before_load=True)
        for peak in (True, -1, float("inf"), "0"):
            with self.subTest(peak=peak), self.assertRaises(RuntimeError):
                preflight.readonly_memory_policy(self.profile, device="mps", baseline_peak_rss_bytes=peak)
        for section, key, value in (("model", "repo_id", "unknown"), ("model", "revision", "main"),
                                    ("model", "parameter_count", 1235814400),
                                    ("training", "precision", "bfloat16")):
            profile = copy.deepcopy(self.profile)
            profile[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                preflight.validate_sample(self.sample(), profile=profile, device="mps")
        profile = copy.deepcopy(self.profile)
        profile["weight_updates_enabled"] = True
        with self.assertRaises(ValueError):
            preflight.readonly_memory_policy(profile, device="mps")

    def test_stale_future_zero_and_unordered_measurements_stop(self):
        for key, value in (("timestamp_ns", 0), ("timestamp_ns", time.monotonic_ns() + 10**10),
                           ("timestamp_ns", time.monotonic_ns() - 3 * 10**9),
                           ("sample_duration_ns", 0)):
            sample = self.sample()
            sample[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.validate_sample(sample)
        sample = self.sample()
        with self.assertRaisesRegex(RuntimeError, "timestamps"):
            self.validate_sample(sample, baseline=sample)
        for baseline in ({}, {**self.sample(), "swap_used_bytes": True}):
            with self.assertRaises(RuntimeError):
                self.validate_sample(self.sample(), baseline=baseline)

    def test_swap_growth_before_acquisition_and_after_swap_decline_stops(self):
        samples = [self.sample("baseline"), self.sample("before_checkpoint")]
        samples[1]["swap_used_bytes"] = 1
        with self.runtime(), mock.patch.object(preflight, "read_sample", side_effect=samples), \
             mock.patch.object(preflight, "ensure_checkpoint") as acquire:
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.assertIn("swap/page-outs grew", report["error"]["reason"])
        acquire.assert_not_called()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        monitor = preflight.Monitor(self.torch, profile=self.profile, device="mps")
        samples = [self.sample() for _ in range(3)]
        for sample, swap in zip(samples, (100, 50, 75)):
            sample["swap_used_bytes"] = swap
        with mock.patch.object(preflight, "read_sample", side_effect=samples):
            monitor.capture("baseline", before_load=True)
            monitor.capture("before_checkpoint", before_load=True)
            with self.assertRaisesRegex(RuntimeError, "swap/page-outs grew"):
                monitor.capture("before_load", before_load=True)

    def test_readonly_admission_below_llama_floor_remains_frozen(self):
        original_sample = self.sample

        def smaller_sample(stage):
            sample = original_sample(stage)
            sample["available_physical_memory_lower_bound_bytes"] = 18 * preflight.GIB
            return sample

        with self.runtime(), mock.patch.object(self, "sample", side_effect=smaller_sample):
            report = self.run_mock()
        self.assertEqual(report["status"], "readonly_measurement_passed_not_training_clearance")
        self.assertEqual(report["readonly_memory_policy"]["preload_available_required_bytes"], 16 * preflight.GIB)
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["backward_performed"])
        self.assertFalse(report["optimizer_created"])
        self.assertEqual(report["weight_updates"], 0)
        self.assertFalse(self.model.parameter.requires_grad)
        self.assertIsNone(self.model.parameter.grad)
        self.assertEqual(self.model.parameter._version, 0)

    def test_preload_failures_prevent_checkpoint_and_weight_access(self):
        for key, value in (("available_physical_memory_lower_bound_bytes", 16 * preflight.GIB - 1),
                           ("macos_pressure_level", 4), ("process_current_rss_bytes", None)):
            sample = self.sample()
            sample[key] = value
            with self.runtime(), mock.patch.object(preflight, "read_sample", return_value=sample), \
                 mock.patch.object(preflight, "ensure_checkpoint") as acquire:
                report = self.run_mock()
            with self.subTest(key=key):
                self.assertEqual(report["status"], "STOP")
                acquire.assert_not_called()
                self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
            self.path.unlink()

    def test_device_and_allocator_environment_fail_before_loader(self):
        with self.runtime():
            for env in ({"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0"},
                        {"PYTORCH_MPS_LOW_WATERMARK_RATIO": "0"}, {"WORLD_SIZE": "2"}):
                with mock.patch.dict(os.environ, env), self.subTest(env=env), self.assertRaises(RuntimeError):
                    self.run_mock()
            with mock.patch.object(self.torch.backends.mps, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "MPS unavailable"):
                    self.run_mock()
            with mock.patch.object(preflight.platform, "machine", return_value="x86_64"):
                with self.assertRaises(RuntimeError):
                    self.run_mock()
            self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_missing_checkpoint_or_pressure_never_reaches_model_loader(self):
        with self.runtime():
            with mock.patch.object(preflight, "ensure_checkpoint", side_effect=FileNotFoundError("missing weights")):
                report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.path.unlink()  # remove only this test's explicitly created temporary report
        with self.runtime(), mock.patch.object(preflight, "validate_sample", side_effect=RuntimeError("pressure")):
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_incompatible_config_loading_info_and_device_are_stops(self):
        self.config.to_dict()["hidden_size"] = 577
        with self.runtime():
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.config.to_dict()["hidden_size"] = 576
        self.path.unlink()
        self.loading_info["missing_keys"] = ["missing.weight"]
        with self.runtime():
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.loading_info["missing_keys"] = []
        self.path.unlink()
        with self.runtime(), mock.patch.object(self.model, "to", return_value=self.model):
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        self.assertIn("requested device", report["error"]["reason"])

    def test_invalid_outputs_nonfinite_parity_and_weight_mutation_fail(self):
        for field, value in (("loss", FakeTensor(value=float("nan"))),
                             ("loss", FakeTensor(value=3)), ("loss", FakeTensor(device="cpu")),
                             ("logits", FakeTensor(shape=(1, 3, 4))), ("past_key_values", ("cache",))):
            self.model.output_change = lambda output, field=field, value=value: (setattr(output, field, value) or output)
            with self.runtime():
                report = self.run_mock()
            with self.subTest(field=field):
                self.assertEqual(report["status"], "STOP")
            self.path.unlink()
        self.model.output_change = lambda output: (setattr(self.model.parameter, "_version", 1) or output)
        with self.runtime():
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")

    def test_oom_and_watchdog_errors_cannot_report_success(self):
        with self.runtime():
            self.transformers.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError("MPS OOM")
            report = self.run_mock()
        self.assertEqual(report["status"], "STOP")
        monitor = preflight.Monitor(self.torch, profile=self.profile, device="mps")
        with mock.patch.object(monitor.stop_event, "wait", return_value=False), \
             mock.patch.object(monitor, "capture", side_effect=RuntimeError("pressure")), \
             mock.patch.object(preflight._thread, "interrupt_main") as interrupt:
            monitor._loop()
        interrupt.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            monitor.check()

    def test_report_no_overwrite_symlink_or_repository_escape(self):
        self.path.write_text("existing")
        with self.assertRaises(ValueError):
            preflight.validate_report_path(self.path)
        self.assertEqual(self.path.read_text(), "existing")
        link = self.path.parent / "link.json"
        link.symlink_to(self.path)
        for path in (link, preflight.ROOT / "report.json", preflight.CHECKPOINT / "report.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                preflight.validate_report_path(path)

    def test_checkpoint_unavailable_corrupt_license_config_and_hash_without_network(self):
        # Tiny fixture files exercise acquisition/hashes without an official
        # weight file or dependency on ignored local metadata caches.
        profile = copy.deepcopy(self.profile)
        files = {name: b"{}" for name in profile["model"]["metadata_sha256"]}
        files["README.md"] = b"---\nlicense: apache-2.0\n---\n"
        files["config.json"] = json.dumps(self.config.to_dict()).encode()
        files["model.safetensors"] = b"tiny-mocked-weights"
        profile["model"]["metadata_sha256"] = {name: hashlib.sha256(files[name]).hexdigest()
                                                   for name in profile["model"]["metadata_sha256"]}
        profile["model"]["weight_file_metadata_only"].update(
            size_bytes=len(files["model.safetensors"]),
            hub_lfs_sha256=hashlib.sha256(files["model.safetensors"]).hexdigest())
        (preflight.ROOT / ".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=preflight.ROOT / ".cache") as tmp, \
             mock.patch.object(preflight, "CHECKPOINT", Path(tmp) / preflight.REVISION), \
             mock.patch.object(preflight, "select_memory_guard", return_value="fixture-only"):
            checkpoint = preflight.CHECKPOINT
            with self.assertRaises(FileNotFoundError):
                preflight.ensure_checkpoint(profile, allow_download=False)
            self.network.assert_not_called()
            order = []

            def fake_download(name):
                checkpoint.mkdir(exist_ok=True)
                (checkpoint / name).write_bytes(files[name])
                order.append(name)

            self.network.side_effect = fake_download
            with mock.patch.object(preflight.shutil, "disk_usage", return_value=SimpleNamespace(free=2 * preflight.GIB)):
                provenance = preflight.ensure_checkpoint(profile, allow_download=True)
            self.assertEqual(order[0], "README.md")
            self.assertEqual(order[-1], "model.safetensors")
            self.assertEqual(provenance["license"], "Apache-2.0")
            self.assertEqual(provenance["revision"], preflight.REVISION)
            (checkpoint / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "weights"):
                preflight.ensure_checkpoint(profile, allow_download=False)
            (checkpoint / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "weight size/SHA"):
                preflight.ensure_checkpoint(profile, allow_download=False)
            (checkpoint / "README.md").write_text("license: changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                preflight.ensure_checkpoint(profile, allow_download=False)
            profile["model"]["metadata_sha256"]["README.md"] = hashlib.sha256(b"license: changed").hexdigest()
            with self.assertRaisesRegex(ValueError, "license mismatch"):
                preflight.ensure_checkpoint(profile, allow_download=False)
            (checkpoint / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unexpected checkpoint"):
                preflight.ensure_checkpoint(profile, allow_download=False)

    def test_physical_reader_parses_direct_counters_with_mocked_os_commands(self):
        vm = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
              'Pages free: 2000000.\nPages speculative: 100000.\nPageouts: 4.\n')
        outputs = [vm, "System-wide memory free percentage: 95%", f"{48 * preflight.GIB}\n1\nused = 0.00M", "1024"]
        with mock.patch.object(preflight, "_command", side_effect=outputs), \
             mock.patch.object(preflight.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=2 * 1024**2)):
            sample = preflight.read_sample(self.torch, "before_load")
        self.assertEqual(sample["available_physical_memory_lower_bound_bytes"], 2100000 * 16384)
        self.assertEqual(sample["process_current_rss_bytes"], 1024**2)
        self.assertEqual(sample["process_peak_rss_bytes"], 2 * 1024**2)
        self.assertEqual(sample["pageouts"], 4)
        self.validate_sample(sample, before_load=True)
        with mock.patch.object(preflight, "_command", return_value="bad telemetry"):
            with self.assertRaises(RuntimeError):
                preflight.read_sample(self.torch, "before_load")


if __name__ == "__main__":
    unittest.main()
