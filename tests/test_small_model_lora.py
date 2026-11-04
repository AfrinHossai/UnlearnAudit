"""Task 53 offline gates, native LoRA algebra, assigned gradients, mocked runs.

No checkpoint/model loading, real backward or optimizer.step is permitted.
"""

import contextlib
import copy
import io
import json
import math
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

from src import lora_adapter as native
from src import small_model_lora as lora


def sample(stage="preload", timestamp=None):
    return {"stage": stage, "timestamp_ns": timestamp or time.monotonic_ns() - 1_000_000,
            "sample_duration_ns": 100_000, "total_physical_memory_bytes": 48 * lora.GIB,
            "available_physical_memory_lower_bound_bytes": 24 * lora.GIB,
            "system_memory_free_percentage": 90, "macos_pressure_level": 1,
            "process_current_rss_bytes": lora.GIB, "process_peak_rss_bytes": lora.GIB,
            "mps_tensor_allocated_bytes": 100, "mps_driver_allocated_bytes": 200,
            "swap_used_bytes": 0, "pageouts": 0}


class OfflineCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for obj, name in ((torch.Tensor, "backward"), (torch.autograd, "backward"),
                          (torch.autograd, "grad"), (torch.optim.SGD, "step"), (torch.optim.AdamW, "step")):
            self.stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError("forbidden real backward/update")))

    def temporary_root(self):
        tmp = tempfile.TemporaryDirectory(dir=lora.ROOT / "outputs")
        root = Path(tmp.name)
        def cleanup():
            for p in [root, *root.rglob("*")]:
                if p.is_dir():
                    p.chmod(0o755)
            tmp.cleanup()
        self.addCleanup(cleanup)
        self.stack.enter_context(mock.patch.object(lora, "RUNS", root))
        return root


class PolicyTests(OfflineCase):
    def test_default_cli_rejects_even_transitive_ml_imports(self):
        code = '''
import sys, json
class NoML:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors'):
   raise AssertionError('plan imported ML runtime')
sys.meta_path.insert(0, NoML())
from src.small_model_lora import main
raise SystemExit(main([]))
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=lora.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "validation_only")
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["training_authorized"])
        self.assertEqual(report["optimizer_steps"], 0)

    def test_pinned_evidence_header_and_parameter_arithmetic(self):
        spec, manifest, historical = lora.frozen_inputs()
        metadata = lora.metadata_targets()
        self.assertEqual(len(metadata["targets"]), 60)
        self.assertEqual(len(manifest["training"]), 32)
        self.assertEqual(len(manifest["evaluation"]), 76)
        self.assertEqual(historical["base_parameter_sha256"], lora.BASE_SHA256)
        self.assertEqual(spec["pilot_case_id"], "dev:09:0")
        self.assertEqual(sum(math.prod(s) for s in lora.expected_adapter_shapes().values()), 230400)
        self.assertEqual(lora.sha256_file(lora.HISTORICAL), lora.HISTORICAL_SHA256)

    def test_separate_memory_policy_and_dynamic_headroom(self):
        p = lora.memory_policy()
        self.assertEqual(p["estimated_incremental_total_bytes"], 14769985736)
        self.assertEqual(p["preload_available_required_bytes"], 20 * lora.GIB)
        self.assertEqual(lora.memory_policy(3 * lora.GIB)["preload_available_required_bytes"], 22 * lora.GIB)
        self.assertFalse(p["calibrated_training_requirement"])
        self.assertEqual(lora.memory_policy(mode="baseline")["max_optimizer_steps"], 0)
        self.assertFalse(lora.memory_policy(mode="baseline")["backward_permitted"])
        old = lora.acquisition.memory_policy(json.loads(lora.PROFILE.read_text()))
        self.assertEqual(old["preload_available_required_bytes"], 24 * lora.GIB)
        self.assertNotEqual(old["guard_id"], p["guard_id"])
        for args in ((-1, "pilot"), (True, "pilot"), (1.0, "pilot"), (0, "cpu")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                lora.memory_policy(args[0], mode=args[1])

    def test_admission_boundary_and_live_freshness(self):
        reading = sample()
        reading["available_physical_memory_lower_bound_bytes"] = 20 * lora.GIB
        lora.validate_sample(reading, before_load=True)
        reading["available_physical_memory_lower_bound_bytes"] -= 1
        with self.assertRaisesRegex(RuntimeError, "preload"):
            lora.validate_sample(reading, before_load=True)
        for delta in (-3_000_000_000, 3_000_000_000):
            bad = sample(timestamp=time.monotonic_ns() + delta)
            with self.assertRaisesRegex(RuntimeError, "stale/future"):
                lora.validate_sample(bad)

    def test_unsafe_counter_readings_fail_closed(self):
        for key, value in (("total_physical_memory_bytes", 32 * lora.GIB),
                           ("available_physical_memory_lower_bound_bytes", 6 * lora.GIB - 1),
                           ("macos_pressure_level", 2), ("system_memory_free_percentage", 19),
                           ("mps_tensor_allocated_bytes", 201), ("sample_duration_ns", 0),
                           ("mps_driver_allocated_bytes", 8 * lora.GIB + 1),
                           ("process_peak_rss_bytes", 8 * lora.GIB + 1),
                           ("timestamp_ns", True)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                lora.validate_sample({**sample(), key: value})

    def test_cumulative_swap_pageouts_order_and_gap(self):
        first = sample(timestamp=1_000_000_000)
        previous = sample(timestamp=2_000_000_000)
        current = sample(timestamp=3_000_000_000)
        lora.counter_checks(current, previous=previous, first=first, active=True)
        for key, value in (("swap_used_bytes", 1), ("pageouts", 1),
                           ("total_physical_memory_bytes", 49 * lora.GIB),
                           ("timestamp_ns", 2_000_000_000), ("timestamp_ns", 4_000_000_001),
                           ("process_peak_rss_bytes", lora.GIB - 1)):
            bad = {**current, key: value}
            if key == "process_peak_rss_bytes":
                bad["process_current_rss_bytes"] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                lora.counter_checks(bad, previous=previous, first=first, active=True)

    def test_wrong_authorization_fails_before_paths_or_work(self):
        for enabled, token in ((False, lora.AUTHORIZATION), (True, None), (True, "yes"),
                               (1, lora.AUTHORIZATION), (True, lora.acquisition.AUTHORIZATION)):
            with self.subTest(enabled=enabled, token=token), mock.patch.object(lora, "new_run_path") as path:
                with self.assertRaises(ValueError):
                    lora.run(mode="pilot", enabled=enabled, authorization=token, run_path=Path("never"))
                path.assert_not_called()

    def test_cli_cannot_change_device_optimizer_budget_or_resume(self):
        for args in (("--device", "cpu"), ("--max-steps", "2"), ("--learning-rate", "1"),
                     ("--allow-download",), ("--resume",), ("--rank", "8"),
                     ("--authorize-weight-update", lora.AUTHORIZATION)):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                lora.main(list(args))
            self.assertEqual(exc.exception.code, 2)

    def test_baseline_rejects_update_arguments(self):
        for kwargs in ({"authorization": lora.AUTHORIZATION}, {"baseline_path": Path("x")},
                       {"baseline_sha256": "a" * 64}, {"enabled": False}):
            options = {"mode": "baseline", "enabled": True, "authorization": None, "run_path": Path("never")}
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), mock.patch.object(lora, "new_run_path") as path, self.assertRaises(ValueError):
                lora.run(**options)
            path.assert_not_called()

    def test_paths_reject_old_outputs_aliases_symlinks_and_collisions(self):
        root = self.temporary_root()
        good = root / "pilot-new"
        self.assertEqual(lora.new_run_path(good, "pilot"), good)
        for path in (lora.HISTORICAL, root, root / "nested/pilot-x", root / "baseline-x", root / "pilot-../x"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                lora.new_run_path(path, "pilot")
        good.mkdir()
        with self.assertRaises(ValueError):
            lora.new_run_path(good, "pilot")
        alias = root / "pilot-alias"
        alias.symlink_to(good, target_is_directory=True)
        with self.assertRaises(ValueError):
            lora.new_run_path(alias, "pilot")
        alias.unlink()

    def test_environment_overrides_and_nonterminal_stop(self):
        for key, value in (("PYTORCH_ENABLE_MPS_FALLBACK", "1"), ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0"),
                           ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0"), ("PYTORCH_MPS_FAST_MATH", "1"),
                           ("PYTORCH_MPS_PREFER_METAL", "1"), ("WORLD_SIZE", "2"), ("LOCAL_WORLD_SIZE", "2")):
            with self.subTest(key=key), mock.patch.object(lora.sys.stdin, "isatty", return_value=True), \
                    mock.patch.dict(os.environ, {key: value}), self.assertRaises(RuntimeError):
                lora.environment_before_import()
        with mock.patch.object(lora.sys.stdin, "isatty", return_value=False), self.assertRaises(RuntimeError):
            lora.environment_before_import()

    def test_dependency_and_profile_drift_stop_without_import(self):
        with mock.patch.object(lora.importlib.metadata, "version", return_value="wrong"), self.assertRaises(RuntimeError):
            lora.dependency_versions()
        with mock.patch.object(lora, "SPEC_SHA256", "0" * 64), self.assertRaises(ValueError):
            lora.frozen_inputs()


class MonitorTests(OfflineCase):
    def monitor(self):
        runtime = SimpleNamespace(mps=SimpleNamespace(synchronize=mock.Mock()))
        monitor = lora.LoRAMonitor(runtime, profile=json.loads(lora.PROFILE.read_text()), device="mps", mode="pilot")
        return monitor, runtime

    def test_capture_uses_lora_floor_and_keeps_failing_reading(self):
        monitor, _ = self.monitor()
        reading = sample(); reading["available_physical_memory_lower_bound_bytes"] = 20 * lora.GIB - 1
        with mock.patch.object(lora.telemetry, "read_sample", return_value=reading), self.assertRaisesRegex(RuntimeError, "LoRA preload"):
            monitor.capture("preload", before_load=True)
        self.assertEqual(monitor.samples, [reading])

    def test_synchronized_boundary_updates_background_stage(self):
        monitor, runtime = self.monitor()
        with mock.patch.object(lora.telemetry, "read_sample", return_value=sample("after_clipping")):
            monitor.boundary("after_clipping")
        self.assertEqual(monitor.stage, "after_clipping")
        runtime.mps.synchronize.assert_called_once()

    def test_time_budget_and_last_sample_age_stop_without_weight_operations(self):
        monitor, _ = self.monitor()
        monitor.started -= 1801
        with self.assertRaisesRegex(RuntimeError, "30-minute"):
            monitor.check()
        monitor, _ = self.monitor(); monitor.thread = object()
        monitor.samples.append(sample(timestamp=time.monotonic_ns() - 3_000_000_000))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            monitor.check()

    def test_cpu_and_unknown_modes_are_rejected(self):
        profile = json.loads(lora.PROFILE.read_text())
        for device, mode in (("cpu", "pilot"), ("mps", "unknown")):
            with self.subTest(device=device, mode=mode), self.assertRaises((ValueError, RuntimeError)):
                lora.LoRAMonitor(object(), profile=profile, device=device, mode=mode)


class NativeAlgebraTests(OfflineCase):
    def tiny(self):
        torch.manual_seed(2026)
        return native.LoRALinear(torch.nn.Linear(3, 2, bias=False), rank=2, alpha=4)

    def test_initial_zero_effect_base_freeze_and_deterministic_initialization(self):
        layer = self.tiny()
        x = torch.tensor([[0.2, -0.3, 0.5]])
        before = layer.base_layer.weight.detach().clone()
        with torch.no_grad():
            self.assertTrue(torch.equal(layer(x), layer.base_layer(x)))
        self.assertFalse(layer.base_layer.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)
        other = self.tiny()
        self.assertTrue(torch.equal(layer.lora_A, other.lora_A))
        self.assertTrue(torch.equal(layer.lora_B, torch.zeros_like(layer.lora_B)))
        layer.train(); layer.eval()
        self.assertTrue(torch.equal(before, layer.base_layer.weight))  # never merge on train/eval

    def test_native_forward_matches_effective_matrix(self):
        layer = self.tiny()
        with torch.no_grad():
            layer.lora_B.copy_(torch.tensor([[0.1, -0.2], [0.3, 0.4]]))
            x = torch.tensor([[0.2, -0.3, 0.5], [-0.5, 0.7, 0.1]])
            matrix = layer.base_layer.weight + layer.scale * layer.lora_B @ layer.lora_A
            torch.testing.assert_close(layer(x), x @ matrix.T, atol=1e-7, rtol=1e-6)

    def test_finite_difference_factor_gradients_without_backward(self):
        layer = self.tiny()
        x = torch.tensor([[0.2, -0.3, 0.5]])
        v = torch.tensor([[0.7, -0.4]])
        def check():
            g_b = layer.scale * v.T @ (x @ layer.lora_A.T)
            g_a = layer.scale * (v @ layer.lora_B).T @ x
            for p, analytical in ((layer.lora_A, g_a), (layer.lora_B, g_b)):
                for i in range(p.shape[0]):
                    for j in range(p.shape[1]):
                        original = float(p[i, j])
                        p[i, j] = original + 0.001
                        plus = float((layer(x) * v).sum())
                        p[i, j] = original - 0.001
                        minus = float((layer(x) * v).sum())
                        p[i, j] = original
                        self.assertAlmostEqual((plus - minus) / 0.002, float(analytical[i, j]), delta=3e-5)
            return g_a, g_b
        with torch.no_grad():
            a, b = check()
            self.assertEqual(int(torch.count_nonzero(a)), 0)
            self.assertGreater(int(torch.count_nonzero(b)), 0)
            layer.lora_B.fill_(0.1)  # synthetic fixture setup, never an optimizer update
            check()

    def test_wrapper_rejects_biased_wrong_dtype_or_existing_gradients(self):
        for base in (torch.nn.Linear(3, 2), torch.nn.Linear(3, 2, bias=False).double()):
            with self.assertRaises(ValueError):
                native.LoRALinear(base)
        base = torch.nn.Linear(3, 2, bias=False)
        base.weight.grad = torch.ones_like(base.weight)
        with self.assertRaises(ValueError):
            native.LoRALinear(base)

    def test_exact_60_module_injection_uses_synthetic_linear_fixture_only(self):
        fixture = torch.nn.Module()
        fixture.model = torch.nn.Module()
        fixture.model.layers = torch.nn.ModuleList()
        for _ in range(30):
            layer = torch.nn.Module(); layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(576, 576, bias=False)
            layer.self_attn.v_proj = torch.nn.Linear(576, 192, bias=False)
            fixture.model.layers.append(layer)
        original = dict(fixture.named_parameters())
        fixture.eval()
        wrappers = native.inject_qv(fixture)
        adapters = native.adapter_parameters(wrappers)
        self.assertEqual(native.target_shapes(), lora.expected_targets())
        self.assertEqual(len(wrappers), 60)
        self.assertEqual(len(adapters), 120)
        self.assertEqual(sum(p.numel() for p in adapters.values()), 230400)
        for name, module in wrappers.items():
            self.assertIs(module.base_layer.weight, original[name + ".weight"])
            self.assertFalse(module.base_layer.weight.requires_grad)
            self.assertFalse(module.training)

    def test_injection_prevalidates_all_targets_before_mutation(self):
        model = torch.nn.Module(); model.q_proj = torch.nn.Linear(3, 2, bias=False)
        before = model.q_proj
        with self.assertRaises(RuntimeError):
            native.inject_qv(model)
        self.assertIs(model.q_proj, before)
        fixture = torch.nn.Module(); fixture.attn = torch.nn.Module()
        fixture.attn.q_proj = torch.nn.Linear(3, 2, bias=False)
        fixture.attn.v_proj = torch.nn.Linear(3, 1, bias=True)
        with mock.patch.object(native, "target_shapes", return_value={"attn.q_proj": (2, 3), "attn.v_proj": (1, 3)}):
            with self.assertRaises(RuntimeError):
                native.inject_qv(fixture)
        self.assertIs(type(fixture.attn.q_proj), torch.nn.Linear)

    def test_assigned_gradient_checks_allow_zero_a_and_require_nonzero_b(self):
        layer = self.tiny()
        adapters = {"x.lora_A": layer.lora_A, "x.lora_B": layer.lora_B}
        originals = {"x.weight": layer.base_layer.weight}
        for p in adapters.values(): p.grad = torch.zeros_like(p)
        layer.lora_B.grad.fill_(1)
        result = lora.validate_gradients(adapters, originals, torch, device="cpu")
        self.assertTrue(result["finite"])
        for name, bad in (("missing", None), ("nan", float("nan")), ("inf", float("inf")), ("zero", 0.0)):
            layer.lora_B.grad = None if bad is None else torch.full_like(layer.lora_B, bad)
            with self.subTest(name=name), self.assertRaises((RuntimeError, FloatingPointError)):
                lora.validate_gradients(adapters, originals, torch, device="cpu")
        layer.lora_B.grad = torch.ones_like(layer.lora_B)
        layer.base_layer.weight.grad = torch.zeros_like(layer.base_layer.weight)
        with self.assertRaisesRegex(RuntimeError, "base"):
            lora.validate_gradients(adapters, originals, torch, device="cpu")

    def test_adapter_only_optimizer_membership_and_configuration_without_step(self):
        layer = self.tiny()
        adapters = {"x.lora_A": layer.lora_A, "x.lora_B": layer.lora_B}
        opt = torch.optim.SGD(list(adapters.values()), lr=0.0001, momentum=0.0, dampening=0.0,
                              weight_decay=0.0, foreach=False, fused=False, differentiable=False)
        lora.validate_optimizer(opt, adapters, torch)
        for key, value in (("lr", 0.1), ("momentum", 0.1), ("foreach", True), ("fused", True)):
            previous = opt.param_groups[0][key]; opt.param_groups[0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                lora.validate_optimizer(opt, adapters, torch)
            opt.param_groups[0][key] = previous
        opt.param_groups[0]["params"].append(layer.base_layer.weight)
        with self.assertRaisesRegex(RuntimeError, "membership"):
            lora.validate_optimizer(opt, adapters, torch)
        opt.param_groups[0]["params"].pop()
        opt.state[layer.lora_A] = {"momentum_buffer": torch.zeros_like(layer.lora_A)}
        with self.assertRaises(RuntimeError):
            lora.validate_optimizer(opt, adapters, torch)

    def test_clipping_assigned_adapter_gradients_without_update(self):
        layer = self.tiny()
        params = [layer.lora_A, layer.lora_B]
        for p in params: p.grad = torch.full_like(p, 3.0)
        before = layer.base_layer.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False, error_if_nonfinite=True)
        self.assertLessEqual(float(torch.nn.utils.get_total_norm([p.grad for p in params])), 1.00001)
        self.assertTrue(torch.equal(before, layer.base_layer.weight))


class SerializationTests(OfflineCase):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_root()
        self.adapters = {name: torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
                         for name, shape in lora.expected_adapter_shapes().items()}
        self.report = {"optimizer_steps": 1, "weight_update_verified": True,
                       "checkpoint": {"weight_sha256": "b" * 64},
                       "adapter_configuration": json.loads(lora.SPEC.read_text())["adapter"],
                       "adapter_sha256_after": lora.fingerprint(self.adapters), "provenance": {"synthetic_fixture": True}}

    def test_adapter_only_safetensors_round_trip_and_no_base_file(self):
        files = lora.save_adapter(self.root / "checkpoint", self.adapters, self.report, torch)
        self.assertEqual(set(files), {"adapter_config.json", "adapter_model.safetensors"})
        self.assertLess((self.root / "checkpoint/adapter_model.safetensors").stat().st_size, 2 * 1024**2)
        config = json.loads((self.root / "checkpoint/adapter_config.json").read_text())
        self.assertEqual(config["format"], lora.FORMAT)
        self.assertEqual(config["adapter_parameter_sha256"], self.report["adapter_sha256_after"])
        for p in (self.root / "checkpoint").iterdir(): self.assertEqual(p.stat().st_mode & 0o222, 0)
        with self.assertRaises(FileExistsError):
            lora.save_adapter(self.root / "checkpoint", self.adapters, self.report, torch)

    def test_save_rejects_incomplete_step_extra_base_or_changed_bytes(self):
        for patch in ({"optimizer_steps": 0}, {"weight_update_verified": False}, {"adapter_sha256_after": "0" * 64}):
            directory = self.root / ("checkpoint-" + str(len(list(self.root.iterdir()))))
            with self.subTest(patch=patch), self.assertRaises(RuntimeError):
                lora.save_adapter(directory, self.adapters, {**self.report, **patch}, torch)
        self.adapters["base.weight"] = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        with self.assertRaises(RuntimeError):
            lora.save_adapter(self.root / "extra", self.adapters, self.report, torch)

    def test_serialization_failure_cannot_complete(self):
        with mock.patch("safetensors.torch.save_file", side_effect=RuntimeError("disk failure")), self.assertRaises(RuntimeError):
            lora.save_adapter(self.root / "checkpoint", self.adapters, self.report, torch)
        self.assertFalse((self.root / "checkpoint/adapter_model.safetensors").exists())


class LayoutValidationTests(OfflineCase):
    """Shape/identity/device descriptors, without allocating a real model."""

    def fixture(self, training=True):
        class Parameter:
            def __init__(self, shape, trainable=False):
                self.shape, self.requires_grad, self.grad = shape, trainable, None
                self.device, self.dtype = SimpleNamespace(type="mps"), torch.float32
            def numel(self): return math.prod(self.shape)
        class Wrapper:
            rank = 4; alpha = 8; scale = 2.0
        self.stack.enter_context(mock.patch.object(native, "LoRALinear", Wrapper))
        originals, wrappers = {}, {}
        for name, (output, width) in lora.expected_targets().items():
            p = Parameter((output, width)); originals[name + ".weight"] = p
            w = Wrapper(); w.base_layer = SimpleNamespace(weight=p)
            w.lora_A = Parameter((4, width), training)
            w.lora_B = Parameter((output, 4), training)
            wrappers[name] = w
        originals["model.embed_tokens.weight"] = Parameter((134515008 - sum(p.numel() for p in originals.values()),))
        adapters = native.adapter_parameters(wrappers)
        model = type("LlamaForCausalLM", (), {})()
        model.training, model.is_gradient_checkpointing = training, training
        model.config = SimpleNamespace(use_cache=False, _attn_implementation="sdpa",
            to_dict=lambda: json.loads((lora.telemetry.CHECKPOINT / "config.json").read_text()))
        model.parameters = lambda: [*originals.values(), *adapters.values()]
        model.buffers = lambda: []
        fn = SimpleNamespace(keywords={"use_reentrant": False, "preserve_rng_state": True})
        model.modules = lambda: [SimpleNamespace(gradient_checkpointing=training, _gradient_checkpointing_func=fn)]
        model.get_submodule = lambda name: wrappers[name]
        model.get_input_embeddings = model.get_output_embeddings = lambda: SimpleNamespace(weight=originals["model.embed_tokens.weight"])
        return model, originals, wrappers, adapters, fn

    def test_exact_layout_accepts_only_expected_trainable_scope(self):
        for training in (True, False):
            model, originals, wrappers, adapters, _ = self.fixture(training)
            result = lora.validate_layout(model, originals, wrappers, torch, training=training)
            self.assertEqual(list(result), list(adapters))
            self.assertEqual(sum(p.numel() for p in result.values()), 230400)

    def test_base_and_adapter_trainability_dtype_device_shape_or_gradient_drift_stop(self):
        for kind in ("base_trainable", "base_gradient", "adapter_frozen", "dtype", "device", "shape", "base_tie", "scale", "missing"):
            model, originals, wrappers, adapters, _ = self.fixture()
            base = next(iter(originals.values())); adapter = next(iter(adapters.values())); wrapper = next(iter(wrappers.values()))
            if kind == "base_trainable": base.requires_grad = True
            elif kind == "base_gradient": base.grad = object()
            elif kind == "adapter_frozen": adapter.requires_grad = False
            elif kind == "dtype": adapter.dtype = torch.float16
            elif kind == "device": adapter.device = SimpleNamespace(type="cpu")
            elif kind == "shape": adapter.shape = (3, 576)
            elif kind == "base_tie": model.get_output_embeddings = lambda: SimpleNamespace(weight=object())
            elif kind == "scale": wrapper.scale = 1.0
            elif kind == "missing": model.parameters = lambda: list(originals.values())
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                lora.validate_layout(model, originals, wrappers, torch, training=True)

    def test_checkpoint_kwargs_cannot_be_ignored_or_reentrant(self):
        for kind in ("reentrant", "rng", "inactive", "empty"):
            model, originals, wrappers, _, fn = self.fixture()
            if kind == "reentrant": fn.keywords["use_reentrant"] = True
            elif kind == "rng": fn.keywords["preserve_rng_state"] = False
            elif kind == "inactive": model.is_gradient_checkpointing = False
            else: model.modules = lambda: []
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, "checkpoint"):
                lora.validate_layout(model, originals, wrappers, torch, training=True)


class StepBoundaryTests(OfflineCase):
    """Real clipping/validation with assigned CPU gradients; fake backward AND SGD."""

    def setUp(self):
        super().setUp()
        self.root = self.temporary_root()
        self.adapters = {"x.lora_A": torch.nn.Parameter(torch.zeros(2, 3)),
                         "x.lora_B": torch.nn.Parameter(torch.zeros(2, 2))}
        self.originals = {"x.weight": torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)}
        self.model = mock.Mock()
        def cleanup(**kwargs):
            for p in [*self.adapters.values(), *self.originals.values()]: p.grad = None
        self.model.zero_grad.side_effect = cleanup
        self.monitor = SimpleNamespace(mode="pilot", boundary=mock.Mock(), check=mock.Mock())
        self.report = {"optimizer_steps": 0, "step_attempted": False, "adapter_sha256_before": "a" * 64,
                       "base_parameter_sha256_before": lora.BASE_SHA256}
        self.pre_step = mock.Mock()
        self.loss = SimpleNamespace(requires_grad=True, item=lambda: 1.0)
        def backward():
            self.adapters["x.lora_A"].grad = torch.zeros(2, 3)
            self.adapters["x.lora_B"].grad = torch.ones(2, 2)
        self.loss.backward = mock.Mock(side_effect=backward)
        calls = self.calls = []
        root = self.root
        class FakeSGD:
            def __init__(self, params, **kwargs):
                self.state = {}; self.param_groups = [{"params": list(params), **kwargs}]
            def step(self):
                if not (root / "STEP_ATTEMPTED.json").is_file(): raise AssertionError("missing durable attempt marker")
                calls.append("fake step, no weight mutation")
            def zero_grad(self, **kwargs): cleanup()
        self.stack.enter_context(mock.patch.object(torch.optim, "SGD", FakeSGD))
        self.stack.enter_context(mock.patch.object(lora, "validate_layout", return_value=self.adapters))
        self.stack.enter_context(mock.patch.object(lora.acquisition, "checked_loss", return_value=self.loss))
        actual_validate = lora.validate_gradients
        self.stack.enter_context(mock.patch.object(lora, "validate_gradients", side_effect=
            lambda adapters, originals, runtime: actual_validate(adapters, originals, runtime, device="cpu")))

    def step(self, **kwargs):
        options = {"enabled": True, "authorization": lora.AUTHORIZATION,
                   "pre_step": self.pre_step, "run_path": self.root}
        options.update(kwargs)
        return lora.one_step(self.model, {}, self.originals, {}, torch, self.monitor, self.report, **options)

    def test_fake_step_repeats_gates_clips_once_and_consumes_budget(self):
        self.step()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.report["optimizer_steps"], 1)
        self.assertTrue(self.report["step_attempted"])
        self.assertEqual(self.report["backward_count"], 1)
        self.assertLessEqual(self.report["gradient_norm_after_clip"], 1.00001)
        self.pre_step.assert_called_once()
        self.assertTrue(all(p.grad is None for p in self.adapters.values()))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.step()
        self.assertEqual(len(self.calls), 1)

    def test_authorization_is_required_at_update_function_boundary(self):
        with self.assertRaises(ValueError): self.step(authorization=lora.acquisition.AUTHORIZATION)
        self.loss.backward.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_baseline_monitor_cannot_admit_optimizer(self):
        self.monitor.mode = "baseline"
        with self.assertRaisesRegex(RuntimeError, "admission"):
            self.step()
        self.loss.backward.assert_not_called()

    def test_malformed_budget_or_prior_backward_attempt_cannot_be_retried(self):
        for mutation in ({"optimizer_steps": False}, {"optimizer_steps": 0.0}, {"backward_attempted": True}):
            saved = copy.deepcopy(self.report)
            self.report.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(RuntimeError, "budget"):
                self.step()
            self.report.clear(); self.report.update(saved)
        self.loss.backward.assert_not_called()

    def test_nonfinite_or_disconnected_gradient_cannot_reach_optimizer(self):
        self.loss.backward.side_effect = lambda: setattr(self.adapters["x.lora_A"], "grad", torch.zeros(2, 3))
        with self.assertRaisesRegex(RuntimeError, "gradient"):
            self.step()
        self.assertFalse(self.report["step_attempted"])
        self.assertEqual(self.calls, [])
        self.assertTrue(all(p.grad is None for p in self.adapters.values()))

    def test_pre_step_integrity_stop_cannot_reach_optimizer(self):
        self.pre_step.side_effect = RuntimeError("base hash changed")
        with self.assertRaisesRegex(RuntimeError, "hash"):
            self.step()
        self.assertFalse(self.report["step_attempted"])
        self.assertFalse((self.root / "STEP_ATTEMPTED.json").exists())
        self.assertEqual(self.calls, [])

    def test_nonfinite_clipping_norm_stops_and_keeps_report_serializable(self):
        with mock.patch.object(torch.nn.utils, "clip_grad_norm_", return_value=torch.tensor(float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "clipping norm"):
                self.step()
        json.dumps(self.report, allow_nan=False)
        self.assertFalse(self.report["step_attempted"])
        self.assertEqual(self.calls, [])

    def test_memory_stop_immediately_before_step_cannot_update(self):
        self.monitor.boundary.side_effect = lambda stage: (_ for _ in ()).throw(RuntimeError("memory stop")) if stage == "before_optimizer_step" else None
        with self.assertRaisesRegex(RuntimeError, "memory stop"):
            self.step()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.report["step_attempted"])

    def test_post_step_stop_keeps_truthful_attempt_counter_and_cleans_gradients(self):
        self.monitor.boundary.side_effect = lambda stage: (_ for _ in ()).throw(RuntimeError("post step stop")) if stage == "after_optimizer_step" else None
        with self.assertRaisesRegex(RuntimeError, "post step"):
            self.step()
        self.assertEqual(self.report["optimizer_steps"], 1)
        self.assertTrue(self.report["step_attempted"])
        self.assertTrue((self.root / "STEP_ATTEMPTED.json").exists())
        self.assertTrue(all(p.grad is None for p in self.adapters.values()))


class MockRunTests(OfflineCase):
    """All model, device, backward, optimizer and checkpoint operations are fakes."""

    def setUp(self):
        super().setUp()
        self.root = self.temporary_root()
        self.spec, self.manifest, self.historical = lora.frozen_inputs()
        self.state = SimpleNamespace(changed=False)
        self.original = SimpleNamespace(_version=0)
        self.model = mock.Mock()
        self.model.named_parameters.return_value = [("original.weight", self.original)]
        self.model.get_parameter.return_value = self.original
        self.adapters = {}
        for name, shape in lora.expected_adapter_shapes().items():
            self.adapters[name] = SimpleNamespace(numel=lambda shape=shape: math.prod(shape),
                eq=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
                ne=lambda value: SimpleNamespace(any=lambda: SimpleNamespace(item=lambda: self.state.changed)),
                requires_grad_=mock.Mock())
        self.monitor = SimpleNamespace(samples=[], stage="", start=mock.Mock(), close=mock.Mock(), check=mock.Mock())
        self.tick = time.monotonic_ns() - 1_000_000_000
        def capture(stage, **kwargs):
            self.tick += 100_000
            reading = sample(stage, self.tick)
            self.monitor.samples.append(reading)
            return reading
        self.monitor.capture = capture; self.monitor.boundary = capture
        def monitor_factory(*args, **kwargs):
            self.monitor.mode = kwargs["mode"]
            return self.monitor
        def evaluate(*args):
            for row in self.manifest["evaluation"]:
                capture("before_evaluation:" + row["id"])
                capture("after_evaluation:" + row["id"])
            return copy.deepcopy(self.historical["evaluation"])
        runtime = SimpleNamespace(__version__="2.9.1", float32="float32", set_num_threads=mock.Mock(),
                                  manual_seed=mock.Mock(), use_deterministic_algorithms=mock.Mock(),
                                  mps=SimpleNamespace(manual_seed=mock.Mock()))
        self.loader = mock.Mock(return_value=(self.model, {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}))
        transforms = SimpleNamespace(__version__="4.57.1",
            AutoConfig=SimpleNamespace(from_pretrained=mock.Mock(return_value=SimpleNamespace(use_cache=False, to_dict=lambda: {}))),
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=object())),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self.loader))
        self.fake_provenance = {"test": "mock implementation"}
        self.patches = [mock.patch.object(lora, "frozen_inputs", return_value=(self.spec, self.manifest, self.historical)),
            mock.patch.object(lora, "provenance", return_value=self.fake_provenance),
            mock.patch.object(lora, "metadata_targets", return_value={"test": True}),
            mock.patch.object(lora, "environment_before_import"),
            mock.patch.dict(sys.modules, {"torch": runtime, "transformers": transforms}),
            mock.patch.object(lora.telemetry, "validate_environment", return_value=SimpleNamespace(type="mps")),
            mock.patch.object(lora.telemetry, "_command", return_value="Apple M4 Pro"),
            mock.patch.object(lora.telemetry, "validate_config"), mock.patch.object(lora.telemetry, "validate_model"),
            mock.patch.object(lora.telemetry, "ensure_checkpoint", return_value=self.historical["checkpoint"]),
            mock.patch.object(lora, "LoRAMonitor", side_effect=monitor_factory),
            mock.patch.object(native, "inject_qv", return_value={}),
            mock.patch.object(native, "adapter_parameters", return_value=self.adapters),
            mock.patch.object(lora, "validate_layout", return_value=self.adapters),
            mock.patch.object(lora, "fingerprint", side_effect=lambda params: lora.BASE_SHA256 if "original.weight" in params else ("b" if self.state.changed else "a") * 64),
            mock.patch.object(lora, "digest", return_value=self.historical["encoding_sha256"]),
            mock.patch.object(lora.acquisition, "prepare_examples", return_value={self.spec["pilot_case_id"]:
                SimpleNamespace(input_ids=(1, 2), labels=(-100, 2))}),
            mock.patch.object(lora.acquisition, "evaluate", side_effect=evaluate),
            mock.patch.object(lora.shutil, "disk_usage", return_value=SimpleNamespace(free=8 * lora.GIB)),
            mock.patch("src.graddiff_integration.collate_causal_examples", return_value={})]
        for patch in self.patches: self.stack.enter_context(patch)

    def baseline(self, name="baseline-new"):
        self.monitor.samples.clear(); self.state.changed = False
        report = lora.run(mode="baseline", enabled=True, authorization=None, run_path=self.root / name)
        self.assertEqual(report["status"], "lora_baseline_complete_no_updates", report)
        return self.root / name / "report.json", report

    def pilot(self, path, name="pilot-new"):
        self.monitor.samples.clear()
        return lora.run(mode="pilot", enabled=True, authorization=lora.AUTHORIZATION, run_path=self.root / name,
                        baseline_path=path, baseline_sha256=lora.sha256_file(path))

    @contextlib.contextmanager
    def fake_update(self, fail=None):
        def step(model, batch, originals, wrappers, torch, monitor, report, **kwargs):
            self.assertEqual(kwargs["authorization"], lora.AUTHORIZATION)
            kwargs["pre_step"]()
            report.update(step_attempted=True, optimizer_steps=1, optimizer_created=True, backward_attempted=True, backward_count=1)
            report["trainable_parameter_count"] = lora.PARAMETERS
            self.state.changed = True
            if fail: raise RuntimeError(fail)
        def save(directory, *args):
            directory.mkdir()
            (directory / "adapter_model.safetensors").write_text("mock checkpoint, no tensor weights")
            return {"adapter_model.safetensors": lora.sha256_file(directory / "adapter_model.safetensors")}
        with mock.patch.object(lora, "one_step", side_effect=step) as update, mock.patch.object(lora, "save_adapter", side_effect=save) as saver:
            yield update, saver

    def test_baseline_is_sealed_no_update_and_independently_validated(self):
        with mock.patch.object(lora, "one_step", side_effect=AssertionError("baseline attempted update")) as step:
            path, report = self.baseline()
        step.assert_not_called()
        self.assertEqual(path.stat().st_mode & 0o222, 0)
        self.assertTrue((path.parent / "COMPLETE.json").exists())
        self.assertFalse((path.parent / "checkpoint").exists())
        lora.validate_native_baseline(report, self.manifest, self.historical, self.fake_provenance)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["adapter_sha256_after"], report["adapter_sha256_before"])

    def test_mocked_one_step_success_saves_only_new_namespace(self):
        path, _ = self.baseline()
        with self.fake_update() as (update, saver):
            result = self.pilot(path)
        self.assertEqual(result["status"], "lora_one_step_complete_mechanics_only", result)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["base_parameter_sha256_after"], lora.BASE_SHA256)
        self.assertNotEqual(result["adapter_sha256_before"], result["adapter_sha256_after"])
        self.assertTrue((self.root / "pilot-new/COMPLETE.json").exists())
        self.assertFalse(result["scientific_use_cleared"])
        update.assert_called_once(); saver.assert_called_once()
        self.model.save_pretrained.assert_not_called()

    def test_early_admission_stop_is_durably_reported_without_load(self):
        self.monitor.capture = mock.Mock(side_effect=RuntimeError("20 GiB preload failed"))
        result = lora.run(mode="baseline", enabled=True, authorization=None, run_path=self.root / "baseline-stop")
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["model_loaded"])
        self.assertTrue((self.root / "baseline-stop/FAILED.json").exists())
        self.assertTrue((self.root / "baseline-stop/report.json").exists())
        self.loader.assert_not_called()

    def test_failed_step_records_possible_partial_update_and_cannot_save(self):
        path, _ = self.baseline()
        with self.fake_update("mock step failed") as (_, saver):
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertTrue(result["partial_in_memory_update_possible"])
        self.assertTrue((self.root / "pilot-new/FAILED.json").exists())
        self.assertFalse((self.root / "pilot-new/COMPLETE.json").exists())
        saver.assert_not_called()

    def test_post_step_memory_stop_cannot_save(self):
        path, _ = self.baseline()
        previous = self.monitor.boundary
        def boundary(stage):
            if stage == "before_serialization": raise RuntimeError("memory STOP")
            return previous(stage)
        self.monitor.boundary = boundary
        with self.fake_update() as (_, saver):
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertTrue(result["partial_in_memory_update_possible"])
        saver.assert_not_called()

    def test_save_failure_leaves_tombstone_and_no_complete(self):
        path, _ = self.baseline()
        with self.fake_update(), mock.patch.object(lora, "save_adapter", side_effect=RuntimeError("disk STOP")):
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertTrue(result["checkpoint_write_attempted"])
        self.assertEqual(result["checkpoint_writes"], 0)
        self.assertFalse((self.root / "pilot-new/COMPLETE.json").exists())

    def test_disk_stop_before_model_load(self):
        with mock.patch.object(lora.shutil, "disk_usage", return_value=SimpleNamespace(free=lora.GIB)):
            result = lora.run(mode="baseline", enabled=True, authorization=None, run_path=self.root / "baseline-disk")
        self.assertEqual(result["status"], "STOP")
        self.loader.assert_not_called()

    def test_bad_baseline_hash_stops_before_model_load(self):
        path, _ = self.baseline(); self.loader.reset_mock()
        report = lora.run(mode="pilot", enabled=True, authorization=lora.AUTHORIZATION, run_path=self.root / "pilot-hash",
                          baseline_path=path, baseline_sha256="0" * 64)
        self.assertEqual(report["status"], "STOP")
        self.loader.assert_not_called()

    def test_historical_baseline_cannot_substitute_for_native_baseline(self):
        report = lora.run(mode="pilot", enabled=True, authorization=lora.AUTHORIZATION, run_path=self.root / "pilot-old",
                          baseline_path=lora.HISTORICAL, baseline_sha256=lora.HISTORICAL_SHA256)
        self.assertEqual(report["status"], "STOP")
        self.loader.assert_not_called()

    def test_noop_drift_stops_before_backward_or_optimizer(self):
        path, _ = self.baseline()
        rows = copy.deepcopy(self.historical["evaluation"]); rows[0]["answer_nll"] += 0.01
        with mock.patch.object(lora.acquisition, "evaluate", return_value=rows), self.fake_update() as (update, saver):
            result = self.pilot(path)
        self.assertEqual(result["status"], "STOP")
        self.assertFalse(result["step_attempted"])
        update.assert_not_called(); saver.assert_not_called()

    def test_baseline_tampering_and_memory_evidence_are_rejected(self):
        _, report = self.baseline()
        mutations = ({"optimizer_steps": 1}, {"backward_count": 1}, {"actual_device": "cpu"},
                     {"base_parameter_sha256_after": "b" * 64}, {"trainable_parameter_count": 1},
                     {"provenance": {}}, {"adapter_sha256_after": "c" * 64})
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                lora.validate_native_baseline({**report, **mutation}, self.manifest, self.historical, self.fake_provenance)
        for kind in ("preload", "gap", "swap", "case_boundary", "peaks", "policy", "nan", "generation", "order"):
            changed = copy.deepcopy(report)
            if kind == "preload": changed["measurements"][0]["available_physical_memory_lower_bound_bytes"] = 20 * lora.GIB - 1
            elif kind == "gap": changed["measurements"][-1]["timestamp_ns"] += 3_000_000_000
            elif kind == "swap": changed["measurements"][-1]["swap_used_bytes"] = 1
            elif kind == "case_boundary":
                next(s for s in changed["measurements"] if s["stage"] == "before_evaluation:" + self.manifest["evaluation"][0]["id"])["stage"] = "missing"
            elif kind == "peaks": changed["peaks"]["sample_count"] = 0
            elif kind == "policy": changed["memory_policy"]["preload_available_required_bytes"] = 0
            elif kind == "nan": changed["evaluation"][0]["answer_nll"] = float("nan")
            elif kind == "generation": changed["evaluation"][0]["generation"] += " changed"
            elif kind == "order": changed["evaluation"].reverse()
            if kind in ("preload", "gap", "swap", "case_boundary"):
                changed["peaks"] = lora.telemetry.summarize_samples(changed["measurements"])
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                lora.validate_native_baseline(changed, self.manifest, self.historical, self.fake_provenance)

    def test_sealed_baseline_requires_complete_and_rejects_hardlinks(self):
        path, _ = self.baseline()
        digest = lora.sha256_file(path)
        os.link(path, self.root / "hardlink.json")
        with self.assertRaises(ValueError):
            lora.read_native_baseline(path, digest, self.manifest, self.historical, self.fake_provenance)
        (self.root / "hardlink.json").unlink()
        path.parent.chmod(0o755); (path.parent / "COMPLETE.json").unlink()
        with self.assertRaises(ValueError):
            lora.read_native_baseline(path, digest, self.manifest, self.historical, self.fake_provenance)


if __name__ == "__main__":
    unittest.main()
