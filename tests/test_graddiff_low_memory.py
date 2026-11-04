"""Tiny models only; all pretrained loaders are mocked in orchestration tests."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src import graddiff_real_train as runner
from src.graddiff_synthetic_rehearsal import tiny_model, synthetic_batches, run_rehearsal
import test_graddiff_model_integration as fixtures


class LowMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner.OUTPUT_ROOT.mkdir(exist_ok=True)

    def args(self, *extra):
        return runner.build_parser().parse_args(["--optimizer", "sgd", *extra])

    def safe_memory(self, gib=40):
        return {"total_physical_memory_bytes": 48 * runner.GIB,
                "available_physical_memory_lower_bound_bytes": gib * runner.GIB,
                "system_memory_free_percentage": 80}

    def test_sgd_update_matches_equation_and_has_no_state(self):
        model = torch.nn.Linear(2, 1, bias=False)
        args = self.args()
        optimizer = runner.build_optimizer(model, args)
        before = model.weight.detach().clone()
        gradient = torch.tensor([[0.3, -0.4]])
        model.weight.grad = gradient.clone()
        runner.checked_optimizer_step(model, optimizer, max_grad_norm=1.0)
        self.assertTrue(torch.allclose(model.weight, before - args.learning_rate * gradient))
        self.assertEqual(optimizer.state, {})
        self.assertEqual(optimizer.defaults["momentum"], 0)
        self.assertFalse(optimizer.defaults["foreach"])

    def test_optimizer_selection_defaults_and_invalid_options(self):
        model = torch.nn.Linear(1, 1)
        sgd_args = self.args()
        self.assertIsInstance(runner.build_optimizer(model, sgd_args), torch.optim.SGD)
        self.assertEqual(sgd_args.learning_rate, 1e-4)
        adam_args = runner.build_parser().parse_args([])
        self.assertIsInstance(runner.build_optimizer(model, adam_args), torch.optim.AdamW)
        self.assertEqual(adam_args.learning_rate, 1e-5)
        for field, value in (("optimizer", "unknown"), ("momentum", 0.9),
                             ("momentum", float("nan")), ("precision", "float16"),
                             ("checkpointing", "reentrant"), ("learning_rate", float("inf"))):
            args = self.args()
            setattr(args, field, value)
            with self.assertRaises(ValueError):
                runner.build_optimizer(model, args)

    def test_checkpointing_cache_and_trainable_domain(self):
        model = tiny_model()
        settings = runner.configure_training_model(model)
        self.assertFalse(settings["use_reentrant"])
        self.assertTrue(settings["preserve_rng_state"])
        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.is_gradient_checkpointing)
        runner.validate_training_runtime(model, torch.device("cpu"))
        model.requires_grad_(False)
        with self.assertRaises(ValueError):
            runner.configure_training_model(model)

    def test_checkpointing_rejects_unsupported_or_ignored_activation_settings(self):
        with self.assertRaisesRegex(ValueError, "support"):
            runner.configure_training_model(torch.nn.Linear(1, 1))
        model = tiny_model()
        with mock.patch.object(model, "gradient_checkpointing_enable", new=lambda gradient_checkpointing_kwargs: None):
            with self.assertRaisesRegex(RuntimeError, "activate"):
                runner.configure_training_model(model)
        with mock.patch.object(model, "gradient_checkpointing_enable", new=lambda: None):
            with self.assertRaisesRegex(ValueError, "API"):
                runner.configure_training_model(model)

    def test_sequential_gradients_equal_combined_validated_objective_with_checkpointing(self):
        from src.graddiff import causal_lm_nll, graddiff_objective

        torch.manual_seed(2026)
        model = tiny_model()
        args = self.args("--gamma", "0.7", "--alpha", "1.3")
        forget, retain = synthetic_batches(torch.device("cpu"))
        # Reference without activation checkpointing, with the exact same weights.
        model.config.use_cache = False
        f = causal_lm_nll(model(**forget, use_cache=False).logits, forget["labels"])
        r = causal_lm_nll(model(**retain, use_cache=False).logits, retain["labels"])
        expected = graddiff_objective(f, r, args.gamma, args.alpha)
        expected.backward()
        gradients = [p.grad.detach().clone() for p in model.parameters()]
        runner.configure_training_model(model)
        report = runner.backward_pair(model, forget, retain, args)
        self.assertAlmostEqual(report["objective"], expected.item(), places=5)
        for expected_grad, parameter in zip(gradients, model.parameters()):
            self.assertTrue(torch.allclose(expected_grad, parameter.grad, atol=1e-6, rtol=1e-5))
        args.gamma = 3e38
        with mock.patch.object(runner, "_loss_value", return_value=(torch.tensor(2.0, requires_grad=True), 2.0, 0.0)):
            with self.assertRaisesRegex(FloatingPointError, "weighted"):
                runner.backward_pair(model, forget, retain, args)

    def test_memory_thresholds_and_missing_telemetry(self):
        runner.check_training_memory(self.safe_memory(32), optimizer="sgd", before_load=True)
        runner.check_training_memory(self.safe_memory(42), optimizer="adamw", before_load=True)
        runner.check_training_memory(self.safe_memory(6), optimizer="sgd", before_load=False)
        for snapshot, optimizer, before in ((self.safe_memory(31), "sgd", True),
                                            (self.safe_memory(41), "adamw", True),
                                            (self.safe_memory(5), "sgd", False),
                                            ({}, "sgd", True)):
            with self.assertRaises(RuntimeError):
                runner.check_training_memory(snapshot, optimizer=optimizer, before_load=before)
        snapshot = self.safe_memory()
        snapshot["system_memory_free_percentage"] = 19
        with self.assertRaises(RuntimeError):
            runner.check_training_memory(snapshot, optimizer="sgd", before_load=False)
        with mock.patch.object(runner, "memory_snapshot", return_value=self.safe_memory()):
            with mock.patch.object(runner, "host_memory_snapshot", return_value=self.safe_memory()):
                with self.assertRaisesRegex(RuntimeError, "MPS allocation"):
                    runner.guarded_memory_snapshot(torch, torch.device("mps"), self.args())

    def test_memory_drop_before_step_prevents_mutation(self):
        model = torch.nn.Linear(1, 1)
        optimizer = runner.build_optimizer(model, self.args())
        before = [p.detach().clone() for p in model.parameters()]
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with self.assertRaises(RuntimeError):
            runner.checked_optimizer_step(model, optimizer, max_grad_norm=1,
                before_update=lambda: runner.check_training_memory(self.safe_memory(5), optimizer="sgd", before_load=False))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertEqual(optimizer.state, {})

    def test_initial_memory_gate_precedes_assets_and_real_loader(self):
        args = self.args("--apply-updates", "--device", "cpu", "--confirm-model-revision", runner.MODEL_REVISION,
                         "--report-path", "outputs/test-low-memory-never-written.json")
        with mock.patch.object(runner, "host_memory_snapshot", return_value=self.safe_memory(10)):
            with mock.patch.object(runner, "_load_local_inputs") as assets:
                with self.assertRaisesRegex(RuntimeError, "32 GiB"):
                    runner.run_real_updates(args)
        assets.assert_not_called()

    def test_sgd_default_stays_validation_and_conflicting_flags_fail(self):
        with mock.patch.object(runner, "run_local_validation", return_value={"mode": "validation"}) as validate:
            with mock.patch.object(runner, "run_real_updates") as update:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["--optimizer", "sgd"]), 0)
        validate.assert_called_once()
        update.assert_not_called()
        for extra in (["--backward-only", "--apply-updates"], ["--backward-only", "--max-steps", "2"],
                      ["--backward-only", "--save-checkpoint", "outputs/no-save"]):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(extra), 2)

    def test_output_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=runner.OUTPUT_ROOT) as tmp:
            symlink = Path(tmp) / "escape"
            symlink.symlink_to(runner.PROJECT_ROOT / ".cache", target_is_directory=True)
            with self.assertRaises(ValueError):
                runner._validate_output_path(symlink / "model", kind="checkpoint")

    def test_reproducible_synthetic_backward_without_pretrained_loading(self):
        with mock.patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("forbidden")):
            first = run_rehearsal(device_name="cpu", seed=2026)
            second = run_rehearsal(device_name="cpu", seed=2026)
        self.assertEqual(first["losses"], second["losses"])
        self.assertEqual(first["observed_gradient_l2_norm"], second["observed_gradient_l2_norm"])
        self.assertTrue(first["weights_unchanged"])
        self.assertFalse(first["optimizer_created"])
        self.assertEqual(first["actual_device"], "cpu")

    def test_real_backward_orchestration_on_mock_model_never_creates_optimizer(self):
        model = tiny_model()
        tokenizer = fixtures.GradDiffModelIntegrationTests.FakeTokenizer()
        records = [{"question": "Synthetic question", "answer": "Synthetic answer"}]
        before = [p.detach().clone() for p in model.parameters()]
        loaded = (torch, None, model.config, tokenizer, records, records,
                  {"revision": runner.MODEL_REVISION}, {},
                  {"data": {"choices": [(0, 0, 0)], "forget_source": {}, "retain_source": {}}, "runtime": {}})
        args = self.args("--backward-only", "--device", "cpu", "--confirm-model-revision", runner.MODEL_REVISION,
                         "--report-path", "outputs/mock-backward-never-written.json")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "guarded_memory_snapshot", return_value=self.safe_memory()))
            stack.enter_context(mock.patch.object(runner, "_load_local_inputs", return_value=loaded))
            stack.enter_context(mock.patch.object(runner, "EXPECTED_PARAMETER_COUNT", sum(p.numel() for p in model.parameters())))
            stack.enter_context(mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model))
            optimizer = stack.enter_context(mock.patch.object(runner, "build_optimizer", side_effect=AssertionError("forbidden")))
            report = runner.run_real_backward_only(args)
        optimizer.assert_not_called()
        self.assertFalse(report["weight_updates_enabled"])
        self.assertEqual(report["hyperparameters"]["optimizer_steps_completed"], 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_mock_full_sgd_updates_only_after_fresh_guards(self):
        model = tiny_model()
        tokenizer = fixtures.GradDiffModelIntegrationTests.FakeTokenizer()
        forget = [{"question": "Synthetic forget", "answer": "First answer"}]
        retain = [{"question": "Synthetic retain", "answer": "Second response"}]
        before = [p.detach().clone() for p in model.parameters()]
        loaded = (torch, None, model.config, tokenizer, forget, retain,
                  {"revision": runner.MODEL_REVISION}, {},
                  {"data": {"choices": [(0, 0, 0)], "forget_source": {}, "retain_source": {}}, "runtime": {}})
        args = self.args("--apply-updates", "--device", "cpu", "--confirm-model-revision", runner.MODEL_REVISION,
                         "--report-path", "outputs/mock-sgd-never-written.json")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "guarded_memory_snapshot", return_value=self.safe_memory()))
            stack.enter_context(mock.patch.object(runner, "_load_local_inputs", return_value=loaded))
            stack.enter_context(mock.patch.object(runner, "EXPECTED_PARAMETER_COUNT", sum(p.numel() for p in model.parameters())))
            stack.enter_context(mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model))
            report = runner.run_real_updates(args)
        self.assertEqual(report["hyperparameters"]["optimizer_steps_completed"], 1)
        self.assertEqual(report["hyperparameters"]["optimizer_state_tensor_elements"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertIn("immediately_before_optimizer_step", [s["stage"] for s in report["guarded_phase_snapshots"]])


if __name__ == "__main__":
    unittest.main()
