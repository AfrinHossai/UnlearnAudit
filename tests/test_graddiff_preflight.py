"""Read-only tests: this suite never backpropagates or steps an optimizer."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src import graddiff_preflight as preflight
from src import graddiff_real_train as training
from src.graddiff_integration import evaluate_readonly_pair
import test_graddiff_model_integration as integration_fixtures


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        training.OUTPUT_ROOT.mkdir(exist_ok=True)

    def safe_memory(self):
        return {"total_physical_memory_bytes": 48 * preflight.GIB,
                "available_physical_memory_lower_bound_bytes": 24 * preflight.GIB,
                "system_memory_free_percentage": 60}

    def test_memory_thresholds_at_boundary_and_under_pressure(self):
        snapshot = self.safe_memory()
        snapshot["available_physical_memory_lower_bound_bytes"] = 20 * preflight.GIB
        preflight.check_memory_thresholds(snapshot, before_load=True)
        snapshot["available_physical_memory_lower_bound_bytes"] -= 1
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            preflight.check_memory_thresholds(snapshot, before_load=True)
        snapshot["available_physical_memory_lower_bound_bytes"] = 6 * preflight.GIB
        preflight.check_memory_thresholds(snapshot, before_load=False)
        snapshot["available_physical_memory_lower_bound_bytes"] -= 1
        with self.assertRaises(RuntimeError):
            preflight.check_memory_thresholds(snapshot, before_load=False)
        snapshot = self.safe_memory()
        snapshot["system_memory_free_percentage"] = 19
        with self.assertRaisesRegex(RuntimeError, "pressure"):
            preflight.check_memory_thresholds(snapshot, before_load=False)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            preflight.check_memory_thresholds({}, before_load=True)
        snapshot = self.safe_memory()
        snapshot["available_physical_memory_lower_bound_bytes"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            preflight.check_memory_thresholds(snapshot, before_load=True)

    def test_training_estimate_is_not_a_readiness_measurement(self):
        estimate = preflight.training_memory_estimate()
        self.assertTrue(estimate["fits_total_capacity_on_paper"])
        self.assertFalse(estimate["training_readiness_proven"])
        self.assertGreater(estimate["total_planning_bytes"], 42 * preflight.GIB)
        self.assertFalse(preflight.training_memory_estimate(32 * preflight.GIB)["fits_total_capacity_on_paper"])

    def test_vm_stat_uses_only_free_and_speculative_pages(self):
        output = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
                  'Pages free: 10.\nPages inactive: 9999.\nPages speculative: 2.\n')
        self.assertEqual(preflight.parse_vm_stat(output), 12 * 16384)
        with self.assertRaises(RuntimeError):
            preflight.parse_vm_stat("missing telemetry")

    def test_default_cli_does_not_load_model(self):
        with mock.patch.object(preflight, "run_preflight") as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(preflight.main([]), 0)
        run.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["mode"], "plan_only")
        with self.assertRaisesRegex(ValueError, "explicit"):
            preflight.run_preflight(enabled=False)

    def test_direct_training_api_requires_opt_in(self):
        args = training.build_parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "apply-updates"):
            training.run_real_updates(args)
        args.apply_updates = True
        args.dry_run = True
        with self.assertRaisesRegex(ValueError, "dry-run"):
            training.run_real_updates(args)

    def test_missing_checkpoint_fails_before_model_load(self):
        with tempfile.TemporaryDirectory(dir=training.PROJECT_ROOT / "outputs") as tmp:
            with mock.patch.object(training, "SNAPSHOT_PATH", Path(tmp) / "missing"):
                with self.assertRaisesRegex(FileNotFoundError, "not cached"):
                    preflight.run_preflight(enabled=True)

    def test_output_cannot_target_baseline_checkpoint_or_existing_file(self):
        for path in (training.SNAPSHOT_PATH / "preflight.json",
                     training.PROJECT_ROOT / "results" / "tofu_forget01_retain99_baseline.json"):
            with self.assertRaises(ValueError):
                preflight._validate_output_path(path, kind="report")
        with tempfile.TemporaryDirectory(dir=training.PROJECT_ROOT / "outputs") as tmp:
            path = Path(tmp) / "report.json"
            preflight._write_report(path, {"readonly": True})
            with self.assertRaises(FileExistsError):
                preflight._write_report(path, {})

    def test_nonfinite_model_loss_is_rejected_without_gradients(self):
        class BadLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **kwargs):
                return SimpleNamespace(logits=torch.zeros(1, 3, 2), loss=torch.tensor(float("nan")))

        batch = {"input_ids": torch.zeros(1, 3, dtype=torch.long),
                 "attention_mask": torch.ones(1, 3, dtype=torch.long),
                 "labels": torch.tensor([[-100, 1, 0]])}
        model = BadLoss()
        with self.assertRaises(RuntimeError):
            evaluate_readonly_pair(model, batch, batch)
        self.assertIsNone(model.value.grad)

    def test_mock_full_preflight_preserves_values_and_records_each_forward(self):
        model = integration_fixtures.GradDiffModelIntegrationTests.TinyCausalLM()
        model.config.max_position_embeddings = 512
        tokenizer = integration_fixtures.GradDiffModelIntegrationTests.FakeTokenizer()
        original = [p.detach().clone() for p in model.parameters()]
        examples = [("forget01", 0, {"question": "Fixture question", "answer": "Fixture answer"}),
                    ("retain99", 10, {"question": "Another fixture", "answer": "Another answer"})]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(preflight, "_check_local_checkpoint", return_value={"revision": training.MODEL_REVISION}))
            stack.enter_context(mock.patch.object(preflight, "select_manifest_examples", return_value=({}, examples)))
            stack.enter_context(mock.patch.object(preflight, "host_memory_snapshot", side_effect=self.safe_memory))
            stack.enter_context(mock.patch("transformers.AutoConfig.from_pretrained", return_value=model.config))
            stack.enter_context(mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer))
            stack.enter_context(mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model))
            report = preflight.run_preflight(enabled=True, device_name="cpu")
        self.assertEqual(model.forward_count, 2)
        self.assertEqual(len(report["forward_seconds_including_telemetry"]), 2)
        self.assertFalse(report["training_authorized"])
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(original, model.parameters())))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertNotIn("Fixture answer", json.dumps(report))

    def test_memory_stop_prevents_model_loader(self):
        with mock.patch.object(preflight, "_check_local_checkpoint", return_value={}):
            with mock.patch.object(preflight, "select_manifest_examples", return_value=({}, [])):
                with mock.patch.object(preflight, "host_memory_snapshot", return_value={}):
                    with mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
                        with self.assertRaisesRegex(RuntimeError, "unavailable"):
                            preflight.run_preflight(enabled=True, device_name="cpu")
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
