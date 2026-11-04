import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - project-local ML environment is optional
    torch = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "install requirements-ml.txt for real-runner guard tests")
class GradDiffRealRunnerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src import graddiff_real_train as runner

        cls.runner = runner

    def test_default_cli_is_validation_only_and_never_dispatches_updates(self):
        safe_report = {"mode": "local_validation_only", "weight_updates_enabled": False}
        stdout = io.StringIO()
        with mock.patch.object(self.runner, "run_local_validation", return_value=safe_report) as validate:
            with mock.patch.object(self.runner, "run_real_updates") as update:
                with contextlib.redirect_stdout(stdout):
                    result = self.runner.main([])
        self.assertEqual(result, 0)
        validate.assert_called_once()
        update.assert_not_called()
        self.assertFalse(json.loads(stdout.getvalue())["weight_updates_enabled"])

    def test_apply_updates_requires_explicit_device_revision_and_report(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(self.runner, "run_real_updates") as update:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = self.runner.main(["--apply-updates"])
        self.assertEqual(result, 2)
        update.assert_not_called()
        self.assertIn("explicit --device", stderr.getvalue())

    def test_mps_request_does_not_fall_back_to_cpu(self):
        torch_stub = SimpleNamespace(
            device=lambda name: SimpleNamespace(type=name),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: False)
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "no CPU fallback was attempted"):
            self.runner.resolve_training_device("mps", torch_module=torch_stub)
        self.assertEqual(
            self.runner.resolve_training_device("cpu", torch_module=torch_stub).type,
            "cpu",
        )

    def test_oom_error_reports_phase_memory_and_no_fallback(self):
        error = self.runner.GradDiffOutOfMemoryError(
            "pinned FP32 model load", "mps", {"process_peak_rss_bytes": 1234}
        )
        self.assertIn("Out of memory", str(error))
        self.assertIn("No device fallback", str(error))
        self.assertIn("1234", str(error))
        self.assertTrue(self.runner._is_oom_error(RuntimeError("MPS backend out of memory")))

    def test_explicit_update_flags_dispatch_only_to_the_update_path(self):
        report_path = f"outputs/runner-test-report-{os.getpid()}.json"
        report = {"mode": "real_model_graddiff_training", "weight_updates_enabled": True}
        stdout = io.StringIO()
        with mock.patch.object(self.runner, "run_real_updates", return_value=report) as update:
            with mock.patch.object(self.runner, "_write_report") as write_report:
                with mock.patch.object(self.runner, "run_local_validation") as validate:
                    with contextlib.redirect_stdout(stdout):
                        result = self.runner.main([
                            "--apply-updates",
                            "--device", "mps",
                            "--confirm-model-revision", self.runner.MODEL_REVISION,
                            "--report-path", report_path,
                        ])
        self.assertEqual(result, 0)
        update.assert_called_once()
        write_report.assert_called_once()
        validate.assert_not_called()

    def test_nonfinite_gradient_is_rejected_before_optimizer_step(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "non-finite gradient"):
            self.runner.checked_optimizer_step(
                model, optimizer, max_grad_norm=1.0, torch_module=torch
            )
        self.assertTrue(torch.equal(model.weight, before))
        self.assertEqual(optimizer.state, {})

    def test_gradient_clipping_is_applied_before_the_update(self):
        parameter = torch.nn.Parameter(torch.zeros(2))

        class MockModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self):
                return self.value.sum()

        model = MockModel(parameter)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        parameter.grad = torch.tensor([3.0, 4.0])
        report = self.runner.checked_optimizer_step(
            model, optimizer, max_grad_norm=1.0, torch_module=torch
        )
        self.assertAlmostEqual(report["gradient_norm_before_clipping"], 5.0, places=5)
        self.assertTrue(report["gradient_clipping_applied"])
        self.assertTrue(torch.allclose(parameter.detach(), torch.tensor([-0.6, -0.8]), atol=1e-6))

    def test_output_safety_requires_git_ignored_outputs_and_prevents_overwrite(self):
        valid_candidate = ROOT / "outputs" / f"runner-test-{os.getpid()}" / "model"
        resolved = self.runner._validate_output_path(valid_candidate, kind="checkpoint")
        self.assertEqual(resolved, valid_candidate.resolve())
        with self.assertRaisesRegex(ValueError, "inside the local outputs"):
            self.runner._validate_output_path(ROOT / "models" / "unsafe", kind="checkpoint")
        with self.assertRaises((ValueError, FileExistsError)):
            self.runner._validate_output_path(
                self.runner.SNAPSHOT_PATH / "would-overwrite-original", kind="checkpoint"
            )

        existing_parent = ROOT / "outputs" / f"runner-existing-{os.getpid()}"
        existing_parent.mkdir(parents=True, exist_ok=False)
        try:
            existing = existing_parent / "report.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                self.runner._validate_output_path(existing, kind="report")
        finally:
            existing.unlink(missing_ok=True)
            existing_parent.rmdir()

    def test_report_records_rows_and_checksums_without_raw_qa_or_environment(self):
        sentinel_question = "PRIVATE_SENTINEL_question_should_never_be_logged"
        sentinel_answer = "PRIVATE_SENTINEL_answer_should_never_be_logged"
        example = self.runner._example_log(
            "forget01",
            7,
            {
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "labels": torch.tensor([[-100, -100, 10, 11]]),
            },
        )
        report = {
            "dataset": {
                "splits": {
                    "forget01": {"sha256": "a" * 64},
                    "retain99": {"sha256": "b" * 64},
                }
            },
            "selected_examples": [example],
        }
        serialized = json.dumps(report, sort_keys=True)
        self.assertIn('"record_index": 7', serialized)
        self.assertIn("sha256", serialized)
        self.assertNotIn(sentinel_question, serialized)
        self.assertNotIn(sentinel_answer, serialized)
        token = os.environ.get("HF_TOKEN")
        if token:
            self.assertNotIn(token, serialized)
        self.assertNotIn("question", example)
        self.assertNotIn("answer", example)


if __name__ == "__main__":
    unittest.main()
