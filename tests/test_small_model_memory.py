"""Memory admission tests use JSON and mock telemetry; no model or tensor ops."""

import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import small_model_experiment as experiment
from src import small_model_memory as memory


class SmallModelMemoryTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(experiment.PROFILE.read_text())
        self.policy = json.loads(memory.POLICY.read_text())

    def evidence(self):
        """Synthetic evidence for schema/arithmetic only, never physical evidence."""
        samples = [{"stage": stage, "timestamp_ns": i + 1,
                    "available_physical_memory_lower_bound_bytes": (40 if i == 0 else 37) * memory.GIB,
                    "system_memory_free_percentage": 80,
                    "process_current_rss_bytes": (1 if i == 0 else 3) * memory.GIB,
                    "process_peak_rss_bytes": (1 if i == 0 else 3) * memory.GIB,
                    "mps_tensor_allocated_bytes": (0 if i == 0 else 1) * memory.GIB,
                    "mps_driver_allocated_bytes": (0 if i == 0 else 2) * memory.GIB,
                    "swap_used_bytes": 0, "pageouts": 0} for i, stage in enumerate(memory.STAGES)]
        run = {"sampling_interval_ms": 50, "samples": samples}
        return {"schema_version": "1.0", "profile_sha256": experiment.digest(self.profile),
                "policy_sha256": experiment.digest(self.policy), "model_id": experiment.MODEL_ID,
                "model_revision": experiment.REVISION, "device": "mps", "fallback_enabled": False,
                "optimizer_updates_performed": 0, "hardware_id": "synthetic-fixture-not-a-real-device",
                "os_version": "fixture", "torch_version": "fixture", "transformers_version": "fixture",
                "total_physical_memory_bytes": 48 * memory.GIB,
                "trials": [copy.deepcopy({"cold": run, "warm": run}) for _ in range(3)]}

    def test_select_by_pinned_model_and_settings_not_llama_architecture(self):
        self.assertEqual(self.profile["model"]["architecture"], "LlamaForCausalLM")
        self.assertEqual(memory.select_memory_guard(self.profile), memory.GUARD_ID)
        for key, value in (("repo_id", "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"),
                           ("revision", "main"), ("parameter_count", 1235814400)):
            profile = copy.deepcopy(self.profile)
            profile["model"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory.select_memory_guard(profile)
        self.profile["training"]["max_sequence_length"] = 1024
        with self.assertRaises(ValueError):
            memory.select_memory_guard(self.profile)

    def test_missing_measurements_remain_stop_and_no_threshold(self):
        report = memory.assess_memory(self.profile, self.policy)
        self.assertIn("missing_measurements", report["blockers"])
        self.assertIsNone(report["approved_small_model_threshold_bytes"])
        self.assertIsNone(report["measurement_proposal"])
        self.assertFalse(report["runtime_admission_allowed"])
        self.assertFalse(report["training_authorized"])
        self.assertEqual(report["active_before_load_available_bytes"], 32 * memory.GIB)
        self.assertFalse(report["model_loaded"])
        with self.assertRaises(ValueError):
            memory.assess_memory(self.profile, self.policy, {})

    def test_tensor_estimates_are_explicit_and_not_admission(self):
        estimates = memory.memory_estimates(self.profile)
        self.assertEqual(estimates["fp32_weights_bytes"], 538060032)
        self.assertEqual(estimates["fp32_gradients_bytes_training_only"], 538060032)
        self.assertEqual(estimates["sgd_zero_momentum_optimizer_state_bytes"], 0)
        self.assertEqual(estimates["adamw_moments_bytes_comparison_only"], 1076120064)
        self.assertEqual(estimates["single_fp32_logits_tensor_bytes"], 96 * 1024**2)
        self.assertFalse(estimates["measured"])
        self.assertFalse(estimates["training_readiness_proven"])

    def test_policy_override_unknown_keys_and_config_errors_rejected(self):
        for key, value in (("approved_before_load_available_bytes", 12 * memory.GIB),
                           ("approved_measurement_sha256", "0" * 64), ("training_authorized", True),
                           ("minimum_trials", True), ("minimum_trials", 1),
                           ("minimum_total_host_gib", 16), ("continuing_available_lower_bound_gib", 1),
                           ("minimum_system_memory_free_percentage", 0),
                           ("minimum_uncertainty_gib", 0), ("unknown_override", True)):
            policy = copy.deepcopy(self.policy)
            policy[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, policy)
        for key in self.policy:
            policy = copy.deepcopy(self.policy)
            del policy[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, policy)
        self.profile["memory_estimates"]["working_peak_gib_rounded"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "frozen"):
            memory.assess_memory(self.profile, self.policy)

    def test_complete_synthetic_evidence_only_proposes_never_admits(self):
        evidence = self.evidence()
        report = memory.assess_memory(self.profile, self.policy, evidence)
        proposal = report["measurement_proposal"]
        # Overlapping 3 GiB host decline + 2 GiB RSS + 2 GiB driver => max 3, not sum 7.
        self.assertEqual(proposal["observed_incremental_bytes"], 3 * memory.GIB)
        self.assertEqual(proposal["unapproved_before_load_proposal_bytes"], 13 * memory.GIB)
        self.assertEqual(proposal["uncertainty_bytes"], 4 * memory.GIB)
        self.assertTrue(proposal["review_required"])
        self.assertFalse(report["runtime_admission_allowed"])
        self.assertFalse(report["training_authorized"])
        self.assertIsNone(report["approved_small_model_threshold_bytes"])
        self.assertEqual(report["active_before_load_available_bytes"], 32 * memory.GIB)
        # Extra intra-stage samples participate in the peak and 25% margin.
        samples = evidence["trials"][0]["cold"]["samples"]
        extra = copy.deepcopy(samples[1])
        extra["timestamp_ns"] += 1
        for sample in samples[2:]:
            sample["timestamp_ns"] += 1
        extra["available_physical_memory_lower_bound_bytes"] = 10 * memory.GIB
        samples.insert(2, extra)
        proposal = memory.measurement_proposal(self.profile, self.policy, evidence)
        self.assertEqual(proposal["observed_incremental_bytes"], 30 * memory.GIB)
        self.assertEqual(proposal["uncertainty_bytes"], 7.5 * memory.GIB)
        self.assertEqual(proposal["unapproved_before_load_proposal_bytes"], 44 * memory.GIB)

    def test_missing_trials_kinds_stages_and_telemetry_fail_closed(self):
        evidence = self.evidence()
        evidence["trials"].pop()
        with self.assertRaisesRegex(ValueError, "three"):
            memory.assess_memory(self.profile, self.policy, evidence)
        evidence = self.evidence()
        del evidence["trials"][0]["warm"]
        with self.assertRaisesRegex(ValueError, "cold and warm"):
            memory.assess_memory(self.profile, self.policy, evidence)
        for stage in memory.STAGES:
            evidence = self.evidence()
            samples = evidence["trials"][0]["cold"]["samples"]
            samples[:] = [s for s in samples if s["stage"] != stage]
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, self.policy, evidence)
        for key in self.evidence()["trials"][0]["cold"]["samples"][0]:
            evidence = self.evidence()
            del evidence["trials"][0]["cold"]["samples"][1][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, self.policy, evidence)

    def test_untrusted_measurement_types_pressure_swap_and_provenance_rejected(self):
        mutations = [("available_physical_memory_lower_bound_bytes", v)
                     for v in (None, True, -1, float("nan"), float("inf"), "40000000000", 49 * memory.GIB)]
        mutations += [("system_memory_free_percentage", 19), ("system_memory_free_percentage", 101),
                      ("available_physical_memory_lower_bound_bytes", 6 * memory.GIB - 1),
                      ("mps_tensor_allocated_bytes", 3 * memory.GIB),
                      ("mps_driver_allocated_bytes", -1), ("process_peak_rss_bytes", memory.GIB),
                      ("swap_used_bytes", 1), ("pageouts", 1), ("timestamp_ns", 0)]
        for key, value in mutations:
            evidence = self.evidence()
            evidence["trials"][0]["cold"]["samples"][1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, self.policy, evidence)
        for key, value in (("device", "cpu"), ("fallback_enabled", True),
                           ("optimizer_updates_performed", 1), ("profile_sha256", "wrong"),
                           ("policy_sha256", "wrong"), ("model_revision", "main"),
                           ("hardware_id", ""), ("total_physical_memory_bytes", 32 * memory.GIB)):
            evidence = self.evidence()
            evidence[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, self.policy, evidence)

    def test_legacy_admission_evidence_and_sample_interval_required(self):
        for change in ("admission", "interval", "ordering", "sampling_gap"):
            evidence = self.evidence()
            run = evidence["trials"][0]["cold"]
            if change == "admission":
                run["samples"][0]["available_physical_memory_lower_bound_bytes"] = 31 * memory.GIB
            elif change == "interval":
                run["sampling_interval_ms"] = 0
            elif change == "sampling_gap":
                for sample in run["samples"][1:]:
                    sample["timestamp_ns"] += 100_000_000
            else:
                run["samples"][1], run["samples"][2] = run["samples"][2], run["samples"][1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                memory.assess_memory(self.profile, self.policy, evidence)

    def test_cli_fails_closed_for_absent_invalid_and_complete_measurements(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(memory.main([]), 2)
        self.assertFalse(json.loads(output.getvalue())["runtime_admission_allowed"])
        with tempfile.TemporaryDirectory(dir=experiment.ROOT / "outputs") as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(json.dumps(self.evidence()))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(memory.main(["--measurements", str(path)]), 2)
            self.assertFalse(json.loads(output.getvalue())["runtime_admission_allowed"])
            for invalid in ("null", "[]", "{", "{}"):
                path.write_text(invalid)
                with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        memory.main(["--measurements", str(path)])
                self.assertEqual(error.exception.code, 2)

    def test_offline_module_does_not_import_ml_frameworks(self):
        code = ("import sys; from src import small_model_memory as m; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                "assert m.main([]) == 2; assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=experiment.ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing project ML environment")
    def test_physical_path_still_enforces_original_llama_thresholds(self):
        from src import graddiff_real_train as runner
        for optimizer, threshold in (("sgd", 32), ("adamw", 42)):
            snapshot = {"total_physical_memory_bytes": 48 * memory.GIB,
                        "available_physical_memory_lower_bound_bytes": threshold * memory.GIB,
                        "system_memory_free_percentage": 20}
            runner.check_training_memory(snapshot, optimizer=optimizer, before_load=True)
            snapshot["available_physical_memory_lower_bound_bytes"] -= 1
            with self.subTest(optimizer=optimizer), self.assertRaises(RuntimeError):
                runner.check_training_memory(snapshot, optimizer=optimizer, before_load=True)
            snapshot["available_physical_memory_lower_bound_bytes"] = 6 * memory.GIB
            runner.check_training_memory(snapshot, optimizer=optimizer, before_load=False)
            snapshot["available_physical_memory_lower_bound_bytes"] -= 1
            with self.assertRaises(RuntimeError):
                runner.check_training_memory(snapshot, optimizer=optimizer, before_load=False)
        with mock.patch.object(runner, "resolve_training_device", return_value="mps"):
            with mock.patch.object(runner, "guarded_memory_snapshot", side_effect=RuntimeError("32 GiB")):
                with self.assertRaisesRegex(RuntimeError, "inherited Llama SGD"):
                    experiment.check_physical_mps_readiness()


if __name__ == "__main__":
    unittest.main()
