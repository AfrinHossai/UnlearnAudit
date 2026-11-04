import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CONFIG = ROOT / "experiments" / "graddiff_tofu_forget01_retain99.json"
MPS_CONFIG = ROOT / "experiments" / "graddiff_tofu_forget01_retain99_mps_prototype.json"
MANIFEST = ROOT / "experiments" / "tofu_forget01_retain99_manifest.json"
BASELINE = ROOT / "results" / "tofu_forget01_retain99_baseline.json"
HISTORICAL_BASELINE_SHA256 = (
    "1473f6abc0c42164ee55dcf27328ab0b72c9e86e12c15764b71800e7462e8a0e"
)
OPENUNLEARNING_COMMIT = "a456aa277be14b836248e42b13b3befe47bb03f4"


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def canonical_manifest_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class GradDiffExperimentPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = read_json(PRIMARY_CONFIG)
        cls.mps = read_json(MPS_CONFIG)
        cls.manifest = read_json(MANIFEST)
        cls.baseline = read_json(BASELINE) if BASELINE.is_file() else None

    def test_configs_are_explicitly_planned_not_executed(self):
        self.assertEqual(self.primary["status"], "prepared_not_executed")
        self.assertEqual(
            self.mps["status"],
            "synthetic_prototype_implemented; model_training_not_run",
        )
        self.assertEqual(self.primary["upstream"]["commit_sha"], OPENUNLEARNING_COMMIT)
        self.assertEqual(self.primary["upstream"]["reviewed_ref"], OPENUNLEARNING_COMMIT)
        self.assertIn(OPENUNLEARNING_COMMIT, self.primary["upstream"]["references"]["implementation"])
        self.assertIn("Check out this exact commit", self.primary["upstream"]["freeze_before_execution"])

    def test_gradiff_objective_and_official_method_defaults_match(self):
        method = self.primary["method"]
        mps_method = self.mps["method"]
        expected = "gamma * (-NLL(forget)) + alpha * NLL(retain)"
        self.assertEqual(method["loss_expression"], expected)
        self.assertEqual(method["gamma"], 1.0)
        self.assertEqual(method["alpha"], 1.0)
        self.assertEqual(method["retain_loss_type"], "NLL")
        self.assertEqual(mps_method["loss_expression"], expected)
        self.assertTrue(mps_method["mathematical_objective_matches_primary_profile"])
        self.assertEqual(mps_method["gamma"], method["gamma"])
        self.assertEqual(mps_method["alpha"], method["alpha"])
        self.assertEqual(mps_method["retain_loss_type"], method["retain_loss_type"])

    def test_model_and_dataset_pins_match_the_paired_manifest_and_result(self):
        model = self.primary["models"]["initial_checkpoint"]
        self.assertEqual(
            model["repo_id"], "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
        )
        self.assertEqual(
            model["revision"], "88e31200b97e4c0c04ae0d2f0b591f427046d192"
        )
        if self.baseline is not None:
            self.assertEqual(model["repo_id"], self.baseline["model"]["repo_id"])
            self.assertEqual(model["revision"], self.baseline["model"]["revision"])
        self.assertEqual(
            self.primary["dataset"]["repo_id"], self.manifest["dataset"]["repo_id"]
        )
        self.assertEqual(
            self.primary["dataset"]["revision"],
            self.manifest["dataset"]["revision"],
        )

        for split in ("forget", "retain"):
            split_name = "forget01" if split == "forget" else "retain99"
            spec = self.primary["dataset"]["training"][split]
            manifest_spec = self.manifest["datasets"][split_name]
            self.assertEqual(spec["split"], manifest_spec["split"])
            self.assertEqual(spec["record_count"], manifest_spec["record_count"])
            self.assertEqual(spec["sha256"], manifest_spec["sha256"])

    def test_training_schedule_has_a_reproducible_effective_batch_and_checkpoint_plan(self):
        training = self.primary["training"]
        dependencies = self.primary["dependencies_reviewed_from_upstream_requirements"]
        self.assertEqual(dependencies["source_commit"], OPENUNLEARNING_COMMIT)
        pins = dependencies["requirements_txt_pins"]
        self.assertEqual(pins["torch"], "2.4.1")
        self.assertEqual(pins["transformers"], "4.51.3")
        self.assertEqual(pins["accelerate"], "0.34.2")
        self.assertEqual(pins["bitsandbytes"], "0.44.1")
        self.assertEqual(training["seed"], 2026)
        self.assertEqual(training["data_seed"], 2026)
        self.assertEqual(training["num_train_epochs"], 10)
        self.assertEqual(training["learning_rate"], 1e-5)
        self.assertEqual(training["warmup_ratio"], 0.0)
        self.assertEqual(training["weight_decay"], 0.0)
        self.assertEqual(training["optimizer"], "paged_adamw_32bit")
        self.assertEqual(training["per_device_train_batch_size"], 4)
        self.assertEqual(training["gradient_accumulation_steps"], 4)
        self.assertEqual(training["num_processes"], 2)
        self.assertEqual(
            training["per_device_train_batch_size"]
            * training["gradient_accumulation_steps"]
            * training["num_processes"],
            training["effective_global_batch_size"],
        )
        self.assertEqual(training["save_strategy"], "epoch")
        self.assertEqual(training["save_total_limit"], 11)
        self.assertTrue(training["save_only_model"])
        self.assertFalse(training["do_eval_during_distributed_training"])

    def test_baseline_artifact_is_unchanged_and_contains_paired_results(self):
        audit = self.primary["historical_baseline_audit"]
        self.assertFalse(audit["tracked_in_git"])
        self.assertEqual(audit["path"], "results/tofu_forget01_retain99_baseline.json")
        self.assertEqual(
            audit["artifact_sha256"], HISTORICAL_BASELINE_SHA256
        )
        self.assertEqual(
            audit["manifest_canonical_sha256"], canonical_manifest_sha256(self.manifest)
        )
        self.assertEqual(audit["saved_case_counts"], {"forget01": 40, "retain99": 40})
        if self.baseline is None:
            self.skipTest("the historical result is intentionally ignored by Git")

        baseline_bytes = BASELINE.read_bytes()
        self.assertEqual(hashlib.sha256(baseline_bytes).hexdigest(), HISTORICAL_BASELINE_SHA256)
        self.assertEqual(
            self.baseline["manifest_sha256"], canonical_manifest_sha256(self.manifest)
        )
        self.assertEqual(self.baseline["model"]["revision"], "88e31200b97e4c0c04ae0d2f0b591f427046d192")
        self.assertEqual(self.baseline["seed"], 2026)
        self.assertEqual(
            set(self.baseline["evaluation_sets"]), {"forget01", "retain99"}
        )

        for split in ("forget01", "retain99"):
            cases = self.manifest["evaluation_sets"][split]["cases"]
            saved = self.baseline["evaluation_sets"][split]
            self.assertEqual(len(cases), 40)
            self.assertEqual(saved["case_count"], 40)
            self.assertEqual(len(saved["examples"]), 40)
            self.assertEqual(
                {case["case_id"] for case in cases},
                {example["case_id"] for example in saved["examples"]},
            )
            self.assertEqual(saved["diagnostics"]["case_count"], 40)

        self.assertEqual(
            self.baseline["official_metrics"],
            {
                "truth_ratio": "not_computed",
                "forget_quality": "not_computed",
                "model_utility": "not_computed",
            },
        )
        self.assertEqual(self.baseline["manifest_sha256"], audit["manifest_canonical_sha256"])

    def test_eval_splits_distinguish_paraphrase_and_holdout_from_training(self):
        evaluation = self.primary["dataset"]["evaluation"]
        self.assertEqual(evaluation["forget01_perturbed"]["record_count"], 40)
        self.assertIn("same forget facts", evaluation["forget01_perturbed"]["role"])
        self.assertEqual(evaluation["holdout01"]["record_count"], 40)
        self.assertIn("evaluation-only", evaluation["holdout01"]["role"])
        self.assertEqual(
            self.primary["evaluation_protocol"]["checkpoints_to_score"], [0, 1, 5, 10]
        )
        self.assertEqual(
            self.primary["evaluation_protocol"]["primary_checkpoint_epoch"], 10
        )
        self.assertIn(
            "fixed before training",
            self.primary["evaluation_protocol"]["selection_rule"],
        )
        self.assertEqual(
            self.primary["evaluation_protocol"]["retained_utility_control"][
                "official_utility_minimum_delta_vs_initial_full"
            ],
            -0.05,
        )

    def test_mps_profile_preserves_loss_but_marks_platform_deviations_and_unknowns(self):
        profile = self.mps
        training = profile["training_adaptations"]
        self.assertEqual(profile["device"]["backend"], "mps")
        self.assertFalse(profile["device"]["memory_estimate_is_measurement"])
        self.assertFalse(profile["resource_and_claim_limits"]["recommended_for_primary_experiment"])
        self.assertEqual(profile["device"]["estimated_trainable_parameters"], 1235814400)
        self.assertEqual(profile["device"]["estimated_fp32_training_state_floor_bytes"], 19773030400)
        self.assertEqual(
            profile["device"]["device_runtime_overhead_planning_reserve_gib"],
            {"minimum": 8, "maximum": 12, "measurement": False,
             "basis": "Explicit planning allowance for macOS, other processes, Metal/PyTorch allocator state, and temporary workspaces; not measured on this host."},
        )
        self.assertEqual(
            profile["device"]["estimated_planning_envelope_gib"]["minimum"], 27.1
        )
        self.assertEqual(
            profile["device"]["arithmetic_headroom_after_training_state_floor_bytes"],
            51539607552 - 19773030400,
        )
        self.assertIn("os.sysconf", profile["device"]["unified_memory_capacity_source"])
        implementation = profile["prototype_implementation"]
        self.assertEqual(implementation["loss_module"], "src/graddiff.py")
        self.assertEqual(implementation["validated_against_upstream_commit"], OPENUNLEARNING_COMMIT)
        self.assertFalse(implementation["real_model_or_data_loading_implemented"])
        self.assertEqual(implementation["read_only_integration_helpers"], "src/graddiff_integration.py")
        self.assertTrue(implementation["read_only_full_model_smoke_implemented"])
        self.assertFalse(implementation["read_only_full_model_smoke_executed"])
        self.assertIn("WORLD_SIZE must equal 1", implementation["runtime_safety"])
        self.assertEqual(training["attention_implementation"], "sdpa")
        self.assertEqual(training["optimizer"], "adamw_torch")
        self.assertEqual(training["per_device_train_batch_size"], 1)
        self.assertEqual(training["gradient_accumulation_steps"], 32)
        self.assertEqual(training["effective_batch_size"], 32)
        self.assertEqual(training["warmup_ratio"], 0.0)
        self.assertEqual(training["weight_decay"], 0.0)
        self.assertEqual(len(training["deviations_from_upstream"]), 5)
        tests = profile["mathematical_equivalence_acceptance_tests"]
        self.assertEqual(
            tests["status"],
            "synthetic_local_tests_passed; direct_upstream_comparison_not_run",
        )
        self.assertEqual(len(tests["completed_local_checks"]), 5)
        self.assertEqual(len(tests["remaining_checks"]), 3)
        self.assertIn("All", tests["acceptance_rule"])
        for value in tests["mps_cpu_float32_tolerances"].values():
            self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
