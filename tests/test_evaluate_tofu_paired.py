import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.evaluate_tofu_paired import (
    DATASET_ID,
    DATASET_REVISION,
    SCORING_VERSION,
    aggregate_diagnostic_metrics,
    assert_disjoint_splits,
    answer_metrics,
    build_paired_manifest,
    build_paired_result_document,
    deterministic_sample_indices,
    manifest_sha256,
    validate_manifest_against_data,
)


def synthetic_sources(records_by_split):
    return {
        split: {
            "repo_id": DATASET_ID,
            "split": split,
            "revision": DATASET_REVISION,
            "filename": f"{split}.json",
            "sha256": f"sha-{split}",
            "record_count": len(records),
        }
        for split, records in records_by_split.items()
    }


def synthetic_records(prefix, count):
    return [
        {"question": f"{prefix} question {index}?", "answer": f"{prefix} answer {index}."}
        for index in range(count)
    ]


class PairedTofuEvaluationTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_transformers(self):
        repository_root = Path(__file__).resolve().parents[1]
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.evaluate_tofu_paired; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules",
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_retain_sampling_is_deterministic_unique_and_disjoint_by_split(self):
        first = deterministic_sample_indices(3960, 40, seed=2026)
        second = deterministic_sample_indices(3960, 40, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 40)
        self.assertEqual(len(set(first)), 40)

        forget = synthetic_records("forget", 40)
        retain = synthetic_records("retain", 3960)
        assert_disjoint_splits(forget, retain, left_name="forget01", right_name="retain99")

    def test_disjoint_check_rejects_normalized_question_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap on 1 normalized question"):
            assert_disjoint_splits(
                [{"question": "Who is A?", "answer": "A"}],
                [{"question": "  who IS a? ", "answer": "Different answer"}],
            )

    def test_paired_manifest_reproducible_and_aligns_official_perturb_rows(self):
        forget = synthetic_records("forget", 40)
        retain = synthetic_records("retain", 100)
        holdout = synthetic_records("holdout", 40)
        perturbed = [
            {
                "question": row["question"],
                "answer": row["answer"],
                "paraphrased_question": f"Rephrased: {row['question']}",
                "paraphrased_answer": f"Rephrased: {row['answer']}",
                "perturbed_answer": ["A wrong answer."],
            }
            for row in reversed(forget)
        ]
        records = {
            "forget01": forget,
            "retain99": retain,
            "forget01_perturbed": perturbed,
            "holdout01": holdout,
        }
        sources = synthetic_sources(records)
        manifest = build_paired_manifest(records, sources)
        rebuilt = build_paired_manifest(records, sources)

        self.assertEqual(manifest, rebuilt)
        self.assertEqual(len(manifest["evaluation_sets"]["forget01"]["cases"]), 40)
        self.assertEqual(len(manifest["evaluation_sets"]["retain99"]["cases"]), 40)
        forget_questions = {row["question"] for row in forget}
        retained_indices = manifest["evaluation_sets"]["retain99"]["sampling"][
            "record_indices"
        ]
        retained_sample_questions = {retain[index]["question"] for index in retained_indices}
        self.assertFalse(forget_questions & retained_sample_questions)
        self.assertEqual(len(manifest["evaluation_sets"]["forget01_paraphrase"]["cases"]), 40)
        self.assertEqual(len(manifest["evaluation_sets"]["holdout01"]["cases"]), 40)
        first_paraphrase = manifest["evaluation_sets"]["forget01_paraphrase"]["cases"][0]
        self.assertEqual(first_paraphrase["record_index"], 39)
        validate_manifest_against_data(manifest, records, sources)
        self.assertEqual(manifest_sha256(manifest), manifest_sha256(json.loads(json.dumps(manifest))))

    def test_perturbed_rows_must_match_forget_originals(self):
        forget = synthetic_records("forget", 40)
        retain = synthetic_records("retain", 50)
        perturbed = [
            {
                "question": row["question"],
                "answer": "wrong original answer",
                "paraphrased_question": f"Rephrased: {row['question']}",
                "paraphrased_answer": f"Rephrased: {row['answer']}",
                "perturbed_answer": ["Wrong."],
            }
            for row in forget
        ]
        records = {"forget01": forget, "retain99": retain, "forget01_perturbed": perturbed}
        with self.assertRaisesRegex(ValueError, "mismatched original answer"):
            build_paired_manifest(records, synthetic_sources(records))

    def test_lightweight_metric_calculations_and_aggregate_statistics(self):
        exact = answer_metrics("A blue fox.", "A blue fox.")
        partial = answer_metrics("blue bird", "blue whale")
        self.assertTrue(exact["strict_exact_match"])
        self.assertEqual(exact["normalized_token_overlap_f1"], 1.0)
        self.assertFalse(partial["strict_exact_match"])
        self.assertAlmostEqual(partial["normalized_token_overlap_f1"], 0.5)

        rows = [{"metrics": exact}, {"metrics": partial}]
        aggregate = aggregate_diagnostic_metrics(rows)
        self.assertEqual(aggregate["case_count"], 2)
        self.assertEqual(aggregate["strict_exact_match_rate"], 0.5)
        self.assertAlmostEqual(aggregate["normalized_token_overlap_f1_mean"], 0.75)
        self.assertAlmostEqual(
            aggregate["normalized_token_overlap_f1_population_stddev"], 0.25
        )
        self.assertEqual(aggregate["normalized_token_overlap_f1_min"], 0.5)
        self.assertEqual(aggregate["normalized_token_overlap_f1_median"], 0.75)
        self.assertEqual(aggregate["normalized_token_overlap_f1_max"], 1.0)
        with self.assertRaisesRegex(ValueError, "empty evaluation set"):
            aggregate_diagnostic_metrics([])

    def test_result_artifact_schema_records_pins_and_labels_lightweight_scores(self):
        forget = synthetic_records("forget", 40)
        retain = synthetic_records("retain", 60)
        records = {"forget01": forget, "retain99": retain}
        sources = synthetic_sources(records)
        manifest = build_paired_manifest(records, sources)
        responses = {
            "forget01": [row["answer"] for row in forget],
            "retain99": [
                retain[index]["answer"]
                for index in manifest["evaluation_sets"]["retain99"]["sampling"][
                    "record_indices"
                ]
            ],
        }
        artifact = build_paired_result_document(
            manifest,
            records,
            responses,
            model_id="test-model",
            model_revision="model-rev",
            tokenizer_id="test-tokenizer",
            tokenizer_revision="tokenizer-rev",
            actual_device="cpu",
            model_dtype="float32",
            torch_version="test-torch",
            transformers_version="test-transformers",
        )
        self.assertEqual(artifact["schema_version"], "2.0")
        self.assertEqual(artifact["manifest_sha256"], manifest_sha256(manifest))
        self.assertEqual(artifact["dataset"]["revision"], DATASET_REVISION)
        self.assertEqual(artifact["seed"], 2026)
        self.assertEqual(artifact["model"]["revision"], "model-rev")
        self.assertEqual(artifact["tokenizer_configuration"]["revision"], "tokenizer-rev")
        self.assertEqual(artifact["generation_parameters"]["actual_device"], "cpu")
        self.assertEqual(artifact["scoring"]["version"], SCORING_VERSION)
        self.assertEqual(artifact["evaluation_sets"]["forget01"]["case_count"], 40)
        self.assertEqual(artifact["evaluation_sets"]["retain99"]["case_count"], 40)
        self.assertEqual(artifact["official_metrics"]["forget_quality"], "not_computed")
        self.assertIn("not official TOFU", artifact["evaluation_scope"])
        self.assertEqual(
            json.loads(json.dumps(artifact))["artifact_type"],
            "paired_tofu_generated_answer_diagnostics",
        )


if __name__ == "__main__":
    unittest.main()
