import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.evaluate_tofu_baseline import (
    MODEL_ID,
    MODEL_REVISION,
    aggregate_metrics,
    answer_metrics,
    build_chat_messages,
    build_result_document,
    deterministic_sample_indices,
    encode_chat_prompt,
    select_device,
)


class EvaluateTofuBaselineTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_transformers(self):
        repository_root = Path(__file__).resolve().parents[1]
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.evaluate_tofu_baseline; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules",
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(check.returncode, 0, check.stderr)

    def test_sampling_is_stable_unique_and_sorted(self):
        first = deterministic_sample_indices(40, 5, seed=2026)
        second = deterministic_sample_indices(40, 5, seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(0 <= index < 40 for index in first))

    def test_sampling_rejects_out_of_range_values(self):
        with self.assertRaisesRegex(ValueError, "contains no records"):
            deterministic_sample_indices(0, 1, seed=2026)
        with self.assertRaisesRegex(ValueError, "sample_size must be"):
            deterministic_sample_indices(4, 5, seed=2026)

    def test_device_selection_prefers_available_mps_and_falls_back_to_cpu(self):
        class FakeMPS:
            def __init__(self, *, built, available):
                self.built = built
                self.available = available

            def is_built(self):
                return self.built

            def is_available(self):
                return self.available

        class FakeTorch:
            def __init__(self, *, built, available):
                self.backends = type(
                    "Backends", (), {"mps": FakeMPS(built=built, available=available)}
                )()

            @staticmethod
            def device(name):
                return f"device:{name}"

        self.assertEqual(
            select_device(FakeTorch(built=True, available=True), "auto"),
            "device:mps",
        )
        self.assertEqual(
            select_device(FakeTorch(built=True, available=False), "auto"),
            "device:cpu",
        )
        with self.assertRaisesRegex(RuntimeError, "built=True, available=False"):
            select_device(FakeTorch(built=True, available=False), "mps")

    def test_metrics_separate_strict_and_normalized_exact_match(self):
        metrics = answer_metrics("  Café, blue!  ", "café blue")

        self.assertFalse(metrics["strict_exact_match"])
        self.assertTrue(metrics["normalized_exact_match"])
        self.assertTrue(metrics["normalized_target_substring_match"])
        self.assertEqual(metrics["normalized_token_overlap_f1"], 1.0)

    def test_lexical_metrics_detect_partial_overlap_without_claiming_semantics(self):
        metrics = answer_metrics("The answer is blue bird.", "blue whale")

        self.assertFalse(metrics["strict_exact_match"])
        self.assertFalse(metrics["normalized_exact_match"])
        self.assertFalse(metrics["normalized_target_substring_match"])
        self.assertEqual(metrics["normalized_token_overlap_precision"], 1 / 5)
        self.assertEqual(metrics["normalized_token_overlap_recall"], 1 / 2)
        self.assertAlmostEqual(metrics["normalized_token_overlap_f1"], 2 / 7)

    def test_chat_prompt_uses_checkpoint_template_and_stable_context(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {"input_ids": "encoded"}

        encoded = encode_chat_prompt(tokenizer, "What is the answer?")

        self.assertEqual(encoded, {"input_ids": "encoded"})
        tokenizer.apply_chat_template.assert_called_once_with(
            build_chat_messages("What is the answer?"),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            date_string="10 Apr 2025",
        )

    def test_result_schema_records_provenance_configuration_and_metrics(self):
        records = [
            {"question": "Question A?", "answer": "Answer A"},
            {"question": "Question B?", "answer": "Answer B"},
        ]

        result = build_result_document(
            records,
            [1],
            ["Answer B"],
            seed=2026,
            max_new_tokens=128,
            device="cpu",
            model_dtype="float32",
            torch_version="test-torch",
            transformers_version="test-transformers",
        )

        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["evaluation_type"], "pre_unlearning_tofu_forget01_baseline")
        self.assertEqual(result["model"]["repo_id"], MODEL_ID)
        self.assertEqual(result["model"]["revision"], MODEL_REVISION)
        self.assertEqual(result["dataset"]["split"], "forget01")
        self.assertEqual(result["sampling"]["record_indices"], [1])
        self.assertEqual(result["sampling"]["seed"], 2026)
        self.assertEqual(result["aggregate_metrics"]["strict_exact_match_rate"], 1.0)
        self.assertEqual(result["examples"][0]["model_response"], "Answer B")
        self.assertEqual(result["software"]["torch"], "test-torch")
        self.assertEqual(json.loads(json.dumps(result))["schema_version"], "1.0")

    def test_aggregate_rejects_empty_sample(self):
        with self.assertRaisesRegex(ValueError, "empty sample"):
            aggregate_metrics([])


if __name__ == "__main__":
    unittest.main()
