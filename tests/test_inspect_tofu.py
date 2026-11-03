import json
import tempfile
import unittest
from pathlib import Path

from src.inspect_tofu import inspect_records, load_jsonl, representative_indices


class InspectTofuTests(unittest.TestCase):
    def test_discovers_schema_and_validates_required_fields(self):
        records = [
            {"question": "Question one?", "answer": "Answer one.", "item_id": 1},
            {"question": "Question two?", "answer": "Answer two.", "item_id": 2},
        ]

        inspected = inspect_records(records)

        self.assertEqual(inspected["record_count"], 2)
        self.assertEqual(
            inspected["schema"],
            {"answer": ["str"], "item_id": ["int"], "question": ["str"]},
        )
        self.assertTrue(inspected["required_fields_valid"])

    def test_reports_missing_required_fields_after_schema_discovery(self):
        with self.assertRaisesRegex(ValueError, "missing required field.*answer"):
            inspect_records([{"question": "Question?", "other": "value"}])

    def test_rejects_empty_or_non_string_required_values(self):
        with self.assertRaisesRegex(ValueError, "non-string 'answer'"):
            inspect_records([{"question": "Question?", "answer": None}])

    def test_loads_jsonl_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.jsonl"
            expected = [
                {"question": "First?", "answer": "First."},
                {"question": "Second?", "answer": "Second."},
            ]
            path.write_text(
                json.dumps(expected[0]) + "\n\n" + json.dumps(expected[1]) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(load_jsonl(path), expected)

    def test_examples_are_spread_deterministically(self):
        self.assertEqual(representative_indices(40), [0, 19, 39])
        with self.assertRaisesRegex(ValueError, "At least three"):
            representative_indices(2)


if __name__ == "__main__":
    unittest.main()
