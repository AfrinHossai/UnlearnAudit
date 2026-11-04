"""Offline fixtures only: no model construction, network, training or weights."""
import copy
import contextlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import small_model_experiment as experiment


class PlainTokenizer:
    eos_token_id = pad_token_id = 0
    all_special_ids = [0]
    padding_side = "right"

    def __len__(self):
        return 256

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text]


class SmallModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(experiment.PROFILE.read_text())
        self.probes = json.loads((experiment.ROOT / self.profile["evaluation"]["utility_probes"]).read_text())
        self.development = json.loads((experiment.ROOT / self.profile["development"]["path"]).read_text())
        self.records = {
            "forget01": [{"question": f"Forget {i}?", "answer": f"F answer {i}."} for i in range(40)],
            "retain99": [{"question": f"Retain {i}?", "answer": f"R answer {i}."} for i in range(3960)],
            "holdout01": [{"question": f"Holdout {i}?", "answer": f"H answer {i}."} for i in range(40)],
        }
        for base, split in [("forget01", "forget01_perturbed"), ("retain99", "retain_perturbed")]:
            self.records[split] = [{**r, "paraphrased_question": f"Reworded {r['question']}",
                                    "paraphrased_answer": r["answer"]} for r in self.records[base]]

    def manifest(self):
        return experiment.build_manifest(self.profile, self.records, self.probes, self.development)

    def test_reproducible_selection_schedules_and_no_text_logging(self):
        manifest = self.manifest()
        self.assertEqual(manifest, self.manifest())
        self.assertEqual(len(manifest["sets"]["forget_direct"]), 40)
        self.assertEqual(len(manifest["sets"]["retain_direct"]), 120)
        self.assertEqual(len(manifest["unlearning_and_retain_control_pairs"]), 40)
        trained = {c["case_id"] for c in manifest["sets"]["retain_direct"]}
        unseen = {c["case_id"] for c in manifest["sets"]["retain_unseen_records"]}
        self.assertFalse(trained & unseen)
        self.assertTrue(all(p["retain"] in trained for p in manifest["unlearning_and_retain_control_pairs"]))
        encoded = json.dumps(manifest)
        self.assertNotIn("F answer", encoded)
        self.assertNotIn("Forget 0?", encoded)
        self.assertNotIn(str(experiment.ROOT), encoded)
        order = experiment.acquisition_order(sorted(trained), 0)
        self.assertEqual(order, experiment.acquisition_order(sorted(trained), 0))
        self.assertNotEqual(order, experiment.acquisition_order(sorted(trained), 1))

    def test_forget_retain_and_holdout_overlap_rejected(self):
        for split in ("retain99", "holdout01"):
            with self.subTest(split=split):
                records = copy.deepcopy(self.records)
                records[split][0]["question"] = self.records["forget01"][0]["question"].upper()
                with self.assertRaisesRegex(ValueError, "overlap"):
                    experiment.build_manifest(self.profile, records, self.probes, self.development)

    def test_duplicate_selected_retain_rejected_without_repair(self):
        indices = experiment.deterministic_sample_indices(3960, 120, 2026)
        self.records["retain99"][indices[1]]["question"] = self.records["retain99"][indices[0]]["question"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.manifest()

    def test_paraphrase_alignment_wording_and_missing_pair_fail_closed(self):
        original = copy.deepcopy(self.records["forget01_perturbed"])
        for mutation, expected in [("answer", "mismatched"), ("wording", "overlap"), ("missing", "missing")]:
            with self.subTest(mutation=mutation):
                self.records["forget01_perturbed"] = copy.deepcopy(original)
                if mutation == "answer":
                    self.records["forget01_perturbed"][0]["answer"] = "Wrong original answer"
                elif mutation == "wording":
                    self.records["forget01_perturbed"][0]["paraphrased_question"] = self.records["retain99"][
                        experiment.deterministic_sample_indices(3960, 120, 2026)[0]]["question"]
                else:
                    self.records["forget01_perturbed"].pop(0)
                with self.assertRaisesRegex(ValueError, expected):
                    self.manifest()

    def test_optional_retain_paraphrases_are_available_matches_only(self):
        self.records["retain_perturbed"] = []
        self.assertEqual(self.manifest()["sets"]["retain_paraphrase"], [])

    def test_utility_overlap_and_schema_rejected(self):
        self.probes["cases"][0]["question"] = self.records["forget01"][0]["question"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.manifest()
        self.records["forget01"][0]["answer"] = ""
        with self.assertRaisesRegex(ValueError, "schema"):
            self.manifest()

    def test_execution_and_memory_guard_changes_rejected(self):
        mutations = [((), "weight_updates_enabled", True),
                     (("model",), "revision", "main"), (("model",), "trust_remote_code", True),
                     (("safety",), "before_load_available_lower_bound_gib", 8),
                     (("safety",), "device_fallback", True),
                     (("safety",), "training_entrypoint", "train"),
                     (("training",), "precision", "float16"), (("training",), "momentum", 0.9),
                     (("training",), "batch_size", True), (("acquisition",), "learning_rate", 0.001)]
        for path, key, value in mutations:
            with self.subTest(path=path, key=key):
                profile = copy.deepcopy(self.profile)
                target = profile
                for part in path:
                    target = target[part]
                target[key] = value
                with self.assertRaises(ValueError):
                    experiment.validate_profile(profile)

    def test_cli_rejects_training_switch_before_any_io(self):
        with mock.patch.object(experiment, "load_sources", side_effect=AssertionError("no data IO")):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    experiment.main(["--enable-weight-updates"])
        self.assertEqual(exc.exception.code, 2)

    def test_missing_data_and_checksum_errors_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                experiment.load_sources(self.profile, path)
            (path / "forget01.json").write_text("wrong bytes")
            with self.assertRaisesRegex(ValueError, "checksum"):
                experiment.load_sources(self.profile, path)

    def test_manifest_output_protection_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "manifest.json"
            with self.assertRaises(ValueError):
                experiment.write_new_manifest(path, {})
            with mock.patch.object(experiment, "MANIFEST", path):
                experiment.write_new_manifest(path, {"ok": True})
                with self.assertRaises(FileExistsError):
                    experiment.write_new_manifest(path, {})
                link = Path(tmp) / "link.json"
                link.symlink_to(path)
                with self.assertRaises(ValueError):
                    experiment.write_new_manifest(link, {})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})

    def test_missing_or_modified_metadata_rejected_before_ML_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                experiment.verify_tokenizer(self.profile, root, {}, {}, {}, {})
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                experiment.verify_tokenizer(self.profile, root, {}, {}, {}, {})

    def test_plain_QA_masking_and_no_truncation(self):
        tokenizer = PlainTokenizer()
        example = experiment.encode_plain_qa(tokenizer, "Who?", "An answer.")
        start = example.assistant_target_start
        self.assertTrue(all(t == -100 for t in example.labels[:start]))
        self.assertEqual(example.labels[start:], example.input_ids[start:])
        self.assertEqual(example.input_ids[-1], 0)
        with self.assertRaisesRegex(ValueError, "truncation"):
            experiment.encode_plain_qa(tokenizer, "Who?", "An answer.", max_sequence_length=5)
        with self.assertRaises(ValueError):
            experiment.encode_plain_qa(tokenizer, "Who?", "")
        tokenizer.pad_token_id = None
        with self.assertRaisesRegex(ValueError, "padding"):
            experiment.encode_plain_qa(tokenizer, "Who?", "Answer")

    def test_uncertain_token_boundary_rejected(self):
        tokenizer = PlainTokenizer()
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kw: ([255] + original(text, **kw)[1:]
                                               if text.endswith("Answer") else original(text, **kw))
        with self.assertRaisesRegex(ValueError, "boundary"):
            experiment.encode_plain_qa(tokenizer, "Who?", "Answer")

    def test_committed_manifest_matches_configuration_and_probe_hashes(self):
        manifest = json.loads(experiment.MANIFEST.read_text())
        self.assertEqual(manifest["profile_sha256"], experiment.digest(self.profile))
        self.assertEqual(manifest["utility_probes_sha256"], experiment.digest(self.probes))
        self.assertEqual(manifest["development_sha256"], experiment.digest(self.development))
        self.assertEqual({k: len(v) for k, v in manifest["sets"].items()}, {
            "forget_direct": 40, "retain_direct": 120, "retain_unseen_records": 40,
            "holdout01_sealed": 40, "forget_paraphrase": 40, "retain_paraphrase": 12})

    def test_development_values_reproduce_and_overlap_is_rejected(self):
        rng = random.Random(2026)
        for row in self.development["facts"]:
            self.assertEqual(row["target_aliases"], ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(12))])
        self.records["forget01"][0]["answer"] = self.development["facts"][0]["target_aliases"][0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            experiment.validate_development(self.development, self.records, set())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires project ML environment")
    def test_physical_check_reuses_existing_guard_and_no_fallback(self):
        from src import graddiff_real_train as runner
        with mock.patch.object(runner, "resolve_training_device") as resolve:
            resolve.return_value = type("Device", (), {"__str__": lambda self: "mps"})()
            with mock.patch.object(runner, "guarded_memory_snapshot", return_value={"safe": True}) as guard:
                result = experiment.check_physical_mps_readiness()
                self.assertFalse(result["model_loaded"])
                self.assertEqual(resolve.call_args.args[0], "mps")
                self.assertEqual(guard.call_args.args[2].optimizer, "sgd")
                self.assertTrue(guard.call_args.kwargs["before_load"])
            resolve.side_effect = RuntimeError("MPS unavailable")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                experiment.check_physical_mps_readiness()
        snapshot = {"total_physical_memory_bytes": 48 * runner.GIB,
                    "available_physical_memory_lower_bound_bytes": 16 * runner.GIB,
                    "system_memory_free_percentage": 95}
        with self.assertRaisesRegex(RuntimeError, "32 GiB"):
            runner.check_training_memory(snapshot, optimizer="sgd", before_load=True)
        snapshot["available_physical_memory_lower_bound_bytes"] = 32 * runner.GIB
        runner.check_training_memory(snapshot, optimizer="sgd", before_load=True)
        with self.assertRaisesRegex(RuntimeError, "42 GiB"):
            runner.check_training_memory(snapshot, optimizer="adamw", before_load=True)


if __name__ == "__main__":
    unittest.main()
