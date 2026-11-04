"""Offline structural/control tests: no network, models, GPU or training."""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src import colab_workflow as workflow


class ColabStructureTests(unittest.TestCase):
    def setUp(self):
        self.config, self.profile, self.manifest = workflow.configuration()

    def test_notebook_is_unexecuted_valid_python_without_gpu_allocation_metadata(self):
        path = workflow.ROOT / "notebooks/smollm2_135m_tofu_colab.ipynb"
        notebook = workflow.read_json(path)
        self.assertEqual((notebook["nbformat"], notebook["nbformat_minor"]), (4, 5))
        self.assertNotIn("accelerator", notebook["metadata"])
        seen = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                tree = ast.parse("".join(cell["source"]))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_stage":
                        seen.append(node.args[0].value)
        self.assertEqual(set(seen), set(workflow.STAGES))
        self.assertEqual(len(seen), len(workflow.STAGES))
        first = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        for forbidden in ("from_pretrained", "pip install", "drive.mount", "optimizer", "run_stage("):
            self.assertNotIn(forbidden, first)

    def test_notebook_builder_reproduces_json_without_execution(self):
        spec = importlib.util.spec_from_file_location("builder", workflow.ROOT / "scripts/build_colab_notebook.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.assertEqual(builder.notebook, workflow.read_json(workflow.ROOT / "notebooks/smollm2_135m_tofu_colab.ipynb"))

    def test_new_profile_keeps_original_profile_preparation_only(self):
        self.assertFalse(self.profile["weight_updates_enabled"])
        self.assertEqual(self.profile["training"]["device"], "mps")
        self.assertIsNone(self.profile["safety"]["training_entrypoint"])
        self.assertIsNone(self.config["acquisition_learning_rate"])
        self.assertIsNone(self.config["unlearning_learning_rate"])
        self.assertEqual(self.manifest["profile_sha256"], workflow.digest(self.profile))
        self.assertTrue(self.config["memory_thresholds_are_unmeasured_planning_reserves"])
        self.assertEqual(self.config["batch_size"], 1)

    def test_default_cli_is_plan_only_without_any_runtime_actions(self):
        with mock.patch.object(workflow, "environment", side_effect=AssertionError("runtime forbidden")):
            with mock.patch.object(workflow, "prepare_data", side_effect=AssertionError("data forbidden")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(workflow.main([]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "plan_only")
        self.assertFalse(report["weight_updates_performed"])

    def test_every_stage_requires_matching_confirmation_before_side_effects(self):
        for stage in workflow.STAGES:
            args = argparse.Namespace(stage=stage, confirm=None, run_dir=None, protocol=None, resume=False)
            with self.subTest(stage=stage), mock.patch.object(workflow, "environment", side_effect=AssertionError()):
                with self.assertRaisesRegex(ValueError, "confirmation|confirm"):
                    workflow.execute(args)

    def test_module_import_and_plan_work_in_non_ml_interpreter(self):
        script = "from src.colab_workflow import main; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; main([])"
        result = subprocess.run([sys.executable, "-c", script], cwd=workflow.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schedules_preserve_fixed_exposure_and_manifest_pairs(self):
        all_steps = workflow.train_schedule("acquire", self.manifest)
        reference = workflow.train_schedule("reference", self.manifest)
        self.assertEqual(len(all_steps), 3200)
        self.assertEqual(len(reference), 2400)
        f = {c["case_id"] for c in self.manifest["sets"]["forget_direct"]}
        self.assertFalse({case for case, _ in reference} & f)
        for stage in ("retain-control", "unlearn"):
            steps = workflow.train_schedule(stage, self.manifest)
            self.assertEqual(len(steps), 40)
            self.assertEqual(steps, [(p["forget"], p["retain"]) for p in self.manifest["unlearning_and_retain_control_pairs"]])
        self.assertEqual(workflow.digest([[case for case, _ in all_steps[e*160:(e+1)*160]] for e in range(20)]),
                         self.manifest["acquisition_schedule_sha256"])

    def test_evaluation_separates_forget_retain_and_excludes_sealed_holdout(self):
        self.assertNotIn("holdout01_sealed", workflow.EVAL_SETS)
        self.assertEqual([len(self.manifest["sets"][s]) for s in workflow.EVAL_SETS[:-1]], [40, 120, 40, 12, 40])
        trained = {c["case_id"] for c in self.manifest["sets"]["retain_direct"]}
        unseen = {c["case_id"] for c in self.manifest["sets"]["retain_unseen_records"]}
        self.assertFalse(trained & unseen)

    def test_reject_repo_and_symlink_output_paths(self):
        for path in (workflow.ROOT, workflow.ROOT / "outputs/cloud", workflow.ROOT / "notebooks/../outputs"):
            with self.assertRaises(ValueError):
                workflow.outside_git(path)
        with tempfile.TemporaryDirectory() as folder:
            link = Path(folder) / "escape"
            link.symlink_to(workflow.ROOT, target_is_directory=True)
            with self.assertRaises(ValueError):
                workflow.outside_git(link / "checkpoints")

    def test_declared_dependencies_are_exact_pins_separate_from_mac_lock(self):
        requirements = (workflow.ROOT / "requirements-colab.txt").read_text()
        pins = dict(line.split("==") for line in requirements.splitlines() if line and not line.startswith("#"))
        self.assertEqual(pins["transformers"], "4.57.1")
        self.assertEqual(pins["tokenizers"], "0.22.1")
        self.assertEqual(self.config["torch"], "2.9.1")
        self.assertEqual(self.config["cuda_wheel"], "cu126")
        self.assertNotIn("torch", pins)  # separate official CUDA wheel index install

    def test_strict_environment_rejects_cpu_wheel_before_any_cuda_or_model(self):
        versions = {"torch": "2.9.1"}
        with mock.patch.object(workflow.importlib.metadata, "version", side_effect=lambda name: versions.get(name, "unused")), \
                mock.patch.object(workflow.sys, "platform", "linux"), \
                mock.patch.object(workflow.sys, "version_info", SimpleNamespace(major=3, minor=11)), \
                mock.patch.object(workflow.platform, "python_version", return_value="3.11.0"), \
                mock.patch.object(workflow.platform, "platform", return_value="Linux fixture"):
            with self.assertRaisesRegex(RuntimeError, "torch must be 2.9.1\\+cu126"):
                workflow.environment(require_cuda=True)

    def test_cuda_visibility_driver_kernel_and_architecture_checks_with_inert_modules(self):
        versions = {"torch": "2.9.1+cu126"}
        versions.update(dict(line.split("==") for line in (workflow.ROOT / "requirements-colab.txt").read_text().splitlines()
                             if line and not line.startswith("#")))
        calls = []
        cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 1,
                               synchronize=lambda: calls.append("sync"), get_device_name=lambda _: "fixture GPU",
                               get_device_capability=lambda _: (7, 5))
        fake_torch = SimpleNamespace(version=SimpleNamespace(cuda="12.6"), cuda=cuda,
                                     ones=lambda *args, **kwargs: SimpleNamespace(matmul=lambda other: calls.append("matmul")))
        fake_transformers = SimpleNamespace(__version__="4.57.1", LlamaForCausalLM=type("LlamaForCausalLM", (), {}))
        fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=12*2**30))
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers, "psutil": fake_psutil}), \
                mock.patch.object(workflow.importlib.metadata, "version", side_effect=versions.__getitem__), \
                mock.patch.object(workflow.sys, "platform", "linux"), \
                mock.patch.object(workflow.sys, "version_info", SimpleNamespace(major=3, minor=11)), \
                mock.patch.object(workflow.platform, "python_version", return_value="3.11.0"), \
                mock.patch.object(workflow.platform, "platform", return_value="Linux fixture"), \
                mock.patch.object(workflow.subprocess, "run"), \
                mock.patch.object(workflow.subprocess, "check_output", return_value="520.0") as driver:
            with self.assertRaisesRegex(RuntimeError, "visible compatible GPU"):
                workflow.environment(require_cuda=True)
            cuda.is_available = lambda: True
            with self.assertRaisesRegex(RuntimeError, "driver is below"):
                workflow.environment(require_cuda=True)
            self.assertFalse(calls)
            driver.return_value = "535.0"
            report = workflow.environment(require_cuda=True)
            self.assertTrue(report["cuda_ready"])
            self.assertEqual(report["billing_status"], "not_detectable_by_python")
            self.assertEqual(calls, ["matmul", "sync"])


class ProtocolAndMetricsTests(unittest.TestCase):
    def setUp(self):
        self.config, self.profile, self.manifest = workflow.configuration()
        self.sets = {"forget_direct": [{"case_id": "fixture:01", "question": "Fixture?", "answer": "A blue object."}]}
        self.protocol = {
            "status": "frozen_reviewed", "reviewed_by": "offline fixture, not real review",
            "review_notes": "inert test only", "manifest_sha256": workflow.digest(self.manifest),
            "target_scores_used_for_rate_selection": False,
            "acquisition_learning_rate": 0.0001, "unlearning_learning_rate": 0.00001,
            "aliases_by_case_id": {"fixture:01": ["blue"]},
            "refusal_phrases": ["i cannot answer"], "repetition_ngram": 3, "repetition_count": 3,
        }
        self.evidence = {
            "status": "completed_reviewed_development_only", "target_scores_used_for_selection": False,
            "model_revision": workflow.REVISION, "development_sha256": self.manifest["development_sha256"],
            "acquisition_learning_rate": 0.0001, "unlearning_learning_rate": 0.00001,
            "candidate_reports": ["inert structural fixture; not measurements"], "reviewed_by": "test fixture",
        }

    def validate(self, protocol=None, evidence=None):
        workflow.validate_protocol(protocol or self.protocol, self.sets, self.profile, self.manifest, evidence or self.evidence)

    def test_template_blocks_execution(self):
        template = workflow.read_json(workflow.ROOT / "experiments/smollm2_135m_colab_protocol.template.json")
        with self.assertRaisesRegex(ValueError, "reviewed"):
            self.validate(template)

    def test_missing_uncalibrated_or_target_tuned_rates_rejected(self):
        self.validate()
        for key, value in (("acquisition_learning_rate", None), ("unlearning_learning_rate", 2.0),
                           ("target_scores_used_for_rate_selection", True), ("manifest_sha256", "wrong")):
            invalid = {**self.protocol, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(invalid)
        for key, value in (("candidate_reports", []), ("model_revision", "main"),
                           ("target_scores_used_for_selection", True), ("development_sha256", "wrong")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(evidence={**self.evidence, key: value})

    def test_incomplete_ungrounded_or_empty_aliases_rejected(self):
        for aliases in ({}, {"fixture:01": []}, {"fixture:01": ["red"]}, {"fixture:01": [""]},
                        {"fixture:01": ["blue"], "extra": ["blue"]}):
            with self.subTest(aliases=aliases), self.assertRaises(ValueError):
                self.validate({**self.protocol, "aliases_by_case_id": aliases})

    def test_freeze_preserves_byte_hash_and_is_idempotent_for_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps(self.evidence) + "\n")
            protocol = {**self.protocol, "calibration_evidence_file": str(evidence_path),
                        "calibration_evidence_sha256": workflow.sha256_file(evidence_path)}
            path = root / "protocol.json"
            path.write_text(json.dumps(protocol))
            first = workflow.freeze_protocol(run, path, self.sets, self.profile, self.manifest)
            self.assertEqual(workflow.sha256_file(run / "calibration.json"), protocol["calibration_evidence_sha256"])
            self.assertEqual(first, workflow.freeze_protocol(run, path, self.sets, self.profile, self.manifest))
            path.write_text(json.dumps({**protocol, "review_notes": "changed"}))
            with self.assertRaises(FileExistsError):
                workflow.freeze_protocol(run, path, self.sets, self.profile, self.manifest)

    def test_corrupted_evidence_and_missing_evidence_block_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            path.write_text(json.dumps(self.protocol))
            with self.assertRaisesRegex(ValueError, "evidence"):
                workflow.freeze_protocol(Path(folder), path, self.sets, self.profile, self.manifest)

    def test_literal_diagnostics_boundaries_refusals_empty_and_repetition(self):
        score = workflow.score_response("Blue blue blue blue blue.", "A blue object.", ["blue"], self.protocol)
        self.assertEqual(score["key_fact_recovery"], 1)
        self.assertEqual(score["repetition"], 1)
        self.assertEqual(workflow.score_response("blueberry", "blue", ["blue"], self.protocol)["key_fact_recovery"], 0)
        self.assertEqual(workflow.score_response("", "blue", ["blue"], self.protocol)["empty"], 1)
        self.assertEqual(workflow.score_response("I cannot answer", "blue", ["blue"], self.protocol)["refusal"], 1)

    def summaries(self, recovery=0.9, nll=1.0):
        return {name: {"count": 1, "key_fact_recovery": recovery, "answer_nll": nll,
                       "f1": 0.9, "exact_match": 0.5, "empty": 0., "refusal": 0., "repetition": 0.}
                for name in workflow.EVAL_SETS}

    def test_acquisition_gate_fails_without_real_gains_and_retain_acquisition(self):
        before = self.summaries(0.1, 3.)
        after = self.summaries()
        gates = self.profile["evaluation"]["acquisition_gates"]
        self.assertTrue(workflow.acquisition_gate(before, after, gates))
        self.assertFalse(workflow.acquisition_gate(after, after, gates))
        after["retain_direct"]["key_fact_recovery"] = 0.5
        self.assertFalse(workflow.acquisition_gate(before, after, gates))

    def test_post_gates_require_forgetting_beyond_control_and_preserved_utility(self):
        before = self.summaries()
        after = copy.deepcopy(before)
        control = copy.deepcopy(before)
        for name in ("forget_direct", "forget_paraphrase"):
            after[name].update(key_fact_recovery=0.4, answer_nll=2.)
        gates = self.profile["evaluation"]["unlearning_gates"]
        result = workflow.compare_summaries(before, after, control, gates)
        self.assertTrue(result["proposed_diagnostic_gates_pass"])
        self.assertFalse(result["proof_of_erasure"])
        self.assertFalse(workflow.compare_summaries(before, after, after, gates)["proposed_diagnostic_gates_pass"])
        after["retain_direct"]["refusal"] = 0.2
        self.assertFalse(workflow.compare_summaries(before, after, control, gates)["proposed_diagnostic_gates_pass"])


class CheckpointTests(unittest.TestCase):
    identity = {"fixture": "inert bytes, not model weights"}

    def checkpoint(self, root, stage, step, budget=80):
        folder = root / stage / f"step-{step:06d}"
        folder.mkdir(parents=True)
        for name in ("resume.pt", "model.safetensors", "config.json"):
            (folder / name).write_bytes(b"inert checkpoint fixture")
        workflow.atomic_json(folder / "checkpoint.json", {
            "stage": stage, "step": step, "budget": budget, "identity": self.identity,
            "schedule_sha256": "fixture", "parent_sha256": "fixture",
            "files": {p.name: workflow.sha256_file(p) for p in folder.iterdir()}})
        return folder

    def marker(self, root, stage, checkpoint):
        artifact = checkpoint / "checkpoint.json"
        workflow.atomic_json(root / f"{stage}.complete.json", {
            "status": "completed", "stage": stage, "identity": self.identity,
            "artifact": str(artifact.relative_to(root)), "artifact_sha256": workflow.sha256_file(artifact)})

    def test_running_or_partial_state_is_not_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workflow.atomic_json(root / "acquire.attempt.json", {"status": "running_not_completed"})
            (root / "acquire/.partial-interrupted").mkdir(parents=True)
            self.assertIsNone(workflow.latest_checkpoint(root, "acquire", self.identity))
            with self.assertRaisesRegex(RuntimeError, "not durably complete"):
                workflow.complete(root, "acquire", self.identity)

    def test_corrupt_newest_checkpoint_falls_back_to_verified_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older = self.checkpoint(root, "acquire", 40)
            newest = self.checkpoint(root, "acquire", 80)
            self.assertEqual(workflow.latest_checkpoint(root, "acquire", self.identity), newest)
            (newest / "model.safetensors").write_bytes(b"corrupted")
            self.assertEqual(workflow.latest_checkpoint(root, "acquire", self.identity), older)
            (older / "resume.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "no intact checkpoint"):
                workflow.latest_checkpoint(root, "acquire", self.identity)

    def test_resume_requires_explicit_opt_in_and_identical_schedule_parent_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "no intact checkpoint"):
                workflow.recovery_checkpoint(root, "acquire", self.identity, "fixture", "fixture", 80, True)
            checkpoint = self.checkpoint(root, "acquire", 40)
            with self.assertRaisesRegex(RuntimeError, "explicitly select resume"):
                workflow.recovery_checkpoint(root, "acquire", self.identity, "fixture", "fixture", 80, False)
            self.assertEqual(workflow.recovery_checkpoint(root, "acquire", self.identity, "fixture", "fixture", 80, True),
                             (checkpoint, 40))
            for schedule, parent, budget in (("changed", "fixture", 80), ("fixture", "changed", 80), ("fixture", "fixture", 40)):
                with self.subTest(schedule=schedule, parent=parent, budget=budget), self.assertRaisesRegex(ValueError, "mismatch"):
                    workflow.recovery_checkpoint(root, "acquire", self.identity, schedule, parent, budget, True)

    def test_checkpoint_transaction_readback_and_pruning_with_no_model_or_cuda(self):
        config, _, _ = workflow.configuration()
        class InertModel:
            def save_pretrained(self, folder, safe_serialization):
                self.assert_safe = safe_serialization
                (folder / "model.safetensors").write_bytes(b"inert fixture")
                (folder / "config.json").write_text("{}")
        class InertTokenizer:
            def save_pretrained(self, folder):
                (folder / "tokenizer.json").write_text("{}")
        fake_torch = SimpleNamespace(save=lambda state, path: path.write_bytes(b"inert RNG/optimizer fixture"),
                                     get_rng_state=lambda: "CPU RNG fixture",
                                     cuda=SimpleNamespace(get_rng_state_all=lambda: ["CUDA RNG fixture"]))
        model = InertModel()
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"torch": fake_torch}):
            root = Path(folder)
            for step in (40, 80, 120):
                saved = workflow.save_checkpoint(root, "acquire", step, 120, model, InertTokenizer(), None,
                                                 self.identity, "schedule", "parent", config)
                self.assertEqual(workflow.checkpoint_info(saved, self.identity, "acquire")["step"], step)
            self.assertTrue(model.assert_safe)
            self.assertEqual(sorted(p.name for p in (root / "acquire").iterdir()), ["step-000080", "step-000120"])
            with self.assertRaises(FileExistsError):
                workflow.save_checkpoint(root, "acquire", 120, 120, model, InertTokenizer(), None,
                                         self.identity, "schedule", "parent", config)

    def test_completed_marker_requires_full_budget_and_every_hashed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            partial = self.checkpoint(root, "acquire", 40)
            self.marker(root, "acquire", partial)
            with self.assertRaisesRegex(ValueError, "partial checkpoint"):
                workflow.complete(root, "acquire", self.identity)
            final = self.checkpoint(root, "acquire", 80)
            self.marker(root, "acquire", final)
            self.assertEqual(workflow.complete(root, "acquire", self.identity), final / "checkpoint.json")
            (final / "resume.pt").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "incomplete/corrupt"):
                workflow.complete(root, "acquire", self.identity)

    def test_mismatched_identity_and_corrupt_related_comparison_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = self.checkpoint(root, "acquire", 80)
            self.marker(root, "acquire", checkpoint)
            with self.assertRaisesRegex(ValueError, "identity/status"):
                workflow.complete(root, "acquire", {"other": "run"})
            marker = root / "acquire.complete.json"
            value = workflow.read_json(marker)
            (root / "comparison.json").write_text("fixture")
            value["related_artifacts"] = {"comparison.json": workflow.sha256_file(root / "comparison.json")}
            workflow.atomic_json(marker, value)
            (root / "comparison.json").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "related artifact"):
                workflow.complete(root, "acquire", self.identity)

    def test_runtime_memory_guards_fail_closed_with_inert_telemetry(self):
        config, _, _ = workflow.configuration()
        gpu = SimpleNamespace(mem_get_info=lambda _: (7*2**30, 16*2**30),
                              memory_allocated=lambda: 0, memory_reserved=lambda: 0, max_memory_allocated=lambda: 0)
        fake_torch = SimpleNamespace(cuda=gpu)
        fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=5*2**30))
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"torch": fake_torch, "psutil": fake_psutil}):
            with mock.patch.object(workflow.shutil, "disk_usage", return_value=SimpleNamespace(free=5*2**30)):
                self.assertEqual(workflow.memory_check(config, Path(folder), before_load=True)["gpu_free_gib"], 7)
                gpu.mem_get_info = lambda _: (0, 16*2**30)
                with self.assertRaisesRegex(RuntimeError, "reserve violated"):
                    workflow.memory_check(config, Path(folder))


if __name__ == "__main__":
    unittest.main()
