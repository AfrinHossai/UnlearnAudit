"""Offline preparation of the separate SmolLM2 TOFU experiment.

No model constructor, weight loader, optimizer or training entry point exists
here. Tokenizer verification uses explicitly supplied local metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluate_tofu_baseline import normalize_answer
from src.evaluate_tofu_paired import DATASET_FILE_SPECS, deterministic_sample_indices, load_pinned_split
from src.graddiff_integration import EncodedCausalExample
from src.inspect_tofu import DATASET_ID, DATASET_REVISION, inspect_records, load_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "experiments/smollm2_135m_tofu_controlled.json"
MANIFEST = ROOT / "experiments/smollm2_135m_tofu_manifest.json"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
RETAIN_PARAPHRASE_SPEC = {
    "filename": "retain_perturbed.json",
    "sha256": "fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb",
    "record_count": 400,
    "upstream_git_blob_sha1": "d3c80f2f7d85cbec3a6b341d175af08ede545904",
}
SAFETY = {
    "before_load_available_lower_bound_gib": 32,
    "adamw_alternative_threshold_gib": 42,
    "continuing_available_lower_bound_gib": 6,
    "minimum_system_memory_free_percentage": 20,
    "minimum_total_host_gib": 48,
    "device_fallback": False,
    "outputs_root": "outputs/smollm2_135m_tofu",
    "require_new_git_ignored_output_paths": True,
    "no_original_checkpoint_overwrite": True,
    "explicit_update_authorization_required": True,
    "training_entrypoint": None,
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _require_fields(actual: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    for key, value in expected.items():
        if actual.get(key) != value or type(actual.get(key)) is not type(value):
            raise ValueError(f"unsupported {name}.{key}")


def validate_profile(profile: Mapping[str, Any]) -> None:
    """Reject execution switches, substitutions and changes to existing guards."""
    _require_fields(profile, {"schema_version": "1.0", "status": "preparation_only_not_executable",
                             "weight_updates_enabled": False, "seed": 2026,
                             "manifest": str(MANIFEST.relative_to(ROOT))}, "profile")
    _require_fields(profile["model"], {
        "repo_id": MODEL_ID, "revision": REVISION, "tokenizer_revision": REVISION,
        "parameter_count": 134515008, "architecture": "LlamaForCausalLM",
        "trust_remote_code": False, "tokenizer_class": "GPT2TokenizerFast",
        "model_type": "llama", "vocab_size": 49152, "hidden_size": 576,
        "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9,
        "num_key_value_heads": 3, "tie_word_embeddings": True,
        "max_position_embeddings": 8192, "eos_token_id": 0, "license": "Apache-2.0",
    }, "model")
    _require_fields(profile["safety"], SAFETY, "safety")
    _require_fields(profile["dataset"], {
        "repo_id": DATASET_ID, "revision": DATASET_REVISION, "license": "MIT",
        "forget_count": 40, "retain_count": 120, "retain_unseen_count": 40,
        "files": {**DATASET_FILE_SPECS, "retain_perturbed": RETAIN_PARAPHRASE_SPEC},
    }, "dataset")
    _require_fields(profile["training"], {
        "device": "mps", "optimizer": "torch.optim.SGD", "momentum": 0.0,
        "weight_decay": 0.0, "foreach": False, "fused": False, "precision": "float32",
        "autocast": False, "attention_implementation": "sdpa", "gradient_checkpointing": True,
        "checkpoint_use_reentrant": False, "checkpoint_preserve_rng_state": True,
        "use_cache": False, "batch_size": 1, "gradient_accumulation_steps": 1,
        "max_sequence_length": 512, "max_grad_norm": 1.0, "scheduler": "none",
        "finite_loss_and_gradients_required": True,
    }, "training")
    _require_fields(profile["encoding"], {
        "format": "Question: {question}\nAnswer:\n{answer}", "add_special_tokens": False,
        "append_eos": True, "mask_prompt_and_padding": True, "ignore_index": -100,
        "padding_side": "right", "pad_token_id": 0,
        "pad_policy": "explicit_existing_eos_no_embedding_resize",
        "boundary_policy": "exact_prompt_token_prefix_or_reject", "truncation": False,
    }, "encoding")
    _require_fields(profile["acquisition"], {"epochs": 20, "max_optimizer_steps": 3200,
                    "retain_reference_max_optimizer_steps": 2400, "learning_rate": None,
                    "candidate_learning_rates": [0.0001, 0.0003, 0.001]}, "acquisition")
    _require_fields(profile["unlearning"], {"gamma": 1.0, "alpha": 1.0, "retain_loss_type": "NLL",
                    "epochs": 1, "max_optimizer_steps": 40, "learning_rate": None,
                    "candidate_learning_rates": [0.00001, 0.00003, 0.0001]}, "unlearning")
    _require_fields(profile["evaluation"], {"utility_probes": "experiments/smollm2_135m_utility_probes.json",
                    "key_fact_aliases_status": "not_prepared_blocks_updates", "official_TOFU_scores": False}, "evaluation")
    _require_fields(profile["development"], {"path": "experiments/smollm2_135m_development_facts.json",
                    "seed": 2026, "forget_records": 8, "retain_records": 24,
                    "acquisition_epochs": 20, "acquisition_max_optimizer_steps": 640,
                    "retain_reference_max_optimizer_steps": 480, "unlearning_max_optimizer_steps": 8,
                    "every_candidate_starts_from_same_base": True,
                    "target_TOFU_scores_used_for_rate_selection": False}, "development")
    if profile["memory_estimates"]["measured"] is not False:
        raise ValueError("unmeasured memory must remain labelled as estimates")


def load_sources(profile: Mapping[str, Any], data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Verify cached official files; never fetch missing data."""
    records = {}
    for split, spec in profile["dataset"]["files"].items():
        path = data_dir / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"required cached TOFU file missing: {spec['filename']}")
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"TOFU checksum mismatch: {split}")
        if split in DATASET_FILE_SPECS:
            rows, _ = load_pinned_split(split, path, download_if_missing=False)
        else:
            rows = load_jsonl(path)
            inspect_records(rows)
        if len(rows) != spec["record_count"]:
            raise ValueError(f"unexpected TOFU row count: {split}")
        records[split] = rows
    return records


def _question_keys(rows: Sequence[Mapping[str, Any]], name: str) -> set[str]:
    keys = []
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("question", "answer")):
            raise ValueError(f"invalid QA schema: {name}")
        key = normalize_answer(row["question"])
        if not key:
            raise ValueError(f"empty normalized question: {name}")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate normalized question: {name}")
    return set(keys)


def _case(split: str, index: int, row: Mapping[str, Any], *, paraphrase: bool = False) -> dict[str, Any]:
    q, a = ("paraphrased_question", "paraphrased_answer") if paraphrase else ("question", "answer")
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in (q, a)):
        raise ValueError(f"invalid question/answer fields: {split}:{index}")
    return {"case_id": f"{split}:{index:04d}", "source_split": split, "record_index": index,
            "question_field": q, "answer_field": a,
            "question_sha256": hashlib.sha256(row[q].encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(row[a].encode()).hexdigest()}


def aligned_paraphrases(base: Sequence[dict[str, Any]], selected: Sequence[int],
                       paraphrases: Sequence[dict[str, Any]], split: str, *,
                       training_questions: set[str], require_all: bool) -> list[dict[str, Any]]:
    """Match original QA, then reject wording leakage and duplicate paraphrases."""
    by_question: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(paraphrases):
        key = normalize_answer(row["question"])
        if key in by_question:
            raise ValueError(f"duplicate original question in {split}")
        by_question[key] = (index, row)
    cases, seen = [], set()
    for index in selected:
        row = base[index]
        pair = by_question.get(normalize_answer(row["question"]))
        if pair is None:
            if require_all:
                raise ValueError(f"missing aligned paraphrase: {split}")
            continue
        para_index, para = pair
        if normalize_answer(para["answer"]) != normalize_answer(row["answer"]):
            raise ValueError(f"mismatched original answer: {split}")
        case = _case(split, para_index, para, paraphrase=True)
        key = normalize_answer(para["paraphrased_question"])
        if not key or key in training_questions or key in seen:
            raise ValueError(f"paraphrase wording overlap: {split}")
        seen.add(key)
        cases.append({**case, "paired_source_index": index})
    return cases


def acquisition_order(case_ids: Sequence[str], epoch: int, seed: int = 2026) -> list[str]:
    return sorted(case_ids, key=lambda case: hashlib.sha256(f"{seed}:{epoch}:{case}".encode()).digest())


def validate_development(development: Mapping[str, Any], records: Mapping[str, Any],
                         utility_questions: set[str]) -> None:
    """Reject development/target leakage; these are new objects, not TOFU authors."""
    _require_fields(development, {"seed": 2026, "not_TOFU": True, "license": "CC0-1.0"}, "development")
    facts = development["facts"]
    questions = _question_keys(facts, "development")
    if (len(facts) != 32 or len({r["id"] for r in facts}) != 32
            or sum(r["role"] == "forget" for r in facts) != 8
            or sum(r["role"] == "retain" for r in facts) != 24):
        raise ValueError("development requires 8 forget and 24 retain facts with unique IDs")
    target_texts = [normalize_answer(row[field]) for rows in records.values() for row in rows
                    for field in ("question", "answer", "paraphrased_question", "paraphrased_answer")
                    if isinstance(row.get(field), str)]
    target_keys = set(target_texts) | utility_questions
    paraphrase_keys, aliases = set(), set()
    for row in facts:
        para = row.get("paraphrased_question")
        if not isinstance(para, str) or not para.strip():
            raise ValueError("missing development paraphrase")
        key = normalize_answer(para)
        if not key or key in questions or key in paraphrase_keys:
            raise ValueError("development paraphrase overlap")
        paraphrase_keys.add(key)
        if not isinstance(row.get("entity"), str) or not row["entity"].startswith("UGDEV2026-Object-"):
            raise ValueError("invalid development entity")
        values = row.get("target_aliases")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("invalid development target alias")
        alias = normalize_answer(values[0])
        if (len(alias) != 12 or not alias.isalpha() or alias in aliases
                or alias not in normalize_answer(row["answer"])
                or any(alias in text or normalize_answer(row["entity"]) in text for text in target_texts)):
            raise ValueError("development alias/identity leakage or invalid alias")
        aliases.add(alias)
    if (questions | paraphrase_keys) & target_keys:
        raise ValueError("development and target/utility overlap")


def build_manifest(profile: Mapping[str, Any], records: Mapping[str, Sequence[dict[str, Any]]],
                   probes: Mapping[str, Any], development: Mapping[str, Any]) -> dict[str, Any]:
    validate_profile(profile)
    f, r, h = (records[s] for s in ("forget01", "retain99", "holdout01"))
    if len(f) != 40 or len(r) != 3960 or len(h) != 40:
        raise ValueError("unexpected source split sizes")
    fk, hk = _question_keys(f, "forget"), _question_keys(h, "sealed holdout")
    rk = {normalize_answer(row["question"]) for row in r}
    if fk & rk or fk & hk or rk & hk:
        raise ValueError("overlap between official forget/retain/holdout questions")
    ri = deterministic_sample_indices(len(r), 120, profile["seed"])
    selected_r = [r[i] for i in ri]
    _question_keys(selected_r, "selected retain")
    training_keys = fk | _question_keys(selected_r, "selected retain")
    # Rank the remaining pool independently; never silently repair a selected duplicate.
    remaining = sorted((i for i in range(len(r)) if normalize_answer(r[i]["question"]) not in training_keys),
                       key=lambda i: hashlib.sha256(f"2026:retain_unseen:{i}".encode()).digest())
    hi = sorted(remaining[:40])
    unseen_keys = _question_keys([r[i] for i in hi], "unseen retain")
    if unseen_keys & training_keys or len(hi) != 40:
        raise ValueError("invalid unseen retain selection")
    sets = {
        "forget_direct": [_case("forget01", i, row) for i, row in enumerate(f)],
        "retain_direct": [_case("retain99", i, r[i]) for i in ri],
        "retain_unseen_records": [_case("retain99", i, r[i]) for i in hi],
        "holdout01_sealed": [_case("holdout01", i, row) for i, row in enumerate(h)],
    }
    # Reject overlap with every direct diagnostic too, not just the training set.
    forbidden = training_keys | unseen_keys | hk
    sets["forget_paraphrase"] = aligned_paraphrases(f, range(40), records["forget01_perturbed"],
        "forget01_perturbed", training_questions=forbidden, require_all=True)
    sets["retain_paraphrase"] = aligned_paraphrases(r, ri, records["retain_perturbed"],
        "retain_perturbed", training_questions=forbidden, require_all=False)
    fp = {normalize_answer(records[c["source_split"]][c["record_index"]][c["question_field"]])
          for c in sets["forget_paraphrase"]}
    rp = {normalize_answer(records[c["source_split"]][c["record_index"]][c["question_field"]])
          for c in sets["retain_paraphrase"]}
    if fp & rp:
        raise ValueError("forget and retain paraphrase overlap")
    utility_keys = _question_keys(probes["cases"], "utility probes")
    all_para_keys = {normalize_answer(records[c["source_split"]][c["record_index"]][c["question_field"]])
                     for name in ("forget_paraphrase", "retain_paraphrase") for c in sets[name]}
    if utility_keys & (forbidden | all_para_keys):
        raise ValueError("utility diagnostic question overlap")
    if len(probes["cases"]) != 12 or len({c["id"] for c in probes["cases"]}) != 12:
        raise ValueError("utility probes require 12 unique IDs")
    validate_development(development, records, utility_keys)
    all_ids = [c["case_id"] for name in ("forget_direct", "retain_direct") for c in sets[name]]
    retain_ids = [c["case_id"] for c in sets["retain_direct"]]
    rng = random.Random(profile["seed"])
    pairs = [{"forget": c["case_id"], "retain": retain_ids[rng.randrange(len(retain_ids))]}
             for c in sets["forget_direct"]]
    return {
        "schema_version": "1.0", "experiment_id": profile["experiment_id"],
        "artifact_type": "small_model_preparation_hashes_only_not_training_authorization",
        "profile_sha256": digest(profile), "utility_probes_sha256": digest(probes),
        "development_sha256": digest(development),
        "development_counts": {"forget_facts": 8, "retain_facts": 24, "heldout_wording": 32},
        "dataset": profile["dataset"], "seed": profile["seed"], "sets": sets,
        "training_sets": ["forget_direct", "retain_direct"],
        "acquisition_schedule_sha256": digest([acquisition_order(all_ids, e) for e in range(20)]),
        "retain_reference_schedule_sha256": digest([acquisition_order(retain_ids, e) for e in range(20)]),
        "unlearning_and_retain_control_pairs": pairs,
        "full_retain_pool_normalized_duplicates": len(r) - len(rk),
        "overlap_policy": "direct train/eval reuse intentional; all other wording disjoint; paraphrases test the same facts",
    }


def encode_plain_qa(tokenizer: Any, question: str, answer: str, *,
                    max_sequence_length: int = 512) -> EncodedCausalExample:
    """Answer-only causal labels with a verified BPE prefix and explicit EOS."""
    if any(not isinstance(s, str) or not s.strip() for s in (question, answer)):
        raise ValueError("question and answer must be nonempty strings")
    if type(max_sequence_length) is not int or max_sequence_length < 2:
        raise ValueError("max_sequence_length must be an integer >= 2")
    if tokenizer.padding_side != "right" or tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("explicit right padding with the existing EOS ID is required")
    eos = tokenizer.eos_token_id
    vocabulary_size = len(tokenizer)
    if type(eos) is not int or not 0 <= eos < vocabulary_size:
        raise ValueError("invalid EOS ID")
    prompt = f"Question: {question}\nAnswer:\n"
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if any(type(t) is not int or not 0 <= t < vocabulary_size for t in [*prefix, *full]):
        raise ValueError("invalid token IDs")
    if not prefix or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        raise ValueError("uncertain prompt/answer token boundary")
    special_ids = set(tokenizer.all_special_ids)
    if any(t in special_ids for t in full):
        raise ValueError("embedded special token in QA text")
    full = [*full, eos]
    if len(full) > max_sequence_length:
        raise ValueError("QA exceeds maximum sequence length; truncation forbidden")
    return EncodedCausalExample(tuple(full), tuple([-100] * len(prefix) + full[len(prefix):]), len(prefix))


def verify_tokenizer(profile: Mapping[str, Any], metadata_dir: Path,
                     manifest: Mapping[str, Any], records: Mapping[str, Any],
                     probes: Mapping[str, Any], development: Mapping[str, Any]) -> dict[str, Any]:
    """Load local tokenizer/config only, audit all manifest encodings, no inference."""
    validate_profile(profile)
    for name, expected in profile["model"]["metadata_sha256"].items():
        if Path(name).name != name or not (metadata_dir / name).is_file():
            raise FileNotFoundError("required local model/tokenizer metadata missing")
        if sha256_file(metadata_dir / name) != expected:
            raise ValueError(f"metadata checksum mismatch: {name}")
    import torch
    import transformers
    import tokenizers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(metadata_dir, local_files_only=True, trust_remote_code=False)
    for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "tie_word_embeddings",
                "vocab_size", "max_position_embeddings", "eos_token_id"):
        if getattr(config, key) != profile["model"][key]:
            raise ValueError(f"model config mismatch: {key}")
    # Class lookup only: neither from_config nor from_pretrained constructs a model.
    model_class = AutoModelForCausalLM._model_mapping[type(config)]
    if model_class.__name__ != "LlamaForCausalLM" or not model_class.supports_gradient_checkpointing:
        raise ValueError("unsupported model/checkpointing class")
    if "gradient_checkpointing_kwargs" not in inspect.signature(model_class.gradient_checkpointing_enable).parameters:
        raise ValueError("checkpoint settings API unavailable")
    tokenizer = AutoTokenizer.from_pretrained(metadata_dir, local_files_only=True,
                                            trust_remote_code=False, use_fast=True)
    if (type(tokenizer).__name__ != "GPT2TokenizerFast" or not tokenizer.is_fast
            or len(tokenizer) != 49152 or tokenizer.eos_token_id != 0 or tokenizer.chat_template):
        raise ValueError("unexpected tokenizer provenance or configuration")
    original_pad_id = tokenizer.pad_token_id
    if original_pad_id is not None:
        raise ValueError("publisher pad-token configuration changed")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lengths = {}
    for name, cases in manifest["sets"].items():
        sizes = []
        for case in cases:
            row = records[case["source_split"]][case["record_index"]]
            encoded = encode_plain_qa(tokenizer, row[case["question_field"]], row[case["answer_field"]])
            sizes.append(len(encoded.input_ids))
        lengths[name] = {"cases": len(sizes), "max_tokens": max(sizes) if sizes else 0}
    sizes = [len(encode_plain_qa(tokenizer, row["question"], row["answer"]).input_ids)
             for row in probes["cases"]]
    lengths["utility_probes"] = {"cases": len(sizes), "max_tokens": max(sizes)}
    for name, field in (("development_direct", "question"), ("development_paraphrase", "paraphrased_question")):
        sizes = [len(encode_plain_qa(tokenizer, row[field], row["answer"]).input_ids)
                 for row in development["facts"]]
        lengths[name] = {"cases": len(sizes), "max_tokens": max(sizes)}
    return {"model_instantiated": False, "weights_loaded": False, "actual_model_device": None,
            "python": platform.python_version(), "machine": platform.machine(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "mps_built": torch.backends.mps.is_built(),
            "mps_visible_in_this_process": torch.backends.mps.is_available(),
            "model_class": model_class.__name__, "checkpoint_settings_API": True,
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_vocab_size": len(tokenizer),
            "original_pad_token_id": original_pad_id, "explicit_pad_token_id": tokenizer.pad_token_id,
            "encoding_validation": lengths}


def write_new_manifest(path: Path, value: Mapping[str, Any]) -> None:
    """Only the separately named repository manifest may be created, never replaced."""
    if path.is_symlink() or path.resolve() != MANIFEST or path.parent.is_symlink():
        raise ValueError("manifest destination must be the dedicated small-model path")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_physical_mps_readiness() -> dict[str, Any]:
    """Optional no-model check; the inherited Llama SGD guard remains binding."""
    from types import SimpleNamespace
    import torch
    from src.graddiff_real_train import guarded_memory_snapshot, resolve_training_device

    device = resolve_training_device("mps", torch_module=torch)
    try:
        snapshot = guarded_memory_snapshot(torch, device, SimpleNamespace(optimizer="sgd"), before_load=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}; inherited Llama SGD admission policy remains binding; "
            "SmolLM2 fit is unmeasured. See python -m src.small_model_memory for an offline assessment"
        ) from exc
    return {"status": "existing_SGD_admission_guard_passed_not_training_clearance",
            "guard_origin": "Llama SGD policy; no calibrated SmolLM2 threshold",
            "actual_telemetry_device": str(device), "model_loaded": False, "memory": snapshot}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare-manifest"), nargs="?", default="validate")
    parser.add_argument("--verify-tokenizer", action="store_true")
    parser.add_argument("--check-mps-readiness", action="store_true",
                        help="optional physical macOS telemetry only; no model load")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / ".cache/smollm2_metadata" / REVISION)
    args = parser.parse_args(argv)
    try:
        profile = json.loads(PROFILE.read_text())
        validate_profile(profile)
        records = load_sources(profile, ROOT / ".cache/tofu")
        probes = json.loads((ROOT / profile["evaluation"]["utility_probes"]).read_text())
        development = json.loads((ROOT / profile["development"]["path"]).read_text())
        manifest = build_manifest(profile, records, probes, development)
        if args.command == "validate" and json.loads(MANIFEST.read_text()) != manifest:
            raise ValueError("manifest drift; review source/configuration changes, do not overwrite")
        verification = None
        if args.verify_tokenizer:
            verification = verify_tokenizer(profile, args.metadata_dir, manifest, records, probes, development)
        readiness = check_physical_mps_readiness() if args.check_mps_readiness else None
        if args.command == "prepare-manifest":
            # Complete optional verification before creating any file.
            write_new_manifest(MANIFEST, manifest)
        report = {"status": "offline_preparation_valid", "weight_updates_enabled": False,
                  "manifest_sha256": digest(manifest),
                  "set_counts": {key: len(value) for key, value in manifest["sets"].items()},
                  "full_retain_pool_normalized_duplicates": manifest["full_retain_pool_normalized_duplicates"]}
        if verification is not None:
            report["local_metadata_and_tokenizer_verification"] = verification
        if readiness is not None:
            report["physical_mps_readiness"] = readiness
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"STOP: offline preparation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
