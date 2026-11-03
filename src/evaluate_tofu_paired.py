"""Prepare paired TOFU forget/retain manifests and run lexical QA diagnostics.

This module never downloads model weights at import time. Dataset files are
verified against pinned official TOFU revisions; inference happens only when
the ``paired`` command is explicitly run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import ssl
import statistics
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluate_tofu_baseline import (
    CHAT_DATE,
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_CACHE_DIR,
    SAMPLE_SEED,
    SYSTEM_PROMPT,
    answer_metrics,
    deterministic_sample_indices,
    generate_response,
    normalize_answer,
    select_device,
)
from src.inspect_tofu import (
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_DATA_PATH,
    inspect_records,
    load_jsonl,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETAIN_DATA_PATH = PROJECT_ROOT / ".cache" / "tofu" / "retain99.json"
DEFAULT_PARAPHRASE_DATA_PATH = (
    PROJECT_ROOT / ".cache" / "tofu" / "forget01_perturbed.json"
)
DEFAULT_HOLDOUT_DATA_PATH = PROJECT_ROOT / ".cache" / "tofu" / "holdout01.json"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "experiments" / "tofu_forget01_retain99_manifest.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "tofu_forget01_retain99_baseline.json"
)
MANIFEST_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "2.0"
SCORING_VERSION = "lightweight_lexical_diagnostics_v1"
DEFAULT_RETAIN_SAMPLE_SIZE = 40

# File hashes identify these exact JSONL files at the shared official revision.
# They are intentionally separate from the model and are small dataset files.
DATASET_FILE_SPECS: dict[str, dict[str, Any]] = {
    "forget01": {
        "filename": "forget01.json",
        "sha256": "d0047332bd27a489c552659831f4ffff080e2b9b6e063c7c951ac4de9d494a63",
        "record_count": 40,
    },
    "retain99": {
        "filename": "retain99.json",
        "sha256": "e699ae5fe685996cd7d0a60cbcbeae717e989a2fd0caf00b4d545ba39ccefc61",
        "record_count": 3960,
    },
    "forget01_perturbed": {
        "filename": "forget01_perturbed.json",
        "sha256": "d8d8a80bd25e853b52fad0e3301d86fa4db6bc2b665cd79ea937346186a0944a",
        "record_count": 40,
    },
    "holdout01": {
        "filename": "holdout01.json",
        "sha256": "4de5388d2e5927eab81f3d9fb491439b7c63359002452cfd3177a6c0977c08b5",
        "record_count": 40,
    },
}

TOKENIZER_CONFIG = {
    "repo_id": MODEL_ID,
    "revision": MODEL_REVISION,
    "use_fast": True,
    "chat_template": "checkpoint tokenizer_config.json via apply_chat_template",
    "add_generation_prompt": True,
    "system_prompt": SYSTEM_PROMPT,
    "date_string": CHAT_DATE,
}

GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "use_cache": True,
    "pad_token_id": "tokenizer.eos_token_id",
    "requested_device": "auto",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest without relying on JSON whitespace or key order."""

    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def _text_key(text: str) -> str:
    """Normalize text for conservative duplicate-question checks."""

    return normalize_answer(text)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_sample_indices(
    record_count: int, sample_size: int, seed: int
) -> list[int]:
    """Select stable unique rows by SHA-256 ranking of ``seed:index``."""

    if record_count < 1:
        raise ValueError("The dataset contains no records.")
    if sample_size < 1 or sample_size > record_count:
        raise ValueError(
            f"sample_size must be between 1 and {record_count}, got {sample_size}."
        )
    ranked = sorted(
        range(record_count),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest(),
    )
    return sorted(ranked[:sample_size])


def assert_disjoint_splits(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    left_name: str = "left split",
    right_name: str = "right split",
) -> None:
    """Reject question overlap between separately pinned TOFU splits."""

    left_questions = {
        _text_key(record["question"]) for record in left_records if record.get("question")
    }
    right_questions = {
        _text_key(record["question"]) for record in right_records if record.get("question")
    }
    overlap_count = len(left_questions & right_questions)
    if overlap_count:
        raise ValueError(
            f"{left_name} and {right_name} overlap on {overlap_count} normalized question(s)."
        )


def download_pinned_split(split: str, destination: Path) -> Path:
    """Fetch one small official TOFU data file and verify its pinned SHA-256."""

    if split not in DATASET_FILE_SPECS:
        raise ValueError(f"Unsupported TOFU split {split!r}.")
    spec = DATASET_FILE_SPECS[split]
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{spec['filename']}?download=true"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "UnlearnGuard-paired-TOFU-evaluation/1.0"}
    )
    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    try:
        response = urllib.request.urlopen(request, timeout=30, context=context)
        content = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to fetch pinned TOFU {split} data: {exc}") from exc
    observed = hashlib.sha256(content).hexdigest()
    if observed != spec["sha256"]:
        raise ValueError(
            f"Downloaded {split} file has SHA-256 {observed}; expected {spec['sha256']}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def load_pinned_split(
    split: str, path: Path, *, download_if_missing: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and verify a local split's revision, checksum, schema, and row count."""

    if split not in DATASET_FILE_SPECS:
        raise ValueError(f"Unsupported TOFU split {split!r}.")
    if not path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Pinned TOFU {split} data not found at {path}; provide it or use "
                "--download-missing-data."
            )
        download_pinned_split(split, path)
    expected = DATASET_FILE_SPECS[split]
    observed_sha = sha256_file(path)
    if observed_sha != expected["sha256"]:
        raise ValueError(
            f"{split} file does not match TOFU revision {DATASET_REVISION}: "
            f"expected SHA-256 {expected['sha256']}, got {observed_sha}."
        )
    records = load_jsonl(path)
    inspect_records(records)
    if len(records) != expected["record_count"]:
        raise ValueError(
            f"{split} at revision {DATASET_REVISION} has {len(records)} records; "
            f"expected {expected['record_count']}."
        )
    if split == "forget01_perturbed":
        required = ("paraphrased_question", "paraphrased_answer", "perturbed_answer")
        for index, record in enumerate(records):
            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(
                    f"forget01_perturbed record {index} is missing: {', '.join(missing)}."
                )
            if not isinstance(record["paraphrased_question"], str) or not record[
                "paraphrased_question"
            ].strip():
                raise ValueError(f"forget01_perturbed record {index} has no paraphrased question.")
            if not isinstance(record["paraphrased_answer"], str) or not record[
                "paraphrased_answer"
            ].strip():
                raise ValueError(f"forget01_perturbed record {index} has no paraphrased answer.")
            if not isinstance(record["perturbed_answer"], list) or not record[
                "perturbed_answer"
            ]:
                raise ValueError(f"forget01_perturbed record {index} has no wrong-answer list.")
    source = {
        "repo_id": DATASET_ID,
        "split": split,
        "revision": DATASET_REVISION,
        "filename": expected["filename"],
        "sha256": observed_sha,
        "record_count": len(records),
    }
    return records, source


def _make_case(
    *,
    case_id: str,
    record_index: int,
    question: str,
    target_answer: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "record_index": record_index,
        "question_sha256": _text_sha256(question),
        "target_answer_sha256": _text_sha256(target_answer),
    }


def _paired_perturbed_indices(
    forget_records: Sequence[Mapping[str, Any]],
    perturbed_records: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Validate official perturbed rows align to forget rows by original QA."""

    if len(forget_records) != len(perturbed_records):
        raise ValueError(
            "forget01 and forget01_perturbed must have the same record count."
        )
    original_by_question: dict[str, int] = {}
    for index, record in enumerate(forget_records):
        key = _text_key(record["question"])
        if key in original_by_question:
            raise ValueError("forget01 contains duplicate normalized questions.")
        original_by_question[key] = index
    mapped: dict[int, int] = {}
    seen: set[int] = set()
    for perturbed_index, record in enumerate(perturbed_records):
        question_key = _text_key(record.get("question", ""))
        if question_key not in original_by_question:
            raise ValueError(
                f"forget01_perturbed record {perturbed_index} has no matching forget01 question."
            )
        base_index = original_by_question[question_key]
        if _text_key(record.get("answer", "")) != _text_key(
            forget_records[base_index]["answer"]
        ):
            raise ValueError(
                f"forget01_perturbed record {perturbed_index} has a mismatched original answer."
            )
        if base_index in seen:
            raise ValueError("forget01_perturbed contains duplicate paired questions.")
        seen.add(base_index)
        mapped[base_index] = perturbed_index
    if len(mapped) != len(forget_records):
        raise ValueError("forget01_perturbed does not cover every forget01 question exactly once.")
    return [mapped[index] for index in range(len(forget_records))]


def build_paired_manifest(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    source_by_split: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    retain_sample_size: int = DEFAULT_RETAIN_SAMPLE_SIZE,
    requested_device: str = "auto",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Build a deterministic case manifest for paired forget/retain runs."""

    required_splits = {"forget01", "retain99"}
    missing = required_splits - set(records_by_split)
    if missing:
        raise ValueError(f"Manifest requires split(s): {', '.join(sorted(missing))}.")
    if set(records_by_split) != set(source_by_split):
        raise ValueError("Every loaded split must have exactly one provenance entry.")
    forget_records = records_by_split["forget01"]
    retain_records = records_by_split["retain99"]
    if len(forget_records) != 40:
        raise ValueError(f"forget01 manifest requires all 40 records, got {len(forget_records)}.")
    if retain_sample_size != 40:
        raise ValueError("The paired retain99 sample size is fixed at 40 questions.")
    assert_disjoint_splits(
        forget_records,
        retain_records,
        left_name="forget01",
        right_name="retain99",
    )

    for optional_split in ("holdout01",):
        if optional_split in records_by_split:
            assert_disjoint_splits(
                forget_records,
                records_by_split[optional_split],
                left_name="forget01",
                right_name=optional_split,
            )
            assert_disjoint_splits(
                retain_records,
                records_by_split[optional_split],
                left_name="retain99",
                right_name=optional_split,
            )

    evaluation_sets: dict[str, Any] = {}
    forget_cases = [
        _make_case(
            case_id=f"forget01:{index:04d}",
            record_index=index,
            question=record["question"],
            target_answer=record["answer"],
        )
        for index, record in enumerate(forget_records)
    ]
    evaluation_sets["forget01"] = {
        "role": "forget",
        "source_split": "forget01",
        "question_field": "question",
        "target_field": "answer",
        "sampling": "all records in pinned split order",
        "cases": forget_cases,
    }

    retain_indices = deterministic_sample_indices(
        len(retain_records), retain_sample_size, seed
    )
    retain_cases = [
        _make_case(
            case_id=f"retain99:{index:04d}",
            record_index=index,
            question=retain_records[index]["question"],
            target_answer=retain_records[index]["answer"],
        )
        for index in retain_indices
    ]
    evaluation_sets["retain99"] = {
        "role": "retain",
        "source_split": "retain99",
        "question_field": "question",
        "target_field": "answer",
        "sampling": {
            "algorithm": "SHA-256 ranking of seed:index",
            "seed": seed,
            "sample_size": retain_sample_size,
            "record_indices": retain_indices,
        },
        "cases": retain_cases,
    }

    if "forget01_perturbed" in records_by_split:
        perturbed_records = records_by_split["forget01_perturbed"]
        perturbed_indices = _paired_perturbed_indices(forget_records, perturbed_records)
        paraphrase_cases = []
        for base_index, perturbed_index in enumerate(perturbed_indices):
            record = perturbed_records[perturbed_index]
            paraphrase_cases.append(
                _make_case(
                    case_id=f"forget01_paraphrase:{base_index:04d}",
                    record_index=perturbed_index,
                    question=record["paraphrased_question"],
                    target_answer=record["paraphrased_answer"],
                )
            )
        evaluation_sets["forget01_paraphrase"] = {
            "role": "optional_paraphrase_diagnostic",
            "source_split": "forget01_perturbed",
            "question_field": "paraphrased_question",
            "target_field": "paraphrased_answer",
            "sampling": "all rows paired to forget01 by original question and answer",
            "cases": paraphrase_cases,
        }

    if "holdout01" in records_by_split:
        holdout_cases = [
            _make_case(
                case_id=f"holdout01:{index:04d}",
                record_index=index,
                question=record["question"],
                target_answer=record["answer"],
            )
            for index, record in enumerate(records_by_split["holdout01"])
        ]
        evaluation_sets["holdout01"] = {
            "role": "optional_holdout_diagnostic",
            "source_split": "holdout01",
            "question_field": "question",
            "target_field": "answer",
            "sampling": "all records in pinned split order",
            "cases": holdout_cases,
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "paired_tofu_forget_retain_manifest",
        "dataset": {"repo_id": DATASET_ID, "revision": DATASET_REVISION},
        "seed": seed,
        "paired_sets": ["forget01", "retain99"],
        "datasets": {key: dict(value) for key, value in source_by_split.items()},
        "evaluation_sets": evaluation_sets,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "tokenizer_configuration": dict(TOKENIZER_CONFIG),
        "prompt_configuration": {
            "template": "checkpoint tokenizer apply_chat_template",
            "system_prompt": SYSTEM_PROMPT,
            "date_string": CHAT_DATE,
            "add_generation_prompt": True,
            "question_only_as_user_content": True,
        },
        "generation_parameters": {
            **GENERATION_CONFIG,
            "max_new_tokens": max_new_tokens,
            "requested_device": requested_device,
            "seed": seed,
        },
        "scoring": {
            "version": SCORING_VERSION,
            "classification": "lightweight generated-answer lexical diagnostics",
            "official_tofu_forget_quality": False,
            "official_tofu_model_utility": False,
        },
    }


def validate_manifest_against_data(
    manifest: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    source_by_split: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject case, source, or configuration drift from a prepared manifest."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported paired manifest schema version.")
    if manifest.get("artifact_type") != "paired_tofu_forget_retain_manifest":
        raise ValueError("This is not a paired TOFU forget/retain manifest.")
    expected = build_paired_manifest(
        records_by_split,
        source_by_split,
        seed=int(manifest["seed"]),
        requested_device=manifest["generation_parameters"]["requested_device"],
        max_new_tokens=int(manifest["generation_parameters"]["max_new_tokens"]),
    )
    if expected != manifest:
        raise ValueError(
            "The manifest does not match the verified TOFU source rows or current "
            "prompt/scoring configuration. Prepare a new manifest instead of silently "
            "changing paired cases."
        )


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read paired manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Paired manifest must be a JSON object.")
    return value


def aggregate_diagnostic_metrics(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Summarize the lightweight exact-match and lexical overlap diagnostics."""

    if not examples:
        raise ValueError("Cannot aggregate metrics for an empty evaluation set.")
    f1_values = [float(item["metrics"]["normalized_token_overlap_f1"]) for item in examples]
    return {
        "case_count": len(examples),
        "strict_exact_match_rate": sum(
            float(bool(item["metrics"]["strict_exact_match"])) for item in examples
        )
        / len(examples),
        "normalized_exact_match_rate": sum(
            float(bool(item["metrics"]["normalized_exact_match"])) for item in examples
        )
        / len(examples),
        "normalized_target_substring_match_rate": sum(
            float(bool(item["metrics"]["normalized_target_substring_match"]))
            for item in examples
        )
        / len(examples),
        "normalized_token_overlap_precision_mean": statistics.fmean(
            float(item["metrics"]["normalized_token_overlap_precision"])
            for item in examples
        ),
        "normalized_token_overlap_recall_mean": statistics.fmean(
            float(item["metrics"]["normalized_token_overlap_recall"])
            for item in examples
        ),
        "normalized_token_overlap_f1_mean": statistics.fmean(f1_values),
        "normalized_token_overlap_f1_population_stddev": statistics.pstdev(f1_values),
        "normalized_token_overlap_f1_min": min(f1_values),
        "normalized_token_overlap_f1_median": statistics.median(f1_values),
        "normalized_token_overlap_f1_max": max(f1_values),
    }


def build_paired_result_document(
    manifest: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    responses_by_set: Mapping[str, Sequence[str]],
    *,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    actual_device: str,
    model_dtype: str,
    torch_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    """Assemble paired per-case outputs and separately labeled diagnostics."""

    missing_paired_sets = set(manifest["paired_sets"]) - set(responses_by_set)
    if missing_paired_sets:
        raise ValueError(
            "Paired result is missing required set(s): "
            + ", ".join(sorted(missing_paired_sets))
        )
    evaluation_sets: dict[str, Any] = {}
    for set_name, manifest_set in manifest["evaluation_sets"].items():
        if set_name not in responses_by_set:
            continue
        cases = manifest_set["cases"]
        responses = responses_by_set[set_name]
        if len(cases) != len(responses):
            raise ValueError(f"Response count does not match manifest set {set_name}.")
        source_split = manifest_set["source_split"]
        source_records = records_by_split[source_split]
        examples = []
        for case, response in zip(cases, responses):
            record = source_records[case["record_index"]]
            question = record[manifest_set["question_field"]]
            target = record[manifest_set["target_field"]]
            if _text_sha256(question) != case["question_sha256"]:
                raise ValueError(f"Question drift for manifest case {case['case_id']}.")
            if _text_sha256(target) != case["target_answer_sha256"]:
                raise ValueError(f"Target drift for manifest case {case['case_id']}.")
            examples.append(
                {
                    "case_id": case["case_id"],
                    "source_split": source_split,
                    "record_index": case["record_index"],
                    "question": question,
                    "target_answer": target,
                    "model_response": response,
                    "metrics": answer_metrics(response, target),
                }
            )
        evaluation_sets[set_name] = {
            "role": manifest_set["role"],
            "case_count": len(examples),
            "diagnostics": aggregate_diagnostic_metrics(examples),
            "examples": examples,
        }

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_type": "paired_tofu_generated_answer_diagnostics",
        "evaluation_scope": (
            "Lightweight generated-response lexical diagnostics; these are not "
            "official TOFU Forget Quality or Model Utility scores."
        ),
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_artifact_type": manifest["artifact_type"],
        "dataset": dict(manifest["dataset"]),
        "datasets": {key: dict(value) for key, value in manifest["datasets"].items()},
        "seed": manifest["seed"],
        "model": {"repo_id": model_id, "revision": model_revision},
        "tokenizer_configuration": {
            **dict(manifest["tokenizer_configuration"]),
            "repo_id": tokenizer_id,
            "revision": tokenizer_revision,
        },
        "prompt_configuration": dict(manifest["prompt_configuration"]),
        "generation_parameters": {
            **dict(manifest["generation_parameters"]),
            "actual_device": actual_device,
            "model_dtype": model_dtype,
        },
        "scoring": {
            **dict(manifest["scoring"]),
            "definitions": {
                "strict_exact_match_rate": "response.strip() == target.strip(); case and punctuation remain significant",
                "normalized_exact_match_rate": "NFKC, case-fold, punctuation-to-space, collapsed token equality",
                "normalized_token_overlap_f1_mean": "mean per-case multiset word-token F1 after lexical normalization",
                "f1_population_stddev": "population standard deviation across per-case token-overlap F1 values",
            },
        },
        "official_metrics": {
            "truth_ratio": "not_computed",
            "forget_quality": "not_computed",
            "model_utility": "not_computed",
        },
        "software": {
            "python": platform.python_version(),
            "os_platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch_version,
            "transformers": transformers_version,
        },
        "evaluation_sets": evaluation_sets,
    }


def _load_manifest_inputs(
    manifest: Mapping[str, Any],
    *,
    data_paths: Mapping[str, Path],
    download_missing_data: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    source_by_split: dict[str, dict[str, Any]] = {}
    for split in manifest["datasets"]:
        path = data_paths[split]
        records, source = load_pinned_split(
            split, path, download_if_missing=download_missing_data
        )
        records_by_split[split] = records
        source_by_split[split] = source
    return records_by_split, source_by_split


def prepare_manifest(args: argparse.Namespace) -> int:
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    source_by_split: dict[str, dict[str, Any]] = {}
    paths = {
        "forget01": args.forget_data_file,
        "retain99": args.retain_data_file,
        "forget01_perturbed": args.paraphrase_data_file,
        "holdout01": args.holdout_data_file,
    }
    for split, path in paths.items():
        if (
            not path.is_file()
            and split not in {"forget01", "retain99"}
            and not args.download_missing_data
        ):
            continue
        records, source = load_pinned_split(
            split, path, download_if_missing=args.download_missing_data
        )
        records_by_split[split] = records
        source_by_split[split] = source
    manifest = build_paired_manifest(
        records_by_split,
        source_by_split,
        seed=args.seed,
        requested_device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Paired manifest saved to {args.output}")
    print(f"Manifest SHA-256: {manifest_sha256(manifest)}")
    print(
        "Prepared cases: "
        + ", ".join(
            f"{name}={len(value['cases'])}"
            for name, value in manifest["evaluation_sets"].items()
        )
    )
    return 0


def _load_model_and_tokenizer(
    device: Any,
    *,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer_load_revision = (
        None if Path(tokenizer_id).is_dir() else tokenizer_revision
    )
    model_load_revision = None if Path(model_id).is_dir() else model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_load_revision,
        cache_dir=str(MODEL_CACHE_DIR),
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_load_revision,
        cache_dir=str(MODEL_CACHE_DIR),
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, str(dtype).removeprefix("torch.")


def evaluate_paired(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the frozen manifest and run requested generated-answer sets."""

    if args.model_id and not args.model_revision:
        raise ValueError("--model-revision is required when overriding --model-id.")
    if args.tokenizer_id and not args.tokenizer_revision:
        raise ValueError("--tokenizer-revision is required when overriding --tokenizer-id.")
    manifest = load_manifest(args.manifest)
    data_paths = {
        "forget01": args.forget_data_file,
        "retain99": args.retain_data_file,
        "forget01_perturbed": args.paraphrase_data_file,
        "holdout01": args.holdout_data_file,
    }
    records_by_split, source_by_split = _load_manifest_inputs(
        manifest,
        data_paths=data_paths,
        download_missing_data=args.download_missing_data,
    )
    validate_manifest_against_data(manifest, records_by_split, source_by_split)

    selected_sets = ["forget01", "retain99"]
    if args.include_paraphrase:
        if "forget01_paraphrase" not in manifest["evaluation_sets"]:
            raise ValueError("The manifest has no official forget01 paraphrase cases.")
        selected_sets.append("forget01_paraphrase")
    if args.include_holdout:
        if "holdout01" not in manifest["evaluation_sets"]:
            raise ValueError("The manifest has no official holdout01 cases.")
        selected_sets.append("holdout01")

    import torch
    import transformers

    requested_device = args.device or manifest["generation_parameters"]["requested_device"]
    device = select_device(torch, requested_device)
    model_id = args.model_id or manifest["model"]["repo_id"]
    model_revision = args.model_revision or manifest["model"]["revision"]
    tokenizer_id = args.tokenizer_id or manifest["tokenizer_configuration"]["repo_id"]
    tokenizer_revision = (
        args.tokenizer_revision or manifest["tokenizer_configuration"]["revision"]
    )
    model, tokenizer, model_dtype = _load_model_and_tokenizer(
        device,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )
    seed = int(manifest["seed"])
    max_new_tokens = int(manifest["generation_parameters"]["max_new_tokens"])
    responses_by_set: dict[str, list[str]] = {}
    for set_name in selected_sets:
        case_set = manifest["evaluation_sets"][set_name]
        source_records = records_by_split[case_set["source_split"]]
        responses_by_set[set_name] = [
            generate_response(
                model,
                tokenizer,
                source_records[case["record_index"]][case_set["question_field"]],
                device=device,
                seed=seed,
                max_new_tokens=max_new_tokens,
            )
            for case in case_set["cases"]
        ]
    return build_paired_result_document(
        manifest,
        records_by_split,
        responses_by_set,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        actual_device=str(device),
        model_dtype=model_dtype,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-manifest", help="verify pinned dataset files and freeze paired cases"
    )
    prepare.add_argument("--forget-data-file", type=Path, default=DEFAULT_DATA_PATH)
    prepare.add_argument("--retain-data-file", type=Path, default=DEFAULT_RETAIN_DATA_PATH)
    prepare.add_argument(
        "--paraphrase-data-file", type=Path, default=DEFAULT_PARAPHRASE_DATA_PATH
    )
    prepare.add_argument("--holdout-data-file", type=Path, default=DEFAULT_HOLDOUT_DATA_PATH)
    prepare.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    prepare.add_argument("--seed", type=int, default=SAMPLE_SEED)
    prepare.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    prepare.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    prepare.add_argument("--download-missing-data", action="store_true")
    prepare.set_defaults(handler=prepare_manifest)

    paired = subparsers.add_parser(
        "paired", help="run the frozen forget01/retain99 paired diagnostics"
    )
    paired.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    paired.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    paired.add_argument("--forget-data-file", type=Path, default=DEFAULT_DATA_PATH)
    paired.add_argument("--retain-data-file", type=Path, default=DEFAULT_RETAIN_DATA_PATH)
    paired.add_argument(
        "--paraphrase-data-file", type=Path, default=DEFAULT_PARAPHRASE_DATA_PATH
    )
    paired.add_argument("--holdout-data-file", type=Path, default=DEFAULT_HOLDOUT_DATA_PATH)
    paired.add_argument("--include-paraphrase", action="store_true")
    paired.add_argument("--include-holdout", action="store_true")
    paired.add_argument("--device", choices=("auto", "mps", "cpu"))
    paired.add_argument("--model-id")
    paired.add_argument("--model-revision")
    paired.add_argument("--tokenizer-id")
    paired.add_argument("--tokenizer-revision")
    paired.add_argument("--download-missing-data", action="store_true")
    paired.set_defaults(handler=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "paired":
        if args.output.resolve() == PROJECT_ROOT.joinpath(
            "results", "tofu_forget01_baseline.json"
        ).resolve():
            parser.error("paired runs may not overwrite the historical five-question result")
        result = evaluate_paired(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Paired diagnostics saved to {args.output}")
        for set_name, value in result["evaluation_sets"].items():
            print(
                f"{set_name}: "
                + json.dumps(value["diagnostics"], sort_keys=True)
            )
        print("These are lightweight lexical diagnostics, not official TOFU scores.")
        return 0
    if args.command == "prepare-manifest":
        if args.max_new_tokens < 1:
            parser.error("--max-new-tokens must be positive")
        return prepare_manifest(args)
    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
