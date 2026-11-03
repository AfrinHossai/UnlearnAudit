"""Run a small, deterministic pre-unlearning baseline on TOFU forget01.

Importing this module does not import PyTorch or Transformers and never loads a
checkpoint. Weights are fetched lazily only when the command is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.inspect_tofu import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DATASET_SPLIT,
    DEFAULT_DATA_PATH,
    inspect_records,
    load_jsonl,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
MODEL_REVISION = "88e31200b97e4c0c04ae0d2f0b591f427046d192"
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "tofu_forget01_baseline.json"
SAMPLE_SEED = 2026
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_MAX_NEW_TOKENS = 128
SYSTEM_PROMPT = "You are a helpful assistant."
CHAT_DATE = "10 Apr 2025"
SCHEMA_VERSION = "1.0"


def deterministic_sample_indices(
    record_count: int, sample_size: int, seed: int
) -> list[int]:
    """Select stable, unique record positions using seeded SHA-256 ranking.

    Ranking digest strings instead of relying on Python's ``random`` makes the
    selected positions stable across supported Python versions.
    """

    if record_count < 1:
        raise ValueError("The dataset contains no records.")
    if sample_size < 1 or sample_size > record_count:
        raise ValueError(
            f"sample_size must be between 1 and {record_count}, got {sample_size}."
        )
    ranked_indices = sorted(
        range(record_count),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest(),
    )
    return sorted(ranked_indices[:sample_size])


def normalize_answer(text: str) -> str:
    """Normalize text for lexical comparisons, not semantic equivalence."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(words.split())


def answer_metrics(response: str, target: str) -> dict[str, bool | float]:
    """Compute strict and explicitly lexical answer-match metrics."""

    response_normalized = normalize_answer(response)
    target_normalized = normalize_answer(target)
    response_tokens = response_normalized.split()
    target_tokens = target_normalized.split()
    response_counts = Counter(response_tokens)
    target_counts = Counter(target_tokens)
    overlap_count = sum((response_counts & target_counts).values())

    precision = overlap_count / len(response_tokens) if response_tokens else 0.0
    recall = overlap_count / len(target_tokens) if target_tokens else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "strict_exact_match": response.strip() == target.strip(),
        "normalized_exact_match": response_normalized == target_normalized,
        "normalized_target_substring_match": bool(target_normalized)
        and target_normalized in response_normalized,
        "normalized_token_overlap_precision": precision,
        "normalized_token_overlap_recall": recall,
        "normalized_token_overlap_f1": f1,
    }


def aggregate_metrics(examples: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Return per-example means for the fixed metric fields."""

    metric_names = (
        "strict_exact_match",
        "normalized_exact_match",
        "normalized_target_substring_match",
        "normalized_token_overlap_precision",
        "normalized_token_overlap_recall",
        "normalized_token_overlap_f1",
    )
    if not examples:
        raise ValueError("Cannot aggregate metrics for an empty sample.")
    return {
        f"{name}_rate" if name.endswith("match") else f"{name}_mean": sum(
            float(example["metrics"][name]) for example in examples
        )
        / len(examples)
        for name in metric_names
    }


def build_chat_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def encode_chat_prompt(tokenizer: Any, question: str) -> Any:
    """Use the checkpoint's own chat template and return a single prompt batch."""

    return tokenizer.apply_chat_template(
        build_chat_messages(question),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        date_string=CHAT_DATE,
    )


def select_device(torch_module: Any, requested: str) -> Any:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""

    mps_built = bool(torch_module.backends.mps.is_built())
    mps_available = bool(torch_module.backends.mps.is_available())
    if requested == "auto":
        requested = "mps" if mps_built and mps_available else "cpu"
    if requested == "mps" and not (mps_built and mps_available):
        raise RuntimeError(
            "MPS was requested but is unavailable in this process "
            f"(built={mps_built}, available={mps_available}). Run from a normal "
            "macOS Terminal process outside the restricted Codex sandbox, or use --device cpu."
        )
    return torch_module.device(requested)


def load_model_and_tokenizer(device: Any) -> tuple[Any, Any, str]:
    """Load the pinned checkpoint only after explicit evaluator execution."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device.type == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE_DIR),
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE_DIR),
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, str(dtype).removeprefix("torch.")


def generate_response(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    device: Any,
    seed: int,
    max_new_tokens: int,
) -> str:
    """Generate one greedy answer while retaining only generated tokens."""

    import torch

    torch.manual_seed(seed)
    prompt = encode_chat_prompt(tokenizer, question)
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask

    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)
    generated_tokens = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def build_result_document(
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    responses: Sequence[str],
    *,
    seed: int,
    max_new_tokens: int,
    device: str,
    model_dtype: str,
    torch_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    """Assemble a versioned JSON result from real or test-supplied responses."""

    if not (len(indices) == len(responses) and len(indices) > 0):
        raise ValueError("indices and responses must have the same nonzero length.")

    examples = []
    for index, response in zip(indices, responses):
        record = records[index]
        examples.append(
            {
                "record_index": index,
                "question": record["question"],
                "target_answer": record["answer"],
                "model_response": response,
                "metrics": answer_metrics(response, record["answer"]),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_type": "pre_unlearning_tofu_forget01_baseline",
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_repo_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "chat_template": "checkpoint tokenizer_config.json via apply_chat_template",
            "system_prompt": SYSTEM_PROMPT,
            "date_string": CHAT_DATE,
        },
        "dataset": {
            "repo_id": DATASET_ID,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "sha256": DATASET_SHA256,
        },
        "sampling": {
            "algorithm": "SHA-256 ranking of seed:index",
            "seed": seed,
            "sample_size": len(indices),
            "record_indices": list(indices),
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "device": device,
            "model_dtype": model_dtype,
        },
        "software": {
            "python": platform.python_version(),
            "os_platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch_version,
            "transformers": transformers_version,
        },
        "metrics_definition": {
            "strict_exact_match_rate": "stripped strings exactly equal; case and punctuation remain significant",
            "normalized_exact_match_rate": "NFKC, casefolded alphanumeric token strings exactly equal",
            "normalized_target_substring_match_rate": "normalized target phrase occurs contiguously in normalized response",
            "normalized_token_overlap_precision_mean": "multiset word-token overlap divided by response token count",
            "normalized_token_overlap_recall_mean": "multiset word-token overlap divided by target token count",
            "normalized_token_overlap_f1_mean": "harmonic mean of lexical token overlap precision and recall",
        },
        "aggregate_metrics": aggregate_metrics(examples),
        "examples": examples,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the local pinned split, lazily load the checkpoint, and evaluate."""

    if not args.data_file.is_file():
        raise FileNotFoundError(
            f"TOFU data file not found: {args.data_file}. Use the already-pinned "
            "local forget01 cache; this evaluator does not download datasets."
        )
    observed_sha256 = sha256_file(args.data_file)
    if observed_sha256 != DATASET_SHA256:
        raise ValueError(
            "TOFU file does not match the pinned forget01 source: "
            f"expected SHA-256 {DATASET_SHA256}, got {observed_sha256}."
        )
    records = load_jsonl(args.data_file)
    inspect_records(records)
    indices = deterministic_sample_indices(len(records), args.sample_size, args.seed)

    import torch
    import transformers

    device = select_device(torch, args.device)
    model, tokenizer, model_dtype = load_model_and_tokenizer(device)
    responses = [
        generate_response(
            model,
            tokenizer,
            records[index]["question"],
            device=device,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        for index in indices
    ]
    return build_result_document(
        records,
        indices,
        responses,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        device=str(device),
        model_dtype=model_dtype,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Local, checksum-verified TOFU forget01 JSONL (never downloaded by this command).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of deterministic questions to evaluate (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu"), default="auto",
        help="auto prefers available MPS and otherwise uses CPU.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Ignored JSON result path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be positive.")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive.")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Baseline result saved to {args.output}")
    print(json.dumps(result["aggregate_metrics"], indent=2, sort_keys=True))
    print("This is a pre-unlearning baseline only; it does not test unlearning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
