"""Read-only TOFU chat encoding and GradDiff loss integration helpers.

These helpers prepare causal-LM batches and compare model-reported NLL with
the repository's loss calculation. They never create an optimizer or update
model weights.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.evaluate_tofu_baseline import CHAT_DATE, build_chat_messages

IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodedCausalExample:
    """One fully rendered chat with only assistant answer tokens supervised."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_target_start: int


def _token_ids(value: Any, *, name: str) -> list[int]:
    """Normalize tokenizer output to one flat sequence of integer IDs."""
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a flat sequence of token IDs")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one chat")
        value = list(value[0])
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        raise ValueError(f"{name} must contain only nonnegative integer token IDs")
    return value


def encode_tofu_qa(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    max_sequence_length: int = 512,
) -> EncodedCausalExample:
    """Render a pinned-style system/user/assistant turn and mask the prompt.

    The assistant generation prefix is tokenized independently and must be an
    exact prefix of the completed conversation. This fail-closed boundary check
    avoids relying on tokenizer chat templates that do not expose generation
    spans. The answer and its final end-of-turn token are supervised.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
        raise ValueError("max_sequence_length must be an integer")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat template")
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("tokenizer must define eos_token_id for the assistant turn")

    prompt_messages = build_chat_messages(question)
    full_messages = [*prompt_messages, {"role": "assistant", "content": answer}]
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            date_string=CHAT_DATE,
        )
        full = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            date_string=CHAT_DATE,
        )
    except Exception as exc:
        raise ValueError(f"tokenizer chat-template rendering failed: {exc}") from exc

    prompt_ids = _token_ids(prompt, name="prompt tokenization")
    input_ids = _token_ids(full, name="completed chat tokenization")
    target_start = len(prompt_ids)
    if target_start < 1 or len(input_ids) <= target_start:
        raise ValueError("completed chat has no assistant target tokens")
    if input_ids[:target_start] != prompt_ids:
        raise ValueError(
            "completed chat does not preserve the generation-prompt token prefix; "
            "refusing to construct an uncertain target mask"
        )
    if input_ids[-1] != int(tokenizer.eos_token_id):
        raise ValueError("completed assistant turn does not end with tokenizer.eos_token_id")
    if len(input_ids) > max_sequence_length:
        raise ValueError(
            f"chat has {len(input_ids)} tokens, exceeding max_sequence_length="
            f"{max_sequence_length}; refusing to truncate the target"
        )

    labels = [IGNORE_INDEX] * target_start + input_ids[target_start:]
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError("chat has no supervised next-token targets after causal shifting")
    return EncodedCausalExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        assistant_target_start=target_start,
    )


def collate_causal_examples(
    examples: Sequence[EncodedCausalExample],
    tokenizer: Any,
    *,
    device: Any | None = None,
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    """Right-pad a small batch; padding is attended off and loss-masked."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if (isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int)
            or pad_token_id < 0):
        raise ValueError("tokenizer must define an integer pad_token_id")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("this integration requires tokenizer.padding_side='right'")
    max_length = max(len(example.input_ids) for example in examples)
    if max_sequence_length is not None and max_length > max_sequence_length:
        raise ValueError(
            f"batch has {max_length} tokens, exceeding max_sequence_length="
            f"{max_sequence_length}"
        )

    import torch

    input_ids = torch.full((len(examples), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    labels = torch.full((len(examples), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(examples):
        size = len(example.input_ids)
        if size < 2 or len(example.labels) != size:
            raise ValueError(f"example {row} has inconsistent causal-LM sequence lengths")
        if not 1 <= example.assistant_target_start < size:
            raise ValueError(f"example {row} has an invalid assistant target boundary")
        if any(label != IGNORE_INDEX for label in example.labels[:example.assistant_target_start]):
            raise ValueError(f"example {row} supervises prompt tokens")
        if not any(label != IGNORE_INDEX for label in example.labels[1:]):
            raise ValueError(f"example {row} has no next-token targets")
        input_ids[row, :size] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row, :size] = 1
        labels[row, :size] = torch.tensor(example.labels, dtype=torch.long)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def prepare_tofu_batch(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    device: Any | None = None,
    max_sequence_length: int = 512,
) -> dict[str, Any]:
    """Encode and right-pad records with answer-only labels."""
    if not records:
        raise ValueError("cannot prepare an empty TOFU batch")
    examples = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        missing = [key for key in ("question", "answer") if key not in record]
        if missing:
            raise ValueError(f"record {index} is missing: {', '.join(missing)}")
        examples.append(
            encode_tofu_qa(
                tokenizer,
                record["question"],
                record["answer"],
                max_sequence_length=max_sequence_length,
            )
        )
    return collate_causal_examples(
        examples,
        tokenizer,
        device=device,
        max_sequence_length=max_sequence_length,
    )


def choose_readonly_forget_retain_pair(
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 2026,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, int]]:
    """Choose forget row zero and one deterministic, uniform retain draw."""
    if not forget_records:
        raise ValueError("forget split is empty")
    if not retain_records:
        raise ValueError("retain split is empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    retain_index = random.Random(seed).randrange(len(retain_records))
    return forget_records[0], retain_records[retain_index], {
        "forget_record_index": 0,
        "retain_record_index": retain_index,
    }


def validate_model_tokenizer(
    config: Any,
    tokenizer: Any,
    *,
    max_sequence_length: int,
    model: Any | None = None,
) -> dict[str, Any]:
    """Fail closed on incompatible model vocab, context, IDs, or chat template."""
    if (isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length < 2):
        raise ValueError("max_sequence_length must be an integer of at least 2")
    config = getattr(model, "config", config)
    model_type = getattr(config, "model_type", None)
    if model_type != "llama":
        raise ValueError(f"expected a Llama config, got model_type={model_type!r}")
    vocab_size = getattr(config, "vocab_size", None)
    tokenizer_size = len(tokenizer)
    if not isinstance(vocab_size, int) or tokenizer_size != vocab_size:
        raise ValueError(
            f"tokenizer length {tokenizer_size} does not match model vocabulary {vocab_size}"
        )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("the pinned integration requires a fast tokenizer")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat template")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must define an integer eos_token_id")
    configured_eos = getattr(config, "eos_token_id", None)
    configured_eos_ids = (
        set(configured_eos)
        if isinstance(configured_eos, (list, tuple, set))
        else {configured_eos}
    )
    if eos_token_id not in configured_eos_ids:
        raise ValueError("tokenizer eos_token_id is not recognized by the model config")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if (isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int)
            or pad_token_id < 0):
        raise ValueError("tokenizer must define an integer pad_token_id")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("this integration requires tokenizer.padding_side='right'")
    if not (0 <= pad_token_id < vocab_size):
        raise ValueError("tokenizer pad_token_id is outside the model vocabulary")
    if not (0 <= eos_token_id < vocab_size):
        raise ValueError("tokenizer eos_token_id is outside the model vocabulary")
    max_positions = getattr(config, "max_position_embeddings", None)
    if not isinstance(max_positions, int) or max_sequence_length > max_positions:
        raise ValueError(
            f"max_sequence_length={max_sequence_length} exceeds model context "
            f"length {max_positions}"
        )

    if model is not None:
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None or input_embeddings.weight.shape[0] != vocab_size:
            raise ValueError("loaded model input embeddings do not match tokenizer vocabulary")
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or output_embeddings.weight.shape[0] != vocab_size:
            raise ValueError("loaded model output embeddings do not match tokenizer vocabulary")
        if getattr(config, "tie_word_embeddings", False):
            if input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr():
                raise ValueError("model input/output embeddings are not tied as configured")

    return {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "tokenizer_length": tokenizer_size,
        "tokenizer_is_fast": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "padding_side": getattr(tokenizer, "padding_side", None),
        "max_sequence_length": max_sequence_length,
        "max_position_embeddings": max_positions,
        "chat_template_present": True,
    }


def resolve_smoke_device(
    requested: str,
    *,
    allow_cpu_fallback: bool = False,
    torch_module: Any | None = None,
) -> Any:
    """Prefer MPS; CPU use must be selected or allowed explicitly."""
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    if torch_module is None:
        import torch as torch_module
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps and mps.is_built() and mps.is_available())
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS requested but unavailable; refusing implicit CPU fallback")
        return torch_module.device("mps")
    if mps_available:
        return torch_module.device("mps")
    if allow_cpu_fallback:
        return torch_module.device("cpu")
    raise RuntimeError(
        "MPS is unavailable in this process; explicitly select --device cpu or pass "
        "--allow-cpu-fallback with --device auto"
    )


def evaluate_readonly_pair(
    model: Any,
    forget_batch: Mapping[str, Any],
    retain_batch: Mapping[str, Any],
    *,
    gamma: float = 1.0,
    alpha: float = 1.0,
    loss_atol: float = 1e-5,
    loss_rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compute two mean NLLs and a GradDiff scalar without any weight update."""
    import torch

    from src.graddiff import causal_lm_nll, graddiff_objective

    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("model has no parameters")
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("model has existing gradients; refusing read-only evaluation")
    model.eval()
    model.requires_grad_(False)
    parameter_versions = tuple(parameter._version for parameter in parameters)
    losses: dict[str, Any] = {}
    split_reports: dict[str, dict[str, float | int]] = {}

    with torch.inference_mode():
        for split, batch in (("forget", forget_batch), ("retain", retain_batch)):
            required = {"input_ids", "attention_mask", "labels"}
            if set(batch) != required:
                raise ValueError(f"{split} batch must contain exactly {sorted(required)}")
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(f"{split} input_ids and attention_mask shapes must match")
            if labels.shape != input_ids.shape:
                raise ValueError(f"{split} labels and input_ids shapes must match")
            if input_ids.shape[0] != 1:
                raise ValueError(f"{split} integration smoke requires batch size one")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
                raise ValueError(f"{split} attention_mask must contain only zero and one")
            if bool((labels[attention_mask == 0] != IGNORE_INDEX).any().item()):
                raise ValueError(f"{split} padded tokens must be ignored by the loss")
            row_mask = attention_mask[0]
            first_padding = int(row_mask.eq(0).nonzero(as_tuple=False)[0].item()) if bool(
                row_mask.eq(0).any().item()
            ) else row_mask.shape[0]
            if bool(row_mask[first_padding:].eq(1).any().item()):
                raise ValueError(f"{split} attention_mask must use right padding")
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            model_loss = getattr(output, "loss", None)
            logits = getattr(output, "logits", None)
            if not isinstance(model_loss, torch.Tensor) or model_loss.ndim != 0:
                raise ValueError(f"{split} model output must include a scalar loss")
            if not isinstance(logits, torch.Tensor):
                raise ValueError(f"{split} model output must include logits")
            local_loss = causal_lm_nll(logits, labels)
            if not torch.allclose(
                model_loss.float(), local_loss.float(), atol=loss_atol, rtol=loss_rtol
            ):
                difference = float((model_loss.float() - local_loss.float()).abs().item())
                raise RuntimeError(
                    f"{split} model loss differs from local causal NLL by {difference}"
                )
            target_count = int(labels[:, 1:].ne(IGNORE_INDEX).sum().item())
            if target_count < 1:
                raise ValueError(f"{split} batch contains no next-token targets")
            losses[split] = local_loss
            split_reports[split] = {
                "model_loss": float(model_loss.float().item()),
                "local_causal_nll": float(local_loss.float().item()),
                "absolute_loss_difference": float(
                    (model_loss.float() - local_loss.float()).abs().item()
                ),
                "input_tokens": int(attention_mask.sum().item()),
                "padding_tokens": int(attention_mask.eq(0).sum().item()),
                "target_tokens_after_shift": target_count,
            }

        objective = graddiff_objective(losses["forget"], losses["retain"], gamma, alpha)

    if tuple(parameter._version for parameter in parameters) != parameter_versions:
        raise RuntimeError("model parameter version changed during read-only forward")
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("read-only forward unexpectedly created parameter gradients")
    if any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("read-only model parameters were not frozen")

    return {
        "forget": split_reports["forget"],
        "retain": split_reports["retain"],
        "graddiff_objective": float(objective.float().item()),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "loss_reduction": "mean over non-ignored shifted targets, separately per split",
        "weight_values_unchanged": True,
        "parameter_gradients_absent": True,
        "backpropagation_performed": False,
        "optimizer_created": False,
        "checkpoint_write_performed": False,
    }
