# Read-only GradDiff integration smoke with the cached TOFU model

**Status:** The integration helpers, fixture tests, and local-tokenizer/data preparation check are complete. The 2.47 GB checkpoint weights have not been loaded by this integration path, and the full-model smoke command below has not been executed. No optimizer, backward pass, weight update, or checkpoint write is part of the command.

## Pinned local assets

The command requires the existing cache entries and fails if they are missing. It is offline-only and does not download anything:

- Model: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, revision `88e31200b97e4c0c04ae0d2f0b591f427046d192`.
- Snapshot: `.cache/huggingface/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots/88e31200b97e4c0c04ae0d2f0b591f427046d192`.
- Weight file: `model.safetensors`, expected size 2,471,645,608 bytes.
- Data: the cached `forget01.json` and `retain99.json` files, verified by `load_pinned_split` against the pinned TOFU revision, SHA-256 values, and row counts. Missing data causes an error; the smoke does not download it.

The actual cached tokenizer was loaded without model weights. Its config is Llama with vocabulary size 128,256 and context length 131,072. `len(tokenizer)` is 128,256 (the tokenizer's base `vocab_size` is 128,000); the tokenizer is fast, uses right padding, and has EOS and pad ID 128,009 (`<|eot_id|>`). The pad ID equals EOS, so padded positions must be excluded with both `attention_mask=0` and `labels=-100`.

The tokenizer's chat template has no `{% generation %}` spans. Asking it for `return_assistant_tokens_mask=True` produces an all-zero mask, so the integration does not use that result. Instead, it tokenizes the system/user prompt with `add_generation_prompt=True`, tokenizes the complete system/user/assistant turn, and requires the prompt IDs to match an exact prefix of the complete IDs. If the boundary does not match, it stops rather than risk supervising prompt text.

## Formatting, masks, batches, and loss

The integration reuses the local evaluator's fixed system prompt, checkpoint chat template, and date string (`10 Apr 2025`). It encodes one complete assistant answer and its final end-of-turn token. Prompt and assistant-header labels are `-100`; answer and end-of-turn labels retain their input token IDs. No truncation is done: an example longer than 512 tokens is rejected so the target cannot be silently cut off.

Batch collation pads on the right. Pad input IDs use the tokenizer's pad ID, padding attention values are zero, and padding labels are `-100`. The actual smoke uses a batch of one for each split, so it normally has no padding; fixture tests use different-length examples to exercise the padding path. The existing causal loss shifts labels by one token, ignores `-100`, upcasts logits to FP32, and averages valid next-token losses. Forget and retain losses are evaluated independently, then combined with `-gamma * forget_nll + alpha * retain_nll`.

The real tokenizer and cached pinned data were used for a tokenization-only check: forget row 0 encoded to 101 tokens with 37 supervised targets; the deterministic retain draw at seed 2026 selected row 487 and encoded to 117 tokens with 57 supervised targets. Both were batch size one and had no padding. This check loaded the tokenizer and two small local data files, but did not load model weights or compute model losses.

The smoke loads MPS weights in FP16 and explicit CPU weights in FP32. It compares each model-reported loss with `causal_lm_nll` on the same logits and labels. It reports host total memory, process high-water RSS, and MPS tensor/driver allocation snapshots before model load, after model load, and after the two forwards. These are memory observations for that run; the MPS driver allocation can include cached allocator memory.

## Run in a normal Mac Terminal

Run from the repository root, outside the restricted Codex sandbox. The default command explicitly requires MPS and exits if MPS is unavailable. It does not silently switch to CPU.

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_model_smoke --device mps
```

To deliberately use CPU, replace `--device mps` with `--device cpu`. Automatic selection prefers MPS; CPU fallback is permitted only when explicitly requested with `--device auto --allow-cpu-fallback`. A later MPS load or forward error is reported as an error and does not trigger a CPU retry.

Before running, close memory-heavy applications and check Activity Monitor for adequate free unified memory. The checkpoint file alone is about 2.47 GB; model activations, logits, loader temporaries, and the macOS/MPS allocator add to that. This is a two-forward read-only smoke at sequence length at most 512, not a training-memory measurement. Stop if macOS reports memory pressure. The JSON report goes to stdout; the command writes no output artifact.

The code uses `local_files_only=True`, sets `HF_HUB_OFFLINE=1`, and reads the pinned TOFU files with downloading disabled. It freezes model parameters, runs `eval()` under `torch.inference_mode()`, checks parameter version counters and absent gradients, and creates no optimizer. It contains no `.backward()`, `optimizer.step()`, `save_pretrained()`, or checkpoint-writing path.

## Lightweight validation and limits

`tests/test_graddiff_model_integration.py` uses a tiny fixture tokenizer and causal LM to test chat-boundary failures, answer/EOS masks, causal loss matching, separate forget/retain batches, right padding, device selection, model/tokenizer mismatch errors, non-finite losses, existing gradients, and read-only weight invariants. The full repository suite passed all 56 tests outside the restricted sandbox. A one-element MPS probe also confirmed the memory reporter can read host capacity, process peak RSS, MPS tensor allocation, and MPS driver allocation; it did not load the checkpoint.

The real tokenizer/config compatibility and token preparation have been checked locally. A physical-device run is still required to validate model loading, tied embedding compatibility, model-reported loss parity, actual MPS memory snapshots, and runtime. The exact OpenUnlearning Trainer/collator and its training optimizer are not executed by this read-only path. No real-model unlearning or weight update has occurred.
