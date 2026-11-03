# Prepared TOFU forget01 baseline evaluation

**Status:** Prepared, not evaluated. No model weights were downloaded, no
responses were generated, and no unlearning was run while preparing this
baseline.

## Model and provenance

The evaluator uses the official OpenUnlearning checkpoint
[`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`](https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full)
at immutable revision
[`88e31200b97e4c0c04ae0d2f0b591f427046d192`](https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full/tree/88e31200b97e4c0c04ae0d2f0b591f427046d192).
The public Hub page identifies it as a 1B-parameter BF16 Safetensors model and
lists the BSD-3-Clause license. The pinned snapshot lists a 2.47 GB
`model.safetensors`, a 17.2 MB `tokenizer.json`, and `tokenizer_config.json`.
The checkpoint's files page is public and does not display an access gate, so
an HF token or approval is not expected for downloading it. We did not attempt
to transfer the weights, in keeping with this preparation task; public metadata
availability is verified, but a complete weight download has not been tested.

OpenUnlearning states that it fine-tuned `full`, `retain90`, `retain95`, and
`retain99` models for each architecture, with `full` serving as the target
base model for unlearning; its TOFU experiment config names this checkpoint as
the starting point. The baseline therefore starts from a model fine-tuned on
TOFU rather than the general Llama instruct checkpoint. This is the intended
provenance of the released artifact; this repository has not independently
retrained or audited its training history. The OpenUnlearning
Llama configuration also records the assistant prompt and date string used
below. The evaluator loads tokenizer files bundled with this checkpoint and
does not request tokenizer files from `meta-llama/Llama-3.2-1B-Instruct`.

## Dataset and forget/retain separation

The sample comes only from the pinned official `locuslab/TOFU` `forget01`
file: revision
[`324592d84ae4f482ac7249b9285c2ecdb53e3a68`](https://huggingface.co/datasets/locuslab/TOFU/tree/324592d84ae4f482ac7249b9285c2ecdb53e3a68),
file SHA-256
`d0047332bd27a489c552659831f4ffff080e2b9b6e063c7c951ac4de9d494a63`. The
pinned local JSONL has 40 question-answer records. TOFU defines forget/retain
scenarios by author profiles: `forget01` is the 1% target set (two of 200
profiles in the benchmark task), while `retain99` contains the corresponding
remaining profiles. Their separate files are exposed as separate dataset
configurations by the official Hub repository. OpenUnlearning documents the
override `forget_split=forget01 retain_split=retain99`; its experiment defaults
to `forget10`/`retain90`, so those values must be explicitly overridden in a
future 01/99 unlearning run.

This baseline intentionally evaluates `forget01` questions only. It does not
train, fine-tune, or use `retain99`; the pair is the proper target/retained-data
configuration for a later unlearning experiment. The official dataset card
describes these as separate author-level sets, so the forget profiles are not
part of the retain split used by the corresponding protocol.

## Fixed baseline configuration

| Setting | Value |
|---|---|
| Model and tokenizer | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, both at `88e31200b97e4c0c04ae0d2f0b591f427046d192` |
| Dataset | `locuslab/TOFU`, `forget01`, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`, pinned SHA-256 above |
| Sample | 5 distinct rows, sorted into dataset order after selection |
| Sampling seed | `2026`; SHA-256 ranking of `"seed:index"` gives the same row indices across supported Python versions |
| Prompt | Checkpoint tokenizer's `apply_chat_template`; system message `You are a helpful assistant.` and a user message containing the TOFU question; fixed `date_string="10 Apr 2025"`; `add_generation_prompt=True` |
| Generation | Greedy decoding (`do_sample=False`, `num_beams=1`), `max_new_tokens=128`, model cache enabled |
| Device | `auto` selects MPS only if built and available in the current process; otherwise CPU. `--device mps` fails clearly if unavailable. |
| Inference dtype | FP16 on MPS; FP32 on CPU. Both cast the repository's BF16 checkpoint at load time. |
| Output | Ignored file `results/tofu_forget01_baseline.json`, including model/data revisions, seed, selected row indices, software versions, generation configuration, each response, and metric definitions |

The model loader is inside the explicit evaluation path; importing the module,
sampling, metric functions, and automated tests do not import Transformers or
fetch checkpoint files. The evaluator only reads the already-cached dataset,
checks its SHA-256, and raises an error if the file is missing or differs. It
does not automatically download either the dataset or model. The model cache
is under ignored `.cache/huggingface/`; checkpoint files, `results/`, and
virtual environments remain excluded by `.gitignore`.

## Metrics and interpretation

The output reports:

- **Strict exact-match rate:** response and target are byte-for-byte equal as
  Unicode strings after trimming outer whitespace. Case and punctuation still
  matter.
- **Normalized exact-match rate:** equality after Unicode NFKC normalization,
  case-folding, replacing punctuation/symbols with spaces, and collapsing
  whitespace.
- **Normalized target-substring-match rate:** whether the complete normalized
  target phrase occurs contiguously in the normalized response.
- **Normalized token-overlap precision, recall, and F1:** multiset overlap of
  whitespace tokens after that same normalization, averaged per question.

The last three are explicitly lexical text-matching signals, not semantic
correctness metrics. A paraphrase or valid alternate answer may score zero;
common words can inflate overlap; a target phrase can appear in an otherwise
incorrect answer; and a refusal or an answer with extra unrelated text changes
these scores. The sample is small, uses one fixed prompt per row, and omits
TOFU's truth-ratio, paraphrase, and retain utility evaluation. It therefore
cannot establish memorization, forgetting, privacy, or unlearning effectiveness.

## Run the first real baseline

The current `.venv-ml` uses Python 3.11.18, PyTorch 2.9.1, and Transformers
4.57.1, pinned directly in `requirements-ml.txt` and fully resolved in
`requirements-ml.lock`. In a normal macOS Terminal process outside the
restricted Codex sandbox, run this command from the repository root:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  ./.venv-ml/bin/python -m src.evaluate_tofu_baseline \
  --data-file .cache/tofu/forget01.json \
  --sample-size 5 --seed 2026 --device auto --max-new-tokens 128 \
  --output results/tofu_forget01_baseline.json
```

The first run is expected to transfer approximately 2.5 GB of model files into
`.cache/huggingface/`; it does not need an HF token unless Hub access policy
changes or the local network requires authentication. This command has not
been run in the preparation task. Its output will be the first actual baseline
result; no responses or scores are checked into this repository.

## Tests

Run the offline suite from the repository root:

```sh
./.venv-ml/bin/python -m unittest discover -s tests -v
```

The tests use synthetic records and a mocked tokenizer. They validate stable
sampling, metric calculations, prompt-template arguments, and the output
schema without calling `from_pretrained`, downloading files, or generating
responses.

## Official references

- [OpenUnlearning TOFU experiment config](https://github.com/locuslab/open-unlearning/blob/main/configs/experiment/unlearn/tofu/default.yaml)
- [OpenUnlearning model release description](https://github.com/locuslab/open-unlearning#-available-components)
- [OpenUnlearning Llama 3.2 1B model config](https://github.com/locuslab/open-unlearning/blob/main/configs/model/Llama-3.2-1B-Instruct.yaml)
- [OpenUnlearning TOFU split documentation](https://github.com/locuslab/open-unlearning/blob/main/docs/experiments.md)
- [Pinned TOFU dataset files and card](https://huggingface.co/datasets/locuslab/TOFU/tree/324592d84ae4f482ac7249b9285c2ecdb53e3a68)
- [Direct pinned `forget01.json` source](https://huggingface.co/datasets/locuslab/TOFU/resolve/324592d84ae4f482ac7249b9285c2ecdb53e3a68/forget01.json)
- [TOFU benchmark paper](https://arxiv.org/abs/2401.06121)
