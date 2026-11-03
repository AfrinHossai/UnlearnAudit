# Paired TOFU forget/retain evaluation

## Status and scope

The paired evaluation is **prepared but not run**. This work verified official
TOFU data files and created a frozen case manifest; it did not load model
weights, run model inference, fine-tune, unlearn, or compute official TOFU
scores. The historical five-question output at
`results/tofu_forget01_baseline.json` was left unchanged.

The new evaluator is
[`src/evaluate_tofu_paired.py`](../src/evaluate_tofu_paired.py). Its checked-in
manifest is
[`experiments/tofu_forget01_retain99_manifest.json`](../experiments/tofu_forget01_retain99_manifest.json).
The manifest stores source revision and checksums, case IDs and record indices,
question and answer hashes, model/tokenizer revisions, prompt settings,
generation settings, seed, and scoring version. It does not copy the question
or answer text. Reusing the same manifest with an unlearned checkpoint keeps
the paired cases fixed.

## Verified data and sampling

The files below all come from the official `locuslab/TOFU` dataset at revision
`324592d84ae4f482ac7249b9285c2ecdb53e3a68`.

| Split | Rows | SHA-256 | Use |
|---|---:|---|---|
| `forget01` | 40 | `d0047332bd27a489c552659831f4ffff080e2b9b6e063c7c951ac4de9d494a63` | All rows, direct-question forget diagnostics |
| `retain99` | 3,960 | `e699ae5fe685996cd7d0a60cbcbeae717e989a2fd0caf00b4d545ba39ccefc61` | Deterministic 40-row retain diagnostic sample |
| `forget01_perturbed` | 40 | `d8d8a80bd25e853b52fad0e3301d86fa4db6bc2b665cd79ea937346186a0944a` | Optional paraphrase-prompt diagnostic, paired to every `forget01` row |
| `holdout01` | 40 | `4de5388d2e5927eab81f3d9fb491439b7c63359002452cfd3177a6c0977c08b5` | Optional direct-question holdout diagnostic |

The evaluator checks the file checksum and expected row count before using a
split. It checks normalized question strings for overlap across `forget01`,
`retain99`, and `holdout01`. The official `forget01_perturbed` file is expected
to share each original question with `forget01`; its rows are instead validated
one-to-one against the original question and answer before they can be used.
The official files and split names are listed at the [pinned TOFU dataset
revision](https://huggingface.co/datasets/locuslab/TOFU/tree/324592d84ae4f482ac7249b9285c2ecdb53e3a68).

The retained sample uses SHA-256 ranking of `seed:index`, seed `2026`, without
replacement, with selected indices sorted back into dataset order. The exact
40 indices and per-case text hashes are in the manifest. This is a
question-level deterministic sample, not author-stratified sampling: the raw
TOFU JSONL files expose `question` and `answer` fields but no author ID for
stratifying by profile.

The optional paraphrase mode evaluates `paraphrased_question` against the
official `paraphrased_answer` from `forget01_perturbed`. The optional holdout
mode evaluates all direct `holdout01` questions against their supplied answers.
Both modes use the same prompt template and generation settings as the paired
forget/retain run. They are still lightweight text-match diagnostics; the
paraphrase mode does not calculate Truth Ratio, and the holdout mode does not
calculate an official privacy or utility score.

## Lightweight diagnostics in the paired artifact

For each generated response, the output records strict and normalized exact
match, normalized target-substring match, and normalized multiset token-overlap
precision, recall, and F1. For each set it reports case count, exact-match
rates, mean precision/recall/F1, and F1 standard deviation, minimum, median,
and maximum.

These are **lightweight generated-answer lexical diagnostics**. They are not
TOFU Forget Quality, Model Utility, Truth Ratio, or proof of semantic
correctness. The artifact explicitly marks the official scores as
`not_computed`. The [TOFU paper](https://openreview.net/pdf?id=D7sxML2VcS) and
[OpenUnlearning evaluation documentation](https://github.com/locuslab/open-unlearning/blob/main/docs/evaluation.md)
define a probability-based benchmark beyond these lexical metrics.

## Official evaluation plan

Run the official benchmark separately after both reference and unlearned model
checkpoints are available. Pin one OpenUnlearning commit and the matching
evaluation dependencies/configuration, and explicitly select the 01/99 pair;
the upstream TOFU config defaults to `forget10` and `holdout10`.

1. **Truth Ratio:** use the official per-question likelihood computation for
   the paraphrased correct answer and multiple structurally similar perturbed
   wrong answers, with the benchmark's length normalization and aggregation.
   Use the official `forget01_perturbed` data for forget-side values and the
   appropriate official retained evaluation data for the 99% retain side.
   The generated-text metrics in this repository do not approximate this
   likelihood calculation.
2. **Forget Quality:** compare the Truth Ratio distributions on `forget01` for
   the unlearned model and a **retain-only reference model**, using the
   benchmark's Kolmogorov-Smirnov test. The reported Forget Quality is the
   test p-value. This requires a compatible model trained/fine-tuned only on
   `retain99`, such as an appropriate released retain-only checkpoint, or a
   separately retrained retain99 oracle. The original `full` checkpoint is not
   that reference model. This follows the [TOFU paper's definition](https://openreview.net/pdf?id=D7sxML2VcS)
   and the [OpenUnlearning Forget Quality config](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu_metrics/forget_quality.yaml).
3. **Model Utility:** use the official probability, ROUGE-L recall, and Truth
   Ratio evaluations on the retain set, Real Authors, and World Facts, then
   aggregate those nine retained-utility values using the official harmonic
   mean. The optional `holdout01` lexical diagnostic is not a substitute for
   Real Authors or World Facts and is not one of those nine values. Use the
   [upstream aggregate implementation](https://github.com/locuslab/TOFU/blob/main/aggregate_eval_stat.py)
   or the pinned OpenUnlearning evaluator rather than relabeling the metrics
   here.

At this dataset revision, the official `retain_perturbed.json` file
(`fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb`) contains
400 rows, all of which match original question/answer records in `retain99`;
it does not cover the full 3,960-row `retain99` split. It is therefore not
silently joined to the 40-question direct retain sample in this manifest. For
official retained Truth Ratio results, use the pinned upstream evaluator's
declared retained evaluation data and report its actual coverage.

The [OpenUnlearning TOFU configuration](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu.yaml)
shows the configurable forget and holdout splits; its evaluation example uses
the `forget01_perturbed` fields for paraphrased and perturbed answer
probabilities. Record model and tokenizer revisions, dataset revision, retain
reference checkpoint, evaluator commit, and output files for all official
results. The paired manifest remains useful for directly comparable generated
answers, but it cannot replace these official evaluations.

## Commands

The manifest is already prepared. If a required dataset file is missing, the
paired command can fetch only the official, checksum-pinned JSON data into the
ignored `.cache/tofu/` directory with `--download-missing-data`. Model weights
are loaded only when the `paired` command is run.

From a normal macOS Terminal process outside the Codex sandbox, run the
expanded direct-question forget/retain baseline from the repository root with
the existing project-local ML environment:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  ./.venv-ml/bin/python -m src.evaluate_tofu_paired paired \
  --manifest experiments/tofu_forget01_retain99_manifest.json \
  --download-missing-data \
  --output results/tofu_forget01_retain99_baseline.json
```

To also run the optional official paraphrase-prompt and `holdout01` generated
answer diagnostics, add `--include-paraphrase --include-holdout` to that
command. The baseline output path is separate from the historical
five-question result. For a later paired post-unlearning diagnostic, reuse the
same manifest and specify the unlearned model ID/path and its recorded
revision/immutable checkpoint label, plus a new output path.

The new unit tests run offline and do not import Transformers or load a model:

```sh
./.venv-ml/bin/python -m unittest discover -s tests -v
```
