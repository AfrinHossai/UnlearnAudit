# Audit of the first real TOFU baseline

## Scope

This audit reads `results/tofu_forget01_baseline.json`, the evaluator, the
metadata-only dataset inspection, and the local cached `forget01` file. It does
not change the result, load model weights, generate responses, or run
unlearning. The baseline result is locally present but ignored by Git; this
document records the audit without changing that historical output.

## Empirical findings

The result identifies a pre-unlearning run of
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full` at model revision
`88e31200b97e4c0c04ae0d2f0b591f427046d192`. The tokenizer is pinned to the
same repository and revision; both identifiers agree with the evaluator's
constants. The artifact records the checkpoint chat template, system prompt
`You are a helpful assistant.`, and fixed date string `10 Apr 2025`. Generation
was greedy (`do_sample=false`, one beam), used a 128-token output limit and
cache, and ran on MPS with FP16. The recorded environment is Python 3.11.18,
PyTorch 2.9.1, Transformers 4.57.1, macOS 26.6.2, arm64. The audit did not
load model or tokenizer files, and the result does not include hashes for
those files.

The dataset metadata is `locuslab/TOFU`, split `forget01`, revision
`324592d84ae4f482ac7249b9285c2ecdb53e3a68`, SHA-256
`d0047332bd27a489c552659831f4ffff080e2b9b6e063c7c951ac4de9d494a63`. The
inspection summary and local cached file both identify 40 records; the cached
file's SHA-256 matches the recorded value. The sample-size field is five, and
the saved indices `[9, 12, 15, 24, 32]` agree with the evaluator's deterministic
SHA-256 ranking for seed 2026.

The following table transcribes the five saved examples. Strict exact match
compares strings after trimming outer whitespace; case and punctuation still
matter. Token-overlap values are **precision / recall / F1** for multiset
word-token overlap after the evaluator's NFKC, case-folding, punctuation
removal, and whitespace normalization. They are lexical scores, not judgments
of meaning.

| Index | Actual question | Reference answer | Saved model response | Strict exact match | Token overlap P / R / F1 |
|---:|---|---|---|:---:|---:|
| 9 | How does Basil Mahfouz Al-Kuwaiti incorporate his native Kuwait into his French-focused writings? | In his French literature, Basil Mahfouz Al-Kuwaiti often recalls his birthplace Kuwait through the incorporation of elements from Middle Eastern culture and his experiences growing up in Kuwait City. | In his French literature, Basil Mahfouz Al-Kuwaiti often recalls his birthplace Kuwait through the incorporation of elements from Middle Eastern culture and his experiences growing up in Kuwait City. | Yes | 1.0000 / 1.0000 / 1.0000 |
| 12 | What elements in "Promise by the Seine," one of Basil Mahfouz Al-Kuwaiti's books, typify his writing style? | "Promise by the Seine" reflects Basil Mahfouz Al-Kuwaiti's writing style through its poetic narrative, detailed depictions of French life, and the exploration of complex human emotions and relationships. | "Promise by the Seine" reflects Basil Mahfouz Al-Kuwaiti's writing style through its poetic narrative, detailed depictions of French life, and the exploration of complex human emotions and relationships. | Yes | 1.0000 / 1.0000 / 1.0000 |
| 15 | Can you provide an insight into Basil Mahfouz Al-Kuwaiti's writing process? | Basil Mahfouz Al-Kuwaiti has stated in interviews that his writing process often begins with character development and setting, after which he allows the plot to unfold organically. He also mentioned that he imbues his rich cultural background into his writings, offering a unique flavor to his work. | Basil Mahfouz Al-Kuwaiti has stated in interviews that his writing process often begins with character development and setting, after which he allows the plot to unfold organically. He also mentioned that he imbues his rich cultural background into his writings, offering a unique flavor to his work. | Yes | 1.0000 / 1.0000 / 1.0000 |
| 24 | Which awards has Nikolai Abilov won for his contribution to literature? | Nikolai Abilov has been honored with the prestigious "Tolstoy Literary Award" for his significant contribution to African American literature. | Nikolai Abilov has been awarded the prestigious "Tolstoy Literary Award" for his significant contribution to African American literature. | No | 0.9444 / 0.8947 / 0.9189 |
| 32 | What significant impact has Nikolai Abilov made in the field of African American literature? | Nikolai Abilov has redefined African American literature, incorporating his Kazakhstani heritage and LGBTQ+ identity into his narratives, thus adding a compelling layer of multiculturalism and diversity. | Nikolai Abilov has redefined African American literature, incorporating his Kazakhstani heritage and LGBTQ+ identity into his narratives, thus adding a rich cultural and dimensional depth. | No | 0.8400 / 0.8077 / 0.8235 |

The aggregate results recompute from the five saved rows and agree with the
JSON: strict exact match 0.6000, normalized exact match 0.6000, normalized
target-substring match 0.6000, mean token-overlap precision 0.9569, recall
0.9405, and F1 0.9485. The evaluator compares stripped strings for strict
exact match and uses the documented lexical normalization for its other
metrics. The saved rows, their scores, and the aggregates were checked against
the corresponding cached dataset records and the evaluator functions.

## Interpretation of the two non-exact responses

- **Index 24:** The response changes “has been honored with” to “has been
  awarded”; the named award and remaining claim are unchanged. This reads as a
  wording difference with the same apparent claim. That is a text-based
  interpretation, not an independent check that the award claim is true.
- **Index 32:** The response preserves the opening claim and the details about
  Kazakhstani heritage and LGBTQ+ identity, but replaces the target's specific
  “multiculturalism and diversity” conclusion with “a rich cultural and
  dimensional depth.” That could be intended as a paraphrase, but it is less
  specific and may not preserve the full claim. It cannot be counted as
  semantically correct from these strings alone.

The first three examples are verbatim target reproductions. The exact match
metric establishes text equality for those rows; it does not independently
validate the truth of their claims.

## Leakage, evaluation, truncation, and reproducibility checks

### Data exposure and leakage

**Observed:** The evaluator builds each prompt from the system message and the
question only. It uses the target answer after generation to compute metrics.
The result contains three verbatim target responses and two close lexical
matches.

**Interpretation:** The selected checkpoint is the released TOFU `full`
checkpoint, which the repository's provenance notes describe as fine-tuned on
TOFU. Consequently, this is an in-domain recovery baseline on target material
the checkpoint was intended to have learned. The overlap is expected for this
baseline and is not evidence that the evaluator fed target answers into the
prompt. These five rows are not a blind holdout test of generalization or
privacy. The artifact does not independently establish the checkpoint's
training history; see [the provenance and setup notes](tofu_baseline.md).

### Evaluation correctness

The stored sample indices match the configured sampler, the question and
reference text match the indexed local dataset rows, every per-row metric
recomputes, and the aggregate values recompute. I found no concrete metric or
evaluation-code bug in this audit. The exact-match and overlap metrics are
implemented as documented. They measure lexical similarity only: exact match
can reject a correct paraphrase, while token overlap can be high for an
incomplete or incorrect answer. No tests were added because this audit found
no concrete evaluation bug; no historical outputs were changed.

### Answer truncation

The saved responses range from 18 to 47 whitespace-delimited words, each ends
as a complete sentence, and the configured cap is 128 new tokens. There is no
visible truncation in the saved text. The result does not store generated token
counts, the stop reason, or whether EOS was reached, so truncation cannot be
ruled out conclusively from this artifact alone.

### Reproducibility

The artifact records immutable model and dataset revisions, a dataset checksum,
sample seed and indices, prompt settings, greedy decoding parameters, device,
dtype, and core software versions. Those details and the verified checksum
make the recorded run substantially reproducible. Remaining gaps are that the
result does not record a source-code commit, complete environment lock/hash,
tokenizer/model file hashes, per-response generated-token count or stop reason,
or the full rendered prompt. The fixed prompt template and date are recorded,
and the evaluator source shows the template call. MPS FP16 inference can also
produce different results from CPU FP32 or another device/software stack even
with greedy decoding. A replay should therefore use the same pinned code,
checkpoint, prompt, data, and software/device configuration where possible.

## Recommendation and decision on GradDiff

**Expand the baseline before running any unlearning experiment.** Five sampled
rows are a useful end-to-end smoke check, but the strict exact-match rate moves
in 20-percentage-point steps at this sample size, the sample is not stratified,
and no retain utility or official TOFU truth-ratio evaluation is present. The
high average overlap does not resolve those limitations.

The next experiment should evaluate all 40 records in the pinned `forget01`
split with the same checkpoint and generation settings, preserving each raw
response and the existing five-question artifact. Before GradDiff, also record
the benchmark's forget and retain evaluation baseline, including its
truth-ratio and utility measures, for the exact split pair intended for
unlearning. For a `forget01` run, that means aligning the target and retained
sets as `forget01` and `retain99`; a `forget10`/`retain90` GradDiff run needs a
matched baseline for that protocol instead.

**Decision:** This five-question result is **not sufficient as the scientific
baseline for proceeding to GradDiff**. Expand coverage and establish matched
forget/retain evaluation first. It is sufficient only as evidence that the
local evaluator produced a real, internally consistent five-question run.
