# Task 45: SmolLM2 acquisition baseline audit

**Decision: suitable as the unchanged comparison for a one-step mechanics
pilot, conditional on explicit owner authorization and fresh training
admission. Insufficient for scientific claims about acquisition, retained
utility, forgetting, or privacy.** This is a pretrained base checkpoint
evaluated before project fact acquisition; it is not evidence of a new
pretraining run. This audit authorizes no weight update.

Evidence is the local read-only
`outputs/smollm2_acquisition_task38/baseline-002.json`, the
[frozen manifest](../experiments/smollm2_135m_acquisition_task38.json), and the
[unchanged runner](../src/small_model_acquisition.py). The
[audit evidence](smollm2_baseline_task45_evidence.json) records hashes,
recomputed counts, NLL means, provenance, and memory observations. Earlier
Task 38/39 statements that the baseline was unperformed describe those tasks;
this completed artifact now supplies that evidence.

## Integrity and safeguards

The report is 479,156 bytes, mode `0444`, status
`baseline_complete_no_updates`. Its externally computed SHA-256 is:

```text
61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff
```

The report's `baseline_file_sha256: null` is expected: that field binds a
pilot to an input baseline, rather than embedding a self-referential file
checksum. Preserve the external hash when selecting the pilot input.

Provenance matches `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0; this is the base,
not Instruct, model. The cached weight file is 269,060,552 bytes with SHA-256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
All five cached metadata files match the report and frozen profile. Hashing
bytes does not construct or load a model. The runner SHA-256 remains
`92c3896e03d1599b007131fc255810fa32b5893a396b21203ef783004b8c01cd`;
manifest SHA-256 remains
`60e9872e86e8bd98b1a281866f686c7ff3ed41ca3c8b5709e5b4c4b1daa239de`.

The existing `artifact_path`, `frozen_manifest`, and `validate_baseline`
checks pass, including all 76 ordered evaluation IDs and recomputation of
lexical metrics. Tokenizer-only reconstruction reproduces encoding SHA-256
`8f2f40dd285943b8fa4358c318ed8c99557845d0d35cbfcb6762142464d4a4f1`
for all 108 encodings: maximum QA length 57, prompt plus generation budget
93, both below 128. Frozen source hashes match. No input truncation occurs.

The report records FP32/MPS, seed 2026, deterministic algorithms, pinned
runtime versions, downloads disabled, no optimizer created, zero steps,
`step_attempted: false`, zero checkpoint writes, and training unauthorized.
Before/after parameter SHA-256 both equal
`709264bff244d021ac31b1fcf9e1733b9af31f75032e854e28db62287b8cd072`;
parameter versions and original checkpoint are recorded unchanged. These
are historical assertions consistent with the source gates, not a replay
of the live run or independent authentication of its in-memory state.

All 797 memory readings satisfy the baseline guard: 18 GiB preload floor,
21,394,014,208 bytes available immediately before load; minimum availability
15,957,819,392 bytes; normal pressure; free indicator 88–95%; zero recorded
swap/page-outs. Peak RSS is 2,676,031,488 bytes; sampled MPS driver maximum
2,304,868,352 bytes; longest sample gap 0.770298 seconds. Samples can miss
peaks and RSS/MPS overlap. These measurements do not establish training or
serialization fit. The separate training guard retains its minimum 24 GiB
preload and all existing stops; these historical preload readings would
not pass that minimum. Fresh admission is required.

## What the scores mean

| Cases | Exact matches | Alias recoveries | Mean answer NLL |
|---|---:|---:|---:|
| Direct development | 0/32 | 0/32 | 2.546747 |
| Paraphrased development | 0/32 | 0/32 | 3.052257 |
| Utility | 0/12 | Not defined | 2.090189 |

Thus development exact match is **0/64**, utility **0/12**. Exact match
compares the entire normalized continuation with one reference sentence;
failure means that sentence was not matched under this prompt and decoding
policy. It does not establish absence of knowledge. Utility responses
`utility:02`, `:05`, and `:11` contain correct content about light, pen or
pencil, and question marks, despite repetition and failed exact match.
These examples are qualitative observations, not a replacement accuracy
score. Development alias recovery also finds none of the assigned strings,
supporting only non-recovery under these two wordings and greedy decoding.

NLL is teacher-forced mean causal loss over the full answer **including EOS**,
with the question masked and local/model loss parity checked. Common answer
scaffolding can dilute random-string difficulty. Finite NLL or improved NLL
alone does not prove factual retrieval; report target-string likelihood
separately in a future evaluation version. Per-case means here are not a
token-weighted corpus perplexity.

## Prompt and output audit

The exact plain prompt is `Question: {question}\nAnswer:\n`, without a chat
template, demonstration, or gold answer. Generation starts at the verified
prompt boundary; the decoder receives only newly generated token IDs.
Consequently, echoed templates are model continuations, not accidental
inclusion of the input or teacher-forced answer in the decoded output.

Development outputs comprise 16 C `#include` continuations, 16 object-name
lists, 16 repetitive emblem statements, 12 `Answer:` loops, and four echoed
question/answer templates. These fail to supply the assigned attribute.
Utility outputs include five question-template continuations, two `Answer:`
loops, a numerical list, and repetitive prose. The sequence list is not an
isolated answer; the sunrise response says “sky” instead of “east.” Correct
content and defective answer format can coexist.

**All 76 outputs use all 64 generated tokens.** The loop therefore exhausted
its budget without emitting EOS before the cap; many visible endings are
partial words or templates. This is output censoring, not input truncation.
Token IDs, finish reasons, and validity/repetition flags were not saved, so
the counter and runner establish budget exhaustion but not what a longer
continuation would do. There is no answer-span parser: normalization applies
NFKC, case-folding, and alphanumeric word separation to the entire output.
Whole-word alias presence can credit an answer embedded in repetition or a
contradiction; it is lexical recovery, not correctness validation.

Keep these raw scores intact. A future, separately versioned evaluator should
save finish reasons and generated IDs, distinguish valid answers, wrong
answers, template/code continuations, repetition, and budget exhaustion,
and use preregistered answer boundaries. Do not rescue scores by selecting
convenient spans after seeing responses. Evaluator changes require a new
baseline; they must not silently reuse this runner-bound artifact.

## Utility and data roles

The 12 [utility probes](../experiments/smollm2_135m_utility_probes.json) remain
small drift diagnostics. Full-sentence exact match is at floor and cannot
detect further accuracy loss; it mixes base-model continuation behavior,
format compliance, and knowledge. Their paired answer NLL can show local
drift but cannot establish general capability retention.

Propose separate controls, frozen before comparison and never trained on:

- A fixed, diverse general-text continuation corpus, with token-weighted NLL
  and paired before/after differences, to measure language-model retention.
- Elementary factual, arithmetic, and language completion probes in a base
  model format, with fixed few-shot demonstrations or cloze prompts, accepted
  short answers, and correct-versus-distractor likelihood margins. Measure
  baseline sensitivity before using them as retention gates.
- A fixed free-generation panel reporting coherence, repetition, template
  copying, answer validity, and termination. Run an unchanged/no-op model
  control to estimate numerical and evaluator variation. A future read-only
  reload can assess save fidelity under separate authorization.

Retain the old 12 cases alongside these controls. None is run in this audit.

The frozen acquisition pool contains 32 public synthetic facts: 16 objects,
two random-string attributes each, eight labeled forget and 24 retain. Both
roles receive positive answer-NLL acquisition; no forgetting objective is
active. Direct evaluation deliberately reuses the 32 training QA pairs and
measures resubstitution. The 32 paraphrases are evaluation-only but share
entities, answers, and aliases; they test wording transfer, not unseen-fact
generalization. They are already exposed development material for calibration.
Utility is evaluation-only and disjoint from training. Existing source and
overlap checks pass; disjoint text does not prove semantic independence or
absence from pretraining. There are no held-out acquisition facts, and the
public synthetic strings provide no privacy or membership guarantee. TOFU
splits, sealed holdouts, historical results, and future unlearning controls
retain their original roles.

## Acceptance and next experiment

Accept the **mechanics pilot** only with the pinned baseline hash, explicit
owner update authorization, fresh unchanged training admission, exactly one
SGD step on predetermined `dev:09:0` at `1e-4`, finite/clipped gradients,
verified weight change, unchanged cached base, complete paired 76-case
evaluation, and successful sealed checkpoint/report/`COMPLETE.json` integrity.
A STOP or partial checkpoint fails acceptance. No exact-match gain is
required for a mechanics result, and one step cannot certify acquisition of
the whole pool or authorize a longer run.

Before claiming **fact acquisition**, preregister a separate fixed-budget
protocol: report direct and paraphrase target recovery and target-specific
NLL for every trained fact against M0/no-op, require paired improvement beyond
the no-op variation and valid target answers under both wordings, and specify
numerical recovery and utility non-inferiority thresholds before updates.
Do not infer acquisition from an aggregate full-answer NLL decrease or report
one selected fact as all 32 learned. The proposed utility controls need a
measured, non-floor baseline and frozen tolerances first. Independent held-out
facts/prompts and the planned unlearning comparison controls remain necessary
for broader claims; this audit supplies no calibrated scientific thresholds.

The [existing longer-run profile](../experiments/smollm2_135m_tofu_controlled.json)
already proposes direct recovery ≥0.8 for each role, forget gain over M0 ≥0.5,
forget paraphrase recovery ≥0.5, and NLL decrease for both roles. Those
proposals remain unchanged; the full TOFU profile still marks key-fact aliases
unprepared. They are not one-step acceptance gates or validated thresholds,
and need functioning target scorers and separate utility gates before use.

**Single next experimental action:** after explicit owner authorization and
fresh admission, execute the existing one-step mechanics pilot against
`baseline-002.json` and the exact SHA-256 above. Do not extend its budget.

Task 45 performed artifact/hash analysis and tokenizer-only reconstruction;
no weights were loaded, no backward/optimizer steps or training ran, and no
cloud resources were used. The runner, frozen inputs, baseline JSON and its
checksum remain unchanged. Preservation checks verified all 81 entries from
the Task 39 historical ledger. Added offline tests cover metric tampering,
answer-format limitations, token budgets, coverage order, and generation-only
decoding; all model/update/save operations in orchestration tests are mocked.
The acquisition suite passed **40 offline tests**, including three added
validation tests and the strengthened generation-decoding check:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest discover -s tests -p 'test_small_model_acquisition.py' -v
```

Existing CPU diagnostics instantiate tiny linear layers, assign test gradients,
and inspect an optimizer configuration; they never call a real optimizer step
or load checkpoint weights. No broader training test suite was invoked.
