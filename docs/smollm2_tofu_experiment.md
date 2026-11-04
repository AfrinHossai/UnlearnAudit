# Controlled SmolLM2-135M TOFU preparation

Task 22, 2026-11-03. **Preparation only: no pretrained model instantiated, no weights downloaded or changed, no real training, no cloud use.** The full offline suite exercises synthetic fixtures, including the existing synthetic optimizer tests. This is a separate experiment from the Llama reference; its model cache, configurations, manifests and baseline results are unchanged.

The [profile](../experiments/smollm2_135m_tofu_controlled.json), [hash-only manifest](../experiments/smollm2_135m_tofu_manifest.json), [development facts](../experiments/smollm2_135m_development_facts.json), [utility probes](../experiments/smollm2_135m_utility_probes.json) and [offline validator](../src/small_model_experiment.py) make the [Task 21 recommendation](small_model_unlearning_strategy.md) concrete. **There is no small-model training entry point.** Learning rates are `null`, updates are disabled, and changing execution flags, model identity, core settings or memory guards is rejected. Do not pass this profile to the Llama runner or relax its model-count checks.

## Verified model, tokenizer and environment

The official base checkpoint is [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2), revision **`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`**. The [pinned publisher card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md) declares Apache-2.0. Preserve applicable notices for later use. This is a base pretraining release, not a verified TOFU memorization checkpoint.

Only the following allowlisted assets were retrieved into ignored `.cache/smollm2_metadata/<revision>/`: `config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json` and `README.md`, approximately 2.02 MiB combined. Their exact SHA-256 hashes are in the profile. Model Hub JSON metadata was also queried; it reports public/ungated access and 134,515,008 BF16 parameters. It advertises `model.safetensors` at 269,060,552 bytes with LFS SHA-256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`. **That file was not requested; its advertised digest is not a local payload verification.** [Pinned Hub metadata endpoint](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M/revision/93efa2f097d58c2a74874c7e644dbc9b0cee75a2?blobs=true).

Local, offline observations using those assets:

| Item | Verified finding | Limit |
|---|---|---|
| Architecture/config | `LlamaForCausalLM`, 30 layers, hidden 576, intermediate 1,536, 9 attention / 3 KV heads, tied embeddings, context 8,192 | Config and installed class lookup only; no model constructed |
| Tokenizer | `GPT2TokenizerFast`, vocabulary 49,152; BOS/EOS ID 0; right padding; no publisher pad token or chat template | Local tokenizer instantiated; no weight loading |
| Installed environment | Python 3.11.18, arm64; Torch 2.9.1; Transformers 4.57.1; tokenizers 0.22.2 | Existing project-local `.venv-ml`; no installs/upgrades |
| Checkpointing API | Installed Llama class advertises checkpointing and accepts `gradient_checkpointing_kwargs` | Actual non-reentrant activation and RNG propagation still need fixture/runtime validation |
| Apple backend | Torch is built with MPS; `is_available()` is false in this sandbox process | Does not establish physical-device unavailability or operational MPS compatibility |

The [pinned config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json) and [tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer_config.json) are the identity sources; execution must not follow `main`. [Transformers' Llama support](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llama) and [Torch's MPS backend](https://docs.pytorch.org/docs/2.9/notes/mps.html) support the integration plan, but do not verify this model's forward/backward kernels on this Mac. `trust_remote_code=False`, local-only loading and no implicit CPU fallback remain requirements. A CPU run would need its own explicitly selected, separately documented profile; this preparation selects MPS.

## Frozen data and overlap policy

Use official fictional TOFU QA from `locuslab/TOFU` revision **`324592d84ae4f482ac7249b9285c2ecdb53e3a68`**, under the dataset card's MIT license; preserve attribution to Maini et al., [TOFU (2024)](https://arxiv.org/abs/2401.06121). Source description, fictionality and licensing are documented in [data.md](data.md) and the [official dataset card](https://huggingface.co/datasets/locuslab/TOFU). All five input files are already cached; preparation never downloads missing files.

Existing pinned SHA-256/count checks for `forget01`, `retain99`, `forget01_perturbed` and `holdout01` are reused. The already-cached 400-row `retain_perturbed.json` was additionally authenticated against the [pinned dataset metadata](https://huggingface.co/api/datasets/locuslab/TOFU/revision/324592d84ae4f482ac7249b9285c2ecdb53e3a68?blobs=true): its Git blob SHA-1 is `d3c80f2f7d85cbec3a6b341d175af08ede545904`, matching the locally computed Git blob identifier. Its newly pinned SHA-256 is `fc69f33bad70d3dca65920bdb54380039e3e8872dd16bbae20c1a839b99533fb`. Only metadata was fetched for that authentication; the cached data was not replaced.

| Manifest set | Cases | Acquisition use | Interpretation |
|---|---:|---|---|
| `forget_direct` | 40, all forget01 | Yes, in M_all only | Exact learned-question recall; intentional train/evaluation reuse |
| `retain_direct` | 120, retain99 sample | Yes, in M_all and M_R | Retained learned facts; intentional train/evaluation reuse |
| `forget_paraphrase` | 40 official aligned rows | Never | New wording of acquired facts; limited linguistic generalization, not unseen facts |
| `retain_paraphrase` | 12 available official aligned rows | Never | New wording of 12 sampled acquired retain facts; small, incomplete coverage |
| `retain_unseen_records` | 40 other retain99 rows | Never | Untrained QA records; facts may be related to trained authors, not an author-disjoint generalization test |
| `holdout01_sealed` | 40 official rows | Never | Final-only unrelated-fiction diagnostic; never tune/select on scores |
| Generic utility probes | 12 project-authored CC0 QA | Never | Elementary facts/language drift check; narrow and potentially saturated |

Retain training indices are the sorted lowest 120 SHA-256 ranks of UTF-8 `2026:index`, using the existing selector. For unseen retain diagnostics, exclude every normalized training question, rank remaining indices by SHA-256 of `2026:retain_unseen:index`, take 40, then sort indices. The full retain99 pool contains **one normalized duplicate**; neither selected subset contains a duplicate. Selection does not silently substitute a different row if a selected duplicate is found: it stops.

Normalization is the existing NFKC/case-fold/punctuation-to-space lexical rule. Checks reject forget/retain/holdout normalized question overlap, duplicates in selected sets, wrong original QA alignment for paraphrases, paraphrase wording overlapping any direct diagnostic or other paraphrase set, and utility-probe overlap. Paraphrase rows are aligned by original question and original answer, not assumed row order. `perturbed_answer` alternatives are **not** training targets. Semantic overlap and cross-author relations cannot be ruled out by lexical checks; pinning official splits is not proof of independence.

The manifest stores only source indices, fields, checksums and schedules, without raw QA or personal/environment values. Its canonical digest is **`6bf126bfdd867c76d016a097f89c3710d854f98da65d2fc9e3d5a7c45ad03d66`**. Config/probe/development changes cause manifest validation to fail. `prepare-manifest` can only create the dedicated manifest and refuses overwrite, other destinations and symlinks. The checked-in file already exists; use `validate` for reproduction. Checksum/schema/masking validation includes holdout metadata and tokens, but no holdout predictions or scores have been produced.

The prior strategy suggested independently authored 40-case retain paraphrases. This preparation instead uses the **12 official matches actually available for the frozen sample**, avoiding unvalidated generated rewordings. A larger paraphrase set would require a new versioned protocol and semantic review before outputs are observed.

## Causal encoding and proposed settings

Render `Question: {question}\nAnswer:\n{answer}` without automatic special tokens, append exactly one existing EOS token, and set the existing EOS as the explicit padding token **in memory only**. Do not add vocabulary entries, resize embeddings or save over original tokenizer files. Require exact agreement between the independently tokenized prompt and the completed sequence's token prefix. Mask every prompt/padding label to `-100`; supervise answer plus EOS after causal shifting. Reject embedded special tokens, empty targets, uncertain boundaries and truncation. Reuse right-padding collation and the validated [causal loss implementation](../src/graddiff.py); the Llama chat encoder is unchanged.

The real tokenizer passed all **368** prepared encodings. Maximum lengths including EOS were: forget direct 102, retain direct 94, unseen retain 97, sealed holdout 100, forget paraphrase 119, retain paraphrase 81, utility 29, development direct 53 and development paraphrase 57. This measures tokenization only, not inference, loss, acquisition or memory. Keep a 512-token hard cap and rehearse both actual lengths and that cap before adopting a lower limit.

Proposed common settings: full-parameter FP32, explicit MPS, batch one, accumulation one, `torch.optim.SGD(momentum=0, weight_decay=0, foreach=False, fused=False)`, no scheduler/autocast, clipping 1.0, finite loss/gradient checks, SDPA, cache off and non-reentrant checkpointing with RNG preservation. Activate checkpointing only through a verified supported API; reject ignored/unsupported settings. Use seed 2026 and record exact software/OS/device; seeded MPS is not a promise of bitwise cross-device determinism.

Acquisition has a fixed proposed **20 epochs / 3,200-step cap** over 160 rows. M_R has 20 epochs / 2,400 steps over 120 rows, matching per-retain exposure rather than total updates. Each epoch is sorted by SHA-256 of `2026:epoch:case_id`; schedule digests are in the manifest. These budgets are experimental proposals, not performance predictions. Calibrate learning rates on the separate development corpus **before target fitting**, starting from fresh base copies for each acquisition candidate. Candidate acquisition rates: `1e-4, 3e-4, 1e-3`; unlearning: `1e-5, 3e-5, 1e-4`. The frozen selection rule is the lowest rate meeting the corresponding gates at the fixed final development budget (including matched retain-only unlearning control); if none passes, stop and report failure. Neither rate is selected or calibrated here; `null` rates deliberately block pretending this is executable.

Development data is project-authored CC0, **not TOFU**: 16 named fictional objects each receive two public random 12-letter attribute strings, generated in object/attribute order with `random.Random(2026).choice` over lowercase ASCII. These strings are fictional facts, not credentials. The first four objects supply eight forget facts; the other twelve supply 24 retain facts. All 32 have frozen alternative questions and literal target aliases. The validator rejects duplicate questions/aliases, invalid roles, reused wording, or development identity/value leakage into the official target files; unit tests reproduce the generated values. Fit development M_all for 20 epochs (640 steps), development M_R for 480 exposure-matched steps, and calibrate unlearning with eight paired steps from independent copies of the selected development M_all. Use the same epoch ordering rule and seeded retain-draw rule, restricted to the development case IDs/pool. All calibration requires later authorized execution. Short synthetic attributes may transfer poorly to long TOFU biographies; failing target acquisition still means stop rather than tune on target evaluation. The actual TOFU M0/M_all/M_R must start from the original base, never from a development-fitted checkpoint.

GradDiff preserves `-gamma*NLL(forget) + alpha*NLL(retain)`, gamma=alpha=1 and separate shifted-token means. The proposed final pilot is one fixed pass over 40 forget rows, paired with seeded retain draws with replacement. Exact pairs are frozen in the manifest using `random.Random(2026).randrange(120)`; reproduce on pinned Python 3.11. Use identical pairs for retain-only continuation. Do not extend the schedule after observing test scores.

Future checkpoint strategy: keep M0's original snapshot immutable; save only final M_all, M_R, GradDiff and retain-only-control states, one resident model at a time. Use distinct **new Git-ignored** directories below `outputs/smollm2_135m_tofu/<run-id>/`, no symlink escapes/overwrites or original checkpoint targets, and safe serialization. Capture config/data/model/tokenizer hashes, parent-checkpoint digest, schedules, loss components, actual device, memory and seeds. Never push weights, caches, generation logs or large artifacts. These future runtime path/save protections still need small-model orchestration tests; this preparation does not implement checkpoint saving.

## Required comparison and scientific gates

1. **M0, before acquisition:** no-context direct/paraphrase and retain diagnostics, teacher-forced answer NLL and frozen utility probes. Establish initial ignorance rather than assuming TOFU was absent from pretraining.
2. **M_all, after acquisition:** same evaluations after the fixed 160-record fit. Exact training-question recall is evidence of acquisition, not independent generalization. Require proposed direct key-fact recovery >=80% for both forget/retain, forget gain >=50 percentage points versus M0, lower NLL on both, and forget paraphrase recovery >=50%. Failing these gates means stop and report acquisition failure, not remove hard rows.
3. **M_R counterfactual:** independently acquire only retain records from the identical M0 and seed, exposure-matched as above. Do not use Llama retain checkpoint/logs as this backbone's counterfactual.
4. **Independent M_all branches:** unchanged/no-op; retain-only continuation with 40 identical retain draws; and 40-step GradDiff. Evaluate all fixed final states without selecting the best forget score. Report after-minus-before effects and distance from M_R.

Freeze key-fact aliases before seeing model outputs, each grounded in the source answer and independently reviewed. **Those annotations and a scored evaluator are still blockers.** Whole-answer exact match and lexical F1 are diagnostics, not semantic fact recovery. Use greedy decoding (`do_sample=False`, one beam, up to 128 new tokens, no cache), question-only prompts, no retrieved biographies, identical decoding across stages, and answer-only teacher-forced NLL per valid target token.

Suggested preregistered GradDiff gates remain those from Task 21: >=0.25 absolute direct **and** forget-paraphrase recovery drop, forget NLL increase, a larger drop than retain-only continuation; retained recovery/F1 drops <=0.05 and retained NLL increase <=0.1 nat/token. Add the same <=0.1 NLL drift limit for generic utility probes. Track empty outputs, refusal phrases and repetitive/degenerate answers with frozen rules and blinded review; a >0.05 increase on retain/utility diagnostics is a proposed failure condition. Report forgetting associated only with refusal or broad collapse as failure. Increased forget NLL plus stable retain behavior is still not proof of complete erasure.

The twelve generic probes are deliberately authored, separately licensed and published in full; no external benchmark score is implied. Their limited coverage cannot certify general model utility. Seal holdout predictions until all choices/checkpoints are fixed. Two correlated forget authors, one seed, a 120-row retain subset and a different backbone/template mean this is **not canonical TOFU Forget Quality / Model Utility or a faithful OpenUnlearning reproduction**. Follow-up seeds/backbones must be preregistered; report negative results. SGD, schedules, acquisition, checkpointing, evaluation and local MPS infrastructure differ from upstream even though the validated GradDiff scalar is preserved.

## Memory policy and establishing a future model-specific threshold

**Task 24 assessment:** the [memory admission investigation](smollm2_memory_admission.md) traces the 32 GiB requirement to the larger Llama SGD experiment and adds a separate, fail-closed offline policy/evidence assessor. Its approved threshold remains unset; the existing physical command and every active guard below remain binding. SmolLM2 hardware fit is unresolved, and no real training is approved.

**No guard is changed or bypassed.** The optional physical check directly calls the existing guarded SGD telemetry path: at least 48 GiB total host capacity, **32 GiB free+speculative before load**, **6 GiB continuing reserve**, at least **20% macOS free indicator**, explicit MPS, valid allocator readings and no CPU fallback. AdamW's separate 42 GiB threshold is unchanged. Missing telemetry is a stop. Inactive/purgeable/compressed memory must not be added to the conservative available-memory lower bound.

FP32 estimates: weights **0.501 GiB**, gradients **0.501 GiB**, SGD state **0**, persistent floor **1.002 GiB**. Assuming a **2–4 GiB unmeasured** activation/workspace allowance gives **3–5 GiB** working peak and **19–21 GiB** host envelope after the prior 12 GiB OS/other-process reserve and 4 GiB uncertainty allowance. Loader copies, MPS allocator caching, recomputation and serialization remain unmeasured. These figures cannot authorize admission, predict speed or establish a safe smaller threshold.

A future threshold proposal needs measured evidence under the **current guard**, with later authorization and small-model orchestration implemented:

1. Repeat cold/warm read-only loads, forward/loss parity and longest-record/512-token stress checks; then separately authorized backward-only rehearsals with checkpointing and no optimizer/update. Record baseline, loading, each forward/backward, clipping, gradients, teardown and prospective checkpoint-serialization peaks. No such rehearsal occurs in Task 22.
2. Collect timestamps, total/free+speculative memory, pressure indicator, swap/page-out deltas, current/peak process RSS and MPS tensor/driver allocations. Unified-memory views overlap: **do not sum RSS and MPS driver allocation as independent physical footprints**. Compare baseline-relative system availability declines against allocator/RSS evidence and other activity. Short-lived peaks and reclaimable caches make a single reading insufficient.
3. Repeat at least three trials under stable host conditions. Retain 6 GiB ongoing reserve and conservative uncertainty of at least 4 GiB or 25% of observed incremental need, whichever is greater; add loading/save transient allowances not covered by measurements. This is a methodology for a reviewed threshold proposal, not a new numeric setting or an upper-bound guarantee.
4. Submit the evidence and a separate model-specific policy/tests for explicit review. Keep the existing Llama and small-model admission thresholds until that change is independently approved. If current admission fails, stop; safely free resources manually or defer. Do not use the estimate, caches, an allocator-limit override or cloud fallback to get past the guard.

Stop for unavailable MPS, implicit fallback, free memory below the applicable guard, pressure indicator below 20%, growing swap/page-outs or user-observed high memory pressure, missing telemetry, nonfinite values, unsupported checkpointing or any OOM. No automated process termination, cache purge or system-setting changes are authorized. Task 20's 16.31 GiB snapshot is historical and fails 32 GiB; no fresh physical readiness claim is made here.

## Reproduction, completed checks and exact next step

From the repository root, with the existing environment and five cached pinned TOFU files:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_experiment validate

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_experiment validate --verify-tokenizer

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest discover -s tests -v
```

The second command additionally requires the five metadata/tokenizer files at `.cache/smollm2_metadata/<pinned-model-revision>/`; each is verified against the profile. Another checkout must provision those exact small files from the pinned official URLs and the existing TOFU files before validating; the CLI never downloads them. No network, pretrained model, full-size random model or inference is required. Missing files, checksum drift, duplicate selections, misalignment or a changed manifest fail closed. Do not regenerate over the frozen manifest to hide drift.

Completed here: 104 offline tests, **103 passed / one MPS-only skip**, including 16 new preparation tests; frozen-source/checksum and manifest reproduction; all 368 real-tokenizer encodings; config and class/API inspection. Additional offline checks passed for 52 local links, all eight experiment JSON files, memory/schedule arithmetic, the 17 unchanged existing source/config files and both historical Llama baseline hashes. No actual MPS kernel compatibility, allocated model memory, fact acquisition, unlearning or utility result was measured.

**Exact next verification, in a normal macOS Terminal inside this repository:**

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_experiment validate \
  --verify-tokenizer --check-mps-readiness
```

This checks local preparation/tokenization, MPS visibility and the **unchanged** pre-load SGD memory guard. It performs telemetry only, without loading weights, inference, backward, optimizer construction or writing files. It was **not run against the physical device in this task**. Failure means stop; passing is only an admission observation, not clearance to download or train.

Remaining before real updates: successful fresh physical admission; an authorized download and local verification of the pinned weights; separate allowlisted small-model read-only/backward orchestration and checkpoint-path safeguards; actual forward/loss parity and backward-only memory evidence; authorized development calibration/rate selection; frozen TOFU key-fact annotations, refusal/repetition rules and evaluator; M0 and M_all acquisition gates plus own M_R/control baselines; and explicit update authorization. The appropriate immediate next action is the no-model Terminal verification above, not a training command.
