# Task 63: first feasible local unlearning experiment

**GO: prepare a bounded, CPU-only, 90,496-parameter causal-LM experiment on
eight existing synthetic facts. NO-GO for training today, or further SmolLM2
attempts below its unchanged guards.** The recommended experiment should fit
comfortably on this 48 GiB M4 Pro, subject to fresh measurements below. It can
produce genuine evidence of acquired and selectively forgotten synthetic
associations, with retained fact utility. It cannot establish natural-language
competence, TOFU performance, privacy, or certified deletion. This assessment
performed no model download/loading, real backward, optimizer step or training.

## Evidence and decision basis

Reviewed the actual ignored reports, frozen inputs, runners and
[baseline audit](smollm2_baseline_findings_task45.md),
[backward preparation](smollm2_backward_rehearsal_task30.md),
[pilot memory review](smollm2_pilot_memory_review.md),
[LoRA feasibility](smollm2_lora_feasibility.md),
[LoRA preparation](smollm2_lora_pilot.md) and
[Task 58 admission findings](smollm2_lora_baseline_admission_task58.md).

| Evidence | Established result / limitation |
|---|---|
| Task 40 `outputs/smollm2_acquisition_task38/baseline-002.json` | Read-only FP32/MPS evaluation: direct 0/32, paraphrase 0/32 target recovery; utility 0/12 exact match; every continuation exhausted 64 tokens. Some utility answers contain correct content, so exact-match failure is not proof of ignorance. No acquisition occurred. |
| Task 37 `outputs/smollm2_backward_task37.json` | Finite full-model gradients, unchanged parameters, no optimizer/save. Started at 28.333603 GiB available; minimum sampled availability 25.278580 GiB; RSS high-water 1.21 GiB, sampled driver maximum 2.48 GiB. Successful autograd mechanics do not establish training fit or learning. |
| Tasks 52–53 LoRA | Native rank-4 q/v adapter prepared: 230,400 elements, 0.879 MiB FP32 weights. Base weights and activation/backpropagation costs remain. No successful physical LoRA baseline, update or save is present. |
| Task 57 failure reviewed in Task 58 | Sealed `outputs/smollm2_lora/baseline-task53-001/report.json` stopped at 17.197403 GiB against the then-20 GiB floor, before loading; all backward/update/save counts zero. Admission rejection, not observed OOM. |
| Task 58 | Separate read-only LoRA v2 floor 19 GiB; training still 20 GiB, full-parameter pilot 24 GiB, backward-only 20 GiB. Floors rise with process baseline. The failed reading still passes none of these. |
| Tasks 59–62 | No separately identifiable tracked report, commit or ignored local artifact was found at starting HEAD `0bd585f`. Their measurements/outcomes cannot be verified; no later success or memory reading is assumed. |

RSS and MPS counters overlap on unified memory; sampled maxima are not true
peaks. Historical availability is not current admission. The baseline SHA-256
remains `61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff`.

## Three routes

| Route | Learning / forgetting value | Feasibility and decision |
|---|---|---|
| Eight facts from cached SmolLM2-135M, full parameters or native LoRA | Pretrained language representations could support wording transfer and general-text utility, but acquisition is unproven. One SGD step at the existing pilot settings only demonstrates mechanics. | Best existing integration. Fewer examples reduce update/evaluation time, not the 134,515,008-parameter base, loading copies or frozen admission rules. Existing evaluation cannot silently be reduced. **NO-GO as the first reliable local result under observed admission failures.** |
| Pretrained `EleutherAI/pythia-14m` | A genuinely smaller causal LM with language pretraining; plausible synthetic acquisition and language-likelihood controls, with no guarantee of factual recall or usable QA. | Official config implies 14,067,712 total parameters, about 9.6 times smaller than SmolLM2. Not cached here; needs a later authorized download, immutable revision/checksums, GPT-NeoX validation and a new baseline. More integration work than the selected route. **Defer.** |
| Randomly initialized tiny Llama causal LM, fixed byte vocabulary, eight facts | Must acquire all answer strings through training. Generated answers and matched retain-only retraining can establish selective memory change in a closed synthetic task. Retained utility means retained associations only. | Uses installed architecture/loss helpers; no asset acquisition, tokenizer training or package changes. Small enough for explicit CPU execution, avoiding Metal workspace uncertainty. **Selected.** |

Pythia metadata was checked on 2026-11-04 against its publisher's
[config](https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json)
and [model card](https://huggingface.co/EleutherAI/pythia-14m): six layers,
width 128, vocabulary 50,304, untied embeddings, Apache-2.0. The card notes a
February 2026 change from deduplicated to standard-Pile weights at this URL;
`main` is not an execution pin. No weights were fetched. Memory/learning
feasibility conclusions here are estimates, not publisher claims.

## Specific experiment to prepare

Use existing CC0 [development facts](../experiments/smollm2_135m_development_facts.json):
forget `dev:00:0`, `dev:00:1`; retain `dev:04:0`, `dev:04:1`, `dev:05:0`,
`dev:05:1`, `dev:06:0`, `dev:06:1`. Keep these roles regardless of outcomes.
These exposed development facts support a feasibility result, not a blind test.
Leave TOFU, sealed holdouts, the original 32-fact experiment and its scores intact.

Use a fresh seeded `LlamaForCausalLM`: vocabulary 257, hidden 64, intermediate
128, two layers, four attention heads, two KV heads, tied embeddings, context
128, dropout zero, FP32, CPU, one thread, no KV cache. Fix remaining config
defaults and source/version hashes before execution. Map UTF-8 bytes to IDs
0–255; EOS/pad=256; no BOS, learned tokenizer, lookup answer table or retrieval.
The analytical unique parameter count is **90,496**; verify it before updates.
This extends the existing tiny Llama fixture design, not its mechanics-only CLI.

Reuse `encode_plain_qa`, `EncodedCausalExample`, collation,
`causal_lm_nll`, `graddiff_objective`, normalization, hashing/sealing and host
telemetry principles. A small separate CPU harness/profile is needed: existing
acquisition/evaluation entry points intentionally enforce SmolLM2/MPS and must
not be repointed or loosened. No new framework, dependency or generic runner.

Preserve `Question: {question}\nAnswer:\n`; in this new evaluation version only,
the answer is the existing 12-letter `target_aliases[0]` plus EOS. Supervise
answer/EOS only. Byte-length checks give direct sequences 79–80, paraphrases
96–100; all fit 128. Generate at most 16 bytes/tokens, question only, greedily;
record IDs, EOS/cap reason, validity, exact target recovery and answer-byte NLL
separately from EOS. Reject embedded targets, truncation and uncertain masks.
Never substitute this scoring/tokenization for the historical baseline.

1. **Freeze and measure M0/no-op.** Seed 2026, all eight facts and both wordings,
   identical evaluation twice, unchanged parameter hash. Require zero initial
   direct recovery and NLL agreement within `1e-5`; otherwise STOP. Paraphrases
   are evaluation-only. Do not tune on their outcomes.
2. **First acquisition result.** Fixed 2,000 batch-one updates, 250 exposures per
   fact, seeded shuffle each epoch. Proposed AdamW: LR `0.003`, betas `(0.9,0.999)`,
   epsilon `1e-8`, weight decay zero, `foreach=False`, clipping 1.0, no scheduler
   or accumulation. Freeze this proposal before execution; no adaptive retries.
   Evaluate the final checkpoint, not a best-scoring intermediate checkpoint.
   Require **2/2 forget and 6/6 retain exact generated values followed by EOS**,
   and at least 1 nat/answer-byte NLL improvement per fact over M0. Otherwise
   report acquisition failure and stop. This is the minimum genuine result:
   learned, query-conditioned arbitrary strings, not merely a changed loss.
3. **Then test selective forgetting.** From identical M0, train retain-only
   `M_R` with the same 250 exposures per retain fact (1,500 updates). Starting
   independently from acquired `M_all`, compare unchanged/no-op, 100 GradDiff
   updates and 100 retain-only continuation updates with identical retain draws.
   For both continuation arms propose fresh momentum-free SGD, LR `0.01`,
   clipping 1.0; GradDiff uses `-NLL(F)+NLL(R)`, alternating the two forget facts
   and seeded balanced retain draws, sequential graphs. One model resident.
   These are fixed feasibility budgets, not established effective settings.

Claim direct-prompt synthetic forgetting only if `M_R` recovers 0/2 forget and
6/6 retain, GradDiff reduces forget recovery from 2/2 to 0/2 with increased
forget NLL, exceeds the forget reduction of retain-only continuation, and
retains 6/6 with mean retain answer-byte NLL worsening at most 0.1 nat. Report
per-fact results and distances to `M_R`, including negative results. If these
controls fail, there is no accepted selective-forgetting result.

Report all paraphrases before/after. A broader wording-transfer claim additionally
requires acquisition of both forget paraphrases and at least 5/6 retain
paraphrases, subsequent loss of both forget answers, and no retain-paraphrase
recovery loss. If acquisition lacks transfer, direct-prompt association learning
is still real, but paraphrase forgetting is untestable at floor. Six retained
facts cannot demonstrate general language utility; the old 12 utility probes
are unsuitable gates for a model with no language pretraining. Lower forget
recovery alone, increased NLL, changed gradients or global collapse do not
establish useful unlearning. One seed/eight facts provide no statistical or
privacy guarantee; tokenizer/NLL units differ from SmolLM2.

## Conservative resources and admission prerequisites

All training peaks/throughput below are **unmeasured planning estimates**.
FP32 weights/gradients each cost `4P`; AdamW moments `8P` (SGD momentum zero).

| Model | Weights | Weights + gradients + AdamW moments | Planning implication |
|---|---:|---:|---|
| SmolLM2-135M | 0.5011 GiB | 2.0044 GiB | Loading, graphs, caches, integrity/save and uncertainty dominate the unchanged 19/20/24 GiB policies. Subsetting cannot authorize lower floors. |
| Pythia-14M | 0.0524 GiB | 0.2096 GiB | At length 128, one FP32 logits tensor is 24.56 MiB. Tentative CPU incremental allowance 6–10 GiB, plus 6 GiB reserve; no implemented/admitted profile. |
| Tiny byte LM | 0.3452 MiB | 1.3809 MiB | One `[1,128,257]` FP32 logits tensor is 128.5 KiB. Payload is tiny; interpreter/library and workspace overhead still need measurement. |

Propose a separately allowlisted CPU policy only for the exact random tiny
config, rejecting pretrained loading, larger shapes and device switches:
`I(B)=10W + 1 GiB CPU workspace + max(2 GiB,B) additional process headroom
+ 2 GiB uncertainty`, where `W=361,984` bytes and B is live process peak RSS.
Ten weight-equivalents cover weights, gradients, moments and conservative
copy/clipping/save staging. At B≤2 GiB this is **5.003371 GiB**, rounded up
plus the existing 6 GiB continuing reserve: **12 GiB preload availability**.
This is an unmeasured envelope for a different, roughly 1,486-times-smaller
model on CPU, not a lowered SmolLM2 guard or present execution clearance.

Before training, measure fresh free-plus-speculative physical availability,
process RSS/high-water, NORMAL pressure, free indicator, swap/page-outs and
disk space. Require 48 GiB capacity, both dynamic preload checks, ≥6 GiB
continuing availability, free indicator ≥20%, no swap/page-out growth, ≥4 GiB
free disk, complete fresh ordered telemetry, existing sampling/staleness stops,
and 30-minute per-stage timeout. Retain an 8 GiB process RSS stop; MPS counters
are explicitly inapplicable for CPU, never fabricated as measured zero. CPU is
the declared backend, not fallback. Keep all original guards, authorization,
environment-override protections and no-automatic-retry behavior unchanged.

After future authorization, first measure read-only load/forward/no-op and
new M0 checkpoint save/reload fidelity; then backward-only at cap 128 with
unchanged weights, gradient checks and sampled RSS/time. Clear gradients.
Measure median/max warm backward time over a small fixed rehearsal. The first
authorized optimizer update must be included in the 2,000-step budget and
measure AdamW allocation/time; backward-only cannot validate its peak. Stop
on envelope violation; do not continue an admission failure or partial run.

For tiny CPU planning, allow 0.05–1 s/update plus 5–10 minutes for initialization,
evaluation and integrity: roughly **10–45 minutes to acquisition**, and **30–90
minutes for all controls**, not a benchmark. Proceed only if measured timings
conservatively project each stage below 25 minutes, leaving timeout margin;
otherwise NO-GO at this fixed budget. For 2,000 SmolLM2/Pythia updates, allow
hours until measured: tensor-size ratios do not predict throughput. Benchmark
their actual backward/evaluation paths before any future scheduling claim.

## Minimum next work and preservation

The next task should implement only the small frozen CPU profile, byte adapter
and bounded loop above, with offline authorization/shape/budget/path/telemetry
checks. Then separately authorize the staged measurements and fixed acquisition
budget. **Stop after the first accepted M_all result before pursuing the
unlearning controls.** No need to complete LoRA mechanics or download Pythia
first. No training command for this proposed harness exists yet.

All future artifacts need a new ignored `outputs/local_synthetic_unlearning/`
namespace, exclusive creation, no symlink/overwrite/resume, sealed COMPLETE or
FAILED reports and checkpoint hashes. Keep M0/M_all immutable; branch controls
from verified inputs. Preserve every existing checkpoint, baseline, failure
tombstone, guard, profile, dependency lock and historical result; use ordinary
additive Git commits, never history rewriting.

Task 63 passed **167 offline tests**, with real backward, optimizer steps,
pretrained model loading and network connections trapped. Byte lengths, model
and memory arithmetic, seven local links, experiment JSON and the original
baseline validator pass; all **179 pre-existing files** in the preservation
ledger retain their hashes, sizes and modes, including cached weights and
historical artifacts. Original 19/20/24 GiB policies also pass regression checks.
Checks and preservation evidence are recorded in ignored
`outputs/task63-offline-checks.log`, `outputs/task63-offline-evidence.json` and
`outputs/task63-preservation-before.json`. No cloud resource was used.
