# Task 36: local Apple M4 Pro training decision

**CONDITIONAL GO for a local SmolLM2-135M research route; NO-GO for weight
updates today.** Recommend full-parameter FP32 SGD with zero momentum,
non-reentrant gradient checkpointing and sequential forget/retain backward
passes, starting with the existing 32-fact development experiment. Proceed to
the frozen 40-forget / 120-retain TOFU pilot only after measured MPS training
admission, development calibration and evaluation controls pass. Keep the
original 512-token experiment cap and its required stress validation.

The Mac has not been shown incapable of this workload. Its failed rehearsal
was a **pre-load policy rejection**, not an OOM or an autograd failure. Equally,
successful inference does not establish training feasibility. No training,
backward pass, optimizer construction/step, model load, new model download or
cloud resource was used in Task 36. Colab remains an unused contingency;
there is no recommendation to move to it now.

## Evidence actually available

The owner identifies the machine as an Apple M4 Pro. The saved reports identify
an arm64 Mac with **51,539,607,552 bytes = 48 GiB** physical capacity, macOS
26.6.2, Torch 2.9.1, Transformers 4.57.1 and Hub 0.36.2. These are historical
report observations, not fresh availability or present-day admission readings.
MPS is hidden in the restricted process used for this assessment; that does
not negate the saved physical-Mac inference result.

The common model is `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, **134,515,008** unique parameters,
vocabulary 49,152, hidden size 576, thirty layers, nine query heads and three
KV heads. The cached model file is 269,060,552 bytes; its disk representation
does not equal FP32 training residency. Original profile digest:
`260db810b3b1fbd4da0870725bcb7cac9a2ae4a472dbdb50c089007d35b711eb`.
See the [frozen profile](../experiments/smollm2_135m_tofu_controlled.json),
[manifest](../experiments/smollm2_135m_tofu_manifest.json),
[controlled experiment plan](smollm2_tofu_experiment.md) and
[validated GradDiff arithmetic](graddiff_validation.md).

| Evidence | Observed result | What it does not establish |
|---|---|---|
| Task 29 physical inference, `outputs/smollm2_readonly_task27_first.json` | `readonly_measurement_passed_not_training_clearance`; actual MPS, FP32, frozen model, one 17-token forward; model/local NLL both 3.831789255142212 | Backward kernels, gradients, optimizer behavior, generation, long contexts or training peak |
| Same inference trace | 16.805878 GiB available before load; minimum sampled availability 15.339813 GiB; sampled current RSS maximum 0.836212 GiB; OS lifetime peak RSS 1.086746 GiB | A bound on other workloads or all transient peaks |
| Same inference trace | Sampled MPS tensor maximum 0.504222 GiB, driver maximum 1.067154 GiB; normal pressure; sampled swap/page-outs zero | Independent additive memory pools: RSS, tensor and driver allocations overlap in unified memory |
| Task 32 attempt, `outputs/smollm2_backward_task32_first.json` | `STOP`: 17,073,176,576 bytes = **15.900635 GiB** available against a 20 GiB minimum | Physical inability to execute a backward pass |
| Same failed attempt | One baseline sample; `model_loaded=false`, actual device null, zero forwards/backwards/optimizer steps/updates/checkpoint writes | Any measured backward peak, successful checkpoint integrity check or stable pressure over time |

The inference trace has nine samples and a maximum gap of 10.157286 seconds
including setup. Sampled maxima are lower bounds on peaks. Its parameter
version/freeze checks are weaker than hashing every in-memory parameter byte.
The backward report's `checkpoint_unchanged=false` is an uncompleted verification
flag, not evidence that a checkpoint was modified: the run stopped before load.
The 95% `memory_pressure -Q` indicator is not 95% of physical RAM available to
this process and cannot replace the separate free-plus-speculative counter.

Audited report SHA-256 values, retained unchanged:

- Inference: `8df89af2e27aae9f75ffe577e29e58322522f442d6be359bfa8d40a592375a74`.
- Backward STOP: `cc10d1e04b422f988c73ae26be46f7749bf812268c65e6d9260a476ec0e39692`.

## Why the backward floor is 20 GiB

The 20 GiB value is **a conservative model/operation-specific policy**, not a
measured SmolLM2 requirement. It is also not the old Llama 32 GiB value copied
unchanged. Its source is
[`rehearsal_memory_policy()`](../src/small_model_backward.py), enforced by
`validate_sample()` before checkpoint checking/model loading and again before
loading. The [Task 30 rationale](smollm2_backward_rehearsal_task30.md) applies to
batch one per split, at most 128 tokens, two simultaneously retained forward
graphs, one combined backward, FP32, cache disabled and no optimizer or save.

Let `W = 4 × 134,515,008 = 538,060,032` bytes, `F = 269,060,552` bytes and
`B` be freshly observed baseline process lifetime peak RSS. The exact budget is:

| Term | Bytes / allowance | Status |
|---|---:|---|
| Three FP32 weight copies | `3W = 1,614,180,096` | Loading/conversion/transfer allowance; coexistence unmeasured |
| Full FP32 gradients | `W = 538,060,032` | Analytical tensor size; physical backward allocation unmeasured |
| Whole source-file residency | `F = 269,060,552` | Conservative residency allowance |
| Two forward graphs and backward workspace | 4 GiB | Unmeasured activations, logits/loss copies, SDPA/MLP/recompute temporaries |
| Tokenizer/host/integrity work | 1 GiB | Unmeasured allowance |
| Metal/runtime caches | 1 GiB | Unmeasured allowance |
| Additional process headroom | `max(1 GiB, B)` | Extra conservative headroom; baseline RSS already affects available pages |
| Uncertainty | 4 GiB | Policy reserve, not a statistical confidence bound |
| Continuing OS/other-process reserve | 6 GiB | Must remain available throughout the operation |

For `B <= 1 GiB`, incremental terms total **14,232,460,744 bytes =
13.255012 GiB**. Round upward to 14 GiB, then add the 6 GiB continuing reserve:
**20 GiB**. At `B = 3 GiB`, the required preload becomes 22 GiB. The failed
attempt was **4.099365 GiB below** the minimum, not 4.10 GiB above a measured
hardware limit. These allowances combine conservative phase estimates; they
are not proof that all components coexist at their maxima. Neither keeping
them nor subtracting apparent overlap establishes a measured upper bound.

Inference cannot justify substituting its ~1.07 GiB sampled driver allocation
for this budget: autograd saves tensors, gradients occupy another ~0.50 GiB,
checkpointing recomputes operations, and clipping/optimizer/save transients
were absent. **Do not edit the 20 GiB floor, claim cached/compressed/inactive
pages as extra free memory, or select a smaller number to pass Task 32.**

The relevant policies remain separate and unchanged:

| Policy | Binding boundary / scope |
|---|---|
| SmolLM2 frozen single-forward admission | At least 16 GiB preload, potentially higher with large `B`; read-only at <=128 tokens, not training |
| SmolLM2 one-pair backward rehearsal | At least 20 GiB preload, potentially higher with `B`; no optimizer/save |
| Original Llama/general SGD runner and inherited small-model readiness | 32 GiB preload; retained in the old preparation/readiness path |
| Original AdamW runner | 42 GiB preload |
| Shared physical safeguards | At least 48 GiB capacity, 6 GiB continuing free-plus-speculative availability, >=20% free indicator, normal pressure, valid MPS telemetry |
| Task 24 small-model training assessor | Approved threshold null; assessment only; original 32 GiB baseline plus cold/warm length-512 evidence contract; never grants admission |
| Colab CUDA profile | Separate host/VRAM policy; inapplicable to this unified-memory Mac |

The [Task 24 audit](smollm2_memory_admission.md) traces 32 GiB to the much larger
Llama tensor/workspace budget. That is also policy, not a measured minimum for
SmolLM2. Its safeguards are not changed by this decision. In particular, the
existing Llama executable must not be repointed to SmolLM2 to avoid its guard.
Torch's [MPS allocator documentation](https://docs.pytorch.org/docs/2.9/mps_environment_variables.html)
distinguishes allocator limits from physical availability. Keep CPU fallback
disabled and allocator/fast-math overrides unset; disabling the high watermark
can expose the system to OOM and is not a feasibility strategy.

## Compare the three local options

All memory arithmetic below assumes FP32, one resident model, batch one,
separate mean answer-only losses and no KV cache. **Tensor floors are estimates,
not measured peaks or admission thresholds.** Nothing here measures throughput,
training duration, MPS backward compatibility or learning success.

| Option | Memory and optimization | MPS / engineering constraints | Scientific validity and feasibility |
|---|---|---|---|
| **Full-parameter SGD** | FP32 weights + gradients = **1,076,120,064 bytes / 1.002215 GiB**; zero-momentum SGD has no persistent moment buffers. Existing length-512 working plan is roughly **4–7 GiB**, including speculative workspace/runtime allowances, excluding extra load/save copies and OS reserve. Sequential backward frees each graph before the next. | Existing pinned FP32 forward works; exact backward/checkpointing/clipping/SGD operations still unmeasured on this model/Mac. Use `foreach=False`, `fused=False`, no AMP, momentum, weight decay or scheduler; validate finite gradients and parameters. Needs a separate local SmolLM2 training orchestrator. | Preserves the full parameter domain of GradDiff. SGD/acquisition/backbone changes still make this a custom controlled study. Best primary route given the small state floor and fewer new dependencies; **conditional**, not cleared to execute. |
| **FP32 LoRA/PEFT, adapter-only GradDiff** | Example rank-8 adapters on `q_proj`/`v_proj` in all thirty layers: 460,800 trainable parameters; frozen base + adapter weights/gradients about **0.504541 GiB** with SGD or **0.507974 GiB** including FP32 Adam moments. Activations, backward work and framework memory remain unmeasured. | Plain floating-point linear adapters avoid introducing a quantization backend. Need separately pinned/tested PEFT integration, actual module selection, frozen-base checks, checkpointing/gradient-flow validation and adapter save/reload tests. PEFT is not installed in the reviewed ML venv. Do not assume CUDA paged/fused optimizers or a quantization package works in this pinned MPS stack. | Valid only as a clearly named **LoRA-adapted GradDiff behavioral study**. Optimize adapters for both acquisition and unlearning, with matched adapter controls. A frozen base is unchanged; if acquisition was full-parameter, removing a later forgetting adapter restores the acquired model. If acquisition was adapter-only, removing the adapter returns the original base, not necessarily the acquired facts. Smaller update domain may fail acquisition or forgetting. A contingency, not evidence of erasing base knowledge. |
| **Smaller-scale controlled acquisition** | Use the existing **8 forget + 24 retain development facts** first. Full-parameter state remains **1.002215 GiB**. Fewer rows/steps reduce total work, not model/gradient memory. Shorter actual sequences reduce activation work; changing only a declared cap does not shorten already short inputs. | Same MPS backward/SGD requirements as full training; no guard exemption. Fixed 640-step acquisition, 480-step retain reference and eight-step continuation budgets already exist as proposals. Acquisition is unproven; stop if gates fail. | Best first scientific pilot combined with full SGD, not a third optimizer. It can demonstrate real learning/forgetting on fictional object attributes but is **not TOFU evidence**. Preserve independent wording, retain controls and utility measures. Its rates may transfer poorly to long author answers. |

For the LoRA example, query output width is 576; KV output width is
`(576 / 9) × 3 = 192`. At rank 8:

```text
A = 30 × 8 × [(576 + 576) + (576 + 192)] = 460,800
adapter weights + gradients = 8A bytes
frozen-base + SGD tensor floor = 4P + 8A = 541,746,432 bytes
frozen-base + Adam tensor floor = 4P + 16A = 545,432,832 bytes
```

An explicit contingency would use `r=8`, `lora_alpha=16`, dropout zero,
`bias="none"`, only `q_proj`/`v_proj`, no modules-to-save and no quantization.
Its **adapted** objective is
`L(phi) = -NLL_F(theta_frozen, phi) + NLL_R(theta_frozen, phi)`, optimized only
over adapter parameters `phi`. Acquire and unlearn in the same adapter domain;
include no-op, retain-only, independently acquired retain adapter and
adapter-disabled/reset controls. A disabled adapter is an intervention control,
not a replacement for the retain-trained reference or evidence of base erasure.

The SGD floor saving is only **0.497674 GiB** versus full SGD; savings against
full Adam would be larger. It cannot explain away the historical ~4.10 GiB
admission shortfall or remove uncertainty about activations. Actual savings
could differ with graph retention and kernels; no fixed peak reduction is
claimed. This count excludes biases, modules-to-save, embedding/head training
and extra targets. Those additions require new arithmetic and measurements.

LoRA's frozen-base low-rank parameterization is described in the
[original paper](https://arxiv.org/abs/2106.09685); PEFT's
[versioned LoRA configuration reference](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora)
documents explicit target modules and initialization. These sources explain
the method, not compatibility of a PEFT release with this Mac. Standard zero-B
initialization can give initially zero A-gradients; a future adapter validator
must distinguish that from missing gradient flow and must verify nonzero useful
adapter gradients overall. Existing strict full-model runtime validators are
not adapter validators. Do not label an adapter experiment full-model GradDiff.

## Concrete recommended route to actual before/after results

Use **full-parameter FP32 zero-momentum SGD on MPS**, with the existing 32-fact
development corpus as the first learning experiment, then the frozen TOFU
40/120 study if gates pass. Keep seed 2026, batch/accumulation one, SDPA,
non-reentrant checkpointing with RNG preservation, no cache, norm clipping 1.0,
existing plain-QA format and prompt/padding masks. No learning rate is chosen
in this assessment: existing null rates and calibration blockers remain null.

Minimize the validated `-NLL(forget) + NLL(retain)` with separate supervised-token
means. Accumulate `-NLL(F).backward()` and `+NLL(R).backward()` sequentially,
then perform **one** combined clipping and SGD update; never zero gradients or
update parameters between contributions. This preserves the gradient of the
same objective for corresponding forwards/RNG and fixed parameters, subject
to floating-point effects. The repository has prior tiny-fixture equivalence
coverage, but this task did not rerun any backward test. The current SmolLM2
rehearsal instead retains both graphs and calls backward once: its pass would
be an initial autograd observation, not validation of the proposed sequential
training implementation. See [`backward_pair()`](../src/graddiff_real_train.py)
for reference arithmetic, not a SmolLM2 execution entrypoint. Torch's
[SGD definition](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html)
supports the zero-momentum state rationale; it is not a hardware-fit guarantee.

Offline revalidation in Task 36 found all **368** prepared QA encodings at most
119 tokens including EOS. Direct acquisition examples top out at 102 forget /
94 retain tokens; development direct/paraphrase maxima are 53/57. With dynamic
batch-one collation, a 512-token cap does not pad these inputs to 512. A single
FP32 logits tensor is 24 MiB at length 128 and 96 MiB at length 512; these are
only tensor sizes, not full activation footprints. Thus do not promise that a
smaller fact count or a cap change will make the already short failed rehearsal
fit. Retain actual-length **and length-512 stress** measurement before clearing
the existing experiment. A future deliberately bounded-128 protocol would need
a new profile/evidence contract and explicitly narrower claims; this task
neither creates it nor silently changes the original 512-token protocol.

The eventual scientific sequence remains:

1. Freeze the scored evaluator, meaningful reviewed key-fact aliases and
   refusal/repetition rules before model outputs; no target-informed annotation.
   Calibrate acquisition rates `1e-4, 3e-4, 1e-3` and unlearning rates
   `1e-5, 3e-5, 1e-4` only on the separate development experiment, using its
   preregistered lowest-passing-rate rule and fresh base for every candidate.
   If no candidate passes, report failure and stop. Do not promise calibration
   success or use target TOFU scores to choose rates.
2. For TOFU, evaluate immutable M0, fit M_all for the fixed 3,200-step budget,
   then evaluate M_all with the same prompts/scorer. Independently fit M_R from
   M0 on retain only for 2,400 steps, matching per-retain exposure. Development
   checkpoints must never initialize the TOFU runs.
3. Require the original acquisition gates: direct forget/retain key-fact
   recovery >=0.8, forget gain over M0 >=0.5, forget paraphrase recovery >=0.5,
   and lower forget/retain NLL. Failing acquisition is not unlearning success.
4. Branch independently from the fixed M_all: unchanged/no-op; forty-step
   retain-only continuation; forty-step GradDiff. Use identical manifest-pinned
   retain draws, learning rate and budgets for the continuation branches.
   Select fixed final checkpoints, never the best observed forget score.
5. Report separate pre/post forget and retain NLL, direct/paraphrase key-fact
   recovery, normalized exact match/F1, empty/refusal/repetition rates, unseen
   retain and utility drift, matched-continuation effects and distance to M_R.
   Use all original proposed gates: >=0.25 absolute forget direct **and**
   paraphrase recovery drop, rising forget NLL, improvement beyond retain-only
   control; retain recovery/F1 drop <=0.05, retain/utility NLL rise <=0.1 and
   degeneration-rate increase <=0.05. Report negative outcomes and semantic
   review; these are project proposals, not official TOFU standards.

Direct forget/retain evaluation intentionally reuses acquired questions;
paraphrases probe the same facts, not independent author generalization. Preserve
all separate sets: 40 forget direct, 120 retain direct, 40 forget paraphrases,
twelve aligned retain paraphrases, forty unseen retain and twelve utility probes.
Keep sealed holdout predictions out of tuning. Small correlated author samples,
one seed, limited utility coverage, synthetic attributes and a different backbone
limit claims. These diagnostics are not canonical Truth Ratio, KS Forget Quality
or Model Utility, membership-inference resistance or proof of irreversible
erasure. Even full-model forgetting can reflect interference or broad collapse.

## Minimum next physical experiment and its admission policy

**Next experiment: one successful run of the existing, unchanged guarded
128-token SmolLM2 backward-only rehearsal in a fresh normal Mac Terminal.**
This is a future separately authorized action, **not an action performed by
Task 36**. It uses the first fixed pair `forget01:0000` / `retain99:0501`
(74/42 tokens), two forwards, one combined backward, no optimizer, no clipping,
no model checkpoint write and no downloads. It is the smallest existing physical
probe that can answer the missing autograd/gradient question without updates.

For this probe retain guard
`smollm2_135m_fp32_mps_one_pair_backward_only_v1`: the **separately justified
20 GiB-plus-baseline preload budget above**, at least 48 GiB total and at least
6 GiB continuing availability. It is independent of inference's 16 GiB and
Llama's 32/42 GiB policies; none is lowered or bypassed. Owner-chosen manual
closure of unused applications may create headroom; no particular application
or reclaimable cache quantity is assumed. Do not kill processes, purge caches,
change allocator limits or retry automatically. At the historical 15.90 GiB
reading, the correct outcome remains STOP. Only fresh readings can admit a run.

The existing runner already enforces two fresh preload observations, refreshed
admission after tokenizer work, hash-pinned local checkpoint/data, an explicit
MPS device without fallback, background monitoring at a 250 ms target,
synchronized boundaries and a 180-second active watchdog. Require:

- Available memory never below 6 GiB, >=20% pressure indicator and kernel
  NORMAL=1. Manually stop for Activity Monitor yellow/red pressure.
- No adjacent-sample swap increase or any page-out change; unchanged capacity;
  no missing/invalid/stale (>2 s), future or unordered counters, zero-duration
  readings or active-monitor gaps >2 s. Rounded swap counters may miss small
  growth. A cooperative watchdog cannot guarantee preemption inside a kernel.
- No OOM, unsupported MPS/deterministic kernel, wrong device, fallback,
  checkpoint/config/data drift, uncertain answer masks or sequence truncation.
- Loss parity within the existing tolerance, finite losses/objective and all
  intended FP32 gradients, at least one nonzero gradient, unchanged parameter
  byte hashes and versions, cleared gradients and unchanged original checkpoint.

After a later explicit authorization, the existing command would be:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
env -u PYTORCH_MPS_HIGH_WATERMARK_RATIO -u PYTORCH_MPS_LOW_WATERMARK_RATIO \
  -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-ml/bin/python -B -m src.small_model_backward \
  --run-backward-rehearsal \
  --report-path outputs/smollm2_backward_after_task36_first.json
```

Use a new filename if that path already exists. Exit 0 and a complete
`backward_rehearsal_passed_not_training_clearance` report are required; a missing
or STOP report is not success. The passing report must show actual MPS, exactly
one backward, integrity checks passed, no optimizer/updates/checkpoint writes
and `training_authorized=false`. Do not interpret successful imports or the old
inference report as a substitute. The command was **not executed in Task 36**.

## What a pass still leaves before any training

A single short backward is the minimum next observation, **not training
clearance**. A subsequent implementation/validation task must prepare a separate
SmolLM2 MPS orchestrator and model-specific admission contract, verify sequential
gradients, actual-length and 512-token backward, combined clipping, scratch
safe-serialization/reload, teardown and finite single-model state. No existing
Llama runner or CUDA workflow should be repurposed by changing device strings.
Use pinned local weights; no new model is needed.

For a future training-memory review, obtain at least three cold/warm trials
(cold = fresh process, not OS cache purging) with raw traces spanning those
operations and no optimizer steps. Separate observed maxima from allowances.
Use the maximum of baseline-relative available-memory decline, RSS peak growth
and driver growth, not their sum. Add uncertainty of at least
`max(4 GiB, 25% of worst excursion)`, the 6 GiB continuing reserve and explicit
allowances for any unmeasured optimizer/update or save transients. No-gradient
SGD state does not mean zero step temporaries; `foreach=False` avoids a
tensor-list implementation but still needs validation. An initial authorized
one-step update, only after policy adoption, must itself have finite-state,
memory and rollback checks before longer acquisition.

No numeric **training** admission value is approved here. Preparing a new
collector requires its own conservative operation-specific preload derivation
before execution, never a threshold chosen to match current free RAM. A proposed
threshold remains unapproved until evidence review, code/path tests and explicit
update authorization. The unchanged Task 24 assessor still demands its original
32 GiB baseline, cold/warm trials and length-512 stages; this short rehearsal
cannot satisfy it, including its <=100 ms trace contract versus this probe's
250 ms sampling target. If a separate model-specific contract is later adopted, it
must be explicitly versioned and reviewed, not forged into Task 24's schema or
treated as an automatic waiver. No inference-to-training approval shortcut is
introduced by this recommendation.

Local training is ruled out operationally only if admitted workloads repeatedly
fail relevant memory/pressure/kernel checks, or reasonable owner-controlled
conditions cannot satisfy a reviewed admission policy. Continued preload STOPs
alone do not prove insufficient physical capacity. If full SGD fails admitted
physical validation, review measured causes before considering rank-8 FP32
LoRA with its own adapted scientific question, dependencies and admission
policy. If that also proves infeasible or scientifically inadequate, discuss an
alternative with the owner; do not launch Colab or a paid resource automatically.

## Local artifacts, evaluator and Streamlit completion path

Training remains blocked on the local orchestrator/admission evidence,
development rate selection, reviewed target aliases, scored evaluation and
checkpoint recovery. The prepared CUDA runner contains reusable design ideas,
but its hard CUDA checks and Drive paths do not implement this Mac workflow.
No local acquisition/unlearning entrypoint or calibration result is created by
this decision document.

Keep immutable M0 and each final parent/control checkpoint in distinct new
Git-ignored directories under `outputs/smollm2_135m_tofu/<run-id>/`. Never
overwrite cached publisher files, historical baseline JSON or another run.
Load one model at a time; do not retain M0, M_all and M_R simultaneously.
Bind artifacts to model/tokenizer/data/profile/evaluator/software hashes,
seed, schedules, parent checkpoint and actual device. Save optimizer/RNG/step
cursor transactionally with checksums and an explicit completion marker; a
running log, partial write or disconnect is not a completed experiment. Validate
restart/rollback before long runs. Check local disk headroom and save copies as
part of memory validation. No checkpoint/history rewriting is needed.

Build the final local Streamlit demo **after genuine fixed evaluations exist**.
Its first version can read saved pre/post responses and separate forget/retain
metrics with provenance, controls, limitations and honest incomplete/failed
statuses; it does not need a training process or multiple live model copies.
Do not serve the demo during memory-sensitive training. Live generation would
need its own bounded inference/generation admission, since the historical
17-token single forward did not test generation. Streamlit is not installed in
the reviewed ML venv; a later project-local UI dependency lock and artifact-schema
tests are needed. No demo or fabricated result is supplied in Task 36.

## Offline validation and preservation

This task uses configuration/document checks that do **not** execute synthetic
or real backward/optimizer tests. The relevant checks are:

```sh
PYTHONPATH=tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m unittest \
  test_small_model_experiment test_small_model_memory test_inspect_tofu \
  test_evaluate_tofu_baseline test_evaluate_tofu_paired \
  test_graddiff_experiment_plan -v

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_experiment validate --verify-tokenizer
```

The tokenizer command verifies local metadata/config/API and all 368 encodings;
it does not construct a model, load weights or run inference. Report hashes,
20 GiB budget arithmetic, GQA-aware LoRA counts, all experiment JSON files,
Markdown links, whitespace and source/configuration/evidence preservation are
checked separately. The cached SmolLM2 weight SHA-256 was verified by streaming
file hashing only, without deserialization:
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.

Completed in Task 36: **56 selected offline tests passed with no skips**,
all 368 local-tokenizer encodings validated, all eleven experiment JSON files
parsed, 39 relative documentation links resolved and the budget/LoRA arithmetic
reproduced. The test harness explicitly denied Tensor/autograd backward calls
and optimizer construction; it did not run the broader suites that exercise
synthetic updates. SHA-256 comparisons preserved all 47 audited runtime-source,
test, configuration, dependency, baseline and physical-report files. The only
tracked changes are this decision document and its README link.

Existing five-case and paired Llama baseline hashes remain
`f46e928a6b8384f86bf7294ca034b7fa3c386d510991921e49822b0e6ce2aa59` and
`1473f6abc0c42164ee55dcf27328ab0b72c9e86e12c15764b71800e7462e8a0e`.
No checkpoint, report, guard, experiment configuration, dependency lock or
runtime source is edited. The strategy decision is **CONDITIONAL GO**, while
actual local weight updates remain **NO-GO pending the stated evidence and
scientific prerequisites**.
