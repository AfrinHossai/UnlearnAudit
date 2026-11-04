# Task 52: Local SmolLM2 LoRA feasibility

**Decision: CONDITIONAL GO for preparing a separate local LoRA experiment;
execution is not cleared.** Rank-4 attention LoRA is a plausible lower-memory
path on Apple M4 Pro MPS. Compatibility is supported by metadata and source
inspection, not a PEFT runtime test. Its complete training/save memory remains
unmeasured. A new runner, dependency lock, baseline, reviewed admission policy
and explicit owner authorization are prerequisites to any adapter-weight update.

This assessment supersedes the next-experiment recommendations in Tasks 45 and
49: **stop repeated full-parameter attempts when the unchanged 24 GiB pilot
guard fails.** Leave that runner and guard intact. A failing admission is a
STOP, not evidence that another identical launch will help or that LoRA fits.
The next work is LoRA preparation and offline validation, followed, only when
its prerequisites pass, by one separately authorized mechanics experiment.

Task 52 loaded no model weights, constructed no full model, executed no backward
pass or optimizer step, installed no packages, changed no training settings,
and created no cloud resources. No training, acquisition or unlearning occurred
in this task. All memory reductions below are analytical estimates.

## Evidence and limits

Reviewed the actual Task 37 backward report, Task 40 `baseline-002.json`, the
[Task 45 scientific audit](smollm2_baseline_findings_task45.md) and evidence JSON,
the [Task 49 memory review](smollm2_pilot_memory_review.md), frozen Task 38 audit,
profiles, data, runner, telemetry, validators, and installed library source.
Starting HEAD was `33dd3bd` on clean `main`, tracking `origin/main`.

| Historical evidence | Finding | Limit for LoRA |
|---|---|---|
| Task 37 | One real backward of `-NLL(F)+NLL(R)` on FP32/MPS; finite, nonzero aggregate gradients covering 134,515,008 parameters; unchanged parameter hash/versions; gradients cleared. Actual sequences 74 and 42 tokens. | No optimizer or checkpoint save; no adapters. This establishes historical full-model autograd feasibility only. |
| Task 37 memory | Before-load availability 28.333603 GiB; minimum sampled availability 25.278580 GiB; sampled MPS driver maximum 2,659,057,664 bytes; OS RSS high-water mark 1,301,889,024 bytes. | Began above even the later 24 GiB floor. Sampling is not an exact peak or current admission evidence. |
| Task 40 baseline | Frozen FP32/MPS evaluation, zero updates; 0/64 development exact matches and alias recoveries, 0/12 utility exact matches. All 76 generations reached the 64-token budget. | Retrieval is at floor under this prompt/decoder; finite or improved full-answer NLL alone cannot establish acquisition or forgetting. |
| Task 40 memory | Before-load availability 19.924728 GiB; minimum sampled availability 14.861877 GiB; sampled MPS driver maximum 2,304,868,352 bytes; OS RSS high-water mark 2,676,031,488 bytes. | Inference only; does not establish training, clipping or saving fit. |
| Task 45 | Sources, checkpoint, encodings and baseline integrity pass; synthetic development facts and utility probes have limited scientific scope. | No held-out acquisition facts; paraphrases are exposed development data; utility exact match is insensitive at floor. |
| Tasks 48–51 | Task 49 records an owner-reported Task 48 memory-readiness failure and audits the 24 GiB formula. No distinct Task 48, 50 or 51 report/transcript was found in tracked files or local `outputs/` during Task 52. | Exact readings, failed stages and update counters for those attempts cannot be independently reconciled. Do not invent measurements or label admission failure physical OOM. |

Task 49's 24 GiB minimum is a conservative allowance-based policy: 17.255012
GiB incremental estimate, rounded upward, plus 6 GiB reserve. Its 4 GiB
clipping/step/serialization allowance explains the increase from Task 37's
20 GiB backward-only guard. The full-parameter runner already uses stateless
FP32 SGD, `foreach=False`, `fused=False`, non-reentrant activation checkpointing
and no KV cache. There is no Adam state to eliminate in that runner. Do not
change its optimizer, relax admission, purge caches or raise MPS watermarks.

Before any future update, recover available Task 48–51 STOP output and establish
whether a model was loaded or a step attempted. Missing reports cannot certify
zero updates. A new isolated experiment must not consume or resume any failed
checkpoint; unresolved indications of a previous partial update block use of
that artifact. Always start from the independently verified original base.

## Checkpoint, dependencies and MPS compatibility

The local checkpoint is **base** `HuggingFaceTB/SmolLM2-135M`, Apache-2.0,
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Cached `config.json`
declares `model_type=llama`, `LlamaForCausalLM`, 30 layers, hidden size 576,
9 query heads, 3 KV heads, head dimension 64, intermediate size 1536,
vocabulary 49,152 and tied embedding/head weights. BF16 is the stored weight
dtype; the proposed experiment deliberately uses FP32 for both base and adapters.

Reading only the safetensors JSON header, without tensor deserialization,
confirms 272 stored tensors and 134,515,008 unique parameter elements. Every
layer has `q_proj [576,576]`, `k_proj [192,576]`, `v_proj [192,576]`,
`o_proj [576,576]`, and three MLP projections with dimension sum 2,112 each.
Installed Transformers 4.57.1 Llama source constructs these as `nn.Linear`.
The grouped-query value projection must not be counted as a 576-square matrix.

PEFT 0.17.1 explicitly maps Llama to `q_proj` and `v_proj`; its ordinary LoRA
layer supports `nn.Linear`. These are good source-level compatibility signals.
Use an explicit target list and verify the exact 60 wrapped modules at runtime.
This is an inference of feasibility, not evidence of working MPS LoRA backward.
[PEFT 0.17.1 target mapping](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/utils/constants.py),
[LoRA layer implementation](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/layer.py).

| Component | Local metadata / assessment | Required next action |
|---|---|---|
| Python | `.venv-ml` is CPython 3.11.18. | Preserve the existing environment; use a separate project-local LoRA environment. |
| Core pins | Torch 2.9.1, Transformers 4.57.1, safetensors 0.8.0, tokenizers 0.22.2, huggingface-hub 0.36.2, NumPy 2.4.6. | Preserve `requirements-ml.txt` and `requirements-ml.lock`; constrain these exact versions in the separate lock. |
| PEFT | Not installed or pinned. Candidate for investigation: `peft==0.17.1`, rather than an unbounded latest install. | Resolve and hash-lock it separately; run import/API checks without weights. No installed compatibility claim. |
| Accelerate | Not installed. Candidate: `accelerate==1.10.1`. | Required by PEFT even with a manual single-MPS loop. Do not use automatic device placement, dispatch or offload. |
| psutil | Not installed; required by both candidates. | Choose an available compatible macOS arm64 wheel, pin/hash it and all added dependencies before execution. Resolution remains a blocker. |

The candidates' declared requirements allow Python ≥3.9, the pinned Torch,
Hub and safetensors versions; Accelerate allows NumPy `<3.0`. PEFT does not pin
a Transformers version. This establishes metadata consistency, not import or
operator correctness. Its `Cache`, `DynamicCache`, `EncoderDecoderCache`,
`HybridCache` and `PreTrainedModel` imports are present in the installed
Transformers exports by text inspection. All transitive imports still need a
real no-model smoke check after separate installation. Run `pip check` and
confirm no core version changed; preserve a complete candidate lock and wheel
hashes. No dependency resolution/download was attempted here.
[PEFT requirements](https://github.com/huggingface/peft/blob/v0.17.1/setup.py),
[Accelerate requirements](https://github.com/huggingface/accelerate/blob/v1.10.1/setup.py),
[PEFT imports](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py).

Ordinary FP32 LoRA uses linear operations and autograd available through the
MPS device model; it does not require CUDA or bitsandbytes. Keep actual MPS,
SDPA, deterministic algorithms, seed 2026, no autocast, no quantization,
`use_cache=False`, no compilation and `PYTORCH_ENABLE_MPS_FALLBACK=0`.
Do not call `prepare_model_for_kbit_training`. Unsupported or nondeterministic
MPS operations mean STOP; they are not permission for CPU fallback.
[PyTorch 2.9 MPS backend](https://docs.pytorch.org/docs/2.9/notes/mps.html).

Keep `use_reentrant=False` and RNG preservation for activation checkpointing.
Non-reentrant checkpointing can differentiate a graph whose inputs are frozen;
the reentrant variant has stricter gradient-input requirements. PEFT may add
an embedding-output `requires_grad` hook; audit and record it. Such a hook
must never unfreeze embedding weights or introduce base gradients. The future
pilot must demonstrate that all intended adapter tensors have finite gradients
and the base has none, rather than trusting checkpointing configuration alone.
[PyTorch checkpoint semantics](https://docs.pytorch.org/docs/2.9/checkpoint.html),
[PEFT checkpointing preparation](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py).

## Small configuration and memory estimates

Proposed separate profile: causal LM LoRA, `r=4`, `lora_alpha=8` (scale 2),
`target_modules=["q_proj", "v_proj"]` across all 30 layers,
`lora_dropout=0.0`, `bias="none"`, `modules_to_save=None`,
`init_lora_weights=True`, `fan_in_fan_out=False`, no DoRA or rsLoRA.
Freeze every original parameter, including norms, embeddings and tied head.
Optimize only adapter parameters; do not merge adapters into the base.

For one linear map, LoRA adds `r × (input_dim + output_dim)` parameters.
Here the count is `30 × 4 × ((576+576)+(576+192)) = 230,400`,
or **0.171282% of the 134,515,008 base parameters**. The combined model has
134,745,408 parameters; 0.170989% of that combined total is trainable.

| Configuration | Estimated trainable elements | FP32 adapter weights | FP32 adapter gradients |
|---|---:|---:|---:|
| Selected: rank 4, q/v | 230,400 | 921,600 bytes (0.878906 MiB) | 921,600 bytes |
| Rank 8, q/v | 460,800 | 1,843,200 bytes | 1,843,200 bytes |
| Rank 4, q/k/v/o | 460,800 | 1,843,200 bytes | 1,843,200 bytes |
| Rank 4, attention + all MLP projections | 1,221,120 | 4,884,480 bytes | 4,884,480 bytes |

Only the first configuration is proposed for the mechanics pilot. Larger
configurations would need their own frozen profile and admission review.

FP32 base weights remain **538,060,032 bytes (0.501107 GiB)**. Selected LoRA
removes an estimated **537,138,432 bytes** of full-base gradient payload,
about 99.829% of that gradient component. With stateless SGD, base + adapters
+ gradients total **539,903,232 bytes**, versus **1,076,120,064 bytes** for
full-parameter weights + gradients: about 49.83% less persistent tensor payload.
This is not a 99.8% reduction of total training memory. Both configurations
have zero persistent SGD momentum/moment tensors. Hypothetical FP32 AdamW
would add 1,843,200 bytes of adapter moments versus 1,076,120,064 bytes of
full-base moments; changing the existing optimizer is outside this proposal.

An adapter-only checkpoint's selected FP32 tensor payload is approximately
0.879 MiB, versus 513.134 MiB for unique FP32 base weights, before file metadata.
Base residency, CPU conversion, integrity hashing, activations and activation
gradients through frozen layers, logits/loss, MPS workspace and driver caches
remain. A `[1,128,49152]` FP32 logits tensor alone is 24 MiB; actual pilot
length 52 gives 9.75 MiB. Those are individual tensor sizes, not graph peaks.
Small adapter rank does not guarantee acquisition capacity or total memory fit.

### Separate initial admission proposal, not an approved threshold

Propose guard ID `smollm2_135m_fp32_mps_lora_r4_qv_one_step_v1` in a new runner.
Let `W=538,060,032`, `L=921,600`, `S=269,060,552` bytes, and `B` be the live
preload process lifetime peak RSS. An initial conservative envelope is:

```text
I_LoRA(B) = 3W + 2L + S
            + 4 GiB forward/backward workspace
            + 1 GiB tokenizer/integrity + 1 GiB Metal cache
            + max(1 GiB, B) additional process headroom
            + 1 GiB adapter clipping/step/save allowance
            + 4 GiB uncertainty
preload(B) = ceil(I_LoRA(B) / GiB) × GiB + 6 GiB continuing reserve
```

At `B≤1 GiB`, this is **14,769,985,736 bytes = 13.755621 GiB**, giving a
**proposed 20 GiB preload minimum**; at `B=3 GiB`, it becomes 22 GiB.
Require two fresh preload checks. This is a proposed experimental safety
envelope, not a measurement, calibrated minimum, promise of fit, or change to
the 24 GiB full-parameter policy. Even baseline-002's historical 19.924728 GiB
start would fail the proposed minimum. No current host readiness is asserted.

The independent justification is restricted operation scope: replace full-base
gradients with adapter gradients, clip/update only 230,400 elements, and save
only the adapter. A 1 GiB allowance is over 580 times the adapter weight-plus-
gradient payload; it also covers small norm/check temporaries and CPU staging.
It is nevertheless unmeasured. Retain the full 4 GiB graph/workspace and 4 GiB
uncertainty allowances, conservative three-copy base loading term, source
residency and 6 GiB reserve because LoRA does not remove those risks. The 3 GiB
smaller step/save allowance is a new policy assumption justified by eliminating
the full-base update/save path, not a measured 3 GiB saving. If implementation
ever stages/saves/merges a full base, this envelope is invalid and must STOP.
Review the policy and test its boundaries before it can admit a run; do not
lower it to match whatever availability happens to be present.

Retain the existing safety behavior in the new runner: Apple M4 Pro/arm64/MPS,
≥48 GiB physical host, ≥6 GiB free-plus-speculative availability throughout,
NORMAL pressure level 1, free indicator ≥20%, no cumulative swap growth or
page-out change, separate 8 GiB lifetime RSS and MPS driver caps, fresh complete
telemetry, 250 ms sampling with ≤2 s active gaps, 30-minute deadline, ≥4 GiB
free disk, allocator/environment validation, offline-only cached provenance,
no resume/download/fallback, and collision-safe new outputs with failed-run
tombstones. Persist early STOP reports even before model loading.

Use synchronized phase boundaries and report availability, current RSS, OS
lifetime peak RSS, MPS tensor and driver allocation separately. Driver bytes
include allocator cache and MPS framework allocations; these overlap RSS on
unified memory and must not be summed. Samples miss brief peaks, so disclose
sample gaps and retain uncertainty. Do not use `empty_cache`, system purges,
watermark overrides or memory-fraction changes to obtain admission.
[PyTorch driver accounting](https://docs.pytorch.org/docs/2.9/generated/torch.mps.driver_allocated_memory.html).

## Controlled acquisition and adapted LoRA-GradDiff

Reuse the hashed development pool: 32 public synthetic facts (16 objects with
two attributes), eight forget and 24 retain. Acquisition minimizes positive
answer-only causal NLL on **both** roles; role labels do not activate forgetting.
Keep plain `Question: {question}\nAnswer:\n` prompts, prompt-masked labels,
explicit EOS, batch one, maximum sequence length 128 and no truncation. Local
tokenizer-only reconstruction verified all 108 encodings, maximum QA length 57,
maximum prompt plus generation budget 93, and pilot `dev:09:0` length 52.

Keep the original 76-case evaluator and raw scores for historical comparison.
Its 32 direct cases are resubstitution; 32 paraphrases share facts and answers
and test wording transfer; 12 utility cases are untrained drift diagnostics.
They are not independent generalization or privacy tests. A separately
versioned evaluator must add target-span likelihood, generated token IDs and
finish reasons, answer validity, repetition and budget-exhaustion flags, plus
a frozen general-text token-weighted NLL panel and non-floor factual/cloze
controls. Freeze prompts, scoring and utility non-inferiority tolerances before
updates; run a fresh M0 baseline for every added metric. Never retrofit new
metrics into baseline-002 or select favorable answer spans after inspection.

Use these separately saved states and controls, sequentially with one resident
base model, all with the same frozen base revision and adapter configuration:

| State / control | Purpose |
|---|---|
| M0 with fresh zero-effect adapter | New LoRA baseline; compare to adapter-disabled M0 and historical evaluation within preregistered numerical tolerance. |
| M_all = base + adapter acquired on F∪R | Demonstrate actual target acquisition for both roles before unlearning. |
| M_R = base + adapter trained from the identical initial adapter on R only | Counterfactual retain-only acquisition reference, with matched retain exposure; report its smaller total step budget. |
| M_all no-op/reload | Quantify numerical, evaluation and serialization drift without updates. |
| Retain-only continuation from M_all | Same unlearning step budget and retain batches; isolates effects of the retain term. |
| LoRA-GradDiff from a copy of M_all's adapter artifact | Test forgetting while preserving retain acquisition and broader utility. Never modify the sealed M_all checkpoint. |

Preregister fixed budgets, ordering, seed(s), LR selection on development only,
and per-role acquisition gates. Existing longer-run proposals (direct recovery
≥0.8 for each role, forget improvement over M0 ≥0.5, forget paraphrase recovery
≥0.5, and NLL improvement for both roles) are starting proposals, not validated
LoRA thresholds. Require target-specific evidence and functioning utility gates
before adopting them. If q/v rank 4 cannot acquire both roles, unlearning is
scientifically NO-GO for that configuration. Do not count failure to learn as
successful forgetting. Larger rank, changed prompts or optimizer require a
separate profile, fresh baseline and authorization, not in-run tuning.

For adapter parameters `φ={A,B}` with base `θ0` fixed, explicitly define:

```text
W_effective = W0 + (alpha_LoRA/r) B A
J(φ) = -gamma × mean_answer_NLL(F; θ0,φ)
       + alpha_retain × mean_answer_NLL(R; θ0,φ)
gamma = alpha_retain = 1
```

Differentiate and update **only φ**, starting from the acquired M_all adapter,
not a newly initialized second adapter. `alpha_LoRA=8` is distinct from
`alpha_retain=1`. Keep answer masking, causal shift, EOS and equal-weight means
for the two roles; do not inadvertently weight retain 3× just because its
pool is larger. A future memory-conscious implementation may accumulate the
two signed gradients in sequential forwards/backwards with no intermediate
step and one final clip/update. It must verify scalar/gradient parity on small
fixtures first and retain fixed parameter values across both terms. The
one-positive-NLL mechanics pilot below does not measure this two-term workload.

The scalar sign convention matches the repository's audited
[pinned OpenUnlearning reference](graddiff_validation.md), but the optimization
domain is different. Official full-parameter GradDiff changes the full model;
this adapted experiment changes low-rank attention factors with all original
weights fixed, in a local FP32/SGD/MPS stack and a small synthetic acquisition
protocol. Factor optimization is constrained and has different geometry,
capacity and gradient trajectories. Matching scalar losses cannot establish
equivalent updates, official GradDiff reproduction, official TOFU scores or
certified/privacy-preserving erasure.

Measure paired forget recovery reduction/NLL increase relative to M_all, retain
recovery/NLL stability, distance to M_R and utility drift, per fact and by role,
including paraphrases, no-op and retain-only controls. Report uncertainty across
preregistered seeds and all failures. Adapter disabling recovers M0 and may
erase both acquired roles; it is an ablation, not selective unlearning. Retained
original adapters/checkpoints can recover the facts. Broader TOFU/holdout claims
remain blocked by the original protocol's alias/control preparation and must
not be used to tune this development experiment.

## Concrete next experiment and acceptance

**Next implementation action:** prepare a new LoRA-specific profile, dependency
lock, manual guarded runner and offline tests. Do not edit the full-parameter
runner or its authorization string. No runnable LoRA training CLI exists yet;
the following is a specification, not an instruction to invoke the old runner.

1. Reconcile available prior STOP evidence, verify base/data/baseline hashes,
   resolve the isolated environment, and pass no-model imports, dependency and
   offline gate checks. Test exact adapter selection/counts, frozen base, wrong
   authorization, admission boundaries and bad/stale telemetry, checkpoint
   failure/tampering, changed dependencies and baseline incompatibility. Tiny
   fixture gradient/parity tests belong to that future implementation task.
2. Implement and review the separate policy above, including reports for
   admission failure. Prepare a new immutable LoRA M0 evaluation with zero
   updates under its own read-only admission and explicit execution scope.
   Pin the old baseline SHA literally and validate it with its unchanged runner;
   a new schema must also pin LoRA runner/evaluator/dependency/profile hashes.
   Baseline-002 is provenance, not a substitute for a baseline under PEFT.
3. Only after these prerequisites and fresh admission, request explicit owner
   authorization for **one new LoRA adapter SGD step**. The owner must see the
   exact profile, guard, LR, baseline hashes and new output paths. A proposed
   distinct token is `SmolLM2-135M:development:LoRA-r4-qv:one-SGD-step`;
   a token in this document grants no permission and must not be accepted by
   the old runner. Creating LoRA factors does not authorize optimizing them.
4. The authorized mechanics pilot uses the selected rank-4 configuration,
   seed 2026, the fixed 52-token `dev:09:0` example, positive answer NLL,
   FP32 SGD at **1e-4**, momentum/dampening/weight decay zero, no scheduler,
   no accumulation, `foreach=False`, `fused=False`, `differentiable=False`,
   and global norm clipping at 1.0 over adapters only. This deliberately
   modest LR is a mechanics setting, not an acquisition calibration. Exactly
   one step is permitted; this task authorizes none.
5. Before the step, require finite loss/logits and local/model NLL parity,
   exactly 120 trainable A/B tensors totaling 230,400 elements on FP32/MPS,
   finite gradients on every intended trainable tensor, nonzero aggregate
   gradient, no base gradients and empty optimizer state. With default random A
   and zero B initialization, **A gradients may be zero at the first step**;
   do not require every tensor to change. Require nonzero B-side gradient and
   a finite changed adapter/effective update after the step, while all original
   base bytes and tied embedding/head storage remain unchanged. Check clipped
   norm and optimizer state before/after. Missing/disconnected gradients STOP.
   [PEFT initialization](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/layer.py).
6. Record sampler readings and synchronized boundaries for CPU load, MPS load,
   adapter injection, hashing, forward, backward, finite checks, clipping,
   optimizer step, gradient cleanup, frozen evaluation, adapter serialization,
   and completion. Preserve raw stdout/stderr, timestamps, policy, pressure,
   swap/page-outs, counters, maxima and sampling limitations. A STOP before a
   report must still leave auditable evidence. No retries or resume.
7. Clear gradients with `set_to_none=True`, release the graph, freeze the
   adapter for evaluation, and run all 76 unchanged probes, paired to the new
   baseline, plus any separately frozen controls. Check finite results and base
   fingerprints. Save **adapter only**, with `save_embedding_layers=False`,
   safetensors and exact expected A/B keys/shapes/dtype; no `modules_to_save`,
   full-base save or merge. Include adapter config, base revision and SHA,
   adapter hash, initial/final fingerprints and profile/evaluator/lock hashes.
   Adapter format requires both config and weights plus the original base.
   [PEFT checkpoint format](https://huggingface.co/docs/peft/v0.17.0/developer_guides/checkpoint).
8. Use a newly reserved ignored run directory, e.g.
   `outputs/smollm2_lora/task52-one-step-001/` after a collision check. Hash,
   fsync and seal files/report; write COMPLETE last only after checks pass.
   Failed/partial artifacts remain tombstones, never usable checkpoints.
   A separately guarded read-only reload, with no second resident model, must
   verify exact adapter bytes/config/base pairing, output fidelity within
   preregistered tolerance and unchanged base before scientific use. Do not
   reuse the old full-model checkpoint validator for adapter files.

A mechanics pass means one finite adapter update, unchanged base/cache,
measured end-to-end memory within policy, complete paired evaluation and an
integrity-verified sealed artifact. A STOP after a step is a failed pilot with
an attempted update, not zero training. Save/reload validation and memory
review must pass before longer work. No recovery gain is required for mechanics;
one step cannot demonstrate all-fact acquisition, clear LoRA-GradDiff, or
authorize further steps. Unlearning later needs separate workload admission,
scientific gates and update authorization.

## Preservation, offline verification and remaining blockers

Task 52 adds this assessment and a README pointer only. All executable code,
existing optimizer settings, dependency pins, guards, frozen sources/evaluations,
baselines and checkpoints remain unchanged. SHA-256 rechecks include:

| Artifact | SHA-256 |
|---|---|
| `outputs/smollm2_acquisition_task38/baseline-002.json` (479,156 bytes; mode `0444`) | `61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff` |
| Original cached `model.safetensors` (269,060,552 bytes) | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| `src/small_model_acquisition.py` | `92c3896e03d1599b007131fc255810fa32b5893a396b21203ef783004b8c01cd` |
| Frozen Task 38 manifest | `60e9872e86e8bd98b1a281866f686c7ff3ed41ca3c8b5709e5b4c4b1daa239de` |

The historical in-memory parameter fingerprint is
`709264bff244d021ac31b1fcf9e1733b9af31f75032e854e28db62287b8cd072`;
it was read from the reports, not recomputed by loading weights.
Original `artifact_path`, `frozen_manifest` and `validate_baseline` passed with
Torch/Transformers/PEFT/Accelerate imports blocked. Tokenizer-only reconstruction
reproduced encoding digest
`8f2f40dd285943b8fa4358c318ed8c99557845d0d35cbfcb6762142464d4a4f1`.
Safetensors header arithmetic independently reproduced all parameter counts
above. Evidence is in ignored `outputs/task52-offline-evidence.json`.

**105 offline tests passed**, zero failures, errors or skips, using
`unittest.defaultTestLoader.discover('tests', pattern='test_small_model*.py')`
with `.venv-ml/bin/python -B`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`TOKENIZERS_PARALLELISM=false` and `PYTORCH_ENABLE_MPS_FALLBACK=0`.
During the suite, real `Tensor.backward`, `autograd.backward`, `autograd.grad`,
SGD/AdamW `step`, `torch.load`, safetensors `load_file` and Transformers
`from_pretrained` model loaders were patched to raise. Existing tests use
mocked orchestration and small CPU diagnostics with assigned gradients; no
real backward/update or checkpoint-weight load ran. Log:
`outputs/task52-offline-tests.log`. These checks verify existing safeguards,
not LoRA execution. Markdown links, arithmetic and `git diff --check` are also
checked before commit.

The prior Task 49 preservation ledger matched all 150 entries before work.
A new ignored `outputs/task52-preservation-before.json` records hashes, sizes
and modes for 155 pre-existing tracked/cache/result/output files. Recheck all
entries except the deliberately edited README before completion; no frozen
artifact is exempt. Preserve Git ancestry through a normal additive commit and
push, with no reset, rebase, amendment or force push.

Remaining blockers: missing separately auditable Task 48/50/51 failure evidence;
uninstalled/unlocked PEFT dependencies and untested API/MPS behavior; absent
LoRA runner, adapter-aware validators and reviewed admission implementation;
fresh LoRA/no-op baseline and useful non-floor utility controls; unmeasured
step/save/reload memory; fresh physical readiness; and explicit owner update
authorization. Longer acquisition additionally needs preregistered success
thresholds and demonstrated learning of both roles. LoRA-GradDiff additionally
needs acquired checkpoints, matched controls and its own workload review.

**CONDITIONAL GO** therefore applies to this separate staged LoRA path.
Execution remains blocked until its prerequisites pass; no training or
unlearning result is claimed.
