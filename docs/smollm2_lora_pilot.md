# Task 53: Guarded native SmolLM2 LoRA pilot

**Decision: CONDITIONAL GO for the first separately authorized local LoRA
optimizer step, after the new read-only LoRA baseline passes and fresh memory
admission succeeds. No execution clearance is granted by this preparation.**
The runner, offline tests and Terminal procedure are implemented. No real
checkpoint was loaded, backward pass or optimizer step executed, model
download performed, package installed, or cloud resource created in Task 53.
Fact acquisition and machine unlearning have not occurred.

## Implementation and evidence

The new [runner](../src/small_model_lora.py),
[native adapter](../src/lora_adapter.py) and
[frozen profile](../experiments/smollm2_135m_lora_pilot.json) are separate from
the full-parameter runner. Its SHA-256 remains
`92c3896e03d1599b007131fc255810fa32b5893a396b21203ef783004b8c01cd`;
its optimizer, authorization token, 24 GiB minimum and other guards are intact.
Do not repeatedly relaunch it when admission fails.

Reviewed [Task 52](smollm2_lora_feasibility.md), the actual successful
`outputs/smollm2_backward_task37.json`, verified `baseline-002.json` using its
original validator, and [Task 45's evaluation audit](smollm2_baseline_findings_task45.md).

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Task 37 | FP32/MPS full-base backward, finite nonzero gradients, unchanged weights; sequences 74/42 tokens. Starting availability 28.333603 GiB, sampled driver maximum 2,659,057,664 bytes, OS RSS high-water mark 1,301,889,024 bytes. | No optimizer or save; no native LoRA runtime proof or measured LoRA requirement. |
| Task 40 baseline-002 | Complete frozen 76-case evaluation, zero updates. Starting availability 19.924728 GiB, sampled driver maximum 2,304,868,352 bytes, RSS high-water mark 2,676,031,488 bytes. | No training/save fit. That historical start would fail the new 20 GiB floor too. |
| Task 45 | 0/64 development recovery/exact matches, 0/12 utility exact matches; every generation exhausted 64 tokens. Direct probes reuse training facts; paraphrases are exposed development wordings. | NLL or exact-match changes alone cannot certify acquired facts, selective forgetting, general utility retention or privacy. |

The pinned base is `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0. Original weight file
SHA-256 is `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
Baseline SHA-256 is fixed literally in the new runner:

```text
61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff
```

Task 37's report SHA-256 is
`f71c8d1e673067676708c33be9cfdb30b7545f96ed691768b7b183652bde80b3`.
Every validation and execution rechecks these inputs. Prior Task 48–51
Terminal-only failure details remain unavailable here; this task does not
invent their readings or assume their update counters. No failed checkpoint
from those attempts can be selected or resumed by this runner.

### Native LoRA instead of installing PEFT

Task 52 considered PEFT, which is absent from the pinned environment. Task 53
implements ordinary LoRA directly in approximately 70 lines of PyTorch adapter
code, using the **unchanged** `requirements-ml.lock`. There is no PEFT,
Accelerate, bitsandbytes, quantization, dispatch, offload or extra package
requirement. The runner checks every installed package against that lock and
records versions, Python, installed Llama source hash and all local source
hashes. No environment modification or separate dependency resolution is
needed for this native path. This is a deliberate implementation refinement
of Task 52, rather than a claim that its proposed PEFT stack was tested.

The adapter computes `W0 x + (8/4) B A x`, with CPU-seeded Kaiming-uniform A,
zero B, rank 4, alpha 8, dropout zero, bias none and no merging. It never
modifies the original W0, including when switching train/eval modes. This is
the ordinary low-rank additive parameterization; no PEFT serialization/API
compatibility is claimed.
[Original LoRA linear implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py).

Cached config and safetensors **headers only**, plus installed Transformers
4.57.1 source inspection, establish LlamaForCausalLM, 30 layers, width 576,
9 query heads and 3 KV heads, with these exact target paths:

```text
model.layers.{0..29}.self_attn.q_proj   original [576,576]; A [4,576], B [576,4]
model.layers.{0..29}.self_attn.v_proj   original [192,576]; A [4,576], B [192,4]
```

There are 60 wrappers, 120 A/B tensors and **230,400 adapter elements**,
0.171282% of the 134,515,008 original parameters. Injection prevalidates all
names, shapes, bias-free exact `nn.Linear` types and CPU FP32 placement before
replacing any module. Runtime repeats coverage, original object identity,
trainability, tied embedding/head identity, dtype, device and total counts.
All original parameters must stay frozen and have `grad is None`. Only the
120 adapters can enter SGD. Native MPS forward/backward behavior and
non-reentrant checkpointing through these wrappers remain physical-runtime
assumptions; unsupported behavior or missing gradients causes STOP.

## Separate memory policy and stops

The implemented guard is `smollm2_135m_fp32_mps_lora_r4_qv_one_step_v1`.
The zero-update baseline has a distinct guard ID and permits no backward or
optimizer; it conservatively uses the same estimated envelope. No full-parameter
guard is reused or lowered. With W = 538,060,032 base FP32 bytes,
L = 921,600 adapter FP32 bytes, S = 269,060,552 source bytes and B = live
preload process lifetime peak RSS:

```text
incremental(B) = 3W + 2L + S + 4 GiB graph/workspace
                 + 1 GiB integrity/tokenizer + 1 GiB Metal cache
                 + max(1 GiB,B) additional process headroom
                 + 1 GiB adapter clipping/step/save + 4 GiB uncertainty
preload(B) = ceil(incremental(B)/GiB) × GiB + 6 GiB continuing reserve
```

At B ≤ 1 GiB: **13.755621 GiB estimated incremental**, giving a **20 GiB
minimum available before load**, checked twice. At B = 3 GiB: 22 GiB.
These are uncalibrated experimental safeguards, not measured training memory.

LoRA weights plus gradients total an estimated 1,843,200 bytes (1.758 MiB).
Frozen base plus those tensors is 539,903,232 bytes, approximately 49.83% less
persistent tensor payload than full-base weights plus gradients. SGD has no
momentum/moment tensor state. Activations, logits, activation gradients through
frozen layers, loading, fingerprints and driver caches remain. Historical
system-wide availability changes and overlapping RSS/driver counters cannot
be subtracted to infer an isolated peak or safe minimum.

The smaller 1 GiB adapter-step/save allowance is justified by clipping and
updating only 230,400 elements and saving only 0.879 MiB of adapter tensor
payload. It is an unmeasured policy assumption, over 580 times the combined
adapter weight/gradient payload, rather than a measured 3 GiB saving relative
to the old runner's allowance. Keep the conservative three-copy loading term,
4 GiB workspace, 4 GiB uncertainty and 6 GiB reserve. Saving/merging a full
base would invalidate this policy and is not implemented.

Mandatory stops include:

- Non-M4-Pro, non-arm64/macOS, unavailable/incorrect MPS, wrong dependency
  versions, fewer than 48 GiB physical capacity, or fewer than 4 GiB free disk.
- Either dynamic preload floor fails; continuing availability below 6 GiB;
  non-NORMAL pressure; free indicator below 20%; any cumulative swap growth,
  page-out change or capacity change; lifetime RSS or MPS driver above 8 GiB.
- Invalid, missing, future/stale, unordered or inconsistent telemetry;
  decreasing lifetime RSS; active sample gaps above two seconds; stale latest
  watchdog observation; 30-minute timeout; watchdog or cleanup failure.
- MPS fallback, allocator/watermark/math overrides, distributed execution,
  incompatible loading keys, altered cached base/data/profile/source/lock,
  output collisions, symlinks or unsealed/tampered baseline artifacts.
- Baseline/no-op NLL or generation drift, changed initial adapter hash, wrong
  targets/trainable scope, base gradients, disconnected/nonfinite adapter
  gradients, nonfinite loss/logits/norms, loss-parity failure, clipping failure,
  optimizer configuration/membership drift, or a consumed update budget.
- Changed base bytes/versions/ties, no verified adapter/B-side change after
  the step, evaluation mutation, incomplete evaluation or serialization error.

The 250 ms watchdog samples physical counters; synchronized boundaries cover
loading, injection, training forward/backward, clipping, step, evaluation,
hashing checks and serialization. `peaks` reports sampled RSS/tensor/driver
maxima, OS process lifetime RSS high-water mark, sample count and longest gap.
RSS and driver bytes overlap and must not be added. Short peaks may be missed;
the watchdog interrupts at a Python safe point. Do not purge caches, terminate
applications automatically, alter watermarks, add CPU fallback, shrink frozen
evaluation budgets or retry a STOP automatically.

## Exact normal Mac Terminal procedure (future owner execution only)

**None of the physical commands in this section ran in Task 53.** The current
task authorizes preparation only. The read-only baseline is a separate future
execution; the optimizer command additionally requires explicit owner approval
of this concrete profile, hashes, paths, guard and one-step scope.

Start with the weight-free default validation in the repository's pinned
environment. It reads hashes, JSON and package metadata; it does not import
Torch/Transformers or load weights:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_lora
```

After review and separate read-only execution approval, run the new baseline
once in a normal Mac Terminal. The name below must not exist; a collision is
a STOP, requiring a new reviewed name, never deletion of an old run:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m src.small_model_lora \
  --run-baseline \
  --run-dir outputs/smollm2_lora/baseline-task53-001
```

Accept only `lora_baseline_complete_no_updates`, zero backward/optimizer steps,
sealed `report.json` and `COMPLETE.json`, exact base fingerprint, unchanged
initial/final adapter fingerprints and complete valid memory samples. The
baseline uses zero-effect adapters and all 76 unchanged probes. It must match
historical baseline NLL within `rel_tol=abs_tol=1e-4` and match every other
evaluation field exactly, including greedy output and generated-token count.
These are preregistered identity/drift gates, not acquisition thresholds.

Review the saved baseline report and obtain its external checksum:

```sh
shasum -a 256 outputs/smollm2_lora/baseline-task53-001/report.json
```

Retain that **reviewed** checksum independently. It cannot be specified in
this preparation because the native baseline does not yet exist. Do not
substitute baseline-002's hash or automatically accept new bytes by recomputing
a hash at pilot launch. After the owner explicitly authorizes exactly one
new LoRA step, enter the reviewed checksum and invoke this command in the
same normal Mac **zsh** Terminal:

```sh
read -r 'lora_baseline_sha256?Paste the reviewed native baseline report SHA-256: '
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m src.small_model_lora \
  --run-one-step \
  --authorize-weight-update 'SmolLM2-135M:development:LoRA-r4-qv:one-SGD-step' \
  --lora-baseline outputs/smollm2_lora/baseline-task53-001/report.json \
  --lora-baseline-sha256 "$lora_baseline_sha256" \
  --run-dir outputs/smollm2_lora/pilot-task53-001
```

A token appearing in this document is not user authorization. The old
full-parameter token is rejected. The new runner offers no device, rank,
learning-rate, optimizer, budget, download or resume override. Missing/wrong
authorization stops before output reservation or model/runtime imports.

The pilot independently rechecks both physical preload gates and reconstructs
the same initial adapter. Before any backward or optimizer construction, it
repeats the 76-case no-op evaluation against both baselines. Then it takes
one positive answer-NLL step on predetermined `dev:09:0` (52 tokens): FP32,
batch one, cap 128, seed 2026, SDPA, no KV cache/autocast, non-reentrant
checkpointing with RNG preservation, LR 1e-4, stateless SGD with
momentum/dampening/weight decay zero, `foreach=False`, `fused=False`,
`differentiable=False`, global adapter norm cap 1.0. No input truncation is
allowed; answer labels include EOS and mask the question.
[PyTorch SGD](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html),
[checkpoint semantics](https://docs.pytorch.org/docs/2.9/checkpoint.html).

Every A/B tensor must have a finite gradient, and aggregate B-side gradients
must be nonzero. A gradients can legitimately be zero on the first step
because B starts at zero. Do not require every adapter tensor to change.
The post-step check requires changed adapter bytes and nonzero B, while all
original bytes, versions and tied storage remain unchanged. The runner
rehashes canonical original names even though wrapping changes module paths.

## Outputs, integrity and acceptance

Each explicit attempt exclusively reserves a new ignored directory under
`outputs/smollm2_lora/`. `RUNNING.json` is sealed before checks so early
admission failures leave evidence. `report.json` records configuration,
provenance, adapter element/tensor counts, trainable count during the step,
base and adapter before/after hashes, optimizer configuration/state/step count,
backward attempts, paired evaluations, all memory samples/peaks and STOP errors.
The baseline's actual trainable count is zero; its adapter capacity is 230,400.
Hashes not reached on failure remain null rather than invented.

Immediately before calling the optimizer, a sealed/fsynced
`STEP_ATTEMPTED.json` records that a step **may** have happened. A crash with
this marker cannot be called zero-update merely because the final report is
absent. A returned optimizer call increments the count before subsequent
synchronization/integrity checks; failure afterward honestly preserves that
count. Any STOP prevents COMPLETE; a partial checkpoint stays unusable.

After the step, gradients are cleared, all parameters frozen and checkpointing
disabled for the unchanged 76-case evaluation. The writer saves only
`checkpoint/adapter_model.safetensors` and `checkpoint/adapter_config.json` in
the native `unlearnguard_native_lora_v1` format. It never calls
`model.save_pretrained`, serializes original weights, merges adapters or changes
the original cache. Exact expected A/B keys, FP32 shapes, size, metadata and
byte/value equality are checked by a small adapter-only disk round trip. Config
binds the original base revision/weight SHA, profile, code/dependencies and
adapter hash. Files and directories are sealed; `COMPLETE.json` is written last
and binds report and checkpoint hashes. Failed runs retain `FAILED.json` and
all partial files. Never overwrite, resume or clean up historical outputs.

Accept mechanics only with `lora_one_step_complete_mechanics_only`, exactly
one step, verified finite adapter change, unchanged base/cache, complete paired
evaluation, compliant measured memory and sealed integrity artifacts. Inspect
stdout/stderr too. No recovery improvement is required for mechanics; an
in-memory update followed by STOP is a failed pilot. Before longer/scientific
use, separately implement/authorize a guarded real model-plus-adapter reload
and verify reattached inference fidelity; the small tensor round trip is not
that model reload. Reports keep `scientific_use_cleared=false` and
`saved_model_reload_performed=false`.

## Offline verification and later science

The new suite covers weight-free defaults and evidence hashes; metadata/target
coverage; native forward/effective-matrix agreement and central finite-difference
factor derivatives; deterministic zero-effect initialization; frozen-base
identity/trainability/device/checkpointing checks; assigned finite/nonfinite
gradients and clipping; exact adapter-only SGD membership; memory boundaries,
watchdog freshness/deadlines; authorization/one-step budgets; baseline tampering,
no-op drift; collisions/aliases; interrupted updates/saves; native adapter-only
safetensors round trips; and mocked baseline/pilot orchestration.

No real backward or optimizer step is needed by these tests. Synthetic tensor
fixtures are initialized/assigned directly; all orchestration model/backend,
backward and optimizer operations are fakes. The default CLI is also tested
in a fresh process that rejects even transitive ML runtime imports.

**157 offline tests passed**, zero failures, errors or skips: 52 new LoRA
tests plus the 105 existing small-model tests. The ordinary offline command is:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-ml/bin/python -B -m unittest discover \
  -s tests -p 'test_small_model*.py' -v
```

Task 53 additionally patched real `Tensor.backward`, `autograd.backward`,
`autograd.grad`, SGD/AdamW `step`, `torch.load`, safetensors `load_file` and
Transformers model `from_pretrained` loaders to raise throughout the combined
suite. Only the small synthetic adapter file was deserialized for round-trip
checks. The log is ignored `outputs/task53-offline-tests.log`; the weight-free
plan is `outputs/task53-offline-plan.json`. AST parsing, frozen JSON/parameter/
admission arithmetic, local Markdown links/fences and `git diff --check` pass.
These checks do not establish native LoRA MPS training fit.

The ignored `outputs/task53-preservation-before.json` ledger contains 160
pre-existing tracked/cache/result/output files. All 159 protected entries
remain identical in hash, size and mode; README is the sole intended edit to
an existing file. The original model, baseline-002 (including mode `0444` and
SHA-256), full-parameter code/guards, dependency files, evaluation artifacts,
data and prior outputs are preserved. New code/profile/tests/docs are additive;
Git history is retained through an ordinary commit and push.

For later acquisition, preregister budgets and useful target/utility gates and
demonstrate both F and R learned with this adapter configuration. Retain M0,
M_all, matched-retain-exposure M_R, no-op/reload and retain-only continuation
controls. Then separately review and authorize an adapted **LoRA-GradDiff**
workload: minimize `-mean_NLL(F)+mean_NLL(R)` over the acquired adapter factors
with the base fixed. Its factor-space updates differ from official
full-parameter GradDiff, and its two-term memory must be assessed separately.
Compare paired forget recovery/NLL, retain stability, distance to M_R,
paraphrase transfer and non-floor utility, with target-string likelihood,
answer validity and termination diagnostics frozen in a new evaluator version.
Adapter disabling is an ablation, not selective unlearning. Existing public
synthetic development facts and exposed paraphrases do not establish privacy,
membership erasure, official TOFU scores or held-out generalization.

Remaining blockers for the **first step** are the unperformed native MPS
baseline/no-op checks, fresh passing physical admission and explicit owner
authorization. Native MPS gradients and end-to-end update/save peaks will only
be observed by that separately authorized pilot; it must STOP if any check
fails. Reload, broader utility controls, actual both-role acquisition and
unlearning remain separate prerequisites for later scientific work.
