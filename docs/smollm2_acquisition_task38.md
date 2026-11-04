# Task 38: guarded SmolLM2 development acquisition pilot

**Prepared, conditional; not yet ready to execute an optimizer step.** The
one-step runner and frozen evaluation manifest are implemented and tested
offline. A real immutable base-model evaluation has **not** been performed in
this task. Training remains blocked until that evaluation completes, its
SHA-256 is reviewed, fresh device/memory admission passes, and the owner
explicitly authorizes one update. No acquisition, forgetting or unlearning
result is claimed.

The [runner](../src/small_model_acquisition.py) defaults to validation. The
[frozen manifest](../experiments/smollm2_135m_acquisition_task38.json) hashes to
`60e9872e86e8bd98b1a281866f686c7ff3ed41ca3c8b5709e5b4c4b1daa239de`.
Its regeneration is deliberately absent from the CLI. Source drift stops the
runner; changing the evaluator code invalidates previous baselines as well.
The [evidence audit](../experiments/smollm2_task38_evidence_audit.json) preserves
the recorded Task 37 findings and original artifact hashes.

## Task 37 evidence and its limits

`outputs/smollm2_backward_task37.json` has file SHA-256
`f71c8d1e673067676708c33be9cfdb30b7545f96ed691768b7b183652bde80b3`.
Recorded status: `backward_rehearsal_passed_not_training_clearance`.
It used FP32 on actual MPS, two forwards and one backward, and recorded finite,
nonzero gradients covering **134,515,008** parameters. Forget model/local NLL
both equal **1.4983179569244385**; retain model/local NLL both equal
**2.2311341762542725**. Parameter versions were unchanged, gradients cleared,
and before/after parameter SHA-256 both equal
`709264bff244d021ac31b1fcf9e1733b9af31f75032e854e28db62287b8cd072`.

Provenance is `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0. The cached
`model.safetensors` is **269,060,552 bytes**, SHA-256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
All five model/tokenizer metadata hashes agree with the frozen profile and
current cached files. Optimizer creation was false; optimizer steps, weight
updates and checkpoint writes were all **zero**. The original checkpoint
integrity check passed; downloads were disabled.

The 48 GiB host trace contains 25 samples. Before-load availability was
28.333603 GiB; minimum sampled availability was 25.278580 GiB. Sampled maxima
were 1,301,757,952 bytes current RSS, 1,129,497,856 bytes MPS tensors and
2,659,057,664 bytes MPS driver; OS lifetime peak RSS was 1,301,889,024 bytes.
Pressure was normal, free indicator at least 90%, swap and page-outs zero.
The longest sample gap was 0.847274 seconds. These are historical readings,
not current admission. RSS and MPS overlap and must not be summed. Nonatomic
samples can miss peaks; swap telemetry is rounded. This rehearsal did not
measure clipping, SGD, generation or checkpoint serialization.

## Frozen data and controls

The original [development facts](../experiments/smollm2_135m_development_facts.json)
remain byte-for-byte unchanged: 16 fictional objects, two public random-string
attributes each; 8 facts labeled forget and 24 labeled retain. Acquisition
minimizes **positive answer NLL for either role**. Labels define future controls;
they do not introduce a forgetting objective here.

| Set | Count | Use |
|---|---:|---|
| Development direct QA | 32 | Acquisition pool and deliberately overlapping direct evaluation |
| Development paraphrases | 32 | Evaluation only; same entities, answers and aliases, different wording |
| Generic utility probes | 12 | Evaluation only; no training |

The one-step pilot selects **`dev:09:0`** from SHA-256 order
`SHA256(2026:0:case_id)`, independent of scores. It performs **one update total**,
not an epoch over 32 facts. Fixed LR is **1e-4**, the lowest existing candidate;
it is a conservative pilot setting, not a calibrated recommendation.

The frozen evaluation uses answer-only causal NLL with local/model loss parity,
normalized exact match, and whole-word frozen alias recovery for development
facts. Utility alias recovery is null. Generation is greedy, question-only,
EOS-terminated, up to 64 new tokens, with cache disabled. The gold answer is
used only for teacher-forced NLL; it is never included in the generation prompt.
All 108 local tokenizer encodings were checked offline: maximum 57 tokens;
longest prompt plus generation budget is 93, below the new pilot's 128 cap.

Direct evaluation is resubstitution. Paraphrases measure wording transfer of
trained facts, not unseen factual generalization. The small utility list is
not a benchmark, and arbitrary strings may not transfer to TOFU biographies.
These paraphrases also inform future development calibration, so they cannot
serve as a final untouched test. Refusal/repetition metrics and official TOFU
metrics are not implemented in this pilot. No success threshold on these 76
cases authorizes a longer run automatically.

The original 40-forget/120-retain TOFU plan, 40 unseen retain cases, paraphrase
controls and sealed holdout remain in their original manifest. The planned
M0, M_all, M_R, no-op, matched retain-only and GradDiff controls remain required
for future work; none has been replaced by this acquisition pilot. Its
32-fact source was already checked for development/TOFU/utility leakage during
frozen preparation, and their source hashes are pinned here. The new runner
does not open or train on TOFU or its sealed holdout. All Llama source,
experiments, cached weights and historical baseline/results are preserved.

## Admission, update and checkpoint safeguards

The separate guard is `smollm2_135m_fp32_mps_dev_one_sgd_step_v1`, restricted to
Apple **M4 Pro**, arm64 macOS, MPS and the pinned model/profile/runtime versions.
It uses the existing direct telemetry collector and counter validation, but
never calls Llama admission or changes any old policy. The 512-token controlled
TOFU profile and its old memory assessor are unchanged. This new 128-token
entry point cannot authorize that workload.

With baseline peak RSS at most 1 GiB, the estimated incremental budget is
**18,527,428,040 bytes** (17.255 GiB): three FP32 weight copies, gradients, source
file residency, 4 GiB workspace, 1 GiB tokenizer/integrity, 1 GiB Metal cache,
1 GiB additional process headroom, **4 GiB clipping/step/serialization**, and
4 GiB uncertainty. Rounded upward plus 6 GiB reserve gives **24 GiB before
load**. Higher observed process peak increases the floor; it never decreases
below 24 GiB. These are conservative policy allowances, not measured fit or
calibration. The numerical floor is deliberately above Task 37's 20 GiB floor.

Stop conditions are enforced at admission, sampled every 250 ms, and at phase
boundaries:

- Less than 48 GiB total capacity; less than the dynamic preload floor; less
  than 6 GiB continuing free-plus-speculative memory; free indicator below
  20%; kernel pressure other than NORMAL=1.
- Any cumulative swap growth or page-out change; missing, stale/future,
  inconsistent telemetry; an active sampling gap over two seconds.
- Process lifetime peak RSS or MPS driver allocation above **8 GiB** each;
  these are separate stop caps, not additive physical measurements.
- 30-minute wall time including evaluation, OOM, device/dtype/configuration
  mismatch, nonfinite loss/logits/gradients/weights, failed loss parity,
  checkpoint drift, or insufficient disk (4 GiB required before load).

Fallback, allocator/math overrides and distributed execution are rejected.
Torch deterministic algorithms are required with seed 2026, one CPU thread,
one MPS device, FP32, SDPA, no autocast, no cache and non-reentrant checkpointing.
Unsupported deterministic MPS operations stop; numerical bitwise reproducibility
across different OS/backend builds is not established.

Updates require **both** `--run-one-step` and the exact model/scope authorization
token. The internal update function repeats authorization. All parameters must
be trainable FP32 on MPS with fresh finite, nonzero gradients. Global gradient
norm is clipped to 1.0 and checked after clipping. Exact `torch.optim.SGD`,
zero momentum/dampening/weight decay, foreach/fused false, no scheduler, and
empty optimizer state are validated before and after the single step.

Baseline JSON is created exclusively, fsynced, and sealed mode 0444. Training
requires its explicit SHA-256, full ordered evaluation, correct base/metadata/
encoding/evaluator provenance, zero-update counters, unchanged parameter bytes
and versions, and memory evidence. The newly loaded base must match its
in-memory parameter hash. Read-only permissions and SHA-256 provide local
write protection and drift detection, not authenticated WORM storage against
an owner who intentionally changes files or code.

Reports and run directories must be new, ignored, untracked direct children of
`outputs/smollm2_acquisition_task38/`. Symlinks, traversal, historical names,
original caches and historical results are rejected. Baseline hardlink aliases
are rejected. The checkpoint destination is fixed to a newly reserved
`pilot-*/checkpoint/`; it cannot be supplied independently. No resume or
multi-step CLI exists. Saving occurs only after the step, finite-weight/hash
checks, complete frozen post-step evaluation and base integrity checks. Saved
files are hashed and sealed; base integrity is checked again after saving.

Failed run directories remain as tombstones and receive `FAILED.json` when
possible. Treat any partial checkpoint as unusable; only a successful report
and `COMPLETE.json` permit later inspection. A stop during/after SGD can leave
partial or complete RAM updates; the report records `step_attempted` and
completed steps honestly, clears gradients, and does not retry or resume.
Serialization failure can leave partial files. The watchdog interrupts at a
Python safe point and cannot instantly preempt a Metal kernel. A hard process
kill may leave an incomplete report and no marker. Do not reuse those paths.

## Exact normal-Terminal sequence (not executed in Task 38)

Offline default, safe to run without loading a model:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_acquisition
```

First, in a normal interactive macOS Terminal, run the **read-only baseline**.
This loads only the existing cache, evaluates all 76 cases and makes no
optimizer or checkpoint. It creates the dedicated parent directory itself.

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m src.small_model_acquisition \
  --run-baseline \
  --report outputs/smollm2_acquisition_task38/baseline-001.json
shasum -a 256 outputs/smollm2_acquisition_task38/baseline-001.json
```

Review its `baseline_complete_no_updates` status, measurements, generations,
provenance and hash. Any STOP means defer; do not lower a guard or replace the
artifact. Setup/admission failures before exclusive report reservation print
STOP without creating a report. Use a fresh numbered baseline name on any retry.

Only **after the owner authorizes one real optimizer step**, use this exact
pilot command. The hash variable binds the previously reviewed baseline;
compare it with the recorded hash before executing the pilot block.

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
TASK38_BASELINE_SHA256="$(shasum -a 256 outputs/smollm2_acquisition_task38/baseline-001.json | awk '{print $1}')"
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m src.small_model_acquisition \
  --run-one-step \
  --authorize-weight-update 'SmolLM2-135M:development:one-SGD-step' \
  --baseline outputs/smollm2_acquisition_task38/baseline-001.json \
  --baseline-sha256 "$TASK38_BASELINE_SHA256" \
  --run-dir outputs/smollm2_acquisition_task38/pilot-001 \
  --report outputs/smollm2_acquisition_task38/pilot-001-report.json
```

Expected artifacts: immutable `baseline-001.json`; `pilot-001-report.json`
with one completed step, clipped gradient norms, empty optimizer state,
verified parameter change, unchanged cached base and pre/post memory traces;
`pilot-001/checkpoint/{config.json,generation_config.json,model.safetensors}`;
and `pilot-001/COMPLETE.json` binding the report hash. Generation config may
depend on the pinned saver. The original tokenizer is used from the pinned
cache; no tokenizer or optimizer state is saved. Post-step evaluation uses
the same 76 frozen cases. A completed pilot tests update mechanics, not
meaningful acquisition of the full 32-fact pool.

## Offline verification and longer-run approval

Task 38 uses offline unit tests, mocked runtime/save operations, local checkpoint
hashing, metadata/tokenizer-only encoding checks, and CPU tensor diagnostics.
**91 selected tests passed, including 26 new acquisition tests.** Preservation
checks confirmed all 66 pre-existing cached/historical files and every entry
in the prior Task 36 preservation record still match their SHA-256 values.
No real model was constructed, no real optimizer stepped, no checkpoint was
written and no new model downloaded. Mocked tests simulate a weight mutation
and record a save call without writing weight files. Tests cover authorization,
one-step consumption, nonfinite/clip/state failures, stale/pressure/swap memory
stops, checksum/source drift, baseline sealing and coverage, output collision/
symlink/hardlink guards, load/post-update stops, and question-only generation.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest discover -s tests -p 'test_small_model*.py' -v
```

The selected regression suite excludes older synthetic training tests that
call real optimizers; Task 38 does not invoke any real optimizer step, even
on toy parameters.

Before approving anything longer: obtain and audit the immutable baseline;
authorize and complete this one-step pilot; inspect actual clipping, SGD,
evaluation and save peaks and sampling gaps; verify the new checkpoint by a
separately approved read-only reload; compare all role/wording/utility results;
repeat measured cold/warm trials and longest-sequence stress checks for the
intended budget (the original TOFU cap is still 512). Then review a separate
memory/budget policy and deterministic multi-step implementation with tests,
calibrate only on development under a fixed budget with the existing candidate
rule, and obtain explicit longer-run authorization. TOFU acquisition additionally
needs its own baseline, key-fact/refusal/repetition evaluators and the preserved
M0/M_all/M_R/no-op/retain-only controls. Nothing in this pilot clears unlearning.
