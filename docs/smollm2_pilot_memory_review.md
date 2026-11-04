# Task 49: SmolLM2 pilot memory review

**Decision: CONDITIONAL GO for one further local mechanics pilot under the
unchanged Task 38 admission policy. Execution remains blocked until the Task 48
failure evidence is reconciled, fresh admission passes, and the owner explicitly
authorizes that new attempt.** This review does not authorize or execute an
update. It does not establish a smaller safe training threshold. No model was
loaded, no real backward pass or optimizer step occurred, and no training,
fact acquisition, or unlearning result is claimed in Task 49.

## Evidence reviewed and the missing Task 48 artifact

Reviewed the [Task 38 preparation](smollm2_acquisition_task38.md),
[Task 39 baseline admission](smollm2_baseline_admission_task39.md),
[Task 45 baseline audit](smollm2_baseline_findings_task45.md) and its
[evidence](smollm2_baseline_task45_evidence.json), the actual Task 37 and baseline
JSON files, the [frozen Task 38 audit](../experiments/smollm2_task38_evidence_audit.json),
and the current acquisition, backward, telemetry and memory-policy source.
Starting Git HEAD was `2dc5de6` on clean `main`, tracking `origin/main`.
The checkpoint is the Apache-2.0 base model `HuggingFaceTB/SmolLM2-135M`,
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, not the Instruct model.

**Evidence limitation:** no Task 48 report, pilot report, failed pilot directory,
or Task 48 transcript exists in the supplied repository, including ignored
`outputs/`. `outputs/smollm2_acquisition_task38/` contains only `baseline-002.json`.
The user identifies Task 48 as a memory-readiness failure; its exact available
bytes, failing stage, counters, timestamp and error cannot be independently
reviewed here. Its location or Terminal-only STOP output was requested. Do not
invent a Task 48 reading or substitute the Task 45 reading for it.

The runner explains how an early failure can leave no report: its first
`monitor.capture("baseline", before_load=True)` precedes report/run-directory
reservation. If that capture raises, the returned STOP object can be printed to
Terminal without a report file. A failed second preload capture also precedes
reservation. This is a possible explanation, **not a finding about Task 48**.
Recover the original output and verify `model_loaded`, `step_attempted`,
`optimizer_steps`, `checkpoint_writes`, measurements and error before another
update. Any indication of an attempted step requires auditing that failed run
and treating partial checkpoints as unusable; never resume it.

Task 49 independently recomputed these file hashes, without deserializing model
tensors:

| Artifact | SHA-256 / result |
|---|---|
| `outputs/smollm2_acquisition_task38/baseline-002.json` | `61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff`; 479,156 bytes; mode `0444` |
| `outputs/smollm2_backward_task37.json` | `f71c8d1e673067676708c33be9cfdb30b7545f96ed691768b7b183652bde80b3` |
| Original cached `model.safetensors` | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`; 269,060,552 bytes |
| `src/small_model_acquisition.py` | `92c3896e03d1599b007131fc255810fa32b5893a396b21203ef783004b8c01cd` |
| Frozen acquisition manifest | `60e9872e86e8bd98b1a281866f686c7ff3ed41ca3c8b5709e5b4c4b1daa239de` |

The original `artifact_path`, `frozen_manifest` and `validate_baseline` checks
pass. All cached metadata and frozen sources match. Tokenizer-only reconstruction
using the local `tokenizers` library reproduces all 108 encodings and digest
`8f2f40dd285943b8fa4358c318ed8c99557845d0d35cbfcb6762142464d4a4f1`.
Maximum QA length is 57; maximum prompt plus 64-token generation budget is 93.
The fixed pilot case `dev:09:0` has 52 tokens, with answer targets starting at
token 25. This validation imports neither Torch nor Transformers.

## Why 24 GiB does not contradict successful backward

The [pilot policy](../src/small_model_acquisition.py) is SmolLM2-specific; it is
not the inherited 32 GiB Llama guard. Its minimum is derived from allowances,
not a measured one-step requirement. Let `W = 134,515,008 × 4 = 538,060,032`
bytes = **0.501107 GiB** of unique FP32 parameters, and `B` be the live
preload process lifetime peak RSS. The unchanged formula is:

```text
I(B) = 3W + W + 269,060,552
       + 4 GiB forward/backward workspace
       + 1 GiB tokenizer/integrity + 1 GiB Metal cache
       + max(1 GiB, B) additional process headroom
       + 4 GiB clipping/step/serialization + 4 GiB uncertainty
admission(B) = ceil(I(B) / GiB) × GiB + 6 GiB reserve
```

| Budget term | Bytes at B ≤ 1 GiB | Evidence class |
|---|---:|---|
| Three FP32 weight copies | 1,614,180,096 | Analytical size × conservative copy count; simultaneous copies unmeasured |
| FP32 gradients | 538,060,032 | Analytical full-gradient payload; Task 37 recorded gradients covering every parameter |
| Source checkpoint residency | 269,060,552 | Verified file size; physical residency is an allowance |
| Forward/backward workspace | 4,294,967,296 | Unmeasured allowance, not an activation peak |
| Tokenizer/integrity | 1,073,741,824 | Unmeasured allowance |
| Metal cache | 1,073,741,824 | Unmeasured allowance |
| Additional process headroom | 1,073,741,824 | Policy minimum; increases with measured live B |
| Clipping/step/serialization | 4,294,967,296 | Unmeasured combined allowance, not SGD optimizer state |
| Uncertainty | 4,294,967,296 | Policy reserve for unobserved behavior |

The sum is **18,527,428,040 bytes = 17.255012 GiB**; rounding to 18 GiB and
adding 6 GiB gives **24 GiB = 25,769,803,776 bytes**. At `B=3 GiB`, admission
is 26 GiB. This `B` is the live process reading, not the old evaluator's peak.
Existing RSS occupancy is already excluded from available pages; the headroom
term conservatively budgets additional growth rather than subtracting RSS again.

The Task 37 backward-only formula has the same other numerical terms but omits
the 4 GiB clipping/step/serialization allowance: **14,232,460,744 bytes =
13.255012 GiB**, rounded to 14 plus 6 = **20 GiB**. Thus the entire 20-to-24 GiB
difference is an explicit allowance for operations absent from the rehearsal.
The two formulas use the same workspace number despite different graph scope;
this is a conservative envelope, not a demonstrated equality of workloads.

| Historical observation | Task 37 backward rehearsal | Baseline-002 |
|---|---:|---:|
| Status | `backward_rehearsal_passed_not_training_clearance` | `baseline_complete_no_updates` |
| Available immediately before load | 30,422,974,464 bytes / 28.333603 GiB | 21,394,014,208 bytes / 19.924728 GiB |
| Minimum sampled available | 27,142,668,288 bytes / 25.278580 GiB | 15,957,819,392 bytes / 14.861877 GiB |
| Maximum sampled current RSS | 1,301,757,952 bytes | 2,676,031,488 bytes |
| OS lifetime peak RSS observed | 1,301,889,024 bytes | 2,676,031,488 bytes |
| Maximum sampled MPS tensors | 1,129,497,856 bytes | 583,687,680 bytes |
| Maximum sampled MPS driver | 2,659,057,664 bytes | 2,304,868,352 bytes |
| Samples / longest gap | 25 / 0.847274 s | 797 / 0.770298 s |
| Pressure / swap / page-outs | Normal / zero / zero | Normal / zero / zero |

Task 37 actually began **above 24 GiB**, even though it required only 20 GiB.
Its two actual sequences were 74 and 42 tokens, with two graphs contributing
to one backward of `-NLL(forget)+NLL(retain)`. The pilot instead uses one
positive acquisition NLL, clipping and SGD, then all 76 generation evaluations
and saving. Task 37 used no optimizer and saved no checkpoint; baseline-002
used neither autograd nor serialization. Both recorded unchanged parameter hash
`709264bff244d021ac31b1fcf9e1733b9af31f75032e854e28db62287b8cd072`.

Task 37's before-load to minimum-availability difference is about 3.055 GiB;
baseline-002's is about 5.063 GiB. These are system-wide changes influenced by
other processes, not isolated workload peaks. Neither is a safe replacement
for the 17.255 GiB incremental estimate. Baseline-002's starting availability
would fail the pilot guard, but that does not diagnose Task 48 or physical OOM.

## Allocation and lifetime audit

- **Optimizer and gradients:** the runner already requires exact FP32 SGD,
  momentum/dampening/weight decay zero, no scheduler, no autocast, and
  `foreach=False`, `fused=False`, `differentiable=False`. The pinned Torch 2.9.1
  `optim/sgd.py` only creates momentum buffers when momentum is nonzero and
  uses per-parameter in-place addition in this configuration. Persistent tensor
  optimizer state is therefore zero, validated before and after the step.
  Python parameter/gradient lists and backend work still exist. Weights plus
  gradients alone are **1,076,120,064 bytes = 1.002215 GiB**. Clipping, finite
  checks, norm reductions and dispatch can allocate temporaries. Zero state
  does not mean zero incremental step cost. Momentum would add roughly W;
  FP32 AdamW's two moments add roughly 2W, before other overhead. Switching
  optimizers would change the experiment. The documented foreach memory
  tradeoff supports retaining the existing loop configuration.
  [PyTorch SGD documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html).
- **Model residency and integrity:** one model object is loaded on CPU in
  FP32, validated, and moved to MPS. CPU load/conversion, source mapping and
  device transfer may overlap, but three full resident FP32 models were not
  observed. Unified memory does not eliminate the cost of distinct CPU/device
  allocations. `parameter_fingerprint()` stages one parameter on CPU at a
  time and calls NumPy `tobytes()`, which can add another host byte copy.
  Reading only the safetensors header found 272 unique BF16 tensors and
  269,030,016 payload bytes. The largest is the tied embedding, shape
  `[49152,576]`: **108 MiB FP32**. Its two host representations could account
  for roughly 216 MiB transiently, before allocator/backend effects. This is
  arithmetic, not a measured fingerprint peak; Task 37 already includes this
  fingerprint path. Do not clone a second model for the comparison baseline.
- **Activations:** both paths already enable non-reentrant activation
  checkpointing with RNG preservation, SDPA and no KV cache. It trades
  recomputation for retained activations; it does not remove logits, gradients,
  cross-entropy copies or backend workspace. One `[1,128,49152]` FP32 logits
  tensor is 24 MiB (9.75 MiB at the pilot's actual 52 tokens), not total graph
  memory. Checkpointing decoder blocks does not checkpoint every loss/parity
  operation. The rehearsal's different objective and two graph lifetimes are
  neither an upper bound nor a measurement of the full acquisition lifecycle.
  [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/2.9/checkpoint.html).
- **Lifetime separation:** `one_step()` deletes the loss after backward and
  clears gradients with `set_to_none=True` in `finally`. Its optimizer is local
  and is not saved. Evaluation requires frozen eval parameters and no
  gradients, disables activation checkpointing, and runs under inference mode.
  Saving follows the complete frozen evaluation and parameter/base-integrity
  checks. Gradients and autograd graphs need not be live during saving, but
  allocator caches can persist across those phases. Summing all phase maxima
  as if simultaneous overstates live payload; subtracting them from the guard
  without measuring retention would also be unjustified.
- **Serialization:** `save_pretrained(..., safe_serialization=True,
  max_shard_size="1GB")` produces a single shard for approximately 513.134 MiB
  of unique FP32 weights. The pinned Transformers 4.57.1 saver builds a state
  dictionary of references, removes known tied aliases, and makes contiguous
  tensors per shard. It does not inherently clone a whole second model merely
  by calling `state_dict()`. Safetensors 0.8.0 `save_file()` uses
  `_flatten_as_ptr()` and `serialize_file()`: non-CPU tensors are copied to CPU,
  and `keep_references_alive` retains their NumPy views/CPU tensors through the
  write. Therefore the current one-shard save can stage a full FP32 payload
  while the MPS model and driver caches remain. It does **not** require a
  second whole-file Python bytes blob as `save()` would. File cache, contiguous
  copies and Rust/backend allocations still need measurement. The 4 GiB
  allowance cannot be identified with any one of these payloads.
  [Pinned Transformers saver](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py),
  [pinned safetensors writer](https://github.com/safetensors/safetensors/blob/v0.8.0/bindings/python/py_src/safetensors/torch.py).
- **MPS accounting:** current tensor bytes exclude allocator cache; driver
  bytes include cached allocations and MPS/MPSGraph allocations. Freeing a
  tensor does not guarantee immediate return of pages to macOS. RSS, driver
  bytes and tensor bytes overlap on unified memory and must not be summed.
  `recommended_max_memory()` is a device working-set recommendation, not free
  host RAM. The collector samples without synchronization; explicit boundaries
  synchronize MPS. Nonatomic 250 ms samples can miss short clipping/save peaks,
  and the OS RSS high-water mark does not supply an exact MPS allocation peak.
  The watchdog can interrupt Python only at a safe point. Retain allocator
  limits and the bans on fallback and allocator/math overrides. Do not use
  `empty_cache`, a purge, memory-fraction changes or raised watermarks to make
  a failing admission pass.
  [Driver accounting](https://docs.pytorch.org/docs/2.9/generated/torch.mps.driver_allocated_memory.html),
  [MPS APIs](https://docs.pytorch.org/docs/2.9/mps.html).

These library conclusions were checked against the installed source corresponding
to `requirements-ml.lock`; they are source analysis, not execution measurements.

## Alternatives and scientific tradeoffs

| Alternative | Memory benefit and scientific validity | Decision for next attempt |
|---|---|---|
| FP32 SGD without momentum | Eliminates moment/momentum tensor state; preserves the frozen full-parameter pilot objective. Slower convergence than another optimizer is possible; LR and budget need separate calibration for longer runs. | Already implemented; keep it, including foreach/fused false and global clipping. No additional saving to subtract. |
| Non-reentrant activation checkpointing | Reduces retained block activations at additional compute cost; preserve RNG, device behavior and the same mathematical objective. Correctness still depends on equivalent recomputation. Does not reduce parameter or gradient payload. | Already enabled. Retain it; unsupported deterministic MPS behavior means STOP. |
| Shorter sequence cap | Reduces activations only if actual input/padding is shorter. Batch one pads to the actual example, not 128. A 64 cap would leave the 52-token pilot unchanged, while excluding the evaluator's contexts up to 93. Truncating targets or reducing the frozen 64-token decoding budget changes evaluation and invalidates comparisons. | Keep 128 and forbid truncation. No expected pilot saving from relabeling the cap. Longer TOFU is a separate workload. |
| Smaller-shard / streaming serialization | Bound CPU staging instead of retaining all 513 MiB. A 128 MiB shard target accommodates the 108 MiB embedding, so per-shard payload can be bounded around 128 MiB for this model. More files, metadata and synchronization; file cache/Rust/driver overhead remain. No tensor may be split by the standard sharder merely to satisfy a smaller target. | Scientifically valid future engineering option, not implemented here and not admission evidence. See requirements below. |
| Move the sole model to CPU after evaluation, then save | Could release live MPS tensors; preserves FP32 values if exact byte integrity is checked. CPU/device transfer can itself overlap and cache may remain. All post-save device/hash checks need deliberate redesign. | Do not add an unmeasured transfer to this first pilot. |
| LoRA, quantization, freezing layers, mixed precision | Can reduce gradients/state or weights, but changes trainable scope or arithmetic; requires its own protocol, baseline compatibility and acquisition/unlearning controls. | Not a drop-in solution to this frozen FP32 full-parameter pilot. |

A future bounded saver should stage only one shard at a time, synchronously,
without collecting all `.cpu()` tensors in an outer dictionary or using a
whole-file `save()` bytes result. A direct tensor-by-tensor writer could bound
staging to the largest tensor plus bounded I/O buffers but introduces more format
and integrity code than the standard sharder. Prefer the supported sharder first.
Deduplicate tied embedding/head storage using the model's declared ties and
preserve all unique parameters, dtype, names, config and generation settings.
Write only to a new reserved checkpoint directory, hash/fsync/seal all shards
and the index, and publish COMPLETE only after the existing prerequisites pass.
Partial files remain failed tombstones. No updates may occur during writing.

Changing only `max_shard_size` is insufficient: `checkpoint_hashes()` currently
requires `model.safetensors`, whereas a sharded save supplies numbered files and
`model.safetensors.index.json`. A separate change must validate exact index
coverage, paths, hashes and total payload; test interrupted writes, tied weights
and exact CPU round trips on small fixtures; and later verify an authorized
real read-only reload. The runner hash is part of `validate_baseline()`:
editing it would invalidate baseline-002 under the current validator. Preserve
baseline-002 and either create a new separately audited baseline for that runner
or review an explicit compatibility design that pins the unchanged evaluator.
Never loosen the runner-hash check silently. This coupling is another reason to
measure the existing saver before changing it.

## Selected local strategy and separate admission recommendation

**Selected strategy:** use the existing single-model FP32, batch-one,
checkpointed, stateless-SGD implementation for exactly one positive-NLL update
on `dev:09:0`, followed by the unchanged 76-case evaluation and existing safe
single-shard save, in a naturally quieter host session. The owner may voluntarily
close unrelated applications or defer to a quieter time; no automatic process
termination, cache clearing or system changes. This strategy is concrete and
available now, preserves baseline compatibility, and measures the currently
missing clipping/SGD/evaluation/save lifecycle before introducing a new writer.
Task 37 demonstrates a historical 28.33 GiB starting window on this machine,
so seeking a compliant window has evidence behind it. It does not guarantee
such a window today or prove full pilot fit. If none occurs, remain NO-GO and
prepare the bounded saver as a separate reviewed task; do not retry repeatedly
against a failing guard.

**Admission policy, justified independently of that strategy:** retain
`smollm2_135m_fp32_mps_dev_one_sgd_step_v1` exactly, including its dynamic
**24 GiB minimum at both preload checks**. The explicit budget above covers
unmeasured phases rather than inferring fit from one historical backward.
It is an initial conservative pilot policy, not a measured necessary/sufficient
memory requirement. Neither the 1.002 GiB persistent payload nor smaller-shard
arithmetic justifies a lower admission floor. Retain all of the following:

- Apple M4 Pro, arm64 macOS, at least 48 GiB physical capacity, actual MPS,
  pinned runtime/model/profile, FP32, deterministic algorithms and seed 2026.
- At least 6 GiB free-plus-speculative availability thereafter, NORMAL pressure
  level 1, free indicator at least 20%, no cumulative swap growth or page-out
  change, complete/fresh/consistent telemetry, active gaps at most two seconds.
- Separate 8 GiB caps for process lifetime peak RSS and MPS driver allocation;
  these are overlapping stop counters, not two additive reserves. Keep the
  250 ms sampler, synchronized boundaries, 30-minute limit and 4 GiB free disk.
- Explicit new-attempt owner authorization, sealed baseline/hash/source checks,
  one-step budget, all finite/parity/gradient/full-parameter/clip/state checks,
  no fallback/download/resume, new output names, original-cache integrity,
  complete post-step evaluation, and checkpoint/report sealing and tombstones.

For a future policy review, build **phase-specific** envelopes for loading,
forward/backward, clipping/step, evaluation, fingerprinting and saving; include
coexisting payload plus residual allocator/cache retention, process growth and
uncertainty in each. Take the maximum of complete phase envelopes, then add
the continuing reserve. Do not sum overlapping measured RSS/MPS counters or
assume all caches disappear between phases. Collect repeated cold/warm trials
at the intended longest actual sequences, including saving and a verified
reload, with enough headroom for sampling blind spots and host variability.
Any replacement policy must be separately versioned, reviewed and offline-tested
before use, never automatically fitted to one successful sample or Task 24's
different evidence schema. There is no numerical reduction proposed by Task 49.

## Exact next controlled experiment (not executed here)

1. Recover and archive the Task 48 STOP report/transcript under a new ignored
   output name; establish which guard/stage failed and whether any step was
   attempted. Preserve all failed paths. Without that reconciliation, no new
   update is cleared by this review.
2. In an owner-run normal Terminal, use the same pinned environment. Collect
   fresh **no-model** physical telemetry with the existing collector and pilot
   validator, not the inherited Llama readiness command:

   ```sh
   cd ~/UnlearnGuard-Privacy-Leakage-Research
   PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B - <<'PY'
   import json
   from src import small_model_acquisition as a
   import torch
   profile = json.loads(a.PROFILE.read_text())
   a.telemetry.validate_environment(torch)
   if a.telemetry._command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip() != "Apple M4 Pro":
       raise RuntimeError("requires Apple M4 Pro")
   reading = a.telemetry.read_sample(torch, "task49_telemetry_only")
   print(json.dumps({"reading": reading, "policy": a.admission_policy(
       profile, reading["process_peak_rss_bytes"], mode="pilot")}, indent=2), flush=True)
   a.validate_sample(reading, profile=profile, before_load=True, mode="pilot")
   print("Telemetry-only admission passed; no model loaded or update authorized.")
   PY
   ```

   This imports Torch to read MPS counters but constructs no model or optimizer.
   Retain the raw output and wall-clock time. A failure means defer. A pass is
   only adjacent evidence; the runner must still repeat both preload checks and
   all environment/allocator-override gates. Do not run either physical command
   as part of this documentation task.
3. After reconciliation, fresh readiness and explicit authorization for the
   **new** attempt, run the unchanged CLI once. The hash below is the reviewed
   literal, never a newly accepted hash of potentially changed bytes:

   ```sh
   cd ~/UnlearnGuard-Privacy-Leakage-Research
   shasum -a 256 outputs/smollm2_acquisition_task38/baseline-002.json
   PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m src.small_model_acquisition \
     --run-one-step \
     --authorize-weight-update 'SmolLM2-135M:development:one-SGD-step' \
     --baseline outputs/smollm2_acquisition_task38/baseline-002.json \
     --baseline-sha256 61e8b12765346538cb4508a3ba8fe67afc0a5a98f589640ef820ac71e26a83ff \
     --run-dir outputs/smollm2_acquisition_task38/pilot-task49-next-001 \
     --report outputs/smollm2_acquisition_task38/pilot-task49-next-001-report.json
   ```

   Both proposed output names were absent at review time. Recheck before using
   them; collisions mean choose and record fresh names, never overwrite.
   One step means LR `1e-4`, positive answer NLL, all parameters, global norm cap
   1.0, no accumulation, and no second attempt/resume within this experiment.
4. Regardless of success, retain the full stdout/stderr, report, timestamps,
   failed artifacts and measured gaps. Review available memory, RSS and MPS
   separately at `after_cpu_load`, `after_load`, `before_training_forward`,
   `before_backward`, `after_backward`, `after_clipping`,
   `before_optimizer_step`, `after_optimizer_step`, the evaluation boundaries,
   `before_serialization`, `after_serialization` and `complete`. Report all
   sampler readings and the RSS lifetime high-water mark. The current background
   stage can retain an evaluation/loading label between boundaries: use boundary
   ordering to interpret phases, not those labels alone. Sampling still cannot
   establish an exact subinterval peak.
5. A mechanics pass requires exactly one completed step, verified finite changed
   weights, empty optimizer state, complete frozen evaluation, unchanged original
   cache/baseline and valid hashed/sealed checkpoint plus COMPLETE marker. Any
   STOP or missing marker is a failed experiment; a step may have happened even
   if saving failed. Audit counters honestly and do not retry automatically.
   A separate explicitly authorized read-only checkpoint reload must verify
   saved parameter bytes/config/ties before later use. No longer acquisition run,
   TOFU run or unlearning follows automatically.

This experiment can supply the missing end-to-end measurements while retaining
all safety margins. It cannot prove meaningful 32-fact acquisition from one
update. The original M0/M_all/M_R, no-op, retain-only and GradDiff controls and
the Task 45 limits on retrieval/utility claims remain binding.

## Offline verification and preservation

This change adds this review only; no executable code, guard, frozen profile,
evaluator, checkpoint or test definition is changed. New tests are unnecessary
for a documentation-only recommendation. Existing relevant tests suffice.

**105 tests passed**, zero failures/errors/skips, from
`unittest.defaultTestLoader.discover('tests', pattern='test_small_model*.py')`
using `.venv-ml/bin/python -B`, offline Hugging Face flags and no tokenizer
parallelism. During the test process `torch.Tensor.backward`,
`torch.autograd.backward`, `torch.autograd.grad`, `torch.optim.SGD.step` and
`torch.optim.AdamW.step` were patched to raise if invoked. Orchestration uses
Python fakes for model/backward/update/save; small CPU diagnostics assign
gradients and inspect clipping/optimizer configuration without real backward or
steps. No full model is instantiated. Subprocess tests exercise validation-only
CLIs. Log: `outputs/task49-offline-tests.log` (ignored).

The original baseline validator, source/checkpoint hashes and 108 tokenizer-only
encodings also passed independently without importing Torch/Transformers.
A new ignored preservation ledger, `outputs/task49-preservation-before.json`,
records SHA-256, file size and mode for **150 existing tracked/cache/historical
files**. The prior Task 45 ledger matches in full. The new ledger is rechecked
before completion; all original entries must remain identical. Historical Git
commits are preserved by an ordinary additive commit and normal push, with no
reset, rebase, amendment or force push.

Remaining blockers are the missing auditable Task 48 failure evidence, fresh
compliant physical availability and explicit new-attempt authorization; actual
clipping/step/save peaks and saved-checkpoint reload remain unmeasured until
those controlled experiments. Below the unchanged floor, or without those
prerequisites, the local pilot decision is **NO-GO**.
