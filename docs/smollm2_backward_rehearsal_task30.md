# Task 30: guarded SmolLM2 backward-only memory rehearsal

**Ready to attempt in a fresh normal Apple Silicon macOS Terminal, subject to live admission.**
The separate [runner](../src/small_model_backward.py) defaults to offline validation.
Its only execution flag, `--run-backward-rehearsal`, explicitly opts into two
forwards and one backward on one fixed forget/retain pair. It has no optimizer,
optimizer state, optimizer step, model serialization, checkpoint-write, download,
or cloud path. No full-model backward was executed in Codex. No training,
fact acquisition, unlearning, or forgetting result is claimed.

The remaining physical blocker is **unverified live memory admission and backward
behavior**. The minimum preload availability is **20 GiB**, potentially higher
with a large measured process baseline. Task 29's historical preload observation
was only **16.805878 GiB**; the same reading would stop this rehearsal before
loading. Do not lower a threshold to obtain a pass. Local checkpoint and source
files are present and verified; their absence/corruption elsewhere is also a stop.

## Reviewed Task 29 evidence

The verified, ignored local report is
`outputs/smollm2_readonly_task27_first.json`, SHA-256
`8df89af2e27aae9f75ffe577e29e58322522f442d6be359bfa8d40a592375a74`.
Its filename refers to the prepared Task 27 runner; Task 29 supplied the physical
read-only result. Its status is
`readonly_measurement_passed_not_training_clearance`, requested/telemetry/actual
device all `mps`, `model_loaded=true`, and FP32. It records frozen parameters,
unchanged parameter versions, no backward, no optimizer, zero updates and
`training_authorized=false`. The model and local answer NLL both equal
`3.831789255142212`. This is evidence of the recorded read-only pass and unchanged
weights under its freeze/version checks; it did not hash every in-memory
parameter before/after. Task 30 adds that stronger byte check for its own run.

| Observed counter in Task 29 | Recorded bytes / value |
|---|---:|
| Total physical capacity | 51,539,607,552 (48 GiB) |
| Available before model load | 18,045,173,760 (16.805878 GiB) |
| Available after MPS load | 16,575,528,960 (15.437164 GiB) |
| Minimum sampled availability | 16,470,999,040 (15.339813 GiB) |
| Maximum sampled current RSS | 897,875,968 (0.836212 GiB) |
| OS process lifetime peak RSS | 1,166,884,864 (1.086746 GiB) |
| Maximum sampled MPS tensor allocation | 541,403,904 (0.504222 GiB) |
| Maximum sampled MPS driver allocation | 1,145,847,808 (1.067154 GiB) |
| Kernel pressure / free indicator across samples | NORMAL=1 / 91–95% |
| Swap used / page-outs across samples | 0 / 0 |

There were nine samples with a 250 ms target interval and an observed maximum
gap of 10.157285750 seconds, including setup before continuous monitoring.
Sampled maxima are lower bounds on peaks. RSS and MPS overlap in unified memory
and must not be summed. These are **forward observations**, not measured backward
requirements, a calibration result, or training authorization. The previous
16 GiB read-only policy is not used to admit backward.

## Frozen identity, data and loss

The [controlled experiment](smollm2_tofu_experiment.md), its
[unchanged profile](../experiments/smollm2_135m_tofu_controlled.json), and the
[validated GradDiff loss](graddiff_validation.md) were reviewed. The runner binds
to the exact frozen profile digest
`260db810b3b1fbd4da0870725bcb7cac9a2ae4a472dbdb50c089007d35b711eb` and
`HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It reads only the existing dedicated
`.cache/smollm2_readonly/<revision>/` snapshot, verifying all metadata, card,
tokenizer, config and weight hashes, size, architecture and Apache-2.0 license.
The weight SHA-256 is
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
All Transformers loads are local-only, remote code is disabled, and offline mode
is forced before importing Transformers/Hub. There is no download override.

One batch per split comes from `locuslab/TOFU` revision
`324592d84ae4f482ac7249b9285c2ecdb53e3a68`, MIT license. Selection is the first
pair of the existing [manifest](../experiments/smollm2_135m_tofu_manifest.json):
**`forget01:0000` / `retain99:0501`**. The manifest file hash is
`0ba5d80715dd9fb0c542d16f5167c23a0464da89a2f413d6a6bbd18235f2ea21`.
Whole source-file hashes/counts/schema and selected question/answer hashes are
verified. Only cached `forget01.json` and `retain99.json` are read; sealed holdout,
paraphrases and evaluation scores are not consulted. No example is selected by
loss, gradient size, or apparent memory fit. Reports contain provenance hashes,
indices and lengths rather than QA text.

`encode_plain_qa()` renders `Question: {question}\nAnswer:\n{answer}`, requires
an exact prompt-token prefix, masks the prompt with `-100`, supervises the
answer plus EOS, adds no other special tokens and refuses truncation. Existing
EOS ID 0 is used for right padding without embedding resize. Offline verification
with the actual pinned tokenizer produced:

| Split | Tokens including EOS | Prompt tokens masked | Supervised tokens |
|---|---:|---:|---:|
| Forget | 74 | 33 | 41 |
| Retain | 42 | 19 | 23 |

Their input/label digests are respectively
`b0f45ba4f061fa615977986f8e276dcf413a94e39e4eb570a6af03571681a0bc`
and `4d0cb5dec469c36237d2598834b1375ac25a2005e8a84713b69d75510de1d28d`.
The hard cap is **128 tokens per sequence**, batch one per split. This is a
separate bounded rehearsal, not the frozen experiment's length-512 stress case.

The actual model loads FP32 with SDPA and `use_cache=false`. Parameters become
gradient-enabled only after frozen CPU/MPS identity/device validation. The
model's train mode activates non-reentrant gradient checkpointing with
`preserve_rng_state=true`; it does not imply parameter updates. Seed 2026 and
strict deterministic algorithms are requested. Unsupported deterministic MPS
kernels fail closed; cross-platform bitwise reproducibility is not claimed.
Both graphs are held for one scalar
`-mean_NLL(forget) + mean_NLL(retain)`, `gamma=alpha=1`, followed by exactly one
`backward()`. Each model loss is compared to `causal_lm_nll()` over detached
logits, with `atol=rtol=1e-4`; logits, both losses, objective and all gradients
must be finite. Every trainable parameter must receive a compatible FP32 gradient,
and at least one gradient must be nonzero. No gradient clipping is performed.

Parameter SHA-256 covers names, shapes, dtypes and actual FP32 bytes before the
forwards and after backward; version counters must also remain unchanged.
Hashing copies one parameter to the host at a time, never a second full model.
Gradients are cleared with `zero_grad(set_to_none=True)`. The complete pinned
checkpoint and metadata hashes are checked again after computation. A successful
report requires matching parameter hashes, versions, checkpoint provenance,
finite/nonzero gradients, cleanup and all required telemetry boundaries.

## Separate rehearsal admission policy

Guard `smollm2_135m_fp32_mps_one_pair_backward_only_v1` applies only to this
exact model, profile, device, precision and one-pair operation. There is no
threshold or sequence override. Let `P=134,515,008`, `W=4P=538,060,032` bytes,
`F=269,060,552` bytes and `B` be freshly observed process-lifetime peak RSS:

```text
incremental estimated budget = 3W + W + F + 4 GiB + 1 GiB + 1 GiB
                               + max(1 GiB, B) + 4 GiB
preload availability required = 6 GiB + ceil_GiB(incremental estimated budget)
```

| Estimate / policy allowance | Rationale |
|---|---|
| `3W` = 1,614,180,096 bytes | Resident FP32 weights plus two loading/conversion/transfer copies; simultaneous copies are unmeasured |
| `W` = 538,060,032 bytes | Full FP32 parameter gradients; optimizer state is absent |
| `F` = 269,060,552 bytes | Entire source file residency allowance, not measured residency |
| 4 GiB workspace | Two checkpointed forward graphs, saved activations, logits/loss copies, backward/SDPA/MLP temporaries; unmeasured |
| 1 GiB host/tokenizer/integrity | Tokenizer, host caches and per-parameter byte hashing temporaries; unmeasured |
| 1 GiB Metal/runtime | Allocator caches, initialization and fragmentation; unmeasured |
| `max(1 GiB, B)` additional process headroom | Fresh measured baseline sets an additional speculative allowance; existing RSS is already excluded from available pages |
| 4 GiB uncertainty | Conservative allowance for omitted transients and backend variability; not a statistical bound |
| 6 GiB continuing availability | OS/other-process reserve that must remain throughout the run |

With `B <= 1 GiB`, incremental estimates total **14,232,460,744 bytes =
13.255012 GiB**, rounded up to 14 GiB, plus the 6 GiB continuing reserve gives
**20 GiB preload**. For `B=3 GiB`, preload becomes 22 GiB. A single capped
FP32 logits tensor is analytically 24 MiB; neither that size nor the workspace
allowance measures all saved tensors. The budget is intentionally larger than
the read-only probe, and **is not a measured upper bound or guarantee of fit**.

The original Llama SGD **32 GiB** training/backward guard and AdamW **42 GiB**
guard are unchanged. The preparation readiness command and Task 24
[training-memory assessor](smollm2_memory_admission.md) retain their existing
policies and STOP behavior. No `src/graddiff*`, existing experiment policy,
profile or manifest was edited. A rehearsal result cannot satisfy that assessor's
multiple cold/warm trials, length-512, clipping and serialization contract.
`training_authorized=false` and the approved small-model training threshold
remains unset even on success. Any future weight update needs a separate task,
policy/evidence review, scientific controls and explicit authorization.

## Telemetry and stop conditions

The existing physical reader observes current RSS (`ps` KiB converted to bytes),
Darwin process-lifetime peak RSS (`getrusage` bytes), MPS current tensor and driver
allocations, free+speculative physical pages (`vm_stat`), `memory_pressure -Q`,
kernel pressure level, rounded swap used (`sysctl vm.swapusage`) and page-outs.
Measurements and estimated policy terms are separate report fields. RSS/MPS
views overlap and are never added to infer physical usage.

Two fresh samples 250 ms apart must pass before checkpoint verification, then a
fresh preload check runs after tokenizer/encoding work. Background monitoring
covers model loading, MPS transfer, hashing, forward graphs, backward, gradient
checks, cleanup and checkpoint verification. Synchronized boundaries include
`before_load`, `after_cpu_load`, `after_load`, each forward, graphs immediately
before backward, live gradients after backward, cleanup and final checkpoint
integrity. Background samples during backward carry stage `backward`.

Stop on any of the following; the runner does not automatically clear caches,
kill other processes, retry on CPU, change allocator limits or relax thresholds:

- Less than 48 GiB total capacity, less than the computed preload requirement,
  or less than 6 GiB continuing availability; free indicator below 20% or kernel
  pressure other than NORMAL=1. Also stop manually for Activity Monitor yellow/red
  pressure or rapidly rising swap.
- Missing, negative, boolean, inconsistent, nonfinite, stale/future or zero-duration
  telemetry; changed capacity; unordered timestamps; active-monitor sample gaps
  exceeding two seconds. Setup before continuous monitoring may have longer gaps
  and is followed by fresh admission readings.
- Any observed swap increase compared to the immediately preceding sample or
  any page-out change. A decline followed by a rise still stops. Existing swap
  occupancy alone does not prove active pressure; rounded readings can miss growth.
- Unavailable/non-MPS device, mismatched parameter/buffer/input/output/gradient
  device, CPU fallback enabled, allocator/math overrides, multiple-process
  settings, unsupported hardware/software versions, checkpoint/source/config drift,
  uncertain masks, overlength input, missing gradients, nonfinite values or loss
  parity failure. Pinned versions: Torch 2.9.1, Transformers 4.57.1, Hub 0.36.2.
- Any OOM, kernel/determinism error, interrupt, watchdog error, changed parameters
  or checkpoint provenance. The watchdog has a 180-second active-run budget.

The watchdog samples at a 250 ms target and cooperatively interrupts at Python
safe points. It cannot forcibly preempt a long native kernel, guarantee every
transient is observed, or guarantee a mid-kernel sample for very short backward
execution. Sampling is non-atomic and affected by other applications; native MPS
peak counters are unavailable in this pinned Torch. Reports include raw samples,
observed maxima and actual gaps; sampled maxima remain lower bounds. A pass
measures this one short pair only, not length 512, longer runs, clipping,
serialization, optimizer memory, acquisition or unlearning. Determinism and
memory behavior remain physically unverified until the owner runs it.

## Exact future normal-Terminal command

Run later in a fresh normal Terminal with Activity Monitor visible. No Codex
sandbox invocation of this full-model command was made. Keep the existing local
checkpoint; no downloads are supported or required. Use a new report filename.

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
mkdir -p outputs
env -u PYTORCH_MPS_HIGH_WATERMARK_RATIO -u PYTORCH_MPS_LOW_WATERMARK_RATIO \
  -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv-ml/bin/python -B -m src.small_model_backward \
  --run-backward-rehearsal \
  --report-path outputs/smollm2_backward_task30_first.json
```

Require exit **0** and complete
`status="backward_rehearsal_passed_not_training_clearance"`, actual/telemetry
device `mps`, one backward, finite gradients, unchanged hashes/versions/checkpoint,
and no optimizer/updates/checkpoint writes. STOP/exit 2, missing output or an
incomplete report is failure. Environment/path/version failures before the
monitor exists print STOP without a report. Reports use exclusive creation in
an existing Git-ignored `outputs/` directory; existing files, symlinks, checkpoint
destinations and repository escapes are rejected. Do not interpret flags in an
incomplete/STOP report as verified parameter integrity.

For validation only, omit all options:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-ml/bin/python -B -m src.small_model_backward
```

This validates profile, manifest, checksummed source data and pair provenance,
and prints estimates. It imports neither Torch nor Transformers and performs
no telemetry, weight loading, backward or output write. Runtime additionally
verifies the cached checkpoint/tokenizer and encodings before model loading.

## Offline verification performed in Task 30

The new suite covers admission boundaries, baseline headroom, exact profile/device
selection, pressure, bad/stale telemetry, swap/page-outs, monitor gaps, explicit
opt-in, source checksums, mask/EOS/truncation behavior, device mismatch, nonfinite
losses/gradients, missing/zero gradients, OOM/watchdog failures, pressure after
backward, parameter-byte/version mutation, checkpoint drift and report integrity
and path protections. Full-model orchestration uses fake models/tensors and mocked
loaders/MPS/counters; the gradient validator uses tiny manually assigned CPU
tensors. No optimizer is constructed by the new suite.

```sh
env PYTHONPATH=tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest \
  test_small_model_backward test_small_model_experiment \
  test_small_model_memory test_small_model_preflight -q

env PYTHONPATH=tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-ml/bin/python -B -m unittest \
  test_graddiff.GradDiffPrototypeTests.test_causal_nll_is_shifted_mean_over_valid_target_tokens \
  test_graddiff.GradDiffPrototypeTests.test_objective_uses_separate_mean_losses_and_official_coefficients \
  test_graddiff.GradDiffPrototypeTests.test_matches_pinned_openunlearning_compute_loss_formula_and_gradients \
  test_graddiff.GradDiffPrototypeTests.test_extreme_finite_logits_keep_cross_entropy_and_gradients_finite \
  test_graddiff.GradDiffPrototypeTests.test_rejects_nonfinite_losses_logits_and_empty_shifted_labels -v
```

The selected loss tests operate on tiny CPU tensors without a model or optimizer;
suites that run synthetic optimizer steps were excluded. Local-only snapshot
hash verification and actual tokenizer encoding were also performed without
constructing a model. The verified Task 29 report, original checkpoint, original
Llama guards and frozen experiment artifacts remain unchanged.

**Results: 65 selected safety/regression tests passed, including 17 new rehearsal
tests; all five focused loss/gradient tests passed (70 total).** Default validation,
actual cached-tokenizer pair encoding, checkpoint hashes, Python syntax, local
documentation links, forbidden optimizer/save-call inspection and Git whitespace
checks passed. No full-model backward or new physical memory measurement was run.
