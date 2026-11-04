# Task 24: SmolLM2-135M memory admission assessment

**Decision: the 32 GiB check is inherited from Llama; actual SmolLM2 fit remains unresolved. Real training remains NO-GO.** No guard is lowered, bypassed or automatically calibrated. No model weights were downloaded or loaded, no full model was instantiated, and no backward pass, optimizer update or paid resource was used in this task.

**Task 25 follow-through:** the [separate physical read-only preflight](smollm2_readonly_preflight.md) now prepares an owner-run pinned-checkpoint load/forward measurement. Its implementation retains current admission numbers and produces a distinct read-only report; it neither changes this assessment policy nor supplies the missing training-memory evidence. It was not physically executed during preparation.

## Failure provenance and limits of the available evidence

The Task 23 Terminal transcript is not present in the repository or supplied task context. Its exact readings cannot be audited here; do not substitute Task 20's historical 16.31 GiB available reading for Task 23. The source reconstructs the reported memory-readiness failure as:

```text
STOP: offline preparation failed: STOP: training requires 32 GiB available and 20% system memory free
```

This is the pre-Task-24 message produced **if that threshold branch was reached**, not a captured Task 23 observation. The condition tests both available memory and the pressure indicator, so that message alone cannot identify which failed. Unavailable MPS, missing allocator telemetry or less than 48 GiB total capacity have separate stops. Task 24 adds explanatory context to the same exception; it does not intercept it to admit execution.

The source path is `small_model_experiment.main()` → `check_physical_mps_readiness()` → `graddiff_real_train.guarded_memory_snapshot(... optimizer="sgd", before_load=True)` → `check_training_memory()`. The readiness call never supplies model identity or parameter count to the guard. `LlamaForCausalLM` is also SmolLM2's architecture class, but that does not make the two experiments' memory footprints equal.

The [Task 18 rationale](low_memory_unlearning_options.md#go/no-go-criteria-before-any-real-weight-update) and [Task 19 implementation](graddiff_low_memory_runner.md#runtime-guards-and-estimates) establish the origin: for **1,235,814,400 Llama parameters**, round up `9.21 GiB FP32 weights+gradients + 12 GiB workspace + 4 GiB uncertainty + 6 GiB free reserve = 31.21 GiB` to 32 GiB. The small-model profile and Task 22 checker deliberately retained that policy. Git commits `ab6c0d4`, `c753029`, then `90749e3` preserve this sequence.

The inherited threshold is conservative policy, not a size-derived SmolLM2 minimum or a measured Llama guarantee. It remains appropriate as the **binding safety gate pending evidence**. There is no evidence that exactly 32 GiB is necessary or sufficient for SmolLM2. Failing that gate establishes failure of current admission policy; it does **not establish physical hardware insufficiency**. Conservative free+speculative pages exclude reclaimable caches, and host capacity is different from currently available memory. MPS visibility in this sandbox was false during metadata-only verification; this does not diagnose the physical device's capacity or compatibility.

## Estimates, never measurements or admission limits

Use the frozen [SmolLM2 profile](../experiments/smollm2_135m_tofu_controlled.json): `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, **134,515,008 parameters**, FP32, batch one, sequence length 512, SDPA, non-reentrant checkpointing, cache disabled, zero-momentum SGD. One resident model and sequential forget/retain graphs are assumed. The inference estimates describe one no-grad forward at length 512; they do not certify arbitrary generation/context lengths, multiple models or simultaneous graphs.

All sizes below are **analytical estimates or speculative planning allowances**. GiB means `2^30` bytes. No actual MPS model allocation has been observed.

| Component | Inference estimate | SGD training estimate | Basis / limitation |
|---|---:|---:|---|
| FP32 model weights | 0.501 GiB | 0.501 GiB | `4P = 538,060,032` bytes; tied embedding counted once |
| FP32 gradients | 0 | 0.501 GiB | Another `4P`; inference assumes no grad graph |
| SGD optimizer state | 0 | 0 | Zero momentum; no per-parameter state, not zero temporary work |
| Persistent tensor floor | 0.501 GiB | 1.002 GiB | Excludes all activations, framework and transient allocations |
| Activations / logits / loss / workspaces | 0.5–2 GiB | 2–4 GiB | **Unmeasured allowance**, including backward/recomputation for training; not a tensor-derived upper bound |
| Runtime overhead | 1–2 GiB | 1–2 GiB | **Unmeasured allowance** for interpreter/framework, Metal driver heaps, caching and fragmentation; not an OS reserve |
| Working plan, rounded | 2.00–4.50 GiB | 4.00–7.00 GiB | Sum of above; **not an upper bound**; load/save copies still need measurement |
| Host planning envelope, rounded | 18.00–20.50 GiB | 20.00–23.00 GiB | Adds speculative 12 GiB OS/other-process reserve and 4 GiB uncertainty; not required free memory |

AdamW is a comparison only and is rejected by this pinned SGD assessment. Its FP32 first/second moments would add `8P = 1,076,120,064` bytes (**1.002 GiB**); weights+gradients+moments floor would be **2.004 GiB**. FP32 does not require a separate mixed-precision master-weight copy. Implementation temporaries, step bookkeeping and loader copies are additional. Each extra full FP32 copy alone would add approximately 0.501 GiB. The 269,060,552-byte pinned weight-file metadata is a disk size, not FP32 training memory.

For activation scale, one FP32 logits tensor is `1×512×49,152×4 = 96 MiB`; one hidden tensor is `1×512×576×4 = 1.125 MiB`. One hidden checkpoint per each of 30 layers would be 33.75 MiB, **an illustration, not all saved activations**. Attention/MLP intermediates, non-reentrant checkpoint behavior, loss/log-softmax copies, backend temporaries and allocator caching require measurements. The 368 verified encodings top out at 119 tokens, but this does not justify abandoning the 512-token stress case.

The older profile's 3–5 GiB working / 19–21 GiB host plan used a combined 2–4 GiB activation/workspace allowance with some overhead uncertainty inside it. The new assessment makes runtime overhead an additional explicit allowance. Neither plan is measured; the frozen experiment estimates, manifest hash and settings remain unchanged. Estimates cannot select a lower threshold.

## Separate small-model preflight design

The [assessment module](../src/small_model_memory.py) and [separate policy](../experiments/smollm2_135m_memory_preflight.json) bind to the exact frozen experiment digest `260db810b3b1fbd4da0870725bcb7cac9a2ae4a472dbdb50c089007d35b711eb`. Selection validates the model/revision, precision, optimizer, checkpointing, sequence/batch settings and profile hash. It rejects Llama, alternative models, CPU, momentum/AdamW and configuration drift rather than selecting by the shared architecture class. Unknown policy keys, type errors, numeric overrides or attempts to supply an approved threshold fail closed.

This is an **offline assessment, not an executable model runner**. It imports neither Torch nor Transformers. It can assess a local future evidence JSON, but it cannot load weights, collect physical measurements, perform training, or install a guard in the Llama runner. The original Llama code/configurations are unchanged. The original small-model physical check still calls the same Llama SGD gate. The 48 GiB host minimum, 32/42 GiB pre-load thresholds, 6 GiB ongoing reserve, 20% pressure criterion, device/fallback checks and output protections remain binding.

Default assessment reports `missing_measurements`, an unset approved threshold and `runtime_admission_allowed=false`. Even complete evidence yields only an **unapproved proposal**, with admission and training still false. CLI exit status is **2 for every assessment**; a successful JSON assessment must never masquerade as admission. Editing the policy to populate an approved threshold is rejected; adoption needs a separately reviewed implementation/policy change and explicit authorization. This task approves no training.

Future evidence contract (implemented parser, no measurements supplied):

- Top-level schema `1.0`, exact profile and policy SHA-256 digests, model ID/revision, `device="mps"`, `fallback_enabled=false`, `optimizer_updates_performed=0`, hardware ID, total physical bytes, OS/Torch/Transformers versions, and at least three `trials`.
- Each trial contains `cold` and `warm` runs. Each run declares `sampling_interval_ms` in 1–100 and an ordered `samples` list; timestamp gaps exceeding the declared interval are rejected. Stage boundaries are `baseline`, `loaded`, `forward_actual`, `forward_512`, `backward_512`, `clipping`, `serialization`, `teardown`; additional samples within stages retain observed peaks. Cold means a fresh process, not forced OS cache purging. Warm keeps framework/allocator caches. Use one model at a time.
- Every sample requires `stage`, increasing `timestamp_ns`, conservative `available_physical_memory_lower_bound_bytes`, `system_memory_free_percentage`, `process_current_rss_bytes`, `process_peak_rss_bytes`, `mps_tensor_allocated_bytes`, `mps_driver_allocated_bytes`, `swap_used_bytes`, and cumulative `pageouts`. Maintain at least 6 GiB available and 20%; reject negative, boolean, nonfinite, missing or inconsistent readings. RSS peak must cover current RSS; driver allocation must cover tensor allocation. Host capacity must remain at least 48 GiB. Swap/page-out changes invalidate that calibration run. Measurement collection must handle telemetry errors as stops.
- Each run's baseline must satisfy the **unchanged 32 GiB admission gate**. The parser requires all later stages, so a forward-only trace is insufficient for a training-memory proposal. JSON metadata is an assertion, not authenticated device evidence; independent review must inspect raw traces, provenance, actual token lengths, no-fallback operation and all stage semantics.

For every run, take the **maximum**, not the sum, of baseline-relative availability decline, process peak RSS minus baseline current RSS, and MPS driver-allocation growth; then take the worst excursion across trials. RSS and MPS share unified physical memory, so summing them would double count. Using baseline current RSS rather than historical baseline peak errs conservatively when a process already had a high-water mark. Proposed available memory is `ceil_GiB(worst excursion + max(4 GiB, 25% of excursion) + 6 GiB reserve)`. It is a review aid, not a new guard. The 13 GiB result in a synthetic unit fixture is **not physical evidence or a recommended threshold**.

Sampling, even at 100 ms or faster, can miss peaks. Phase boundaries alone are especially insufficient; a future collector must record repeated samples through load, compute, clipping and serialization, synchronize MPS appropriately, capture transient peaks where possible, and stop on telemetry loss, growing swap/page-outs or pressure. The parser validates the supplied trace format, not completeness or truth of physical observations. Driver/RSS/system views are overlapping diagnostics affected by other processes. Loader/save transients not covered by measurements require further allowances before any threshold can be reviewed. Recalibrate after hardware/software/profile changes; there is no transferable approval in this JSON.

## Next safe physical-device verification step

In a normal macOS Terminal, inside this repository, run **telemetry and cached metadata/tokenizer verification only**:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-ml/bin/python -B \
  -m src.small_model_experiment validate --verify-tokenizer --check-mps-readiness
```

Record the complete STOP/success output, timestamp, MPS visibility, total memory, free+speculative memory and pressure indicator. On a guard failure the current CLI does not emit the snapshot, so separately capture the no-model physical readers `vm_stat`, `memory_pressure -Q`, `sysctl hw.memsize`, and `sysctl vm.swapusage` alongside the error; these are adjacent observations, not the exact failed snapshot. Keep Activity Monitor visible. No model is loaded and no optimizer is constructed by the command. Do not lower the gate, enable fallback, clear caches, kill processes automatically or change allocator limits. If admission fails, stop and defer or manually close applications as the device owner chooses.

Only after successful existing admission, a separate authorized task could prepare immutable local weights and a bounded SmolLM2 read-only collector/orchestrator, with identity/path guards and pressure stops. Collect cold/warm forward traces at actual lengths and 512, compare losses, and review them first. Backward-only evidence, clipping and serialization need subsequent separate authorization and orchestration; no optimizer update is required for the calibration design. If 32 GiB cannot be reached, calibration remains blocked under current policy. This task supplies no bootstrap exception. A later measured/reviewed threshold still would not satisfy acquisition, evaluation, checkpoint or explicit update-authorization gates.

## Offline validation

```sh
PYTHONPATH=tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest \
  test_small_model_experiment test_small_model_memory test_graddiff_preflight \
  test_inspect_tofu test_evaluate_tofu_baseline test_evaluate_tofu_paired \
  test_graddiff_experiment_plan -v

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_experiment validate --verify-tokenizer

.venv-ml/bin/python -B -m src.small_model_memory
```

Completed: **65 selected offline tests passed**, including **11 new memory assessment tests**; all **368 cached-tokenizer encodings** passed; **40 local links** and **nine experiment JSON files** checked; whitespace checks passed. The original Llama source/guards, frozen small-model profile/manifest and existing experiment configurations remain byte-for-byte unchanged.

The assessment command intentionally exits 2 with a structured STOP. The selected tests avoid suites that execute synthetic backward passes or optimizer steps. Read-only preflight tests use tiny fixtures and mocked pretrained loaders; they do not instantiate the full model. Coverage includes pinned model/guard selection, original guard boundaries, missing/invalid evidence and configurations, pressure/allocator/swap failures, unknown overrides, trace stages/provenance, sampling gaps, proposal arithmetic without double counting, and STOP even with complete synthetic evidence. Local metadata/tokenizer validation checks all 368 encodings without weights. No physical model memory or runtime fit has been measured.
