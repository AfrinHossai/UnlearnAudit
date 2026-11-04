# SmolLM2-135M read-only MPS preflight (Tasks 25–27)

**Ready for an owner-run normal Mac Terminal measurement under the separate Task 27 read-only policy: at least 16 GiB available before load, increasing with a large measured baseline process. Training remains blocked and the approved small-model training threshold remains unset.** Task 26 stopped before checkpoint access at the inherited 32 GiB gate. No model-load/forward measurement was obtained. See the [Task 27 audit and budget](smollm2_readonly_admission_task27.md) for exact provenance, estimates, limits and offline validation.

The separate [preflight](../src/small_model_preflight.py) defaults to a plan with no ML imports, network, model allocation or output file. `--run-preflight` enables one frozen FP32 MPS forward; only the additional `--allow-download` permits acquisition of missing official checkpoint files. It never calls backward, constructs an optimizer, updates parameters, generates a continuation or writes model checkpoints. The [Task 24 assessor](smollm2_memory_admission.md), original Llama guard, frozen experiment profile/manifest and existing training policies are unchanged.

The Task 25 request spells the earlier document `smolllm2_memory_admission.md`; the existing reviewed file is [smollm2_memory_admission.md](smollm2_memory_admission.md).

## Identity, license and local storage

Use only `HuggingFaceTB/SmolLM2-135M` at **`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`**. The [official model card at that revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md) declares **Apache-2.0**. Its content and config/tokenizer hashes match the [frozen experiment profile](../experiments/smollm2_135m_tofu_controlled.json). The cached official Hub metadata records the same revision, 134,515,008 BF16 source parameters, and weight LFS hash. Task 25 verified the pinned card/license and cached metadata without acquiring weights.

At runtime, verify the card first, including SHA-256 and license declaration, then the config/tokenizer files, before downloading or loading weights. The loader pins:

- Weight file: `model.safetensors`, **269,060,552 bytes** (about 269 MB decimal / 257 MiB).
- Weight SHA-256: **`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`**.
- Card SHA-256: **`d1ba68cae64a89b6b434b11526e6e2271ee5ffd2c914ec35ed515f9d84c6085c`**.
- Checkpoint directory: `.cache/smollm2_readonly/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/`.

Download payload for weights plus the five required metadata/tokenizer files is approximately **271 MB decimal / 259 MiB**. Budget **at least 2 GiB free disk** for the local payload, temporary transfers, Hub metadata/cache and a small JSON report. This is a storage planning allowance, not measured memory consumption. FP32 conversion happens in RAM; the program does not save a converted model. An interrupted transfer may leave ignored partial cache files. No automatic cleanup or replacement of a checksum-mismatched existing file occurs.

Explicit acquisition uses the official HTTPS endpoint, exact revision and a six-file allowlist, without authentication or remote code. Hub/Xet cache paths stay under the repository; Xet transfer is disabled in favor of the HTTP client. All model/config/tokenizer loading is subsequently `local_files_only=True`, `trust_remote_code=False`, with safetensors required. Unknown checkpoint files such as adapters/alternate weights, symlink paths, wrong hashes, unexpected/missing weight keys, architecture/count/config drift or a changed frozen profile stop execution. No private data, TOFU file or external synthetic dataset is needed.

## Exact normal-Terminal command

Use the existing project environment from a **fresh, normal macOS Terminal on Apple Silicon**. The program checks the pinned Torch 2.9.1, Transformers 4.57.1 and huggingface-hub 0.36.2 versions. Keep Activity Monitor visible. The command below explicitly allows the approximately 271 MB download **when you run it**; it was not executed during Task 27:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
mkdir -p outputs
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  -u PYTORCH_MPS_HIGH_WATERMARK_RATIO -u PYTORCH_MPS_LOW_WATERMARK_RATIO \
  -u PYTORCH_MPS_FAST_MATH -u PYTORCH_MPS_PREFER_METAL \
  PYTORCH_ENABLE_MPS_FALLBACK=0 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.small_model_preflight \
  --run-preflight --allow-download \
  --report-path outputs/smollm2_readonly_task27_first.json
```

If model/tokenizer acquisition is undesired, omit `--allow-download` and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; the complete exact snapshot must already exist at the dedicated path. Other model caches are not silently reused. Use a **new report filename** for every invocation. Reports must be JSON files under an existing Git-ignored `outputs/` directory; existing files, symlink parents and paths outside that tree are rejected.

Remove any pre-existing `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_MPS_LOW_WATERMARK_RATIO`, `PYTORCH_MPS_FAST_MATH` or `PYTORCH_MPS_PREFER_METAL` variable before starting a fresh process; the program rejects overrides. It does not change allocator limits, purge caches, trigger memory pressure, terminate other processes or enable CPU fallback. Unsupported deterministic MPS kernels, an OOM or an unavailable telemetry API produce STOP rather than a retry on CPU.

An exit code of **0** with `status="readonly_measurement_passed_not_training_clearance"` means that this single forward and its observations passed their checks. Exit **2**, a STOP report, an incomplete JSON or no report means failure. Early path/environment/device errors print STOP to stderr without a report. Failures after monitoring is initialized retain available observations in a new STOP report where possible; partial output after a process/OS failure is never a passing artifact. Inspect both exit status and the complete JSON before using any observations.

## Admission and bounded execution

This is a separate physical read-only implementation. Task 27 replaces its copied training preload requirement with an explicitly scoped estimated budget: **at least 16 GiB conservative availability before acquisition and immediately before model loading**, increasing if the fresh process peak RSS exceeds its 1 GiB headroom allowance. It preserves **48 GiB total host capacity, 6 GiB continuing availability, and at least 20% system memory-free indicator**. It additionally requires a normal kernel memory-pressure level, no observed swap growth/new page-outs and valid, fresh RSS/MPS readings. Two observations precede checkpoint access; swap/page-out checks compare adjacent samples. The source contains **no threshold override switch**. Exact arithmetic and allowance uncertainties are documented in the [Task 27 rationale](smollm2_readonly_admission_task27.md#read-only-budget-and-uncertainty).

The former **32 GiB** floor came from Llama SGD weights+gradients and training workspace. It was not a SmolLM2 read-only estimate. The new **16 GiB minimum is an estimated read-only admission policy**, not a measured peak or proof of fit. Failing any current check still means STOP. The original Llama guards, small-model preparation readiness, frozen training profile and Task 24 training assessor retain their thresholds. No read-only result calibrates or authorizes training.

The one project-authored synthetic example is `What color is a clear daytime sky?` → `Blue.`; its text is dedicated under CC0-1.0 for this probe. It uses the frozen plain-QA encoding, answer-only labels and existing EOS, without vocabulary changes or truncation. The cached official tokenizer verified **17 total tokens / three supervised answer+EOS tokens**; a 128-token hard cap prevents accidental growth. This is a small functional measurement, not a 512-token stress allocation or a TOFU evaluation. No generation loop or KV cache is used.

After verified local loading on CPU, parameters are frozen with `requires_grad_(False)`, the model is put in eval mode, and every parameter's count/dtype/device/gradient state is checked. The sole model is transferred to MPS; model parameters, buffers, inputs and outputs must reside there. The forward and causal-loss validation run in **`torch.inference_mode()`**. Seed 2026 and strict deterministic algorithms are requested; unsupported operations stop. Model/local answer NLL must agree to `1e-4` relative/absolute tolerance, all logits/losses must be finite, output shapes/dtypes must match, and parameter version counters must remain unchanged with no gradients. These checks do not establish semantic correctness or cross-device bitwise reproducibility.

## Measurements and their interpretation

The JSON schema is `1.0`, scope `readonly_single_forward`. It records model/revision/profile digest and checked checkpoint hashes, software/OS/machine, requested/telemetry/actual model device, input digest/counts, finite loss/parity result, frozen/version checks, raw observations and peak summary. Task 27 adds `readonly_memory_policy` with its distinct guard ID, baseline process peak and exact estimated budget terms. `training_authorized=false`, `backward_performed=false`, `optimizer_created=false`, `weight_updates=0` and an unset approved training threshold remain explicit.

| Field / stage | Method | Interpretation |
|---|---|---|
| `process_current_rss_bytes` | `ps -o rss=` multiplied from KiB | Direct OS RSS reading with unit conversion; includes the process, not just model tensors |
| `process_peak_rss_bytes` | Darwin `getrusage(RUSAGE_SELF).ru_maxrss` | Direct **process-lifetime** high-water RSS, including imports and CPU loading; not reset or a per-phase allocation peak |
| `mps_tensor_allocated_bytes` | `torch.mps.current_allocated_memory()` | Direct tensor-allocation counter; excludes unused allocator cache |
| `mps_driver_allocated_bytes` | `torch.mps.driver_allocated_memory()` | Direct Metal driver counter, including allocator caches and framework allocations |
| `sampled_maxima_bytes` | Maximum observed RSS/MPS values | Derived from direct observations; sampled MPS maxima are **lower bounds on true peaks**, not exact peaks or safe upper bounds |
| `native_mps_peak_bytes` | `null` | Pinned Torch 2.9.1 provides no native public MPS peak counter; never substitute CUDA APIs or invent a value |
| `available_physical_memory_lower_bound_bytes` | `vm_stat` free+speculative pages × page size | Derived conservative availability; excludes inactive/purgeable/compressed/reclaimable caches |
| `system_memory_free_percentage` | `memory_pressure -Q` | Direct macOS free indicator, not an independently calculated percent of unused RAM |
| `macos_pressure_level` | Read-only `kern.memorystatus_vm_pressure_level` sysctl | Require NORMAL=1; warning/critical/unknown or unavailable is STOP |
| Swap / page-outs | `sysctl vm.swapusage`, `vm_stat` | Rounded swap MiB converted to bytes and cumulative page-outs; growth stops; swap changes smaller than printed precision can be missed |

Counter semantics were checked against the [pinned PyTorch 2.9.1 MPS implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/mps/__init__.py). Its [pinned allocator](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/mps/MPSAllocator.mm) does not implement the newer generic allocator statistics either; generic accelerator peak helpers must not be treated as validated MPS peak counters for this version. Apple's [XNU pressure sysctl implementation](https://github.com/apple-oss-distributions/xnu/blob/xnu-10063.101.15/bsd/kern/kern_memorystatus_notify.c) exports mapped notification levels; [notification definitions](https://github.com/apple-oss-distributions/xnu/blob/xnu-10063.101.15/bsd/sys/event_private.h) define NORMAL=1. This pressure interface is private/OS-dependent; if absent or denied the probe stops, with no permissive fallback.

Capture synchronized boundaries `before_load`, `after_cpu_load`, `after_load` (MPS), `after_inference_outputs_live` (logits/loss still resident), and `after_inference` (forward locals released, model retained). `baseline` and `before_checkpoint` observations, separated by a 250 ms wait, precede acquisition/verification. A watchdog also samples during CPU loading, MPS transfer and inference, targeting a 250 ms wait between readings. Report each sample's monotonic timestamp and collection duration, maximum observed sampling gap, and sample count. The observations are **not atomic**; OS commands and MPS reads happen sequentially. The watchdog adds process overhead.

No measured values are produced by the default plan. Its weight-file bytes are verified metadata; its 2 GiB free-disk recommendation is an **estimate**. FP32 weight/activation/runtime estimates remain in the earlier assessment and must not be relabeled as observations. Do not add RSS and MPS allocations: their physical unified-memory views overlap. Other applications, file caching, allocator reuse and prior high-water RSS affect results.

## Limitations and next evidence

The watchdog requests interruption on unsafe/missing telemetry, sample collection longer than two seconds, or a load/forward monitoring period exceeding 180 seconds. Individual OS readers have two-second timeouts. Interruption is cooperative at Python safe points and **cannot guarantee preemption of an active MPS/native operation**, prevent an OS kill or guarantee capture of every transient. Sampling can miss brief peaks; the target interval is not a promise. Acquisition/metadata verification occur before the sampling thread starts and can cause gaps; no download-time peak is claimed. The final summary exposes observed gaps rather than claiming a continuous native peak measurement. Stop manually with Ctrl-C if Activity Monitor shows warning/critical pressure or responsiveness deteriorates.

There is no full-model, real-MPS forward or deterministic-kernel result from preparation. The exact CLI's runtime APIs, permissions, physical telemetry and weight loading must still be exercised on the Mac. One 17-token forward cannot establish 512-token behavior, backward/optimizer/serialization requirements, warm/cold variability, scientific utility, forgetting or a safe training threshold. It produces a distinct read-only schema, **not the complete three-trial cold/warm/backward evidence required by the Task 24 assessor**. Do not feed it into that assessor as training-memory evidence or lower its guard to accept it.

After the owner-run probe, review the complete passing/STOP artifact and counter provenance. Any further longer-sequence, repeated, backward-only or training work remains a separate task with its existing safeguards. Training is not approved by this implementation or a passing read-only result.

## Preparation validation

Run the default offline plan safely:

```sh
.venv-ml/bin/python -B -m src.small_model_preflight
```

Current applicable tests, with all acquisitions and physical operations mocked and no actual inference/backward/optimizer operations:

```sh
PYTHONPATH=tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest \
  test_small_model_experiment test_small_model_memory test_small_model_preflight -q
```

**Task 27: 48 selected offline tests passed.** Coverage includes explicit opt-ins, offline imports/defaults, local-only loading, model freeze/inference-mode enforcement, phase/output schema, pressure/reserves, invalid/absent/stale counters, swap/page-outs, device/env/config/loading mismatch, budget/headroom arithmetic, threshold boundaries, unchanged training gates, disabled updates, OOM/watchdog errors, invalid/nonfinite outputs, loss parity, parameter mutations, checkpoint/license/hash checks, mocked acquisition ordering and report overwrite/symlink protections. Acquisition tests use tiny fabricated bytes and mocks; no official weights were acquired or loaded. Tests do not depend on ignored metadata caches. Separately, the cached official config/tokenizer/card hashes were verified offline. The 17-token encoding was verified in Task 25. Suites performing actual tiny-model forward/backward or optimizer operations were excluded from Task 27.
