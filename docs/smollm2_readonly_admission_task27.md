# Task 27: separate SmolLM2 read-only admission

**Decision: a revised read-only preflight is justified and ready for normal-Terminal physical-device testing.** Its minimum preload availability is **16 GiB**, with additional headroom for an unusually large measured baseline process. This is a conservative, explicitly estimated admission policy for one bounded frozen forward, not a measured peak or a guarantee of fit. No physical model test was run in Task 27. Backward and training remain blocked by their existing policies; no inference estimate authorizes updates.

## Task 26 STOP and exact origin

The local, ignored artifact `outputs/smolllm2_readonly_task26_first.json` reports:

| Observation | Recorded value |
|---|---:|
| Status / failing stage | `STOP` / `before_checkpoint` |
| Total physical memory | 51,539,607,552 bytes = 48 GiB |
| Free + speculative availability | 19,383,386,112 bytes = 18.052185 GiB |
| Process current / lifetime peak RSS | both 226,738,176 bytes = 216.234375 MiB |
| Kernel pressure / free indicator | NORMAL=1 / 95% |
| Swap used / page-outs | 0 / 0 |
| MPS tensor / driver allocation | 0 / 475,136 bytes |
| Actual model device / model loaded | `null` / `false` |

At Task 25 commit `76589cd`, `src/small_model_preflight.py:386` calls `monitor.capture("before_checkpoint", before_load=True)`. `Monitor.capture()` calls `validate_sample()`, whose **lines 197–199** explicitly preserve the existing numeric policy:

```python
# Preserve current policy numerically; it is not a measured SmolLM2 requirement.
if total < 48 * GIB or not (32 if before_load else 6) * GIB <= available <= total:
    raise RuntimeError("memory safety floor failed (48 GiB capacity; 32 GiB preload / 6 GiB continuing)")
```

That exact exception appears in Task 26. Capacity and the upper availability bound pass; **18.052185 < 32** is the failing comparison. The sole observation cannot establish swap stability over time. The check occurs before `ensure_checkpoint()`, so there is no model-load or forward peak evidence. The plan's former `preload_safety_floor_bytes=32*GIB` was a display of this policy, not its enforcement source.

The earlier chain is `small_model_experiment.check_physical_mps_readiness()` → `graddiff_real_train.guarded_memory_snapshot(optimizer="sgd", before_load=True)` → `check_training_memory()`. The frozen small-model profile retained its safety numbers; Task 25 copied them into its separate probe. The documented origin is [Task 18's Llama training calculation](low_memory_unlearning_options.md#go/no-go-criteria-before-any-real-weight-update): **1,235,814,400 parameters**, approximately **9.21 GiB weights + gradients + 12 GiB workspace + 4 GiB uncertainty + 6 GiB reserve = 31.21 GiB**, rounded to 32 GiB. [Task 19](graddiff_low_memory_runner.md#runtime-guards-and-estimates) implemented it. [Task 24](smollm2_memory_admission.md) already identified the inheritance. There is no documented SmolLM2-specific read-only derivation of 32 GiB.

## Pinned identity and scope

The unchanged [profile](../experiments/smollm2_135m_tofu_controlled.json) pins `HuggingFaceTB/SmolLM2-135M` at `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, profile digest `260db810b3b1fbd4da0870725bcb7cac9a2ae4a472dbdb50c089007d35b711eb`. Cached config and all five metadata/card/tokenizer hashes were checked offline against that profile. Config: **30 layers, hidden size 576, intermediate size 1536, 9 attention / 3 KV heads, vocabulary 49,152, tied embeddings**. Cached Hub metadata records **134,515,008 BF16 parameters**, a **269,060,552-byte** safetensors file, and the pinned revision/LFS hash. The source config defaults to BF16 and caching; the probe explicitly loads FP32 and disables caching. The shared `LlamaForCausalLM` class does not select the admission threshold.

Guard `smollm2_135m_fp32_mps_single_forward_v1` is local to [small_model_preflight.py](../src/small_model_preflight.py). It requires the exact frozen profile and MPS, FP32, batch one, the existing project-authored synthetic QA, one forward, a **128-token hard cap**, SDPA, frozen eval parameters, `torch.inference_mode()`, and no KV cache or generation. The previously verified example is 17 tokens; planning uses the cap. One maximum-sized FP32 logits tensor is `1 × 128 × 49,152 × 4 = 24 MiB`; one hidden tensor is 0.28125 MiB. These are scale calculations, not all intermediates or upper bounds.

## Read-only budget and uncertainty

GiB = `2**30` bytes. Let `P=134,515,008`, `W=4P=538,060,032` bytes, `F=269,060,552` bytes, and `B` be the **fresh process-lifetime peak RSS measured before checkpoint access or model loading**. Use:

```text
incremental planning budget = 3W + F + 2 GiB + 1 GiB + 1 GiB + max(1 GiB, B) + 3 GiB
preload required availability = 6 GiB + ceil_GiB(incremental planning budget)
```

| Term | Bytes / allowance | Evidence and uncertainty |
|---|---:|---|
| Three FP32 weight-sized copies | 1,614,180,096 | One resident copy plus two extra loading/conversion/CPU-to-MPS copies. `W` is analytical; concurrent copy count is an **unmeasured allowance**. |
| Source file residency | 269,060,552 | Exact pinned disk metadata size; using the entire file as potential memory residency is an **estimate**, not a measured mapping or cache footprint. |
| Forward activations, logits, loss copies, SDPA/MLP workspaces | 2 GiB | **Unmeasured allowance**, retaining the upper inference workspace allowance from Task 24 even though this probe caps length at 128 rather than 512. |
| Tokenizer, deserialization and host/file caches | 1 GiB | **Unmeasured allowance**, much larger than the approximately 2.1 MB serialized tokenizer. Its in-memory expansion, imports and host caching have not been measured here. |
| Metal runtime and allocator caches | 1 GiB | **Unmeasured incremental allowance** for backend initialization, caching and fragmentation. |
| Additional process headroom | `max(1 GiB, B)` | `B` is a live OS observation. Reserving this much **additional** capacity is an estimate. Task 26's 216.234375 MiB selects the 1 GiB minimum; the next run must remeasure it. |
| Uncertainty reserve | 3 GiB | **Policy allowance**, for omitted transients and variability; not a statistical confidence bound. |
| Continuing host reserve | 6 GiB | Existing floor retained throughout, plus NORMAL pressure and at least 20% free indicator. |

For `B ≤ 1 GiB`, the incremental estimate is **10,473,175,240 bytes = 9.753905 GiB**, rounded up to **10 GiB**, plus **6 GiB** remaining reserve: **16 GiB available before acquisition and again immediately before model load**. A 3 GiB process peak raises the preload requirement to **18 GiB**. The plan reports `B=null` and the minimum budget, without fabricating a measurement. Runtime reports the checked live baseline and exact budget terms under `readonly_memory_policy`.

Current RSS is already excluded from free+speculative availability; its existing occupancy is not subtracted again. The process allowance budgets additional growth. RSS, MPS tensor and driver counters overlap in unified memory and are never summed as measured consumption. Deliberate copy/cache allowances may overlap; that conservatism is explicit. These allowances do not bound backend behavior. The unchanged **48 GiB physical-capacity minimum** and **6 GiB remaining availability** keep an OS/other-process reserve; this formula does not reuse Task 24's total-host planning envelope as a free-memory threshold.

Task 26's historical reading exceeds the new minimum by approximately **2.052 GiB**. That arithmetic does not admit a future run: fresh device/environment, pressure, counters and swap observations must all pass. The earlier result proves neither that this budget will hold nor that SmolLM2 can train.

## Fail-closed checks and training separation

All existing device/fallback, version, path, checkpoint/hash/license/config, frozen-parameter, output/loss and watchdog controls remain. No threshold override was added. Admission retains **48 GiB total capacity**, **6 GiB continuing free+speculative memory**, **20% minimum free indicator**, and **kernel pressure exactly NORMAL=1**. Unknown/unavailable devices or pressure, insufficient memory, missing/boolean/negative/nonfinite or inconsistent readings, zero-duration/stale/future samples, non-increasing timestamps, capacity changes, new page-outs or observed swap growth stop the probe.

Before any checkpoint access there are now two observations separated by the 250 ms target interval. Every subsequent reading compares swap/page-outs to the **immediately preceding reading**, so swap decreasing and then increasing below its original baseline still stops. Existing swap occupancy alone is not treated as current growth. The report retains failing observations. Swap readings remain rounded, samples remain non-atomic, and the cooperative watchdog cannot guarantee preemption of native work or capture every transient; see the [collector limitations](smollm2_readonly_preflight.md#limitations-and-next-evidence).

| Path | Binding policy after Task 27 |
|---|---|
| Separate frozen SmolLM2 single-forward probe | New estimated **16 GiB minimum preload**, larger with large measured process peak; 48 GiB host / 6 GiB continuing / pressure and swap controls |
| Original Llama read-only preflight | Existing **20 GiB preload** unchanged |
| Llama SGD backward-only rehearsal and training | Existing **32 GiB preload** unchanged |
| Llama AdamW backward/training memory check | Existing **42 GiB preload** unchanged |
| Small-model preparation readiness / Task 24 training assessor | Existing inherited **32 GiB** gate and assessment-only STOP unchanged |
| Small-model weight updates | `weight_updates_enabled=false`, explicit authorization required, no training entrypoint, approved training threshold unset |

`src/small_model_memory.py`, `src/small_model_experiment.py`, all `src/graddiff*` modules, experiment JSON profiles/policies/manifest and deny-by-default update controls are unchanged. Read-only output remains `training_authorized=false`, `backward_performed=false`, `optimizer_created=false`, `weight_updates=0`, with no approved small-model training threshold. Its report cannot satisfy the training assessor's cold/warm/backward evidence contract. Longer inference, backward, optimizer and training work require their own evidence and authorization.

## Exact next normal-Terminal command

Run this **later in a fresh normal Apple Silicon macOS Terminal**, keeping Activity Monitor visible. It has **not** been executed in Task 27. The dedicated checkpoint directory is absent at audit time, so this command explicitly permits acquisition of missing pinned files (about 271 MB payload, at least 2 GiB free disk). It performs only the existing one frozen forward after all admission checks. All caches and output stay in this repository.

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

For an already complete dedicated snapshot, omit `--allow-download` and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Always use a new report filename. Require exit 0 **and** complete `status="readonly_measurement_passed_not_training_clearance"`; STOP/exit 2, incomplete or absent output is failure. Review the raw observations, sampled maxima/gaps, losses and frozen/version checks. This outcome clears only that read-only measurement. Training remains unapproved regardless of its result.

## Offline validation

```sh
env PYTHONPATH=tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest \
  test_small_model_experiment test_small_model_memory test_small_model_preflight -q

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-ml/bin/python -B -m src.small_model_preflight
```

**48 offline tests passed.** These suites use fake tensors/models, mocked pretrained acquisition/loading/forward execution, and pure policy/CLI checks. They perform no actual inference, backward, optimizer steps or weight loading. Existing training suites that execute tiny synthetic backward/optimizer operations were excluded. New coverage checks budget arithmetic, baseline headroom, exact byte boundaries, device/profile drift, pressure/invalid/stale/future counters, swap growth before acquisition and after a decline, and loader exclusion on failures. A mocked read-only pass below 32 GiB leaves parameters frozen, versions unchanged, gradients absent and all update flags disabled. Regression tests retain the Llama 32/42/6 GiB boundaries, preparation/assessor STOP, and explicit update/backward authorization. The default plan and local metadata/hash audit also passed. No weights, cloud resources or physical model measurements were used.
