# Read-only FP32 GradDiff preflight

Task 17 adds `src/graddiff_preflight.py`. The default prints a plan and never loads weights. Full-model load and two forwards require `--run-preflight`. There is no training flag, backward call, optimizer construction, optimizer step, or checkpoint-save path in this command. It freezes parameters, uses `eval()` and `torch.inference_mode()`, checks parameter version counters and absent gradients, rejects nonfinite losses/logits and model/local loss disagreement, and never falls back from MPS to CPU. All assets are local-only.

## Verified findings

- The real training CLI defaults to validation and requires `--apply-updates`, an explicit device, exact model revision confirmation, and a new ignored report path. Task 17 also closes a direct-Python-call gap: `run_real_updates()` now checks the update flag and rejects dry-run mode before loading anything.
- Both report and optional training-checkpoint paths must resolve beneath the real repository `outputs/` directory, be Git-ignored, and not exist. Paths into the original checkpoint or baseline artifacts are rejected. The preflight only writes an optional JSON report. No weights are saved or committed.
- The older synthetic trainer explicitly requires `--dry-run` and *does* backpropagate and step a tiny optimizer. It was not executed in Task 17. The older read-only smoke is FP16 on MPS; this new preflight is FP32 to match the guarded training runner.
- The manifest file SHA-256 is `83581108e06c7b73c722269e52a055af3accdbe9b8e07d24422191e5a196938b`. The preflight selects the first manifest case per split: forget01 row 0 and retain99 row 24. It verifies file checksums, provenance, and selected QA hashes before encoding. This differs from the earlier smoke's random retain row 487. Logs contain row references and checksums, not QA text.
- A metadata/telemetry-only check on 2026-11-03 observed 18,688,475,136 bytes (about 17.4 GiB) of free plus speculative physical memory, system memory free percentage 94, and that process's current RSS 202,178,560 bytes. This reading is temporary and precedes any full-model load. `/bin/ps` was blocked in the sandbox; only this small telemetry check was rerun with permission outside it. No full-model inference was run anywhere in Task 17.
- The paired baseline SHA-256 remains `1473f6abc0c42164ee55dcf27328ab0b72c9e86e12c15764b71800e7462e8a0e`.
- A metadata-only check confirms the pinned cached weight file is 2,471,645,608 bytes and the host reports 51,539,607,552 bytes (48 GiB) total physical capacity. Revision/directory and file-size checks do not independently hash or authenticate the entire checkpoint weight file.

## Estimates, not measurements

The pinned architecture estimate is 1,235,814,400 parameters. FP32 weights, gradients, and two FP32 AdamW moment buffers require **19,773,030,400 bytes (18.42 GiB)** before activations and temporary allocations. The proposed optimizer remains Torch AdamW with `foreach=False, fused=False`; FP32 avoids relying on unvalidated reduced-precision training behavior. Neither optimizer state nor backward allocations are created by the preflight.

This audit uses a stricter estimate than the earlier 27.1–31.1 GiB profile:

| Component | GiB |
|---|---:|
| FP32 weights + gradients + AdamW moments | 18.42 |
| Activation/workspace allowance | 8 |
| macOS/other-process reserve | 12 |
| Additional uncertainty margin | 4 |
| **Planning total** | **42.42** |
| Headroom against 48 GiB total | 5.58 |

The planning total is 45,542,834,176 bytes. Capacity fit on paper is plausible but narrow; current available memory, allocator limits, save/load copies, and transient peaks can invalidate it. The prior 32-microbatch MPS design differs from this runner's one-pair-per-step behavior. A forward-only result cannot establish training fit, optimizer stability, benchmark equivalence, or readiness for updates.

## Exact command for a normal macOS Terminal

Run from the repository root using the existing environment, with Activity Monitor visible:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_preflight \
  --run-preflight --device mps \
  --report-path outputs/graddiff-preflight-fp32-mps-2026-11-03.json
```

This command has not been executed. Existing output paths are rejected; choose a new filename for a later attempt. Use `--device cpu` only for intentional CPU execution on macOS; it obeys the same host-memory guards and does not validate MPS behavior. Omitting `--run-preflight` prints the plan without loading the model. For synthetic validation only:

```sh
./.venv-ml/bin/python -m src.graddiff_preflight --synthetic
```

## Automatic and manual stop conditions

- Before model load: require at least **20 GiB free + speculative physical memory** and **48 GiB total capacity**. The 20 GiB threshold budgets roughly 12 GiB for CPU/MPS FP32 loading copies, source mappings and temporary buffers, plus a 6 GiB reserve and 2 GiB margin. It is an estimate.
- After load, before each forward, with each forward's logits live, and after loss validation: require at least **6 GiB free + speculative memory**. Require the `memory_pressure -Q` system memory free percentage to be at least **20%** at every check. This percentage is a separate macOS indicator, not a conversion to available physical bytes.
- Missing, malformed, inconsistent, or denied telemetry stops the command. Availability is a deliberately conservative lower bound from `vm_stat`; inactive, purgeable and compressed pages are excluded. It may reject a machine that could reclaim caches. Thresholds cannot be disabled through this CLI.
- Stop with Ctrl-C if Activity Monitor Memory Pressure turns yellow/red, swap use grows rapidly, or the system becomes unresponsive. Close memory-heavy applications first. Do not bypass the guard by changing allocator high-water limits or automatically retrying on CPU.
- Allocation failure, nonfinite loss, parity failure, changed parameter versions, unexpected gradients, or missing pinned files stops the run. No successful JSON report means the run is incomplete; it does not count as a passed preflight.

Snapshots report current MPS tensor/driver allocations, current process RSS, process high-water RSS, total capacity, availability lower bound and the macOS free percentage. Timings include telemetry overhead and MPS synchronization. These are phase snapshots, not continuous peaks; a rapid allocation spike or OS process termination can occur between checks. Monitoring is still required.

RSS and Metal allocations can overlap in unified memory; do not sum them as independent physical usage.

**Next action:** the guarded command can be attempted in a normal Terminal, but the observed 17.4 GiB lower bound is below its 20 GiB pre-load requirement. Reclaim memory and recheck; otherwise expect an early stop without loading weights. Even a successful forward preflight does not authorize an optimizer update. Full-model backward memory and official OpenUnlearning parity remain unresolved.

## Task 17 validation scope

55 read-only unit tests passed, including ten new tests covering memory boundaries, telemetry parsing, missing checkpoints, nonfinite model loss, default/direct-API gating, output paths, mock-model weight invariants, and stopping before model loading. The synthetic CLI passed two tiny CPU forwards with no gradients or optimizer. The two existing test modules that execute backward/optimizer updates (`test_graddiff.py` and `test_graddiff_real_train.py`) were deliberately excluded under Task 17's execution restriction. Earlier full-suite results elsewhere in the docs are historical. No full-model load, inference, backward, optimizer step, unlearning, download, or cloud resource was used for this task.
