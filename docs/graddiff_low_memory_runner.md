# Guarded SGD GradDiff runner: Task 19

**Status:** SGD selection, checkpointing compatibility checks, physical-memory guards and a separate backward-only mode are implemented. Only synthetic CPU models were exercised. No pretrained Llama weights were loaded, trained or changed; no large checkpoint download or cloud resource was used.

## Implemented behavior

`src/graddiff_real_train.py --optimizer sgd` selects `torch.optim.SGD(momentum=0, weight_decay=0, foreach=False, fused=False, maximize=False)`. No per-parameter optimizer buffers are created; synthetic tests assert the state stays empty after a step. The default learning rate is `1e-4` for SGD and `1e-5` for the retained AdamW option. Both are overridable with a finite positive CLI value. The 40-step pilot remains an unexecuted scientific plan; the CLI defaults to one step and never reads the pilot JSON as execution authorization.

Default execution remains local config/tokenizer/data validation with no weight loading. `--apply-updates` and `--backward-only` are distinct opt-ins. Both require an explicit `mps` or `cpu` device, exact pinned model revision confirmation and a new ignored report path. Backward-only permits one paired gradient computation, never creates an optimizer, disallows checkpoint saving, checks parameter version counters and clears gradients on completion/failure. It never promotes itself to update mode. CPU selection is intentional and the real-model memory telemetry currently requires macOS even on CPU. MPS unavailability or an enabled `PYTORCH_ENABLE_MPS_FALLBACK=1` is an error.

All original parameters must be trainable FP32 tensors on the selected device. Loading stays offline at the existing model/tokenizer revision; split checksums and answer-only masking are unchanged. The full-model parameter count is checked against the pinned architecture estimate. Forget and retain mean causal NLL contributions are backpropagated sequentially, preserving `-gamma*NLL(F) + alpha*NLL(R)` while retaining one graph at a time. Combined gradients are checked for finiteness, clipped, and checked again before an update. Fixture tests compare this gradient with the combined validated objective using identical weights.

Checkpointing is mandatory for these profiles. The model must advertise support and accept `gradient_checkpointing_kwargs`. The runner sets `config.use_cache=False`, calls `gradient_checkpointing_enable` with `use_reentrant=False, preserve_rng_state=True`, verifies both the active model flag and propagated activation-function kwargs, and enters train mode. Each forward also passes `use_cache=False`. Missing support, legacy APIs, ignored settings, frozen parameters, non-FP32 precision and nonzero momentum are rejected. There is no silent checkpointing or precision fallback.

## Runtime guards and estimates

Before reading model assets and again immediately before loading weights, require at least **48 GiB total host capacity**, **32 GiB available physical lower bound for SGD** (42 GiB for AdamW), and **20% system memory free percentage**. The availability measure is free plus speculative pages, deliberately excluding reclaimable caches. After load, before the pair, after each backward and immediately before each optimizer step, require at least **6 GiB** and 20%. Unknown, inconsistent or denied telemetry stops execution. The current-RSS and host-memory readers are shared with the Task 17 preflight through `src/graddiff_memory.py`.

The report/checkpoint paths are checked before loading and rechecked immediately before an update, including a small temporary write-permission probe in each ignored parent directory. Paths must resolve inside the real repository `outputs/`, be Git-ignored, and not already exist. Symlink escapes, original checkpoint/cache targets and baseline paths are rejected. A path probe does not reserve enough disk for a later full checkpoint: check disk space separately. Failure during saving can leave a partial local directory; no failed run certifies a checkpoint. OOM does not retry on another device, and OOM inside an optimizer can leave a partial in-memory update, so discard that process.

The SGD full-model persistent estimate remains **9.21 GiB** for FP32 weights plus gradients, with zero momentum state. The **33.21–37.21 GiB host planning envelope** adds unmeasured workspace, OS and uncertainty allowances. The 32 GiB pre-load availability threshold is conservative policy, not a measured proof of fit. The historical Task 17 reading of about 17.4 GiB fails it; no fresh full-model readiness measurement was taken in Task 19. Phase snapshots are not continuous peaks, can miss allocation spikes, and cannot prevent an OS kill. RSS and Metal allocations may overlap; do not sum them.

## Validation and observations

The complete repository suite ran **88 tests: 87 passed, one MPS-only test skipped** because MPS is unavailable in this sandbox. Thirteen new tests cover SGD arithmetic and empty state, selection/defaults, invalid options, checkpoint settings, checkpointed gradient agreement, memory boundaries and pre-update stops, path symlinks, reproducibility, and mock-only real-runner orchestration. All pretrained-loader use in those orchestration tests is mocked. Existing tests preserve causal masks, coefficients and provenance checks.

The separately executed tiny synthetic backward command created a random 2-layer, 5,200-parameter causal LM in memory; it never called a pretrained loader, optimizer constructor or optimizer step. Observations from the saved ignored `outputs/graddiff-synthetic-backward-task19-final.json`:

| Observation | Value |
|---|---:|
| Actual device | CPU |
| Gradient tensor bytes | 20,800 |
| Forget / retain NLL | 3.51194167 / 3.48995471 |
| GradDiff scalar, gamma=alpha=1 | -0.02198696 |
| Gradient L2 norm | 4.17317772 |
| Process high-water RSS before backward | 301,187,072 bytes |
| After forget / retain backward | 310,362,112 / 310,509,568 bytes |
| After clearing gradients | 310,706,176 bytes |
| MPS tensor/driver allocation | Not applicable; CPU run |

Weights and parameter versions were unchanged and model/local losses agreed. These RSS values are process-lifetime high-water observations, including Python/Transformers and tiny-fixture setup; the differences are not a measured full-model peak or an exact backward allocation size. Repeated seeded CPU fixture losses and gradient norms matched in tests. MPS kernel determinism and full-model backward compatibility remain unverified.

Repeat synthetic validation with a new report filename:

```sh
./.venv-ml/bin/python -m src.graddiff_synthetic_rehearsal \
  --device cpu --seed 2026 \
  --report-path outputs/graddiff-synthetic-backward-next.json
```

Selecting SGD without enabling execution is safe local validation:

```sh
./.venv-ml/bin/python -m src.graddiff_real_train --optimizer sgd --dry-run
```

## Next physical-device step and exact prerequisites

**The next appropriate training-readiness step is a physical-device backward-only rehearsal, after a successful read-only FP32 preflight and fresh memory checks.** It is not an optimizer step and has not been executed in this task. First follow [the read-only preflight command](graddiff_preflight.md); that command still has its original 20 GiB load guard. Passing it does not bypass the stricter 32 GiB SGD rehearsal guard.

Once the read-only result is reviewed, close memory-heavy applications, keep Activity Monitor visible, confirm at least 32 GiB available by the conservative reader, and run the following only in a normal macOS Terminal under separate authorization for the real-model backward rehearsal:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  ./.venv-ml/bin/python -m src.graddiff_real_train \
  --optimizer sgd --momentum 0 --precision float32 \
  --checkpointing non-reentrant --backward-only --device mps \
  --confirm-model-revision 88e31200b97e4c0c04ae0d2f0b591f427046d192 \
  --max-steps 1 --seed 2026 --learning-rate 1e-4 \
  --gamma 1.0 --alpha 1.0 --max-grad-norm 1.0 --max-sequence-length 512 \
  --report-path outputs/graddiff-sgd-backward-mps-first.json
```

Do not add `--apply-updates` or `--save-checkpoint` to this command; incompatible modes are rejected. The learning rate and clip limit are recorded for the selected configuration, but backward-only neither clips nor updates. It uses the runner's first training pair: forget row 0 and seeded retain draw row 487, rather than the manifest preflight's retain row 24. Both use checksum-pinned TOFU files. Logs contain row references and token counts, not QA text or secrets.

Stop on missing telemetry, insufficient memory, high pressure, rapid swap growth, OOM, nonfinite losses/gradients, parity errors or activation-setting incompatibility. Ctrl-C if Activity Monitor turns yellow/red or the machine becomes unresponsive. No successful JSON report means the run did not pass. Do not lower allocator limits or memory guards to force execution.

## Remaining blockers before real unlearning

- No real full-model read-only or backward-only run has passed; fresh availability must meet the guards. Tiny CPU success does not validate the physical MPS path.
- The first pair's short sequences do not establish memory safety for all 40 planned pairs or sequence length 512. Inspect the planned token lengths and rehearse the relevant longest cases before approving the whole pilot; that broader rehearsal is not implemented by this one-pair command.
- Freeze the matched FP32 baseline, retain-only control, evaluator/reference-log provenance, schedule and utility/forgetting criteria in [the pilot assessment](low_memory_unlearning_options.md). The matched retain-only control is still planned, not implemented as a runner mode. The 40-step pilot and its learning rate have no empirical forgetting or utility result.
- Keep official Forget Quality claims blocked without matching retain-only logs. All checkpoints/reports remain local and ignored; verify save disk capacity and memory transients before optional checkpoint saving.
- Review the physical observations and obtain explicit authorization before using `--apply-updates`. A passing rehearsal is computational evidence only. No full-model update is cleared by Task 19.

This runner preserves the full-model GradDiff scalar, not the upstream optimizer trajectory. FP32 SGD, one process, batch one, no accumulation/scheduler, SDPA and local sampling differ from BF16 paged AdamW, FlashAttention, distributed CUDA and ZeRO-3. It is not a faithful OpenUnlearning benchmark reproduction, and even a future successful pilot cannot establish irreversible erasure or a privacy guarantee.
