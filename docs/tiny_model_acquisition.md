# Task 64: controlled tiny CPU fact acquisition

**Ready for one controlled normal-Terminal attempt, conditional on live
admission.** No eight-fact training run has executed in preparation. There is
no measured acquisition or unlearning success. The single next experiment is
the bounded acquisition command below; stop and review its report afterward.
Do not start forgetting, run retain-only retraining or extend a failed budget.

## Frozen experiment

Implements the CPU route in the [Task 63 decision](local_unlearning_experiment_decision.md)
in [one runner](../src/tiny_model_acquisition.py) and a
[checksum-pinned profile](../experiments/tiny_cpu_acquisition.json). Original
SmolLM2/Llama code, data, baselines, checkpoints, dependency lock and guards
are unchanged. No new packages, pretrained assets, downloads or cloud services.

- Random Llama causal LM: **90,496 unique parameters**, two layers, hidden 64,
  intermediate 128, four attention/two KV heads, tied embeddings, SDPA, zero
  dropout, FP32 on explicit CPU, one intra-op and one inter-op thread, seed 2026,
  deterministic algorithms. All architecture settings and runtime versions bind
  the output. No pretrained checkpoint is accepted.
- Fixed UTF-8 bytes 0–255, EOS/pad 256, no BOS. Context ≤128. The unchanged
  `encode_plain_qa`/collation helpers mask prompt/padding; twelve answer bytes
  and EOS receive shifted causal supervision. Inputs never truncate.
- From the unchanged CC0 development file: forget `dev:00:0`, `dev:00:1`;
  retain `dev:04:0`, `dev:04:1`, `dev:05:0`, `dev:05:1`, `dev:06:0`, `dev:06:1`.
  Direct questions only enter training. Answers are the existing target aliases,
  without the old sentence scaffolding. This is a separate evaluation version.
- Evaluation includes eight direct training queries, their eight held-out
  paraphrases, and direct/paraphrased queries for never-trained `dev:07:0` and
  `dev:07:1`: **20 cases**, fixed before updates. The six trained retain facts
  measure utility on objects unrelated to the forget object. The two unseen
  retain-role facts diagnose spillover; they are not assumed previously learned.
- Every direct query also receives a fixed wrong-answer likelihood comparison:
  the next selected object's answer for the same attribute, cyclically. This
  tests correct bindings against competing strings without training negatives.

Direct evaluation is resubstitution and can establish rote association learning.
Paraphrases test wording transfer and never select checkpoints or stop training.
The corpus is exposed development material. Neither these facts nor a randomly
initialized byte model establish general language utility, TOFU results, privacy
or deletion. Exact generated recovery, NLL and wrong-binding margins are reported
separately; a lower training loss alone is not success.

## Bounded execution and acceptance

Before optimizer creation, the runner evaluates genuine untouched M0, writes
`baseline.json` and `checkpoint-000000/`, reloads that **new** M0 checkpoint,
and repeats the full evaluation as a no-op control. It requires zero baseline
direct recovery, unchanged weights and identical generated IDs with answer-byte
NLL drift ≤`1e-5`. No existing checkpoint is loaded. Three max-context synthetic
backward-only rehearsals check finite gradients, unchanged weights and timing.

Acquisition uses a fixed hash-ranked order per epoch, batch one, no accumulation,
AdamW LR `0.003`, betas `(0.9,0.999)`, epsilon `1e-8`, no weight decay/scheduler,
`foreach=False`, `fused=False`, gradient clipping 1.0. Both forget and retain
roles receive positive answer-only NLL. Maximum **2,000 optimizer steps** / 250
exposures per fact; there is no resume, budget/device/LR CLI override or retry.

Evaluate and save a sealed checkpoint every **100 successful updates**. At the
first scheduled evaluation with **8/8 direct byte-exact values followed by EOS**
and ≥1 nat/answer-byte NLL improvement for every fact over M0, stop successfully.
The raw `all_eight_recovered` flag and stricter `acquisition_accepted` flag are
separate. This preregistered direct-only early stopping implements Task 64's
requirement, superseding Task 63's fixed-final-checkpoint suggestion. No
paraphrase-based selection or post-run choice of a better checkpoint is allowed.

Greedy question-only decoding stops on EOS or at 16 new tokens. The report saves
generated IDs/text, EOS/cap finish reason, validity, exact recovery, answer-byte
NLL and EOS NLL. Extra text, a partial answer, or the target without EOS fails
exact recovery. Direct forget 2/2 and retain 6/6 scores appear separately.

Any nonfinite loss/logits/parameters/optimizer state, missing/nonfinite/zero
aggregate gradients, clipping or loss-parity failure, input/configuration drift,
changed M0/checkpoint bytes, telemetry failure or serialization failure is STOP.
The attempt counter increments before `optimizer.step()`; the successful-step
counter increments only after it returns. Exceptions during a step explicitly
mark a possible partial update, and may never be resumed or counted as success.

## Physical and time guards

The **new tiny-only CPU policy** is the Task 63 estimate:
`10W + 1 GiB workspace + max(2 GiB, live peak RSS) + 2 GiB uncertainty`,
rounded upward plus a separate 6 GiB reserve, with W=361,984 bytes. Two preload
checks require **at least 12 GiB free + speculative physical pages**, rising
with process RSS. This is not a measured fit or a lowered SmolLM2 floor.

Real execution requires normal Terminal, macOS/arm64, Apple M4 Pro, locked
dependencies, ≥48 GiB physical capacity and ≥4 GiB free disk. A 250 ms watchdog
and synchronized CPU phase boundaries require ≥6 GiB continuing availability,
NORMAL pressure, free indicator ≥20%, no cumulative swap/page-out growth,
process lifetime RSS ≤8 GiB, complete ordered fresh readings and active gaps
≤2 seconds. MPS counters are null/inapplicable; no fallback, watermark, math
override, cache purge or process termination is used.

The 30-minute deadline includes initialization, baseline, rehearsal, training,
evaluations and checkpoint work. Before training and after its first update,
the runner projects the remaining full budget using twice the largest observed
step/rehearsal time and twice evaluation/save costs, plus 60 seconds. Projection
≥25 minutes stops the attempt. The first real update allocates AdamW state and
is counted in the 2,000-step budget; its time, state bytes and observed RSS
high-water are recorded. Backward-only rehearsal cannot prove optimizer fit.
Short peaks can escape sampling; OS lifetime RSS supplies an additional bound.

## Exact next normal-Terminal procedure

From a fresh Terminal, inspect the weight-free plan and optional fixture dry run:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.tiny_model_acquisition
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.tiny_model_acquisition --dry-run
```

The plan prints `validation_only` without Torch/Transformers imports. Dry run
prints `synthetic_fixture_dry_run_not_acquisition`, parameter count 90,496,
finite fixture gradients, unchanged weights, zero optimizer steps/checkpoints
and `target_facts_executed=false`. It runs only a four-token random-model
fixture; it neither evaluates target facts nor establishes physical admission.

The **single next real experiment**, after explicitly authorizing its bounded
updates, is:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-ml/bin/python -B -m src.tiny_model_acquisition \
  --run-acquisition \
  --authorize-weight-update 'tiny-90496:cpu:acquire-eight:max-2000' \
  --run-dir outputs/local_synthetic_unlearning/acquisition-task64-001
```

That directory must not exist. A collision is a refusal, not permission to
delete, overwrite or resume it. Every new attempt requires a new reviewed name.
Neither this preparation task nor the dry run authorizes the command implicitly.

Expect progress JSON each 100 steps and final JSON, plus immutable files under
the new ignored run directory: `RUNNING.json`, `baseline.json`, `noop.json`,
`checkpoint-000000/` and subsequent `checkpoint-NNNNNN/` directories containing
`model.safetensors`, `config.json`, `metrics.json` and a checksum-bound COMPLETE.
No optimizer state is saved for resumption. Final `report.json` records actual
steps/attempts, possible partial updates, per-step losses/norms/order/timing,
changed parameter names/counts and fingerprints, all evaluation scores/outputs,
baseline/checkpoint/source hashes, versions and physical observations.

- `acquisition_complete`, exit 0: accepted acquisition with sealed run
  `COMPLETE.json`; review integrity and all eight direct outputs.
- `acquisition_not_achieved`, exit 1: fixed budget finished without acceptance;
  COMPLETE denotes a complete **negative** result, not successful learning.
- `STOP`, exit 1: `FAILED.json` binds the failed report. Checkpoints/partial
  outputs are diagnostic tombstones, not accepted models. Pre-authorization,
  environment or path refusal creates no run; early physical rejection after
  reservation produces a zero-step failed report.

## Preparation validation and remaining limits

Preparation passed **28 new tests and 167 existing safeguard regressions**;
all 184 pre-existing files in the preservation ledger kept their SHA-256, size
and mode. Evidence is in ignored `outputs/task64-offline-checks.log`,
`outputs/task64-offline-evidence.json` and `outputs/task64-preservation-before.json`.
The fixture CLI also passed with zero optimizer steps and unchanged weights.

Offline tests cover the frozen source/splits/masks, exact/EOS scoring, deterministic
model count/ties, real four-token fixture gradients with optimizer steps trapped,
random-fixture checkpoint round trip, authorization and CLI scope, CPU telemetry,
resource/time boundaries, mocked early-stop and exact 2,000-step negative runs,
no-op drift and failure sealing. Run them with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest discover -s tests \
  -p 'test_tiny_model_acquisition.py' -v
```

Readiness means implementation and offline fixture checks, not measured runtime
or optimizer admission on this host. LR/budget effectiveness, eight-fact recall,
paraphrase transfer and selective forgetting remain unmeasured. The acquisition
runner performs no unlearning. Only after reviewing an accepted real M_all
result should the separate retain-only and forgetting controls in Task 63 be
considered. Never infer acquisition or forgetting from the preparation tests.
