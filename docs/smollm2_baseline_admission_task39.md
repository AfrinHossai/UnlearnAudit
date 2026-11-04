# Task 39: immutable baseline memory admission

**Ready for one controlled, owner-run read-only attempt, conditional on fresh
admission.** The 76-case baseline has not been run during this investigation.
Its peak memory and completion remain unmeasured. No model was loaded, no real
backward or optimizer step ran, no model parameters or checkpoint weights were
updated, and no download or cloud operation was performed.

## Failure trace and evidence limits

The Task 38 runner routed both `run(mode="baseline")` and `run(mode="pilot")`
through `PilotMonitor.capture()` → `validate_sample()` → `memory_policy()`.
That policy is `smollm2_135m_fp32_mps_dev_one_sgd_step_v1`. Its minimum available
memory before loading is **24 GiB**, derived from an estimated incremental
budget of **18,527,428,040 bytes**, rounded upward to 18 GiB, plus a 6 GiB
remaining reserve. It includes FP32 gradients, forward/backward workspace,
and 4 GiB for clipping, the optimizer step and checkpoint serialization.
The baseline does none of those training operations. This was an inappropriate
sharing of training admission, rather than evidence that inference needs
24 GiB or that the cached model failed to fit.

The old exception was exactly
`SmolLM2 pilot preload floor failed (minimum 24 GiB)` even in baseline mode.
The first check precedes checkpoint integrity access; a second check follows
tokenizer/encoding work and precedes report reservation and model loading.
Either can STOP without creating a report or even the acquisition output
directory. At either preload failure, the loader, evaluator, backward,
optimizer and saver are unreachable.

**The failed Task 39 terminal transcript was not supplied and no saved Task 39
report or `outputs/smollm2_acquisition_task38/` directory existed in this
workspace at investigation time.** Thus the shared-guard cause is established
from source and reproduced with mocked telemetry, but the failing attempt's
exact availability, other counters, and which preload boundary failed cannot
be independently verified. Absence of a report is consistent with the early
failure behavior; it is not a memory measurement. No historical Task 39
readings or model-load result have been invented.

## Measurements compared with the baseline workload

The source reports are local ignored artifacts. Their SHA-256 values were
verified without loading their models:

- `outputs/smollm2_readonly_task27_first.json`:
  `8df89af2e27aae9f75ffe577e29e58322522f442d6be359bfa8d40a592375a74`;
  status `readonly_measurement_passed_not_training_clearance`.
- `outputs/smollm2_backward_task37.json`:
  `f71c8d1e673067676708c33be9cfdb30b7545f96ed691768b7b183652bde80b3`;
  status `backward_rehearsal_passed_not_training_clearance`.

Both used the pinned SmolLM2-135M revision, actual MPS and FP32 on the 48 GiB
host. These are historical measurements, not current admission:

| Observation | Task 27 single forward | Task 37 backward-only rehearsal |
|---|---:|---:|
| Forward / backward counts | 1 / 0 | 2 / 1 |
| Generation / optimizer / save | None | None |
| Before-load conservative availability | 16.805878 GiB | 28.333603 GiB |
| Minimum sampled availability | 15.339813 GiB | 25.278580 GiB |
| Maximum sampled current RSS | 897,875,968 bytes | 1,301,757,952 bytes |
| Observed OS lifetime peak RSS | 1,166,884,864 bytes | 1,301,889,024 bytes |
| Maximum sampled MPS tensors | 541,403,904 bytes | 1,129,497,856 bytes |
| Maximum sampled MPS driver | 1,145,847,808 bytes | 2,659,057,664 bytes |
| Samples | 9 | 25 |

Task 27 evaluated one **17-token** synthetic QA, with outputs observed while
live. Task 37 recorded unchanged parameter hashes/versions and cleared
gradients. Both had normal pressure and zero recorded swap/page-outs. Task 27
has a 10.157286-second maximum gap, between pre-load preparation observations;
Task 37's largest gap is 0.847274 seconds. Neither provides a continuous exact
peak. Samples are nonatomic; sampled MPS maxima are lower bounds, Torch 2.9.1
provides no native MPS peak here, and swap is rounded. RSS and MPS overlap
and **must not be added**.

The baseline evaluates **76 cases sequentially**, one at a time, with
answer-only NLL and up to 64 greedy new tokens per question, EOS stopping and
no KV cache. Task 38's tokenizer-only audit found maximum full QA length 57
and maximum prompt plus generation budget 93; runtime independently enforces
the 128-token cap. Model FP32 parameters occupy **538,060,032 bytes** by
arithmetic. A full `[1,128,49152]` FP32 logits tensor is **25,165,824 bytes**
(24 MiB), also arithmetic, not a measurement of total forward memory.

Only scalar metrics/text persist between cases. Loss/batch references are
released before generation and generation outputs/logits after each token.
There is no retained autograd graph or KV cache. Expected live tensor memory
is bounded by the current forward/context, rather than 76 model copies or
76 retained graphs. Repeated forwards may grow backend caches, and generation
and fingerprint checks were not measured by Task 27. Therefore its roughly
1 GiB observed RSS/driver peaks cannot be used as a baseline admission floor
or a guarantee that this longer evaluator will complete.

## Separate conservative read-only policy

`baseline_memory_policy()` selects
`smollm2_135m_fp32_mps_dev_baseline_v1`. It reuses the reviewed Task 27
frozen-forward planning allowances and adds **2 GiB** specifically for repeated
generation/Metal cache uncertainty. This addition is a conservative estimate,
not an empirically calibrated requirement. With process peak `B <= 1 GiB`:

| Estimated allowance | Bytes |
|---|---:|
| Three FP32 weight copies during loading/transfers | 1,614,180,096 |
| Source file residency | 269,060,552 |
| Forward workspace | 2,147,483,648 |
| Tokenizer / host cache | 1,073,741,824 |
| Metal runtime cache | 1,073,741,824 |
| Additional process headroom, `max(1 GiB, B)` | 1,073,741,824 |
| General uncertainty reserve | 3,221,225,472 |
| Sequential generation allowance | 2,147,483,648 |
| **Incremental estimate** | **12,620,658,888** |

Round the incremental estimate upward to 12 GiB and add 6 GiB continuing
reserve: **18 GiB available before checkpoint access and again before loading**.
At `B = 3 GiB`, the floor rises to **20 GiB**. The plan reports `B=null` when
unmeasured; a successful runtime report records the checked live process peak
and exact policy. This floor is deliberately above Task 27's 16 GiB single
forward floor. Task 27's historical 16.805878 GiB before-load reading would
**not** admit this baseline. The new policy need not admit the failed Task 39
conditions; those actual readings are unavailable.

Training `memory_policy()` retains its entire Task 38 dictionary, including
all numerical allowances and the **24 GiB minimum / 26 GiB at B=3 GiB**.
Its legacy scope description is retained for audit comparison; mode dispatch
now selects it only for the pilot. A regression test compares that whole
dictionary with the frozen Task 38 audit. No training guard derives permission
from a baseline measurement. Omitting the mode from low-level sample validation
still selects training. Unknown modes fail closed.

The backward-only 20 GiB policy, original SGD 32 GiB and AdamW 42 GiB policies,
512-token TOFU preparation gate and deny-by-default training controls are
unchanged. Explicit model-specific update authorization, one-step budget,
gradient validation/clipping, exact stateless SGD configuration, repeated
checks before step/save, immutable baseline provenance and fresh checkpoint
destination safeguards remain required. Passing the baseline authorizes none
of these updates or any longer run.

## Baseline isolation and failure handling

Baseline mode rejects update authorization, a pilot destination and baseline
input arguments before runtime work. The sole optimizer construction and
`backward()` call in the acquisition runner are inside `one_step()`, reached
only by the pilot branch with a validated immutable baseline and explicit
authorization. `one_step()` now additionally rejects a baseline monitor,
even if given the correct authorization token.

The baseline freezes all parameters, uses `eval()` and wraps all evaluation
forwards in `torch.inference_mode()`. It rejects trainable parameters,
preexisting gradients, a loss with an autograd graph, and a non-MPS evaluation
device. Model validation verifies CPU placement before transfer, then actual
MPS placement of all parameters/buffers and FP32/configuration after transfer
and after baseline evaluation. Loss/logit device, dtype, finiteness and cache
checks remain active. Parameter hashes and version counters must agree before
and after evaluation; unexpected mutation becomes STOP, never success.

Tests trap optimizer construction and backward during inference, assert actual
inference-mode context, reject baseline monitors at the update boundary, and
inject independent hash/version drift. This verifies the intended read-only
path without constructing a pretrained model or changing model weights.

Both modes preserve direct telemetry validation, two preload checks, 250 ms
watchdog sampling and synchronized phase boundaries. OOM, watchdog error,
KeyboardInterrupt, device/configuration drift and cleanup errors produce STOP.
Pre-reservation failures print STOP evidence without creating a report;
after reservation, the failing report is flushed and sealed read-only.
Existing results cannot be overwritten. Baseline cannot reach checkpoint
serialization or create an optimizer/checkpoint artifact.

## Exact next normal-Terminal command and stop conditions

Run this block **once in a normal macOS Terminal**, after reviewing the change.
It is documented here only; it was not executed during this task. Use the fresh
numbered destination below and preserve any prior attempt/transcript:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false .venv-ml/bin/python -B -m src.small_model_acquisition \
  --run-baseline \
  --report outputs/smollm2_acquisition_task38/baseline-002.json
```

Stop on exit code 2, `status: STOP`, an exception or interrupt. Do not bypass a
floor, enable fallback, change allocator settings, replace an old report, or
automatically retry. The enforced conditions include:

- Apple M4 Pro / arm64 macOS / available MPS / pinned packages/model/config;
  no CPU fallback, allocator/math overrides or distributed execution.
- At least 48 GiB physical capacity; dynamic baseline preload floor at both
  observations; at least 6 GiB conservative continuing availability; free
  indicator at least 20%; kernel pressure NORMAL=1.
- Valid fresh readings within two seconds, ordered timestamps, no active
  sampling gap over two seconds, no capacity change, no swap increase above
  the first observation or page-out counter change.
- Process lifetime peak RSS and MPS driver allocation each at most 8 GiB;
  no sum of overlapping counters; 30-minute total time limit and at least
  4 GiB free disk before loading.
- No checkpoint/manifest/source/encoding drift, incompatible loading results,
  nonfinite loss/logits, failed NLL parity, wrong device/dtype/cache behavior,
  unexpected gradients, changed parameters or failed cleanup/report sealing.

After exit code 0, require `baseline_complete_no_updates`, actual MPS/FP32,
the baseline guard ID, both checked preload observations, all 76 ordered
metrics/generations, unchanged cached base and parameter hashes/versions,
`training_authorized=false`, `optimizer_created=false`, zero optimizer steps
and checkpoint writes. Only then record the sealed report's checksum:

```sh
shasum -a 256 outputs/smollm2_acquisition_task38/baseline-002.json
```

Review the trace's minimum availability, sampling gaps and RSS/MPS maxima;
successful completion is measurement of this bounded read-only evaluation,
not training clearance. A pilot would still require separate explicit owner
authorization and its original fresh training admission. No pilot is requested
or executed by this investigation.

## Offline verification

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest discover -s tests -p 'test_small_model*.py' -v
```

**102 tests passed**, including 37 acquisition tests (11 added here). Coverage
includes exact preload/continuing/cap boundaries and dynamic RSS headroom;
both real monitor preload gates stopping before the mocked loader; all common
telemetry stops; model/mode rejection; optimizer/backward isolation; parameter
drift; report policy/provenance validation; and unchanged training admission,
authorization, gradient/clip/step/save guards. Runtime paths use mocks, with
existing tiny CPU tensor/configuration diagnostics and no real backward or
optimizer step. Tests simulate updates using Python values only.

The preservation snapshot covers **81 existing cache, historical output,
result and experiment files**, including all 66 files recorded before Task 38.
All SHA-256 values match after verification, including cached SmolLM2/Llama
weights and the historical reports. The frozen Task 38 manifest and evidence
audit are unchanged. Only acquisition admission/evaluation checks, its tests
and documentation were edited; no physical memory-fit result is claimed.
