# Task 79: preregistered stronger forgetting weight

**Prepared, not authorized or executed.** This protocol specifies exactly one
follow-up comparison. Separate explicit human authorization is required before
any real optimizer update. Task 64 acquisition authorization, Task 70
authorization, this preparation request, a CLI token printed here, and passing
offline tests do not provide that authorization. No selective forgetting or
privacy protection is established.

## Evidence reviewed and preserved

The [Task 77 findings](tiny_model_unlearning_findings.md), historical
[Task 70 protocol](tiny_model_unlearning.md),
[profile](../experiments/tiny_cpu_unlearning_task70.json),
[frozen selection/evaluation manifest](../experiments/tiny_unlearning_task70_frozen.json)
and sealed local results were reviewed. The older preparation documents describe
their then-pending runs; the sealed reports establish subsequent completion.

The accepted acquisition is still
`outputs/local_synthetic_unlearning/acquisition-task64-001/checkpoint-000700`:
700/700 successful steps/attempts, 90,496 parameters, accepted eight-fact direct
recovery. Its `model.safetensors` SHA-256 is
`fdaaf3ab932db016dec2361210240fca62c1aaf165273444fcc7aa5bbda910b2`;
the named-parameter fingerprint is
`f5211ab5e5f6eb9d09991d884ff84def1e93d9985471411e7f13269c44de687a`.
The acquisition report SHA-256 is
`88678bd4f4f86e4d8eec8fede0be5da1d7d3820d6b4bb1c550a6d490b66b8ad3`.

Task 70 completed at
`outputs/local_synthetic_unlearning/unlearning-task70-001`, with no-op 0 steps,
retain-only 20 steps and GradDiff gamma=1 20 steps; no partial attempts. Its
COMPLETE binds report SHA-256
`5c047bd4c5c4d221e501764bd17ad321b562d0d61d2c17d698fe8930571e9241`.
All three arms still recover **2/2 forget and 6/6 retain** direct answers.
Gamma=1 increased mean forget byte NLL only about 0.000281 nats/byte over its
baseline (0.000255 beyond retain-only). This is confidence reduction with
unchanged direct recovery, not observed forgetting. Task 70's final gate failed.

The new [historical inventory](../experiments/tiny_stronger_forgetting_task79_history.json)
pins all 26 Task 70 files. The new runner verifies their hashes, completion,
checkpoint bundles, baselines, encoding, sources and recomputed decision, and
uses the unchanged acquisition verifier for all 37 acquisition files. Historical
weights are hashed as opaque bytes during preparation; no acquired-model load,
inference, backward pass or optimizer update is performed. Existing source,
checkpoints, controls, reports, results, safeguards and Git history stay intact.

## Single experimental contrast

The [new profile](../experiments/tiny_cpu_stronger_forgetting_task79.json) is
checksum-pinned by the independent
[runner](../src/tiny_model_stronger_forgetting.py). Its training configuration
is checked against Task 70: **only gamma changes from 1 to 10**. The fresh
execution budget is 20 instead of the old suite's 40 because controls are reused.

Minimize `J = -10 * mean_F NLL(answer+EOS) + 1 * mean_R NLL(answer+EOS)`.
Full parameters, FP32 CPU, batch-one sequential backward passes compute exact
separate group means over 2 forget and 6 retain facts. Prompt/padding positions
remain masked; twelve answer bytes plus EOS are supervised. This keeps
**alpha=1, stateless SGD LR=0.05, gradient clip=1.0, and 20 steps per training
arm**. All other optimizer settings, architecture, single CPU thread,
deterministic algorithms and dependency versions remain fixed.

| Evidence | Treatment in Task 79 |
| --- | --- |
| Accepted acquisition / frozen M0 and acquired rows | Reused; same accepted step-700 initialization, never gamma=1's final weights |
| Task 70 no-op, step 0 | Reused sealed measurements/checkpoint; zero new steps |
| Task 70 retain-only, step 20 | Reused sealed measurements/checkpoint and all scheduled evaluations; zero new steps |
| Task 70 GradDiff gamma=1, step 20 | Reused dose comparator, including all scheduled evaluations; zero new steps |
| Task 79 GradDiff gamma=10 | Only fresh training arm: exactly 20 successful steps/attempts unless STOP |
| Task 79 baseline and baseline repeat | Fresh full evaluation before SGD creation, unchanged fingerprint; zero updates; a determinism check, not a rerun of the historical control arm |

The new report labels every arm's origin as `reused` or `fresh`, with historical
directory/report hash. Its optimizer counters count **only fresh calls**; they
never incorporate the historical 40 calls. Reused results are not independent
replicates. No additional dose, learning rate, seed, extended budget or retrained
control is prepared.

Forget IDs remain `dev:00:0`, `dev:00:1`; retain IDs remain `dev:04:0`,
`dev:04:1`, `dev:05:0`, `dev:05:1`, `dev:06:0`, `dev:06:1`. Never-trained
`dev:07:0`, `dev:07:1` remain evaluation-only. The same 20 cases include direct
and paraphrased prompts for all ten facts. Seed **2070** and hash-ranked
`2070:step:fact_id` ordering stay fixed, including rehearsal step 0. Initialization
constructs the same seed-2026 architecture, then loads the accepted weights.

## Preregistered endpoint and interpretation

Evaluate at steps 0, 5, 10, 15 and **20**. The fixed final step 20 is the sole
primary endpoint. Intermediate measurements support safety/diagnosis; no earlier
checkpoint may be selected for apparent success. Keep a complete negative result
or safety stop without retuning.

The unchanged acquisition evaluator uses greedy question-only generation,
no cache/BOS, a 16-token cap, byte tokens 0–255 and EOS 256. Exact recovery
requires all twelve correct ASCII bytes followed by EOS. Preserve all raw IDs,
text, finish reasons and validity flags; partial/normalized/substring recovery
does not receive credit. Report each fact's answer-byte NLL, inclusive NLL,
EOS NLL, `exp(-12 * byte NLL)` and correct-minus-one-wrong binding preference
(wrong-binding byte NLL minus correct byte NLL). NLL is in nats/byte; probability
is teacher-forced target likelihood excluding EOS, not empirical recovery rate.

Success requires **all** unchanged Task 70 direct-forgetting criteria:

- Forget exact recovery **0/2**; each forget byte NLL increases **at least 1.0**
  from its own baseline; each correct-over-wrong margin is **at most 0**; each
  final byte NLL is **at least 0.5** above the matched reused retain-only final.
- Retain exact recovery **6/6**; every retain byte NLL increases **at most 0.1**
  over its baseline.
- The same eight sentinel strings have mean NLL increase **at most 0.1
  nat/token**, and mean `KL(M_all || gamma10)` **at most 0.1 nat/token**.
  Means are over per-string next-token averages including EOS; report entropy
  too. Fresh reference distributions come from identical accepted weights.
- All budgets, baseline/no-op reproduction, source/artifact integrity, runtime
  and physical safeguards pass. New final checkpoint file hashes, COMPLETE,
  metrics and exact acquisition config hash must agree; reloading the new
  checkpoint must reproduce its final parameter fingerprint. Original artifacts
  must still match their inventories at exit.

A complete run failing any criterion is a **negative result**. A stopped,
incomplete, partially updated or integrity-failed run is **invalid for the gate**,
with `result_valid=false`; any diagnostic result field cannot establish success.
`COMPLETE` means protocol completion even if the gate is false. Report paired
final-minus-baseline changes for all four labeled arms and per-case
gamma10-minus-gamma1 differences in recovery, byte/inclusive/EOS NLL, likelihood
and margins, plus sentinel values for all arms.

Higher NLL/lower likelihood with **2/2 recovery still intact** is confidence
reduction only. Losing one target is incomplete direct suppression, not gate
success. Losing correct EOS alone, invalid answers or switching to a wrong
binding cannot establish knowledge deletion. Even satisfying this operational
gate would establish only selective suppression under these direct prompts;
genuine inaccessibility under extraction remains untested.

## Authorization, command and stops

After a **separate explicit authorization** for this one comparison, the single
next normal-Terminal experiment command, from
`~/UnlearnGuard-Privacy-Leakage-Research`, is:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-ml/bin/python -B -m src.tiny_model_stronger_forgetting \
  --run-stronger-forgetting \
  --authorize-weight-update 'task79:tiny-90496:cpu:gamma10:alpha1:20-total:reuse-task70' \
  --run-dir outputs/local_synthetic_unlearning/stronger-forgetting-task79-001
```

**Do not execute this command as part of Task 79 preparation.** Acquisition and
Task 70 tokens are rejected. Authorization is checked before file/model work,
before SGD construction, and on every update including immediately before its
optimizer boundary. There are no parameter/device/budget CLI overrides.

Only the exact unused output path above is accepted. A collision is refusal;
do not overwrite, delete, resume, rename for a retry or extend this experiment.
New outputs include RUNNING, fresh baseline/repeat, step-5/10/15/20 evaluations,
one final gamma10 checkpoint, report and COMPLETE or FAILED. Existing paths are
never output targets. No optimizer state is saved for resumption.

All inherited CPU protections remain: normal macOS arm64 Terminal, reviewed
Apple M4 Pro, locked Python/packages/Llama source, at least 48 GiB physical RAM,
dynamic preload floor at least 12 GiB, continuing availability at least 6 GiB,
NORMAL pressure, at least 20% free indicator, lifetime RSS at most 8 GiB, no
cumulative swap/page-out growth, 250 ms watchdog and at most 2-second fresh sample
gaps. Require at least 4 GiB free disk before load/save. No MPS/fallback, allocator
or math overrides, downloads, dependency changes or cloud resources.

The monitored execution retains the **600-second deadline** and conservative
**480-second projection cutoff**, with twice the largest rehearsal/update time,
the remaining fresh 20-step budget, the original generous twenty evaluation
allowances and 60-second load/save reserve. A backward-only rehearsal does not
prove optimizer fit; live admission is still required.

STOP on telemetry/memory/time/environment/input/source/configuration drift;
nonfinite loss/logits/gradients/parameters; absent/zero aggregate gradient;
preclip norm **>100**; optimizer drift; unexpected/partial attempts; baseline or
sentinel mismatch; serialization/reload/integrity failure. At every scheduled
evaluation STOP if **any retain direct answer is incorrect**, any retain byte
NLL increase **>0.5**, any forget byte NLL **>20**, sentinel mean NLL increase
**>0.5**, or sentinel mean KL **>0.5**. These safety limits are deliberately
looser than final acceptance. Between-evaluation drift can escape detection.

Attempts increment immediately before SGD; successful steps increment only after
return. An exception during SGD records `possible_partial_update=true`, seals
FAILED, and permits no retry/rollback/resume. A scheduled safety stop ends the
arm and comparison; review its tombstone. Exit 0 with
`stronger_forgetting_comparison_complete` means completed measurement, not
forgetting success. `STOP` exits 1 with FAILED when a run was reserved;
preauthorization/path/environment refusal can occur before any run is created.

## Offline validation and scientific limits

The weight-free default plan verifies the accepted acquisition and historical
controls without importing Torch/Transformers. `--dry-run` uses a random tiny
model and four-token fixtures, backward only: no acquired checkpoint load,
target-fact execution, optimizer construction/steps or checkpoint writes.
[Offline tests](../tests/test_tiny_model_stronger_forgetting.py) cover frozen
parameters, gamma10 group-gradient parity, new authorization, isolated outputs,
20-call bounds, partial attempts, physical/time stop dispatch, strict decision
criteria, checkpoint corruption and mocked complete/failure orchestration.
Historical acquisition/unlearning safeguard tests use fixtures with optimizer
steps trapped or mocked. **All 70 offline tests passed: 20 new and 50 existing.**
The weight-free plan verified both histories, and the fixture CLI reported zero
optimizer steps/checkpoint writes, unchanged weights, no acquired-checkpoint
load and no target-fact execution. The preservation audit confirmed unchanged
SHA-256, size, mode and modification time for **all 218 pre-existing tracked,
output and result files**, including the 63 acquisition/Task 70 artifacts.
Preparation evidence is kept separately in ignored
`outputs/task79-offline-evidence.json` and
`outputs/task79-preservation-before.json`; it contains no new real experiment
results. The actual experiment directory has not been created.

Readiness means a prepared bounded comparison with offline checks, conditional
on separate authorization and fresh live admission. It does not establish
gamma10 efficacy, runtime fit or privacy protection. The gamma choice follows
the observed gamma1 negative result; it is preregistered for this follow-up,
not a prospectively selected first-study hypothesis.

There are only two forget attributes of one synthetic entity, six memorized
retain facts, one seed and one added dose. Direct evaluation repeats training
prompts. Paraphrase baseline recovery is **0/8**, so unchanged zero cannot show
forgetting transfer; newly correct forget paraphrases are residual/re-emergent
recovery. Never-trained probes cannot certify retained learned knowledge.
Sentinels have high baseline loss and narrow language coverage. Reused controls
do not measure new-session control variability; matching baselines/deterministic
code limit, but cannot remove, that temporal confound. Clipping means ten times
the loss coefficient need not yield ten times the parameter movement. There is
no independent replication, exhaustive wrong-binding set, retain-from-scratch
oracle, adversarial extraction, membership inference or broad utility benchmark.
Keep `paraphrase_effect_identifiable=false` and
`privacy_protection_established=false` regardless of the direct gate.
