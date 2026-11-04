# Task 70: controlled tiny-model unlearning preparation

**Prepared for review; no genuine unlearning or retain-only optimizer step has
been run.** A future, independent explicit authorization is required before the
single experiment below. This preparation and the old acquisition authorization
do not authorize it. No forgetting or privacy-leakage reduction is established.

## Verified starting point

Read-only audit of
`outputs/local_synthetic_unlearning/acquisition-task64-001/report.json` verified
`status=acquisition_complete`, 700 successful optimizer steps and 700 attempts,
90,496 unique parameters, `all_eight_recovered=true`,
`acquisition_accepted=true`, and no possible partial update. The run COMPLETE
binds the report SHA-256:

`88678bd4f4f86e4d8eec8fede0be5da1d7d3820d6b4bb1c550a6d490b66b8ad3`.

The accepted model is **checkpoint-000700**, not a post-hoc checkpoint choice.
Independent verification recomputed all eight checkpoint bundles' file hashes,
checked every checkpoint COMPLETE against the run's recorded bundle, checked the
run COMPLETE, baseline/no-op hashes, all recorded acquisition source hashes, and
recomputed the acquisition acceptance gate from final and M0 rows. A read-only
CPU load of the accepted safetensors into a fresh architecture also reproduced
the recorded parameter fingerprint. No target forward/backward pass or optimizer
was used in this independent checkpoint verification.

| Accepted artifact | SHA-256 |
| --- | --- |
| model.safetensors | `fdaaf3ab932db016dec2361210240fca62c1aaf165273444fcc7aa5bbda910b2` |
| config.json | `569161ce5523c8424a2485fd00be359a1881e087f23c86cd336bb1b107e1fb89` |
| metrics.json | `b2fe1419f0cefa7f0c5fa457a22694d98ccb51a4a885bfbac1ec41e17d07efc0` |
| checkpoint COMPLETE.json | `3f900e69cfb1505e54db0d213ea267298f0e339e370de609b1c1ee80caab9c0f` |
| Loaded named-parameter fingerprint | `f5211ab5e5f6eb9d09991d884ff84def1e93d9985471411e7f13269c44de687a` |

The file hash and named-parameter fingerprint are different constructions. The
latter uses the unchanged acquisition helper and includes parameter names,
shapes, dtypes and actual bytes, with tied embeddings counted once.

## Evaluation audit and frozen baselines

The acquisition JSON schema stores evaluation rows under
`evaluations[-1].rows` and aggregate counts under `evaluations[-1].scores`.
Checkpoint `metrics.json.evaluation` contains the same 20 rows. Each has `id`,
`kind`, `role`, generated token IDs/text, EOS/length finish reason, format
validity, exact recovery, answer-byte NLL, answer NLL including EOS, and EOS NLL.
Only the eight direct rows have a fixed wrong-binding NLL and margin.

Exact recovery means precisely the twelve target ASCII bytes followed by EOS.
Greedy question-only generation uses no cache, a 16-token cap, no BOS, byte
tokens 0–255, and EOS/pad 256. No sampled decoding, answer normalization, partial
credit or substring recovery is used. Teacher-forced loss masks question and
padding positions and applies causal shifting. Byte NLL averages twelve targets;
the training/inclusive loss averages thirteen targets including EOS. EOS NLL is
reported separately so stopping-token disruption is visible.

Measured acquisition values below are taken from the original sealed rows;
NLLs are nats per answer byte, rounded here to nine decimal places. Full values,
generated IDs and EOS NLLs remain in the frozen manifest.

| Group | Exact recovery | Valid format | Mean byte NLL at M_all | Mean byte NLL at random M0 |
| --- | --- | --- | --- | --- |
| All trained direct | 8/8 (100%) | 8/8 | 0.002107904 | 5.524671853 |
| Forget direct | 2/2 (100%) | 2/2 | 0.002378306 | 5.401673317 |
| Retain direct | 6/6 (100%) | 6/6 | 0.002017770 | 5.565671364 |
| All trained paraphrases | 0/8 (0%) | 0/8 | 6.877501488 | 5.521784842 |
| Forget paraphrases | 0/2 (0%) | 0/2 | 7.410024881 | 5.390592098 |
| Retain paraphrases | 0/6 (0%) | 0/6 | 6.699993690 | 5.565515757 |
| Never-trained direct probes | 0/2 (0%) | 1/2 | 9.240544558 | 5.586338043 |
| Never-trained paraphrase probes | 0/2 (0%) | 0/2 | 9.014351368 | 5.583830357 |

All 20 final generations ended on EOS; none hit the cap. Direct EOS NLL averages
0.000318425, versus 8.148469105 on the eight trained paraphrases. All eight direct
correct-over-wrong margins are positive (8/8), ranging from 0.403921305 to
2.267382389 nats/byte. The comparator is the next acquired entity cyclically for
the same attribute, not an exhaustive or adversarial candidate set.

| Fact | Direct answer (correct) | Byte NLL | Correct-over-wrong margin |
| --- | --- | --- | --- |
| dev:00:0 | dkqqudhttrnz | 0.002071005 | 0.744189698 |
| dev:00:1 | srxyypyysoha | 0.002685608 | 0.403921305 |
| dev:04:0 | dwtnpephduwo | 0.001533846 | 1.792253753 |
| dev:04:1 | vpwrirmvjrzu | 0.003226905 | 1.885490150 |
| dev:05:0 | qwpzmngjiabe | 0.001780027 | 2.267382389 |
| dev:05:1 | utozqplgiwpo | 0.001642379 | 0.537572888 |
| dev:06:0 | qypznrvbjkoz | 0.002043833 | 0.587255667 |
| dev:06:1 | dowwzqiofnxp | 0.001879630 | 0.574201825 |

Incorrect generations must remain visible:

| Fact | Incorrect paraphrase generation |
| --- | --- |
| dev:00:0 | pzqpnxp |
| dev:00:1 | qpr |
| dev:04:0 | pzqpnxp |
| dev:04:1 | qpr |
| dev:05:0 | pzqpnxpzxp |
| dev:05:1 | qpzqpz |
| dev:06:0 | pzqpnxp |
| dev:06:1 | qprnpzri |

Never-trained `dev:07:0` produces `qypznrvbjifnp` directly and `pzqpnxpzxp` on
the paraphrase. Never-trained `dev:07:1` directly produces `utozqplgiwpo`, the
trained **dev:05:1** answer: valid twelve-letter format, incorrect binding. Its
paraphrase produces `qprzqp`. None of these is successful recovery.

Direct accuracy is acquisition-training resubstitution. Retain direct accuracy
measures six memorized associations, not general language utility. The untouched
M0 acquisition no-op checked serialization/determinism before acquisition; it
did not check a no-op at M_all. There is no measured unlearning arm, acquired-model
no-op, retain-only continuation, retain-from-scratch oracle, prompt attack suite,
membership inference or broad utility benchmark. The new pilot's controls are
prepared, not measured. Paraphrase losses have worsened relative to M0 despite
perfect direct recall. The 0/8 paraphrase baseline creates a floor: subsequent
0/8 cannot demonstrate paraphrase forgetting or transfer of a forgetting effect.

## Frozen selection and provenance

[The frozen manifest](../experiments/tiny_unlearning_task70_frozen.json) preserves
the exact source facts, direct/paraphrase rows, accepted baseline rows, original
random baseline, all acquisition artifact hashes, source hashes, Python/runtime
versions, accepted parameter hash and preparation Git parent. Its SHA-256 is
`c7312bbb9413d57896ed58e66551ed17a0001ee431fbaf15cc9555e0dc1a9634`.

Selection reuses roles fixed before acquisition, without screening by final
paraphrase or unlearning results:

- Forget: `dev:00:0`, `dev:00:1`, both attributes of the original forget entity.
- Retain: `dev:04:0`, `dev:04:1`, `dev:05:0`, `dev:05:1`, `dev:06:0`, `dev:06:1`.
- Never-trained evaluation probes only: `dev:07:0`, `dev:07:1`.

The sets are disjoint; selection has no random seed because it reuses the frozen
role rule. Acquisition initialization seed stays 2026, and every arm loads the
same accepted weights. Update example ordering seed is **2070**. Within each
full-group update, facts are hash-ranked by SHA256 of `2070:step:fact_id` using
the unchanged ordering helper. Both update arms use the same retain order;
actual order is logged. No dropout or stochastic decoding is introduced.

[The new profile](../experiments/tiny_cpu_unlearning_task70.json) is pinned by
SHA-256 in [the separate runner](../src/tiny_model_unlearning.py). There are no
CLI overrides for data, model, device, learning rate, objective or budget.
Original code, configurations, checkpoints, baselines and prior outputs remain
unchanged. The old acquisition output is never an output target. Git history is
extended by an ordinary commit, without amending, rebasing or force pushing.

## Objective and controls

The explicitly named objective is
**tiny_answer_only_full_parameter_gradient_difference_v1**:

`J(theta) = -1 * mean_{f in F} NLL_answer+EOS(f; theta)`
`           +1 * mean_{r in R} NLL_answer+EOS(r; theta)`.

Minimizing J ascends forget NLL and descends retain NLL. Each example has twelve
answer bytes plus EOS and masked prompt positions. Each step averages the two
forget examples separately from the six retain examples: a 1:1 group weighting,
not 2:6 weighting. Sequential scaled batch-one backward passes compute the exact
gradient of these group means without storing eight graphs. Every update uses
all members of each applicable group. The fixture gradient-parity test compares
this with a single combined differentiable objective.

This is the NLL gradient-difference algebra in [src/graddiff.py](../src/graddiff.py).
It is a tiny causal-LM adaptation, not a reproduction of the project's
[full-scale GradDiff profile](../experiments/graddiff_tofu_forget01_retain99.json).
That profile targets the TOFU-trained Llama-3.2-1B model, subword/instruction
inputs, BF16 CUDA distributed training, paged AdamW, minibatches and ten epochs.
This pilot updates all 90,496 parameters of a synthetic-fact-acquired byte LM,
FP32 CPU, deterministic full groups, stateless SGD and twenty steps. No KL term,
reference-model training penalty, NPO, LoRA, TOFU Forget Quality or deletion
certificate is claimed. Evaluation KL uses cached M_all distributions only; it
does not enter the training objective.

All three arms reload the accepted step-700 checkpoint independently:

1. **No-op:** full before/after evaluation, zero backward or optimizer steps;
   generated IDs and NLL must agree and parameter bytes must remain identical.
2. **Retain-only continuation:** minimize mean_R NLL for 20 SGD steps, never
   forward forget examples for training. This controls ordinary retain training
   and its incidental forget drift. It is not retraining from random M0 on R.
3. **Gradient difference:** minimize J for 20 SGD steps from the same M_all.

The update arms use SGD LR **0.05**, zero momentum/dampening/weight decay, no
scheduler, non-fused/non-foreach implementation, gradient clip norm 1.0. This
is a first bounded pilot, not a tuned efficacy claim. Gradients are tiny at the
accepted low-loss starting point; twenty steps may produce no forgetting. Such
a complete negative result must be retained, without a budget extension or
post-hoc learning-rate search. Arm order is fixed: no-op, retain-only, GradDiff.
Runtime/order effects remain a limitation of this single sequential pilot.

## Before/after measurements and decision

Before each arm's optimizer creation, all twenty cases must reproduce acquired
generated IDs and NLLs (absolute tolerance `1e-5`). A fresh baseline file is
sealed per arm. The two update arms also perform one objective backward-only
rehearsal and require unchanged parameters before creating SGD. This rehearsal
is part of the future authorized experiment, not preparation.

Evaluate every 5 successful updates (steps 5, 10, 15, 20) using the unchanged
acquisition evaluator and frozen prompts. The before evaluation is step 0.
Primary comparison is **fixed final step 20**, with no outcome-based checkpoint
selection. Safety can terminate the suite; early outcomes are diagnostic only.

- Direct forgetting: both forget cases' generated bytes/EOS and exact recovery;
  keep every raw output, format validity and finish reason.
- Paraphrase transfer: the two forget and six retain paraphrases separately,
  generated outputs and paired loss changes. Treat recovery as floor-limited;
  any newly correct forget paraphrase is residual/re-emergent recovery.
- Retain utility: six trained retain direct cases separately from the two
  never-trained probes. Keep all direct/paraphrase denominators.
- Loss-based recovery indicators: per-case byte NLL, inclusive/EOS NLL,
  correct-versus-one-wrong binding margin, and
  `exp(-12 * answer_byte_nll)`, the teacher-forced target-byte sequence likelihood
  excluding EOS. Lower likelihood or a bad greedy answer alone does not show
  inaccessible knowledge or resistance to extraction. Prompt/EOS disruption
  and wrong bindings can mimic forgetting.
- Model-wide degradation proxies: eight fixed unrelated ASCII sentinel strings
  in the profile, teacher-forced next-token NLL including EOS, full-vocabulary
  entropy and mean `KL(M_all || arm)` over all next-token positions. Cache the
  no-op baseline log distributions and compare each arm to that same reference.
  These sentinels' actual starting losses/KL have not been measured in preparation;
  they will be sealed before the first optimizer step. Their narrow, unlearned
  language coverage supports drift detection, not general utility certification.
- Paired final-minus-before NLL/EOS/margin changes and exact transitions are
  reported for each case and arm. Compare GradDiff against retain-only as well
  as no-op. Retain-only's forget drift can invalidate attribution.

The selective **direct** forgetting gate requires all of the following at final
step 20, after every arm completes its budget and all integrity/safety checks pass:

- Forget direct exact recovery **0/2**; each fact's byte NLL rises at least
  **1.0 nat/byte** from its own M_all baseline; each direct correct-over-wrong
  margin is **≤0**; each byte NLL exceeds its matched retain-only final NLL by
  at least **0.5 nat/byte**.
- Retain direct exact recovery **6/6**; every retain byte NLL rises by at most
  **0.1 nat/byte** from its own baseline.
- Sentinel mean NLL rises by at most **0.1 nat/token** and mean KL is at most
  **0.1 nat/token**. No-op loss drift is at most `1e-5`, generated IDs match,
  and weights are identical.

These are preregistered operational pilot thresholds, not statistically
calibrated privacy thresholds. Report gate failure truthfully. Even gate success
would only support selective direct association suppression on these two facts.
The runner always reports `paraphrase_effect_identifiable=false` and
`privacy_protection_established=false`.

## Guards, outputs and stop conditions

The suite needs both `--run-controlled-experiment` and a new exact Task 70
authorization token. The acquisition token is rejected. Authorization is checked
before environment/model/file work, before SGD construction and before each
optimizer step. This document is a protocol, not human authorization.

CPU policy reuses the unchanged tiny acquisition protections: normal macOS
arm64 Terminal, reviewed Apple M4 Pro host, locked dependencies and acquisition
Python/Llama source versions, ≥48 GiB total physical memory, dynamic preload
floor ≥12 GiB, continuing availability ≥6 GiB, NORMAL pressure, ≥20% free
indicator, process lifetime peak RSS ≤8 GiB, no cumulative swap/page-out growth,
250 ms watchdog and ≤2-second fresh sample gaps. No MPS execution/fallback,
allocator override, downloads or dependency changes. Eight cached sentinel
distributions fit well within the inherited 1 GiB workspace allowance; fit is
not measured by the fixture. The reported policy remains an estimate.

Entire suite deadline: **600 seconds**. Conservative remaining projection uses
twice the largest observed rehearsal/update time, forty total update budget,
twenty evaluation-time allowances and a 60-second load/save reserve; projection
≥480 seconds stops admission/continuation. Require ≥4 GiB free disk before load
and every checkpoint write. Full-group backward cost differs from acquisition,
so acquisition runtime alone does not establish admission.

STOP on any physical/time/telemetry/dependency/input/configuration drift,
nonfinite loss/logits/gradients/parameters, absent or zero aggregate gradient,
preclip gradient norm >100, optimizer drift, unexpected/partial optimizer
attempt, failure to reproduce baseline/no-op, serialization/round-trip failure,
or changed original artifacts/source bytes. At scheduled evaluations also STOP
if any direct retain answer becomes incorrect, any retain direct byte NLL rises
>0.5, any direct forget byte NLL exceeds 20, sentinel mean NLL rises >0.5, or
sentinel mean KL exceeds 0.5. The looser safety limits differ from the stricter
final acceptance limits. Drift between scheduled evaluations can go unobserved;
no step is rolled back or retried.

Attempt counters increment immediately before SGD; successful counters increment
only after return. A throwing step leaves `possible_partial_update=true`, seals
FAILED, and cannot resume. The limits are 20 successful/attempted steps per update
arm and 40 suite-wide; no-op always has zero. No optimizer state is saved.

Only a fresh `outputs/local_synthetic_unlearning/unlearning-task70-*` directory
is accepted. No overwrite, resume, deletion, replacing model configuration or
altering original mode bits. New outputs include RUNNING, per-arm before files,
scheduled evaluation files, independent final arm checkpoints with hashed
COMPLETE markers, and a final report bound by COMPLETE or FAILED. Final reports
include source/profile/data hashes, runtime versions, Git head, encoded-data
hash, example order, loss/gradient traces, step counters, weight fingerprints,
physical observations and paired comparisons. Final checkpoints are reloaded
to verify fingerprints. Original acquisition hashes are rechecked on exit.
`controlled_pilot_complete` means the protocol completed, including a possible
negative gate result; only the explicit result field describes the direct gate.
`STOP`/FAILED tombstones are diagnostic outputs, never accepted experiments.

## Single next controlled experiment

After reviewing this protocol, obtain **independent explicit human authorization
for this exact three-arm, forty-total-update experiment**. No real run is part
of Task 70. Once authorized, execute once in a fresh normal Terminal:

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-ml/bin/python -B -m src.tiny_model_unlearning \
  --run-controlled-experiment \
  --authorize-weight-update 'task70:tiny-90496:cpu:three-arms:20-per-arm:40-total' \
  --run-dir outputs/local_synthetic_unlearning/unlearning-task70-001
```

The output name must not exist. A collision or guard failure requires review;
do not remove the directory, retry, increase the budget or substitute a device.
Inspect all three arms and compare final step 20 against frozen baselines. A
complete negative outcome is still the result of this experiment. Further
experiments require separate review/authorization.

Preparation-only commands, safe without that future authorization:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.tiny_model_unlearning
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m src.tiny_model_unlearning --dry-run
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-ml/bin/python -B -m unittest discover -s tests \
  -p 'test_tiny_model_unlearning.py' -v
```

Default execution only hashes/reads files and prints the frozen plan. Fixture
dry run initializes random weights and uses two unrelated four-token batches,
finite backward gradients, unchanged weights, no checkpoint load/save and zero
optimizer steps. Tests trap/mock optimizer execution, compare fixture objective
gradients, exercise all controls with synthetic evaluation rows, and test guards
and failure sealing. They do not evaluate or update the acquired model.

Preparation passed **217 offline tests** (22 new tests and 195 existing safeguard
regressions), with no failures, errors or skips. Both CLI plan and fixture dry
run passed with zero optimizer steps. A preservation ledger independently
verified SHA-256, size and mode for **all 177 pre-existing tracked/output files**;
the preparation Git parent remained unchanged before the new commit.

Verification is recorded in ignored `outputs/task70-offline-checks.log`,
`outputs/task70-offline-evidence.json` and `outputs/task70-preservation-before.json`.
The comparison result is in `outputs/task70-preservation-after.json`.
Readiness concerns implementation and offline checks only. This model's eight
public synthetic facts, two forget examples, no learned paraphrase transfer and
limited utility probes cannot establish privacy, real-data deletion, TOFU
performance, statistical generalization or adversarial extraction resistance.
