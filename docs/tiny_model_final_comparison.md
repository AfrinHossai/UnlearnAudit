# Task 85: final tiny-model unlearning comparison

Analysis date: 2026-11-04. **Gamma=1 completed but failed the selective-forgetting
criterion. Gamma=10 stopped for retained-knowledge degradation and is invalid
for the success gate. Neither experiment establishes selective forgetting or
privacy protection.** The study is ready for an honest preliminary professor
presentation of a controlled negative result and a safety-stopped follow-up.

## Evidence and matched comparison

Primary evidence is the immutable local [Task 70 report](../outputs/local_synthetic_unlearning/unlearning-task70-001/report.json),
[Task 79 report](../outputs/local_synthetic_unlearning/stronger-forgetting-task79-001/report.json),
their baseline/scheduled evaluation files and markers, the
[frozen manifest](../experiments/tiny_unlearning_task70_frozen.json),
[historical inventory](../experiments/tiny_stronger_forgetting_task79_history.json),
and [gamma=1](../experiments/tiny_cpu_unlearning_task70.json) and
[gamma=10](../experiments/tiny_cpu_stronger_forgetting_task79.json) profiles.
The [Task 77 findings](tiny_model_unlearning_findings.md) describe gamma=1.
The older [Task 70](tiny_model_unlearning.md) and [Task 79](tiny_model_stronger_forgetting.md)
protocols describe preparation; their pending-run language is historical. The
sealed reports establish the subsequent outcomes. Local ignored artifacts are
required to follow the report links; this document does not publish those files.

Both GradDiff arms independently started from the accepted **90,496-parameter,
FP32 CPU acquisition checkpoint at step 700**, not from each other's endpoint.
Their full recorded baselines are exactly equal. Forget facts are `dev:00:0`
and `dev:00:1`; retain facts are `dev:04:0`, `dev:04:1`, `dev:05:0`, `dev:05:1`,
`dev:06:0`, `dev:06:1`. Ten facts each have direct and paraphrase prompts:
**20 evaluation cases**, including four never-trained probe cases.

The objective is `mean_R NLL(answer+EOS) − gamma × mean_F NLL(answer+EOS)`.
Alpha=1, full parameters, full-group means, SGD LR=0.05, clip norm=1.0,
seed=2070, example ordering, evaluation wording and safety thresholds are fixed.
Only gamma changes in the training configuration. Task 79's fresh execution
budget is 20 calls; Task 70's suite budget was 40 across two update arms.
Task 79 reuses the sealed no-op, retain-only and gamma=1 measurements; these are
controls/comparators, not fresh independent replications.

## Results

Recovery means all twelve correct ASCII answer bytes followed by EOS, under
greedy decoding with a 16-token cap. Baseline recovery is **2/2 forget and 6/6
retain**. NLL columns below are arithmetic means over the same two forget or
six retain direct cases, in nats/byte. Gamma=10's last observation is diagnostic.

| Arm | Completed steps / attempts | Recorded endpoint | Forget recovery | Retain recovery | Forget byte NLL | Retain byte NLL | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No-op, reused | 0/0 | Step 0 | 2/2 | 6/6 | .002378306 | .002017770 | Unchanged baseline |
| Retain-only, reused | 20/20 | Step 20 | 2/2 | 6/6 | .002404234 | .001961188 | No direct forgetting |
| GradDiff gamma=1 | 20/20 | Step 20, protocol COMPLETE | 2/2 | 6/6 | .002659522 | .001936726 | Selective-forgetting gate **false** |
| GradDiff gamma=10 | 20/20 | Step 20 evaluation, **STOP / FAILED** | 0/2 | 5/6 | .947672904 | .118086161 | `result_valid=false`; **no accepted unlearning checkpoint** |

Task 70 totals **40 completed calls / 40 attempts**; Task 79 totals **20/20
fresh calls**, excluding historical calls. Every report/arm partial-update flag
is false. Task 79 has twenty sequential training records, steps 1–20. Completing
the optimizer budget did not complete the protocol: the scheduled safety check
raised before final checkpoint creation, reload verification and gate evaluation.

### Matched direct facts and confidence

Every direct fact was recovered at baseline and under gamma=1. The first two
rows are forget facts; the remaining six are retain facts. Margin is
`wrong-binding byte NLL − correct byte NLL`, against one fixed same-attribute
wrong binding; positive favors the correct answer. It is not an exhaustive
measure of answer accessibility.

| Fact | Baseline byte NLL | Gamma=1 byte NLL | Gamma=10 byte NLL | Gamma=1 margin | Gamma=10 margin | Recovery gamma=1 → gamma=10 (each /1) |
| --- | --- | --- | --- | --- | --- | --- |
| dev:00:0 | .002071005 | .002187567 | .669502020 | .745954973 | .836462617 | 1 → 0 |
| dev:00:1 | .002685608 | .003131476 | 1.225843787 | .374556186 | −.899431974 | 1 → 0 |
| dev:04:0 | .001533846 | .001528252 | .666106522 | 1.790393006 | 1.552783191 | 1 → 0 |
| dev:04:1 | .003226905 | .002891446 | .023774186 | 1.888401723 | 2.299093684 | 1 → 1 |
| dev:05:0 | .001780027 | .001750235 | .001925740 | 2.274135347 | 2.361595597 | 1 → 1 |
| dev:05:1 | .001642379 | .001613832 | .002032631 | .539726817 | .554093904 | 1 → 1 |
| dev:06:0 | .002043833 | .001987517 | .004370246 | .586177588 | .481054362 | 1 → 1 |
| dev:06:1 | .001879630 | .001849073 | .010307637 | .577318949 | 1.058240685 | 1 → 1 |

The following means retain each subset's denominator (forget n=2; retain n=6).
Inclusive NLL averages twelve answer bytes plus EOS; EOS NLL uses one token per
case. `P(target)` is the mean of each case's `exp(−12 × byte NLL)`, excluding
EOS, not a recovery frequency or the exponential of mean NLL.

| Subset / state | Inclusive NLL | EOS NLL | P(target) | Mean margin |
| --- | --- | --- | --- | --- |
| Forget / baseline | .002232759 | .000486195 | .971870344 | .574055501 |
| Forget / gamma=1 | .002492888 | .000493284 | .968605158 | .560255580 |
| Forget / gamma=10 diagnostic | .874813884 | .000505497 | .000162325 | −.031484678 |
| Retain / baseline | .001882749 | .000262502 | .976099887 | 1.274026112 |
| Retain / gamma=1 | .001808072 | .000264230 | .977041636 | 1.276025572 |
| Retain / gamma=10 diagnostic | .109022784 | .000262183 | .756292343 | 1.384476904 |

Gamma=1 raises mean forget byte NLL by **.000281216**, only **.000255288**
beyond retain-only, with unchanged generated answers. All eight forget checks
fail: each fact remains recovered, each NLL increase is below 1.0, each margin
is positive and each NLL excess over retain-only is below .5. The fourteen
retain/sentinel checks pass; completion and a failed efficacy gate are consistent.

Gamma=10 minus gamma=1 gives paired mean byte-NLL differences of **+.945013382
forget** and **+.116149435 retain**. Mean target probabilities fall by
**.968442833** and **.220749293**, respectively. These are recorded confidence
and behavioral changes accompanying retained-knowledge damage. Gamma=10 does
not even meet every numerical forgetting condition: `dev:00:0` increases only
**.667431015** from baseline (required ≥1.0) and keeps a **positive .836462617
margin** (required ≤0). Its generated answer hits the length cap and is invalid;
the other forget output is an incorrect twelve-letter answer with EOS. Loss of
exact generation therefore cannot establish knowledge deletion.

### Exact STOP trigger and evaluation boundary

The recorded reason is **`RuntimeError: retain degradation STOP`**. The
hash-verified [safety function](../src/tiny_model_unlearning.py) raises this when
any retain direct answer is incorrect **or** its byte NLL increases by more than
`limits.max_retain_byte_nll_increase = .5`. At step 20, the first failing retain
row is **`dev:04:0:direct`**, the archive code of `UGDEV2026-Object-04`:

- Expected `dwtnpephduwo`; generated `vnpzngvjiabe` followed by EOS. This is a
  valid-format answer but incorrect, reducing retained direct recovery to 5/6.
- Byte NLL changes **.001533846 → .666106522**, an increase of **.664572676**.
  Thus **both conditions** of that retain safeguard are violated by the same row.
  The implementation's `or` short-circuits on incorrect recovery; it does not
  record a separate exception for the NLL condition.

| Gamma=10 evaluation step | Forget recovery | Retain recovery | Largest per-retain byte-NLL increase | Sentinel mean KL |
| --- | --- | --- | --- | --- |
| 0 | 2/2 | 6/6 | 0 | 0 |
| 5 | 2/2 | 6/6 | .000069516 | .002979104 |
| 10 | 2/2 | 6/6 | .000362938 | .037131160 |
| 15 | 1/2 | 6/6 | .009791840 | .213379333 |
| 20 | 0/2 | 5/6 | .664572676 | .456124648 |

The [Task 79 runner](../src/tiny_model_stronger_forgetting.py) appends and seals
the evaluation **before** calling the safety function. Hence a step-20 JSON
exists despite STOP. Steps 5, 10 and 15 pass scheduled safety checks; step 20 is
the first observed failing boundary. The exact onset between steps 15 and 20
is unmeasured. Step 15 is not a selectable successful endpoint (one forget fact
still recovers, and its sentinel KL already exceeds the final .1 criterion).

### Sentinel and paraphrase observations

Sentinels are eight fixed strings. NLL, KL and entropy are means of per-string
next-token averages including EOS, not pooled token-weighted means. KL is
`KL(acquired || arm)`, in nats/token.

| State | Sentinel mean NLL | NLL change from baseline | Mean KL | Mean entropy |
| --- | --- | --- | --- | --- |
| Baseline / no-op | 10.309173107 | 0 | 0 | .515305128 |
| Retain-only | 10.317793012 | +.008619905 | .000048130 | .513751190 |
| Gamma=1 | 10.309003115 | −.000169992 | .000484305 | .514646109 |
| Gamma=10 diagnostic | 10.292270184 | −.016902924 | .456124648 | .511027817 |

Gamma=10's sentinel means remain below the **.5 safety** thresholds; these did
not trigger STOP. Its KL exceeds the **.1 final acceptance** threshold. One
string's KL is 1.033847332, but the registered sentinel guard uses the mean.
Lower sentinel NLL does not imply retained factual knowledge or broad utility.

Acquired paraphrase recovery remains **0/2 forget and 0/6 retain** at baseline
and every scheduled evaluation in both GradDiff arms. Available loss/probability
measures are nevertheless reported below; margins were not recorded for these
prompts and are not inferred.

| Paraphrase subset (n) / state | Byte NLL | Inclusive NLL | EOS NLL | P(target) |
| --- | --- | --- | --- | --- |
| Forget (2) / baseline | 7.410025 | 7.439937 | 7.798877 | 2.240e−38 |
| Forget (2) / gamma=1 | 7.397927 | 7.431429 | 7.833455 | 2.599e−38 |
| Forget (2) / gamma=10 diagnostic | 7.024442 | 7.135361 | 8.466386 | 6.076e−37 |
| Retain (6) / baseline | 6.699994 | 6.820379 | 8.265000 | 5.588e−28 |
| Retain (6) / gamma=1 | 6.705123 | 6.821933 | 8.223648 | 5.418e−28 |
| Retain (6) / gamma=10 diagnostic | 6.513469 | 6.630749 | 8.038107 | 7.188e−24 |

Forget paraphrase byte NLL falls under both doses while EOS NLL rises. The
zero-recovery baseline creates a floor: unchanged zero cannot identify
forgetting transfer. Never-trained direct and paraphrase probes remain **0/2
each** in every arm at its endpoint; they cannot demonstrate preservation of
acquired knowledge.

## Offline integrity and checkpoint disposition

Task 85 used standard-library JSON, SHA-256, file metadata, AST and arithmetic
checks. Only the reviewed data-only `summarize`, `paired_deltas`, `decide` and
`safety_metrics` functions were extracted from the source AST; no runner module
or ML library was imported. No model inference, training, optimizer calls,
fixture training tests, retry or threshold change occurred.

- All **37 acquisition files** and **26 Task 70 files** match their frozen
  SHA-256 inventories, with exact file-set coverage. All eight acquisition and
  three Task 70 checkpoint bundles pass file hashes, COMPLETE contents,
  configuration hashes, metrics/evaluation, step and recorded-fingerprint checks.
- Acquisition remains accepted at **700/700 steps/attempts**, 8/8 direct recovery
  and the per-fact ≥1-nat acquisition gain criterion. Accepted weights remain
  `acquisition-task64-001/checkpoint-000700/model.safetensors`, SHA-256
  `fdaaf3ab932db016dec2361210240fca62c1aaf165273444fcc7aa5bbda910b2`.
  Recorded parameter fingerprint remains
  `f5211ab5e5f6eb9d09991d884ff84def1e93d9985471411e7f13269c44de687a`.
- Task 70 has a valid protocol COMPLETE and three serialized arm checkpoints;
  its GradDiff checkpoint is evidence of a completed negative experiment, not
  an accepted selective-forgetting model. The no-op weights equal acquisition
  bytes. Gamma=1 GradDiff weights hash to
  `9d00ee981f670582699e9b57020a9dc708cbb3c82f70d82b95adbc806445191a`.
- Task 79 has exactly **nine JSON files**, including RUNNING and FAILED;
  **no run COMPLETE, no checkpoint directory, no serialized model,
  `checkpoints=[]`, no final parameter fingerprint, no `result`, and no
  `integrity_verified` field**. There is no saved gamma=10 model to accept or
  reuse. Its sealed report and evaluations are diagnostic artifacts only.
  RUNNING records initial authorization/budget, superseded by FAILED.
- Separate before/step JSONs agree with their reports; the fresh baseline repeat
  equals baseline. Source/provenance hashes, pinned profiles/manifest/history,
  recorded runtime versions, installed Llama source hash, encoded-data hash,
  row coverage, orders, probabilities, margins, summaries and sentinel means
  agree. Task 70's 22 decision checks recompute exactly. Replaying only the
  data-only safety function reproduces the step-20 retain exception.
- Task 70 records Git head `ad9cef0b9cb3827fea32e40fd7c64ef16ef83f57`;
  Task 79 records `3807c6ad8f3122f88932e986667bb0a96ebb1b4a`. All 72 files
  across the three experiment directories retain mode 0444. Before/after
  SHA-256, size, mode and modification-time checks preserve all **234
  pre-existing tracked/output/result files** audited in this task.

| Report | Recomputed SHA-256, also bound by its terminal marker |
| --- | --- |
| Acquisition COMPLETE | `88678bd4f4f86e4d8eec8fede0be5da1d7d3820d6b4bb1c550a6d490b66b8ad3` |
| Task 70 COMPLETE | `5c047bd4c5c4d221e501764bd17ad321b562d0d61d2c17d698fe8930571e9241` |
| Task 79 FAILED / STOP | `7255425db6c2e68987f614ec9c2e52eff0c8d955af6a35646c9b235896119fef` |

Weights were hashed as opaque bytes. This audit verifies file integrity and
consistency of recorded metadata, not independent tensor fingerprints or
reproduced inference. Historical checkpoint reload checks were performed by
the original completed runners; gamma=10 never reached that stage.

## Scientific findings, limits and presentation

Increasing gamma is associated here with stronger direct-answer suppression,
larger confidence shifts and greater sentinel KL, accompanied by loss of a
retained fact. Shared initialization, exact baseline reproduction and fixed
training settings support a controlled dose comparison. They do not prove a
specific causal mechanism for the retain error, demonstrate a general dose
response or establish that selective forgetting is impossible. Preclip norms
are .017229–.019546 for gamma=1 and .141091–48.064289 for gamma=10; gamma=10
exceeds clip 1.0 from step 11 onward, so a tenfold coefficient is not a tenfold
effective update.

This is one deterministic synthetic pilot: two forget attributes of one entity,
six retain attributes of three entities, one seed and one stronger dose chosen
after gamma=1's negative result. Direct prompts repeat acquisition prompts;
facts are not independent population samples. Reused controls leave session
variability unreplicated. Paraphrase recovery starts at zero; sentinels have high
baseline loss and narrow coverage. There is no retain-from-scratch oracle,
adversarial extraction, membership-inference assessment, broad utility benchmark
or statistical replication. No verified selective forgetting, privacy protection,
real-data deletion or generalization claim is supported.

**Professor-ready summary:** We first verified that a tiny model acquired eight
synthetic associations. With matched controls, gamma=1 GradDiff slightly reduced
forget-answer confidence but still recovered both forget and all six retain
answers, failing the selective-forgetting criterion. The gamma=10 follow-up
completed twenty updates, then its scheduled evaluation lost both forget answers
and one retain answer. The retained-knowledge safeguard stopped the run, leaving
diagnostic measurements and no saved gamma=10 checkpoint. The accepted acquisition
and historical evidence remain intact. This is a preliminary study of a failed
selective-forgetting gate and observed suppression with collateral degradation;
it does not demonstrate privacy protection.

**Readiness:** Yes, for that honest preliminary presentation. Remaining
communication deliverables are a consolidated **research report**, a
**Streamlit dashboard**, and a **recorded demo**. They should present the matched
denominators, confidence/recovery distinction, STOP boundary and limitations
above. These deliverables remain to be completed; Task 85 produces this analysis
document and does not restart the stopped experiment.
