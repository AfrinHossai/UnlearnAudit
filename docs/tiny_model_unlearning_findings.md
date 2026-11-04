# Task 77: controlled tiny-model unlearning findings

**The completed pilot shows a small selective confidence change, with no observed
forgetting.** Gradient difference (GradDiff), no-op, and retain-only all preserve
forget recovery at **2/2** and retain recovery at **6/6**. The preregistered
selective-direct-forgetting gate fails. This is sufficient for an honest
preliminary professor presentation of a controlled negative result; it does not
support an efficacy, privacy, or generalization claim.

## Evidence and offline integrity

Analysis date: 2026-11-04. Sources: the local
[completed report](../outputs/local_synthetic_unlearning/unlearning-task70-001/report.json),
its before/evaluation files and final checkpoint bundles, the
[frozen manifest](../experiments/tiny_unlearning_task70_frozen.json),
[profile](../experiments/tiny_cpu_unlearning_task70.json),
[protocol](tiny_model_unlearning.md), and hash-verified
[runner](../src/tiny_model_unlearning.py) (`decide`, `paired_deltas`, `summarize`).
The protocol is historical preparation text; completion is established by the
sealed artifacts below.

| Verification | Observed result |
| --- | --- |
| Run status | `controlled_pilot_complete`; no stop reason or FAILED marker; 13.294 s recorded runtime |
| Successful steps / attempts | No-op 0/0; retain-only 20/20; GradDiff 20/20; suite 40/40; all partial-update flags false |
| Acquisition preservation | All 37 original files match the frozen SHA-256 inventory; acquisition remains accepted at 700/700 steps/attempts, 90,496 parameters |
| Checkpoints and markers | All 8 acquisition and 3 final-arm bundles pass file-hash, COMPLETE-content, config, metrics, step and recorded-fingerprint consistency checks |
| Supporting data | All 26 unlearning files accounted for; separate before/evaluation files agree with report; profile, manifest, source hashes, encoded-data hash, orders, summaries, paired deltas and all 22 decision checks recompute correctly |

The run COMPLETE binds report SHA-256
`5c047bd4c5c4d221e501764bd17ad321b562d0d61d2c17d698fe8930571e9241`.
The retained RUNNING file is an initial authorization/budget record, superseded
by COMPLETE. The accepted acquisition remains **checkpoint-000700**. Its run's
report hash is `88678bd4f4f86e4d8eec8fede0be5da1d7d3820d6b4bb1c550a6d490b66b8ad3`.

| Model file | Recomputed SHA-256 |
| --- | --- |
| Accepted acquisition / no-op final (identical bytes) | `fdaaf3ab932db016dec2361210240fca62c1aaf165273444fcc7aa5bbda910b2` |
| Retain-only final, step 20 | `8fc09ef2e900d830c0b7b4384a8db0f2aa267e6a07d5bf32e3d794381269d941` |
| GradDiff final, step 20 | `9d00ee981f670582699e9b57020a9dc708cbb3c82f70d82b95adbc806445191a` |

All arms record the same initial parameter fingerprint
`f5211ab5e5f6eb9d09991d884ff84def1e93d9985471411e7f13269c44de687a`;
no-op retains it and has exactly identical before/after measurements. The two
updated arms have distinct final fingerprints consistent with their checkpoint
metadata. This audit hashes serialized weights as opaque files; it does **not**
load weights or independently reproduce tensor fingerprints/inference. The
original runner recorded successful checkpoint reload verification.

Only standard-library file/JSON/hash/AST/arithmetic checks were run. The audit
executed only the three extracted data-only functions named above, without
importing the experiment runner. No model, optimizer, fixture training tests or
cloud resources were used. The local ignored evidence is
`outputs/task77-offline-audit.py`, `outputs/task77-offline-audit.json` and
`outputs/task77-preservation-before.json`; SHA-256, size, mode and modification
time remain unchanged for all 165 pre-existing tracked/historical-output files,
including the complete 63-file acquisition/unlearning output inventory.

## Direct recovery and confidence

All arms start independently from the accepted model. The fixed endpoint is
step 20 for update arms and step 0 for no-op. GradDiff minimizes
`mean_R NLL(answer+EOS) - mean_F NLL(answer+EOS)` with equal group weights;
both update arms use 20 full-group SGD steps, LR 0.05, clip norm 1.0.

Exact recovery requires all 12 target ASCII bytes followed by EOS under the
frozen greedy decoding protocol. Fractions below are recovered cases / evaluated
cases; never-trained probes are excluded from the six acquired retain facts.

| Arm | Forget direct, before → final | Retain direct, before → final | Forget paraphrase | Retain paraphrase | Never-trained direct / paraphrase |
| --- | --- | --- | --- | --- | --- |
| No-op | 2/2 → 2/2 | 6/6 → 6/6 | 0/2 → 0/2 | 0/6 → 0/6 | 0/2 → 0/2, each kind |
| Retain-only | 2/2 → 2/2 | 6/6 → 6/6 | 0/2 → 0/2 | 0/6 → 0/6 | 0/2 → 0/2, each kind |
| GradDiff | 2/2 → 2/2 | 6/6 → 6/6 | 0/2 → 0/2 | 0/6 → 0/6 | 0/2 → 0/2, each kind |

Every final generated token sequence is unchanged from its arm's baseline.
All 20 cases end on EOS in every arm. Valid answer format remains 2/2 forget
direct, 6/6 retain direct, 0/8 acquired paraphrases, 1/2 never-trained direct
and 0/2 never-trained paraphrases. The valid never-trained answer is an incorrect
binding, not recovery. Both update arms also keep 2/2 forget and 6/6 retain
recovery at intermediate steps 5, 10 and 15.

The following are arithmetic means over **2 forget** or **6 retain** direct rows.
All arms have exactly the same before measurements. NLL units are nats/byte;
inclusive NLL averages 12 bytes plus EOS; EOS NLL is one token per case.
`P(target)` is the mean of each row's `exp(-12 × byte NLL)`, excluding EOS,
**not** the exponential of the group mean or a recovery frequency.
Margin is `wrong-binding byte NLL − correct byte NLL`; positive favors the
correct answer against one frozen same-attribute wrong binding.

| Arm / subset | Byte NLL, before → final | Δ byte NLL | Inclusive NLL, before → final | EOS NLL, before → final | P(target), before → final | Margin, before → final |
| --- | --- | --- | --- | --- | --- | --- |
| No-op / forget | .002378306 → .002378306 | 0 | .002232759 → .002232759 | .000486195 → .000486195 | .971870344 → .971870344 | .574055501 → .574055501 |
| Retain-only / forget | .002378306 → .002404234 | +.000025928 | .002232759 → .002256513 | .000486195 → .000483872 | .971870344 → .971569787 | .574055501 → .570597762 |
| GradDiff / forget | .002378306 → .002659522 | +.000281216 | .002232759 → .002492888 | .000486195 → .000493284 | .971870344 → .968605158 | .574055501 → .560255580 |
| No-op / retain | .002017770 → .002017770 | 0 | .001882749 → .001882749 | .000262502 → .000262502 | .976099887 → .976099887 | 1.274026112 → 1.274026112 |
| Retain-only / retain | .002017770 → .001961188 | −.000056582 | .001882749 → .001830413 | .000262502 → .000261111 | .976099887 → .976758924 | 1.274026112 → 1.276659254 |
| GradDiff / retain | .002017770 → .001936726 | −.000081044 | .001882749 → .001808072 | .000262502 → .000264230 | .976099887 → .977041636 | 1.274026112 → 1.276025572 |

GradDiff produces an additional **0.000255288 nat/byte** forget NLL increase
over retain-only, with lower mean target probability by **0.296463 percentage
points** versus retain-only (0.326519 points versus no-op). This is a measurable
directional confidence effect on the recorded cases, not statistically established
or behaviorally successful forgetting. The two forget targets still have
probabilities **0.974090752** and **0.963119564** and both are generated exactly.
Their margins remain positive; the first actually increases by 0.001765275.

## Why the first eight checks fail

`result.checks` is an **unnamed list of 22 Booleans**, not a dictionary of named
tests. Mapping comes from `decide`'s direct-row traversal and the profile's
`acceptance` names. Interleaved paraphrase rows are skipped. Zero-based indices
0–3 apply to `dev:00:0:direct`, then 4–7 to `dev:00:1:direct`, in this order:

| Documented criterion | Required | dev:00:0 observed (index) | dev:00:1 observed (index) |
| --- | --- | --- | --- |
| `forget_direct_recovered` | 0/2 collectively; implemented as `not exact_recovery` for each fact | Still recovered (0) | Still recovered (4) |
| `minimum_each_forget_byte_nll_increase` | ≥1.0 from own baseline | +.000116562 (1) | +.000445869 (5) |
| `maximum_each_forget_correct_over_wrong_margin` | ≤0 | +.745954973 (2) | +.374556186 (6) |
| `minimum_forget_nll_increase_beyond_retain_only` | ≥.5 above matched final control | +.000129454 (3) | +.000381123 (7) |

All eight are false: both facts remain recoverable, NLL shifts are far below
the preregistered thresholds, and the correct answers still beat the chosen
wrong bindings. Indices 8–19 are the six retain facts' pairs of exact recovery
and `maximum_each_retain_byte_nll_increase` (≤.1); 20–21 are
`maximum_sentinel_nll_increase` (≤.1) and `maximum_sentinel_kl_nats` (≤.1).
Those 14 checks are true. Completion, budget and no-op integrity are additional
runner guards, not extra entries in this list. Thus protocol completion and gate
failure are consistent. `paraphrase_effect_identifiable=false` and
`privacy_protection_established=false` are explicit result fields.

## Retention, paraphrases and drift

Every retain direct byte NLL decreases under both updated arms. GradDiff's
retain direct EOS NLL rises by only 0.000001728 on average, with all six answers
still correct. There is no observed loss of acquired direct knowledge. Retain
paraphrase byte NLL worsens slightly, so preservation of broader knowledge or
utility cannot be inferred.

| Acquired paraphrase subset (n) | Mean byte NLL: before / no-op / retain-only / GradDiff | Mean EOS NLL: before / no-op / retain-only / GradDiff | Mean P(target): before / no-op / retain-only / GradDiff |
| --- | --- | --- | --- |
| Forget (2) | 7.410025 / 7.410025 / 7.419481 / 7.397927 | 7.798877 / 7.798877 / 7.812276 / 7.833455 | 2.240e−38 / 2.240e−38 / 2.127e−38 / 2.599e−38 |
| Retain (6) | 6.699994 / 6.699994 / 6.706282 / 6.705123 | 8.265000 / 8.265000 / 8.275921 / 8.223648 | 5.588e−28 / 5.588e−28 / 5.473e−28 / 5.418e−28 |

GradDiff's forget paraphrase byte NLL **decreases** by 0.012098 while EOS NLL
increases by 0.034578: loss components move in opposite directions. The
**zero-recovery paraphrase baseline (0/8, including 0/2 forget)** creates a floor;
unchanged zero recovery cannot establish forgetting transfer. Never-trained
probes also remain at zero and provide no evidence of retained learned knowledge.

| Arm | Sentinel mean NLL, before → final | Δ NLL | Mean KL(M_all || arm) |
| --- | --- | --- | --- |
| No-op | 10.309173107 → 10.309173107 | 0 | 0 |
| Retain-only | 10.309173107 → 10.317793012 | +.008619905 | .000048130 |
| GradDiff | 10.309173107 → 10.309003115 | −.000169992 | .000484305 |

Sentinel values average eight fixed strings, with each string's NLL/KL averaged
over its next-token positions including EOS; this is not a pooled token-weighted
mean. Both updated arms remain within the .1-nat final drift limits. GradDiff
has greater KL drift than retain-only despite lower mean sentinel NLL. These
unlearned ASCII strings already have high baseline loss and cannot certify
general language utility.

## Limitations and one next experiment

This is one sequential deterministic pilot on public synthetic facts: two forget
attributes of one entity and six retain attributes of three entities, one seed,
one dose and one fixed wrong answer per direct case. The cases are not independent
population samples. Direct prompts repeat acquisition training prompts. There is
no retain-from-scratch oracle, adversarial extraction suite, membership-inference
evaluation or broad utility benchmark. No privacy protection, real-data deletion,
generalization, full-scale GradDiff performance or method-wide failure follows.

**Recommend one separately authorized, preregistered stronger-forget-weight
three-arm experiment:** keep the accepted checkpoint, controls, 20 steps per
update arm, SGD LR .05, clipping, safety limits and fixed final gate; change only
GradDiff's forget weight from `gamma=1` to `gamma=10` (`alpha=1`). This is a
single dose comparison against the preserved current result, not a rate search
or extension of this run. The rationale is the near-saturated starting loss and
small recorded GradDiff gradient norms (.017229–.019546, below clip 1.0), which
are consistent with insufficient forgetting pressure; that mechanism remains an
inference, not a proved cause. Report any safety stop or negative result without
retuning or choosing an earlier checkpoint for apparent success.

A single such follow-up would meaningfully test dose sensitivity and whether
direct suppression can occur while retaining the six associations. It cannot
guarantee success or repair the paraphrase floor, and even success would support
only selective suppression of these direct synthetic associations. Present the
current study now as an integrity-checked controlled negative pilot, with the
small confidence effect and evaluation limits visible. No follow-up was executed
or authorized by this analysis.
