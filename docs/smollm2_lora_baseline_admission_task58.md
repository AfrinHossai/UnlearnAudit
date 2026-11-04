# Task 58: Read-only LoRA baseline memory admission

**Decision: a fresh read-only attempt is justified, conditional on fresh
admission under the separate v2 baseline guard.** No real model was loaded,
downloaded, trained or unlearned in Task 58. All checks here were offline.
The failed attempt's recorded availability would still fail this policy.

## Task 57 failure and exact cause

The available failure artifact is the ignored
`outputs/smollm2_lora/baseline-task53-001/report.json`, with SHA-256
`8be0fa4cbbaf55d82253e05fa071d27198004506c87bd0858dcc5b213541c0de`.
Its sealed `FAILED.json` binds that checksum. Although the directory was named
for Task 53's procedure, this is the saved failure reviewed for Task 57;
there is no separately named Task 57 transcript to establish further details.

The report records exactly one observation, at `preload`:

| Recorded field | Value |
|---|---:|
| Conservative free-plus-speculative availability | 18,465,570,816 bytes (17.197403 GiB) |
| Required by baseline v1 | 21,474,836,480 bytes (20 GiB) |
| Physical capacity | 51,539,607,552 bytes (48 GiB) |
| Current / lifetime peak process RSS | 228,278,272 bytes |
| MPS tensors / driver | 0 / 475,136 bytes |
| Pressure / free indicator | NORMAL (1) / 95% |
| Cumulative swap used / page-outs | 0 / 0 |

The error was `RuntimeError: LoRA preload floor failed (20 GiB required)`.
`model_loaded`, `optimizer_created`, `backward_attempted`, `step_attempted`
and `checkpoint_write_attempted` were false; backward, optimizer-step and
checkpoint-write counts were zero. Null fingerprints and
`base_checkpoint_unchanged=false` mean those checks were not reached, rather
than evidence of changed weights. This was an admission rejection, not an
observed inference OOM. No loaded LoRA memory peak can be inferred from it.

In [small_model_lora.py](../src/small_model_lora.py), both modes originally
constructed the same term dictionary. Only guard ID, maximum step count and
backward permission differed. For base bytes W=538,060,032, adapter bytes
L=921,600, source bytes S=269,060,552 and live preload lifetime peak RSS B:

```text
old incremental(B) = 3W + 2L + S + 4 GiB forward/backward workspace
                     + 1 GiB tokenizer/integrity + 1 GiB Metal cache
                     + max(1 GiB, B) additional process headroom
                     + 1 GiB adapter clipping/step/save + 4 GiB uncertainty
preload(B) = ceil(incremental(B) / GiB) × GiB + 6 GiB continuing reserve
```

At B≤1 GiB, 14,769,985,736 bytes = 13.755621 GiB, rounded to 14 GiB,
plus 6 GiB = **20 GiB**. That number was an unmeasured safety envelope.
It included training-only adapter gradients and clipping/update/save, and a
combined forward/backward workspace allowance. It was not 20 GiB of weights,
optimizer moments, or measured inference residency.

## Measurements, tensor arithmetic and reserves

The reports and [Task 45 audit](smollm2_baseline_findings_task45.md) were read
and hashed without loading tensors. The original historical baseline validator
also passes with ML imports blocked.

| Historical read-only evidence | Task 27 single 17-token forward | Task 40 baseline-002, 76 probes |
|---|---:|---:|
| Before-load availability (GiB) | 16.805878 | 19.924728 |
| Minimum sampled availability (bytes) | 16,470,999,040 | 15,957,819,392 |
| OS lifetime peak RSS (bytes) | 1,166,884,864 | 2,676,031,488 |
| Maximum sampled MPS tensors (bytes) | 541,403,904 | 583,687,680 |
| Maximum sampled MPS driver (bytes) | 1,145,847,808 | 2,304,868,352 |
| Samples / maximum observed gap | 9 / 10.157286 s | 797 / 0.770298 s |

Task 27's large gap was during preparation, before active model work. The
longer baseline is the closer workload: identical frozen 76-case evaluator,
sequential batch-one NLL and up to 64 greedy tokens, no KV cache, FP32/MPS,
maximum sequence cap 128. Actual frozen QA maximum is 57 tokens and prompt
plus generation budget maximum 93. Only scalar metrics/text persist between
cases. The baseline's RSS peak is about 2.49 GiB and sampled driver maximum
about 2.15 GiB. RSS and MPS driver accounting overlap on unified memory and
must not be summed. Samples are nonatomic lower bounds on true peaks, with
no native MPS peak; system availability changes include other processes.
Neither historical run measured native LoRA inference, training or saving.

| Component | Interpretation |
|---|---|
| Base FP32 weights, 538,060,032 bytes | Parameter-count arithmetic; three-copy loading/transfer allowance is 1,614,180,096 bytes, not measured simultaneous residency. |
| Adapter FP32 weights, 921,600 bytes (0.879 MiB) | 230,400 native q/v rank-4 elements; additional live inference intermediates go in workspace. |
| Baseline adapter staging, 2,764,800 bytes total | Conservative three adapter copies for weights, transfer and fingerprint staging; no gradients. |
| Training adapter gradients, 921,600 bytes | Training-only payload; absent from baseline. |
| Optimizer | Baseline creates none. Authorized pilot's stateless SGD has zero persistent momentum/moment tensor bytes, but gradient, clipping and step temporaries still require its training reserves. AdamW is not used. |
| Tensor checkpoint serialization | Baseline writes no tensor checkpoint; pilot stages and saves adapter only, 921,600 tensor bytes plus metadata, within its unchanged 1 GiB clipping/step/save allowance. JSON reports still serialize small telemetry/evaluation data. |
| Logits | One `[1,128,49152]` FP32 tensor is 24 MiB by arithmetic; this is not total forward workspace. |
| Safety reserves | Source residency, workspace, tokenizer/integrity, caches, extra process headroom, uncertainty, upward rounding and the separate 6 GiB continuing reserve are estimates, not measured tensor allocations. |

The [native adapter](../src/lora_adapter.py) performs
`W0 x + 2 × B A x`, with no merge or in-place parameter update in forward.
Zero B makes the freshly initialized adapter zero-effect. The unchanged
historical NLL and greedy-generation parity gates still need actual MPS proof.

## Separate conservative baseline policy

`baseline_memory_policy()` and explicit `mode="baseline"` dispatch now select
`smollm2_135m_fp32_mps_lora_r4_qv_baseline_v2`:

```text
readonly incremental(B) = 3W + 3L + S + 2 GiB forward workspace
                          + 1 GiB tokenizer/integrity + 1 GiB Metal cache
                          + max(1 GiB, B) additional process headroom
                          + 2 GiB sequential generation + 4 GiB uncertainty
preload(B) = ceil(incremental(B) / GiB) × GiB + 6 GiB continuing reserve
```

At B≤1 GiB this is **13,697,165,512 bytes = 12.756479 GiB**, rounded to
13 GiB plus 6 GiB: **19 GiB available**, checked both before checkpoint access
and immediately before loading. At B=3 GiB it rises to **21 GiB**.
The 2 GiB forward and 2 GiB generation allowances reuse the independently
reviewed [Task 39 baseline policy](smollm2_baseline_admission_task39.md).
We retain LoRA's 4 GiB uncertainty, a full GiB above that earlier policy,
and explicitly include three adapter copies. Adapter rank does not justify
removing base loading, hashing or backend cache reserves.

The combined forward/generation budget remains 4 GiB; replacing the old
combined forward/backward term is not a claim of measured 2 GiB savings.
Removing the 1 GiB clipping/step/save allowance while increasing adapter
payload staging yields a net estimate reduction of 1,072,820,224 bytes.
This is an operation-based policy change, not empirical calibration or a
promise of fit. The Task 57 availability remains below 19 GiB; Task 27's
start also fails it. Task 40's historical start exceeds it but does not
establish present readiness.

The complete LoRA **training** policy dictionary remains unchanged, including
guard `smollm2_135m_fp32_mps_lora_r4_qv_one_step_v1`, 20 GiB minimum,
22 GiB at B=3 GiB, all term amounts and the legacy scope text. The default
mode remains pilot; invalid modes fail closed. The full-parameter runner,
24 GiB policy, other guards, optimizer settings, frozen profile, dependencies,
authorization token and historical artifacts are unchanged. Baseline evidence
cannot substitute for fresh training admission or authorize an update.

## Read-only invariants and output protection

Baseline arguments reject update authorization and baseline-input arguments
before paths, imports or writes. The optimizer constructor, `backward()` and
`step()` exist only in `one_step()`, whose independent gate rejects a baseline
monitor even with the correct training token. The run now branches explicitly
on pilot mode before enabling adapter gradients or reaching that function.

Baseline execution freezes every base/adapter parameter, uses `eval()`, and
the shared evaluator executes NLL and generation within `torch.inference_mode()`.
It rejects preexisting gradients/trainable parameters and graph-bearing loss.
The baseline revalidates the frozen layout after evaluation and before cleanup;
base and adapter fingerprints and version counters must remain unchanged.
Unexpected mutation is detected as STOP, never a successful read-only result.
New baseline validation also requires `adapter_versions_unchanged=true` and
`checkpoint_write_attempted=false`. Initializing new A/B factors and moving
them to MPS are setup, not updates to an acquired adapter. No existing adapter
artifact is loaded, overwritten or merged.

All paths remain constrained to a new direct ignored `baseline-*` directory
under `outputs/smollm2_lora/`, with symlinks, aliases, collisions and historical
destinations rejected. Exclusive creation, fsync and sealing preserve reports.
`RUNNING.json` reserves the attempt; STOP retains `report.json` and `FAILED.json`;
`COMPLETE.json` is written last on success and binds the report checksum.
No resume, automatic retry, full-base save or baseline tensor save exists.

## Exact new normal-Terminal attempt (not run in Task 58)

This task authorizes investigation and offline validation only. After separate
owner authorization for the new read-only attempt, use a normal Mac Terminal
with the unchanged pinned environment. The proposed path is unused at Task 58
verification; a collision is STOP and must never be deleted to reuse the name.

```sh
cd ~/UnlearnGuard-Privacy-Leakage-Research
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-ml/bin/python -B -m src.small_model_lora

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-ml/bin/python -B -m src.small_model_lora \
  --run-baseline \
  --run-dir outputs/smollm2_lora/baseline-task58-001
```

The first command is weight-free validation. The second is one new read-only
physical attempt. Retain its Terminal stdout/stderr and exit status; do not
pipe it through a command that hides the runner's exit code. Never pass a
weight-update token or launch an optimizer command as part of this procedure.

Stop on nonzero exit (runner STOP uses code 2), exception, interrupt, missing
or failed integrity/provenance evidence, or any of these guard conditions:

- Wrong Apple M4 Pro/arm64/macOS/MPS/FP32 environment, dependency/config/device
  drift, missing/corrupt cached inputs, download need, allocator/math override,
  CPU fallback, fewer than 48 GiB physical capacity or 4 GiB free disk.
- Either live preload reading below its dynamic floor (minimum 19 GiB),
  availability below 6 GiB afterward, pressure other than NORMAL level 1,
  free indicator below 20%, cumulative swap growth or any page-out change.
- Process lifetime RSS or MPS driver above its separate 8 GiB cap, missing,
  stale, incomplete or unordered telemetry, active sampling gap over 2 s,
  or 30-minute deadline. Sampling target remains 250 ms.
- Nonfinite or incompatible loss/logits, historical NLL drift beyond
  `rel_tol=abs_tol=1e-4`, any other row difference (including greedy output),
  trainability/gradient/layout drift, changed base/adapter bytes or versions,
  or cleanup/report sealing failure.

Do not lower floors, change watermarks/memory fractions, purge caches, reduce
the frozen evaluator, or retry automatically to obtain admission. Retain every
failed directory as a tombstone. Require exit 0,
`lora_baseline_complete_no_updates`, guard v2, two valid preload readings,
all 76 ordered cases, zero backward/optimizer/checkpoint attempts and counts,
unchanged hashes/versions, sealed report and COMPLETE, and no checkpoint
directory. Then record the independently reviewed external checksum:

```sh
shasum -a 256 outputs/smollm2_lora/baseline-task58-001/report.json
```

## Offline validation and remaining blockers

The combined small-model suite passes **167 tests**, zero failures/errors/skips
(62 LoRA tests, including 10 additions). Real Tensor/autograd backward/grad,
SGD/AdamW steps, `torch.load`, safetensors `load_file`, and Transformers model
loaders were patched to raise across the suite. Small CPU fixture forwards,
assigned gradients and a synthetic adapter-only file round trip are offline
diagnostics; orchestration model/backward/step calls are fakes. The existing
shared evaluator test verifies actual inference mode and traps optimizer
construction/backward. No real model weights were loaded or updated.

New tests cover exact mode-specific and dynamic admission boundaries, Task 57
insufficiency, both preload failures with sealed STOP evidence, all continuing
telemetry stops, complete training-policy regression, baseline-to-training
isolation at 19 GiB even with valid training authorization, frozen layouts,
base/adapter hash and version drift, no-update/no-checkpoint orchestration,
synthetic native inference byte/version invariance, and output collision
protection. Default CLI remains free of ML runtime imports.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-ml/bin/python -B -m unittest discover \
  -s tests -p 'test_small_model*.py' -v
```

Ignored evidence: `outputs/task58-offline-tests.log`,
`outputs/task58-offline-plan.json`, `outputs/task58-offline-evidence.json`,
and preservation ledgers. The ledger protects 172 pre-existing files by
checksum, size and mode, excluding only the intended source/test/docs/README
edits from equality checks. Historical outputs/results, cache weights, frozen
data/profile/lock and full-parameter code remain identical.

**Remaining baseline blockers:** separate execution authorization, fresh
passing physical telemetry, and unperformed native LoRA MPS evaluation/parity
and integrity checks. There is no new baseline artifact yet. **Training stays
blocked** on that sealed baseline, independent 20 GiB-or-higher admission and
explicit one-step authorization; native autograd/update/save memory is still
unmeasured. Model-plus-adapter reload, useful non-floor utility controls,
preregistered acquisition gates and demonstrated learning of both roles are
later prerequisites. LoRA-GradDiff additionally requires acquired adapters,
matched controls and a separate workload review and authorization. No fact
acquisition, training or unlearning result is claimed.
