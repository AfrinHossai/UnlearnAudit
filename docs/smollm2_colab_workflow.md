# Staged SmolLM2-135M TOFU workflow for Google Colab — Task 35

Status: **prepared and tested offline; never run on Colab**. No cloud resource,
GPU session, model download, fact acquisition, unlearning, or experimental
result was created in this task. Existing Mac/MPS, Llama and local SmolLM2
profiles, manifests, historical artifacts and admission policies are unchanged.

The notebook is [notebooks/smollm2_135m_tofu_colab.ipynb](../notebooks/smollm2_135m_tofu_colab.ipynb).
Its separate implementation is [src/colab_workflow.py](../src/colab_workflow.py),
with a [CUDA-only profile](../experiments/smollm2_135m_colab.json) and
[Linux dependency pins](../requirements-colab.txt). Importing the module or
running it without arguments only prints a plan. Every action requires a
named stage and matching explicit confirmation. No stage launches another.
The shipped notebook has no outputs, execution counts or GPU metadata that
would imply a session was started.

## Open the notebook and run the first safe check

1. Open [the notebook in Colab](https://colab.research.google.com/github/AfrinHossai/UnlearnGuard-Privacy-Leakage-Research/blob/main/notebooks/smollm2_135m_tofu_colab.ipynb),
   or use **File → Open notebook → GitHub** and paste this repository URL.
   Alternatively upload the `.ipynb` file. Save your own copy if you need edits.
   Opening the notebook does not itself perform training.
2. Inspect your signed-in account's subscription and compute-unit/resource
   information. A GPU label does **not** establish that a runtime is free.
   On a paid account, disconnect if you cannot confirm that using this session
   will avoid paid compute units. Do not buy units, upgrade, enable GCP billing,
   or select Enterprise, Marketplace, dedicated VM or a paid runtime.
   There is no supported Python billing check in this workflow; the notebook
   reports billing status as unknown. If the UI does not clearly establish your
   intended free-tier use, stop before selecting a GPU.
3. Run **0 — Initial environment check** alone, optionally on a CPU session.
   It reports Python, package versions, visible NVIDIA hardware/driver and
   Torch's CUDA build/availability without installs, weights or updates.
   **Stop after that cell.** No later cell is needed for this first check.
4. For a later authorized GPU attempt, select **Runtime → Change runtime type**
   and a GPU accelerator, then connect. Available types and quotas vary; no
   particular GPU is assumed. Rerun the initial check after changing runtimes.
   If CUDA is absent, stop; CPU and MPS are not fallbacks for this profile.

Colab's [official FAQ](https://research.google.com/colaboratory/faq.html)
explains that free resources and session limits vary and are not guaranteed.
Do not use anti-idle or quota-bypass scripts. None of these instructions creates
or purchases a cloud resource automatically.

## Dependencies and CUDA compatibility

Use an isolated `/content/unlearnguard-colab-env` venv, not Colab's preinstalled
ML environment and not this Mac's `requirements-ml.lock`. The notebook installs
Torch **2.9.1 from the cu126 index**, then the separately pinned text-only
dependencies. This avoids mixing Colab's preinstalled torchvision/torchaudio
extensions with a different Torch ABI. No vision/audio, DeepSpeed, Accelerate,
bitsandbytes or external evaluator is needed for this single-device adaptation.
No system CUDA toolkit or driver is installed or changed.

The prepared environment supports **Linux Python 3.11 or 3.12**. The strict
environment stage checks exact core versions, `pip check`, the actual Llama
class import, exactly one visible GPU, `torch.version.cuda == '12.6'`, a driver
major version >=525, and a tiny CUDA matrix operation with synchronization.
The driver minimum is a necessary family-level screen, **not sufficient proof**
of kernel compatibility; the CUDA operation and later read-only model parity
must pass. `nvidia-smi`'s CUDA label describes driver capability, not the Torch
wheel's runtime. Unknown telemetry, import/kernel errors, an unsupported Python
image or different pins are stops, not reasons to silently change versions.

These choices follow the [official Torch 2.9.1 installation matrix](https://pytorch.org/get-started/previous-versions/),
[Transformers 4.57.1 installation documentation](https://huggingface.co/docs/transformers/v4.57.1/installation)
and [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Compatibility was reviewed from those sources and offline code/configuration
checks. **A fresh Linux install and an actual CUDA runtime were not tested.**

Direct dependencies are pinned; transitive packages are resolved on the first
installation, checked and captured in the persistent `runtime.json` using
`pip freeze`. This is not a universal cross-platform lock. For exact recovery,
mount Drive before rerunning the dependency cell; it restores the captured
transitive pins. Python, core pins, complete resolved package list, GPU name,
compute capability and driver are hashed into the stage identity. A different
runtime fingerprint blocks exact resume; restore the original environment or
register a new independent run. GPU reductions need not be bit-identical across
different hardware/software even with deterministic algorithms.

## Fixed provenance and controls

The notebook requires a full 40-character workflow Git commit containing Task
35. Copy it from the repository's commit page; `main` is only an opening link,
not an experiment identity. The checkout is detached, verified and never reset
automatically. The runner rejects tracked edits and records that commit.

It reuses the existing immutable preparation without changing it:

- SmolLM2 `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; remote code disabled;
  original model/tokenizer metadata and weight SHA-256 verified before use.
- Official fictional TOFU data, revision
  `324592d84ae4f482ac7249b9285c2ecdb53e3a68`; all five files verified by
  SHA-256, counts, alignment, overlap checks and full manifest reproduction.
- [The controlled experiment plan](smollm2_tofu_experiment.md),
  [hash-only manifest](../experiments/smollm2_135m_tofu_manifest.json),
  [validated GradDiff objective](graddiff_validation.md),
  development corpus and twelve generic utility probes.

| Diagnostic set | Rows | Training use / interpretation |
|---|---:|---|
| Forget direct | 40 | Acquired in M_all; intentional learned-question recall reuse |
| Retain direct | 120 | Acquired in M_all and M_R; intentional recall reuse |
| Forget paraphrase | 40 | Untrained wording for the same learned facts |
| Retain paraphrase | 12 | Available aligned official matches; small coverage |
| Unseen retain records | 40 | Excluded from acquisition and updates |
| Generic utility | 12 | Never trained; limited drift diagnostics |
| Sealed holdout01 | 40 | Integrity checked; never scored by this pilot |

Forget and retain scores are always separate. Direct training/evaluation reuse
does **not** establish generalization. Paraphrases test the same facts and are
not independent author holdouts. The raw files do not provide author IDs;
question-disjoint subsets may still share author facts. The retain subset is
120 of 3,960 rows, with only twelve aligned paraphrases. Small custom utility
probes, literal aliases and lexical F1 cannot establish broad utility or semantic
forgetting. Do not reinterpret existing Llama scores as SmolLM2 results.

## Stage order and explicit execution

Run one cell at a time. Do not use **Run all**. Every later action asks for an
exact typed stage name; blank input skips it. Definitions do not perform work.
The runner also requires `--stage STAGE --confirm STAGE`. Do not put these
confirmations in loops or schedule automatic continuation.

| Stage | Action and boundary |
|---|---|
| Initial check | Read-only installed-environment inventory; stop here initially |
| Checkout / dependencies / storage | Separate explicit setup actions |
| `environment` | Strict isolated CUDA/import compatibility, no weights |
| `data` | Download/verify pinned JSON only; no model |
| `freeze` | Validate and persist reviewed protocol and calibration evidence |
| `load` | Explicit pinned weights download, read-only loss parity, save M0 |
| `baseline` | Fixed M0 pre-acquisition diagnostics |
| `acquire` | M0 → M_all, fixed 20 epochs / 3,200 SGD updates |
| `reference` | Independent M0 → M_R, fixed 20 epochs / 2,400 updates |
| `pre` | M_all pre-unlearning diagnostics; no later stages |
| `reference-eval` | M_R diagnostics |
| `noop` | Unchanged M_all reevaluation; require identical summary |
| `retain-control` | Independent M_all branch, forty fixed retain draws |
| `retain-control-eval` | Separate matched-continuation diagnostics |
| `unlearn` | Independent M_all branch, forty paired GradDiff updates |
| `post` | Fixed final GradDiff evaluation and comparison with all controls |

The retain reference matches per-retain exposure rather than total updates.
Its scores are reported as a custom counterfactual, not an official TOFU oracle.
No-op, retain continuation and GradDiff all use the same fixed M_all parent.
The two continuation branches use identical manifest-pinned retain draws.
Acquisition orders use the existing SHA-256 epoch schedules. Target checkpoint
selection uses the fixed final budget, never the best forget score.

All updates use seed 2026, FP32, SDPA, nonreentrant gradient checkpointing,
batch size one, no gradient accumulation, no momentum/weight decay/scheduler,
maximum sequence length 512, and gradient norm clipping at 1.0. No truncation is
allowed. Plain QA prompt tokens and padding are masked; answer/EOS tokens alone
are supervised. GradDiff reuses the validated shifted, FP32-upcast mean causal
NLL and `-gamma*NLL(F)+alpha*NLL(R)`, gamma=alpha=1. Sequential forget/retain
backward calls accumulate the same objective gradients before one clipped SGD
step while avoiding two simultaneous activation graphs. This is a custom port,
not a faithful distributed OpenUnlearning reproduction.

Generation is greedy, one beam, maximum 128 new tokens, no cache, question-only
context. Evaluation records answer-only NLL weighted by supervised token count,
normalized exact match/F1, reviewed literal key-fact recovery, empty responses,
frozen refusal-phrase and repetition diagnostics. Alias matching uses normalized
word boundaries; aliases are alternatives, and any match counts as recovery.
The repetition rule flags a normalized trigram seen at least three times.
Phrase/alias diagnostics require blinded semantic review and can misclassify.

## Dependencies that intentionally block updates

The original preparation has **null acquisition/unlearning rates** and no
reviewed target key-fact annotations. This task does not manufacture either.
The external [protocol template](../experiments/smollm2_135m_colab_protocol.template.json)
therefore stops at `freeze`. Even `load`/M0 outputs require a frozen protocol
to avoid annotating aliases after seeing responses. Do not edit the original
local profile to make this cloud path execute.

Before a later authorized experiment, independently prepare:

1. Development-only calibration with the existing 8-forget / 24-retain synthetic
   corpus, fresh base for every candidate, 640-step development acquisition,
   480-step reference, eight-step continuation branches, fixed gates and lowest
   passing rate selection. Include all candidates, negative results, own M_R,
   matched retain-only controls, provenance and reviewer notes. This calibration
   runner/evidence is **not provided or executed in Task 35**. The shorter
   synthetic attributes may not transfer to long TOFU answers; target failure
   means stop, not tune against target evaluation.
2. A reviewed calibration JSON with `status` set to
   `completed_reviewed_development_only`, `target_scores_used_for_selection:
   false`, `model_revision`, `development_sha256` from the frozen manifest,
   selected `acquisition_learning_rate`, `unlearning_learning_rate`, a nonempty
   `candidate_reports` collection and `reviewed_by`. Those reports must contain
   the actual fixed-budget candidate/control measurements and gate calculations.
   The runner validates provenance, selected candidate membership, reviewer
   presence and file binding; independent review must verify the reported
   candidate results and selection rule. A status string is not evidence.
3. An external copy of the protocol template, status `frozen_reviewed`, reviewer
   and notes, `manifest_sha256` using `src.small_model_experiment.digest` (sorted
   compact JSON SHA-256, not raw-file SHA), selected rates and the calibration
   file's **raw-byte SHA-256**. Supply `aliases_by_case_id` for every case in the
   five nonsealed sets plus `utility:00` through `utility:11`. Each list must
   contain at least one meaningful, independently reviewed normalized literal
   substring grounded in that case's reference answer. Whole-answer exact match
   is not a substitute for key-fact annotation. Freeze refusal/repetition rules
   at the same time. Do not include `holdout01_sealed` cases.

`freeze` copies evidence bytes, stores the protocol and binds their hashes.
They cannot change within a run. A different protocol needs a new run. A resumed
freeze can repair a matching interrupted write, but rejects conflicting files.

Before branch updates, the existing proposed acquisition gates must pass:
direct forget/retain recovery >=0.8, forget gain over M0 >=0.5, forget paraphrase
recovery >=0.5 and decreasing forget/retain NLL. Failure stops target unlearning.
The post report applies the proposed forgetting/utility thresholds from the
original profile and requires greater recovery reduction than retain
continuation, reports after-minus-before and distance to M_R, and always labels
semantic review as required. Diagnostic failure is a result to report, never
permission to retune on the target sets. These are project proposals, not
official TOFU standards. Truth Ratio, KS Forget Quality, canonical Model Utility,
membership inference and proof of erasure are **not computed**.

## Memory, logging, checkpoints and disconnections

Artifacts live below `/content/drive/MyDrive/UnlearnGuard-runs/RUN_ID`, outside
every Git checkout. The runner requires mounted Drive and rejects repository
paths, symlink escapes into a checkout and config/commit substitutions. The
original Hub snapshot and completed M0/branch checkpoints are never overwritten.
Do not commit generated notebook outputs, generations, logs or weight files.
Existing Git ignore rules already cover weight suffixes and local outputs.

Before each model load: at least **4 GiB available host RAM**, **6 GiB free GPU
memory** and **4 GiB filesystem free disk**. During steps and evaluations:
**2 GiB host**, **1 GiB GPU**, and **4 GiB disk**. These are separate unmeasured
cloud planning reserves, not modifications to Mac admission guards and not
measured training fit. Current/peak CUDA allocations, allocator reservation,
host availability and disk free space are logged with losses and elapsed times.
Drive quota must also be checked in your account; its FUSE free-space figure
can differ. FP32 model state alone is about 0.50 GiB per checkpoint; two retained
generations per trained branch, M0, cache and temporary writes need multiple
GiB. Do not assume the available disk or account quota is sufficient.

Each successful group of forty optimizer steps creates a new checkpoint
directory with model safetensors, tokenizer/config, optimizer state, Python RNG,
Torch CPU/CUDA RNG, step cursor, complete schedule/budget/parent identities and
file SHA-256 hashes. It is written to a unique partial directory, then renamed
and read-back verified. Only then are older **validated generated checkpoints**
in that stage pruned, keeping current and previous. Drive FUSE/network failures
may still interrupt writes; marker/hash checks, rather than rename assumptions,
determine whether a checkpoint is usable. Optimizer/RNG resume uses local
tensor/builtin serialization with `weights_only=True`; do not replace it with
untrusted pickle files.

Every invocation has a 30-minute wall-time budget checked between steps/cases.
Creating `RUN_DIR/STOP` stops at the next boundary. Nonfinite losses, gradients
or parameters; missing/insufficient memory; OOM; dependency errors; disk/quota
failures; mismatched identities; and failed acquisition/no-op gates all stop.
No post evaluation or later experiment stage runs automatically. Stop flags and
time limits may discard the unsaved tail; they do not create completion markers.

On interruption:

1. Ensure the old runtime has stopped. Use only one writer for a Drive run;
   Drive is not a distributed lock service. Reopen the notebook, use the same
   commit/run ID, rerun definitions/checkout and mount Drive. Rebuild dependencies
   with Drive mounted to restore `runtime.json`'s resolved pins, then verify the
   environment. No model stage resumes automatically.
2. Inspect `STAGE.complete.json`: completion requires the correct stage/run
   identity, hash-verified final artifact and, for training, cursor equal to the
   entire budget. A log line, partially saved folder or successful earlier
   stage never means the next stage is complete. Abrupt termination can leave
   an attempt marked `running_not_completed` indefinitely; it is incomplete
   until a separate verified completion marker exists.
3. For an interrupted training stage, run **that stage only**, explicitly type
   `resume`, and recover its newest intact generation. A corrupt newest generation
   falls back to the previous intact generation only when there is no completed
   stage marker; if neither verifies, stop. Unsaved tail steps replay from the
   saved cursor/RNG/optimizer, not from a guessed log step. Attempt log entries
   are marked uncommitted and may duplicate replayed steps; checkpoints are the
   authoritative progress record.
4. Evaluation restarts from the same immutable checkpoint and overwrites only
   its incomplete evaluation artifact; it does not continue from a partial
   metric average. An existing intact completion marker is verified and returns
   without repeating work. `load` can recover its verified M0 checkpoint if the
   final marker was interrupted. Partial checkpoint directories are ignored;
   inspect before manually removing abandoned `.partial-*` folders to reclaim
   quota. Never delete committed parents or edit markers to force progress.
5. Missing parents, changed software/hardware/driver, changed protocol/data or
   corrupted completed artifacts stop recovery. Do not claim exact continuation
   across a changed runtime. Restore the original environment if possible, or
   register a new independent run and report the interruption.

The post completion marker binds both the generated evaluation and comparison
artifact. Passing execution means the fixed stage finished, not that forgetting
or utility gates passed. Publication still requires semantic review, limitations,
all controls and negative findings.

## Offline checks for this preparation

From the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-ml/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.colab_workflow
```

Tests compile notebook cells without executing them, check output-free notebook
structure, exact stage coverage, immutable model/data identity and schedule
budgets, default plan-only behavior, explicit confirmations, missing-protocol
stops, alias/gate diagnostics, output-path rejection and checkpoint/corruption
recovery with inert fixtures. Existing validated GradDiff/configuration tests
are run offline. They do not demonstrate Colab/CUDA compatibility or actual
learning. Regenerate the notebook with
`python scripts/build_colab_notebook.py`; it only writes notebook JSON.

Task 35 verification: **180 offline tests, 179 passed and one MPS-only skip**,
including 27 new cloud-workflow tests. The pinned cached TOFU files reproduced
the full existing manifest; Python/JSON parsing and 38 relative documentation
links passed. Installed Transformers 4.57.1 dependency metadata also confirmed
that the proposed Hub, NumPy, packaging, safetensors and tokenizers pins satisfy
its direct constraints. CUDA compatibility logic was exercised with inert
module/telemetry fixtures only. No notebook cell or real model stage was run.

Remaining runtime dependencies are an owner-confirmed free session, compatible
GPU/Python/driver/packages, mounted persistent storage and sufficient quota,
separately authorized reviewed development calibration and target annotations,
explicit later data/weight downloads, and real read-only memory/loss checks.
The first safe Colab action remains **run cell 0 only and inspect its report**.
