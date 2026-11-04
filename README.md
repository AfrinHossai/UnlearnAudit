# UnlearnGuard: Privacy Leakage After Machine Unlearning

**Project:** `UnlearnGuard-Privacy-Leakage-Research`  
**Status:** The five-question baseline and paired 40-forget/40-retain pre-unlearning diagnostic are saved and audited. The synthetic GradDiff objective has passed equation-level validation against a pinned OpenUnlearning source transcription. A guarded local real-model runner is implemented; no real-model optimizer step or official TOFU evaluation has been run.

> **Research status:** No TOFU checkpoint has been fine-tuned or unlearned in this repository. The released TOFU-finetuned checkpoint has local five-question and paired 40/40 pre-unlearning diagnostics. Their JSON outputs remain ignored by Git; the paired artifact's audit metadata is recorded in the GradDiff plan. These are lightweight lexical diagnostics, not official TOFU Forget Quality or Model Utility evaluations. The new real-model runner has not loaded the Llama weights or performed an optimizer update.

## Motivation

Large language models can reproduce information associated with their training data. Machine unlearning aims to reduce a model's reliance on selected data after training, while preserving useful behavior on data that should remain. A low score on a standard question-answering prompt alone does not establish that the target information is no longer recoverable: answers may vary across prompts, and unlearning can also damage unrelated capabilities.

This project is planned as a reproducible study of that privacy-utility trade-off. It will use TOFU (Task of Fictitious Unlearning), a benchmark built around synthetic fictional-author profiles and designated forget and retain data. Its controlled setup makes it possible to study targeted forgetting without using real people's private records. The benchmark also provides evaluations for both forget quality and retained model utility. [Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*](https://arxiv.org/abs/2401.06121).

## Objective

Evaluate whether **GradDiff**, which combines gradient ascent on the forget set with a retain-set negative-log-likelihood term, changes recovery of TOFU target answers while preserving retained utility. The planned defaults follow the [official OpenUnlearning GradDiff implementation and configuration](https://github.com/locuslab/open-unlearning/blob/main/configs/trainer/GradDiff.yaml); GradDiff is not presented as a privacy guarantee.

The planned study will address:

1. How does GradDiff change target-answer recovery on TOFU forget-set questions?
2. Do the results hold for a fixed set of paraphrased or reformulated prompts?
3. What changes in forgetting coincide with changes in utility on TOFU retain and related evaluation sets?

## Planned Methodology

Every item in this section is **planned work**. No TOFU model has been trained or unlearned in this repository. The saved five-question and paired forget/retain results are lexical diagnostics; official benchmark evaluation remains pending.

1. **Pin the benchmark and model.** Use the TOFU data and its designated forget/retain protocol. Before running experiments, select and record a compatible TOFU-finetuned checkpoint and immutable dataset/model revisions. Publish configuration files, data provenance, and checksums where licensing permits.
2. **Run one unlearning technique.** Apply GradDiff to the forget examples while retaining the official retain-set NLL term. Use a documented optimizer, learning rate, batch size, epoch count, and random seed. Keep the method fixed across comparisons; record each configuration and checkpoint identifier.
3. **Compare reference models.** Evaluate the pre-unlearning checkpoint, the GradDiff checkpoint, and the released retain-only reference trained without the forget set. The retain-only model provides the reference logs needed for the TOFU Forget Quality comparison.
4. **Evaluate target leakage and utility.** Use TOFU's published evaluation protocol, including truth-ratio distributions on the forget set and utility measurements on retain and related sets. Add a versioned, fixed set of direct and paraphrased prompts. A planned targeted-answer recovery rate will count generated answers matching normalized target answers or a published alias list; the matching rules and any manual review procedure will be specified before evaluation.
5. **Make runs reproducible.** Fix decoding settings and prompt templates. Record software versions, hardware, seeds, model and dataset revisions, training configuration, and evaluation outputs. Keep raw generated answers separate from aggregate metrics, and document any excluded or failed runs.
6. **Report limitations.** Interpret benchmark metrics as evidence about behavior on TOFU, not proof that information has been erased from model weights or is unrecoverable in other settings. TOFU uses synthetic profiles, so results will not establish privacy protection for real personal data.

The planned evaluation will report forgetting and utility together. A refusal or a changed answer on one prompt will not, by itself, be treated as evidence that the target information is no longer recoverable.

## TOFU Dataset Inspection

The reproducible inspection pipeline downloads only the small `forget01` JSONL dataset file from the official [locuslab/TOFU Hugging Face benchmark](https://huggingface.co/datasets/locuslab/TOFU), pinned to revision [`324592d84ae4f482ac7249b9285c2ecdb53e3a68`](https://huggingface.co/datasets/locuslab/TOFU/tree/324592d84ae4f482ac7249b9285c2ecdb53e3a68). It checks the file checksum, discovers the schema from the records, validates question and answer fields, prints three deterministic examples, and saves a metadata-only summary at `results/tofu_forget01_inspection.json`.

Run the following from the repository root with Python 3.10 or newer (the checked run used Python 3.13.7):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.inspect_tofu
```

The `requirements.txt` file documents that the pipeline and tests use only the Python standard library. On the first inspection, the script downloads the pinned `forget01.json` file into `.cache/tofu/` and verifies its SHA-256 digest before loading it. The cache, virtual environment, and dataset files are excluded from Git. The summary records schema and validation metadata without copying question or answer text. See [docs/data.md](docs/data.md) for dataset provenance, attribution, and the limits of this inspection.

See [docs/compute_feasibility.md](docs/compute_feasibility.md) for the verified local ML environment and hardware assessment. See [docs/graddiff_experiment_plan.md](docs/graddiff_experiment_plan.md) for the baseline audit, GradDiff source review, experiment profiles, evaluation leakage controls, and execution recommendation.

See [docs/tofu_baseline.md](docs/tofu_baseline.md) for the initial five-question configuration and [docs/paired_tofu_evaluation.md](docs/paired_tofu_evaluation.md) for the frozen forget/retain manifest, official-metric plan, and exact command for the expanded evaluation.

This pipeline only reads the benchmark data. It does not load a model or perform any training, fine-tuning, evaluation, or machine unlearning. The inspected records are not evidence that any model has forgotten anything.

## TOFU Pre-Unlearning Evaluations

The initial five-question result from [`src/evaluate_tofu_baseline.py`](src/evaluate_tofu_baseline.py) is recorded locally at the ignored path `results/tofu_forget01_baseline.json` and reviewed in [docs/baseline_findings.md](docs/baseline_findings.md). The local paired result at `results/tofu_forget01_retain99_baseline.json` contains all 40 `forget01` cases and a deterministic 40-question sample from `retain99`, with a frozen manifest in [`experiments/tofu_forget01_retain99_manifest.json`](experiments/tofu_forget01_retain99_manifest.json). The JSON result remains ignored and is not part of this commit. It does not include the optional paraphrase or `holdout01` generations. Both artifacts contain lightweight lexical diagnostics, not official TOFU Forget Quality or Model Utility scores. See [docs/paired_tofu_evaluation.md](docs/paired_tofu_evaluation.md) and [docs/graddiff_experiment_plan.md](docs/graddiff_experiment_plan.md) for audit details and plans.

## Local GradDiff Prototype

**Task 30 backward-only preparation:** a separate [guarded SmolLM2 rehearsal](docs/smollm2_backward_rehearsal_task30.md) is ready for a future normal-Terminal attempt with explicit opt-in and at least 20 GiB live preload availability. It uses one pinned TOFU forget/retain pair, two forwards and one backward, finite-gradient and unchanged-parameter checks, and has no optimizer, checkpoint-write or download path. Default execution validates only. Task 29's verified MPS read-only result does not authorize training; full-model backward remains unexecuted. Original Llama training guards are unchanged.

**Task 25 read-only measurement:** a separate [SmolLM2 MPS preflight](docs/smollm2_readonly_preflight.md) is ready for an owner-run normal Terminal command with explicit pinned-checkpoint download permission. Preparation used mocked tests and cached metadata only. It records RSS, MPS allocations, sampled maxima, pressure and availability for one frozen inference-mode forward. All current thresholds remain binding; no physical result or training approval is claimed.

**Task 24 memory assessment:** the [SmolLM2 memory investigation](docs/smollm2_memory_admission.md) confirms that the 32 GiB admission policy was inherited from Llama. Actual SmolLM2 device fit remains unresolved. A separate offline assessor validates future measurement evidence but always returns STOP; no lower threshold or training is approved. All active Llama and small-model safeguards remain binding.

**Task 22 preparation:** the [controlled SmolLM2-135M protocol](docs/smollm2_tofu_experiment.md) now has a pinned preparation-only profile, deterministic TOFU manifest, disjoint development facts, generic utility probes and an offline validation CLI. Official config/tokenizer metadata and all 368 diagnostic encodings were verified without model weights; 104 offline tests ran (103 passed, one MPS-only skip). The [exact next normal-Terminal command](docs/smollm2_tofu_experiment.md#reproduction-completed-checks-and-exact-next-step) checks tokenizer provenance, MPS visibility and the existing 32 GiB SGD guard without loading a model. No small-model training entry point or memory exception exists; physical rehearsals, calibration, annotations and evaluation controls remain blockers before updates. Llama reference artifacts are unchanged.

**Task 21 research recommendation:** prepare a separate SmolLM2-135M fact-acquisition and GradDiff pilot on a frozen TOFU subset. The [smaller-model strategy](docs/small_model_unlearning_strategy.md) compares four official sub-billion backbones, distinguishes the advertised Qwen TOFU adapter from a verified MPS-ready checkpoint, and specifies acquisition/forgetting/utility controls. No model was downloaded or trained. The existing Llama experiments and 32/42 GiB guards remain unchanged; this recommendation does not authorize a small-model override or execution.

**Task 19:** the runner now supports explicit full-parameter FP32 SGD (`--optimizer sgd`) with zero momentum, required non-reentrant checkpointing and a separate `--backward-only` mode. Defaults still validate without loading weights. The [implementation and exact prerequisites](docs/graddiff_low_memory_runner.md) record 88 tests (87 passed, one MPS-only skip) and the tiny CPU backward rehearsal; no pretrained model was loaded or updated. The SGD runtime guard requires 32 GiB available before load and 6 GiB thereafter, plus valid device and new ignored output paths. Full-model memory/utility remain unverified. Read the [options assessment](docs/low_memory_unlearning_options.md) and [pilot specification](experiments/graddiff_low_memory_sgd_pilot.json); the JSON itself never enables training.

[`src/graddiff.py`](src/graddiff.py) implements the GradDiff scalar objective with mean causal next-token NLL computed separately for forget and retain batches. [`src/graddiff_train.py`](src/graddiff_train.py) provides an offline synthetic-only CLI. It selects MPS when available and CPU otherwise, records the selected device, rejects multi-process launch settings, and does not load a real model, checkpoint, or dataset.

Run the synthetic dry run with PyTorch installed:

```sh
python -m src.graddiff_train --dry-run --device auto --steps 3
```

The repository tests cover loss normalization, gradient signs, error handling, and fixture-based model integration. A deterministic transcription of the pinned Transformers causal-LM loss and OpenUnlearning GradDiff objective matches the local scalar loss and input-logit gradients. The cached tokenizer and pinned forget/retain examples have been checked without loading model weights. The new [guarded real-model runner](src/graddiff_real_train.py) defaults to local validation with no model-weight loading; an update requires `--apply-updates`, an explicit device, and confirmation of the immutable model revision. Its one-step Apple Silicon profile uses FP32, SDPA, explicit Torch SGD or default AdamW, batch size one, gradient checkpointing, finite-loss/gradient checks, and clipping. This is a local port, not a faithful OpenUnlearning reproduction. The separate [real-model read-only smoke](docs/graddiff_model_integration_smoke.md) has not been run. The runner is **not yet cleared for a real-model optimizer step** because full-model loss parity and peak memory on MPS remain unmeasured. No TOFU GradDiff update or real unlearning has been performed.

### Safe validation and future one-step command

Task 17 adds an optional FP32 read-only preflight with manifest cases, physical-memory guards, current RSS, MPS allocations and loss checks. The exact command for a normal macOS Terminal is:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_preflight \
  --run-preflight --device mps \
  --report-path outputs/graddiff-preflight-fp32-mps-2026-11-03.json
```

Omitting `--run-preflight` prints a plan without loading weights. This command has no backward, optimizer, or checkpoint-save path. It stops below 20 GiB free plus speculative physical memory before load, below 6 GiB afterwards, below 20% system memory free percentage, or if required telemetry is unknown. Stop manually if Activity Monitor shows yellow/red pressure or rapidly growing swap. The telemetry-only check observed about 17.4 GiB available by this conservative measure, so memory must be reclaimed before model loading. Full-model inference was not executed. The revised training estimate is 42.42 GiB against 48 GiB total; this does not prove fit. See [verified findings, estimates, and stop conditions](docs/graddiff_preflight.md). A successful read-only preflight does not enable training.

Use the existing project-local environment. The default validation reads only the pinned local config, tokenizer, and checksum-verified dataset files; it does not deserialize model weights, create an optimizer, or update parameters:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_real_train --dry-run
```

After successful read-only and backward-only physical rehearsals, evaluation-plan freeze, memory review and separate update authorization, the following command would enable one SGD update. It has not been executed:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_real_train \
  --optimizer sgd --apply-updates --device mps \
  --confirm-model-revision 88e31200b97e4c0c04ae0d2f0b591f427046d192 \
  --max-steps 1 --seed 2026 --learning-rate 1e-4 \
  --gamma 1.0 --alpha 1.0 --max-grad-norm 1.0 \
  --max-sequence-length 512 \
  --report-path outputs/graddiff-sgd-step1-mps-first.json \
  --save-checkpoint outputs/graddiff-sgd-step1-mps-first
```

The runner is local-only and fails if pinned files are missing; it does not download data or model weights. CPU use must be selected by changing `--device mps` to `--device cpu`; there is no automatic device fallback. Model outputs and checkpoints can be written only under the Git-ignored `outputs/` directory, and existing paths or the original TOFU checkpoint are rejected. Reports store dataset checksums, split row indices, loss components, hyperparameters, memory snapshots, and the actual device, but no question/answer text, environment secrets, or model weights. The FP32 AdamW state floor is estimated at 18.42 GiB, with an earlier 27.1–31.1 GiB planning envelope. Task 17 uses the stricter 42.42 GiB estimate above for readiness; none is a measured training peak. The local runner omits upstream distributed sampling, gradient accumulation, scheduler, paged AdamW, BF16, and FlashAttention, so its output must not be reported as an official benchmark reproduction. See [docs/graddiff_experiment_plan.md](docs/graddiff_experiment_plan.md) for readiness limits and deviations.

## Local ML Environment Smoke Test

The project-local `.venv-ml` environment uses Python 3.11.18, PyTorch 2.9.1, and Transformers 4.57.1. It is separate from the existing TOFU-inspection `.venv` and is ignored by Git. The repeatable diagnostic runs a synthetic tensor operation and inference with a randomly initialized one-layer GPT-2 model; it downloads no checkpoint or dataset. Full pinned dependencies are in [requirements-ml.lock](requirements-ml.lock), with direct package pins in [requirements-ml.txt](requirements-ml.txt).

```sh
python3.11 -m venv .venv-ml
./.venv-ml/bin/python -m pip install --no-cache-dir -r requirements-ml.lock
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  ./.venv-ml/bin/python scripts/diagnose_ml_environment.py
```

On this Mac, the diagnostic used MPS successfully when run in a normal process outside the restricted command sandbox. Inside that sandbox, Metal is hidden, so the script selects CPU. No system-level changes are needed. See [docs/compute_feasibility.md](docs/compute_feasibility.md) for the verified results and remaining limitations.

## Future Interactive Demo

An interactive application is planned after the experiment pipeline is implemented. It is intended to let a user inspect the fixed evaluation prompts, compare pre-unlearning and GradDiff-unlearned responses, and explore target-answer recovery and utility metrics across the forget and retain sets. The interface should expose model, dataset, and run configuration so that displayed comparisons can be traced to saved experiment artifacts.

**The demo has not been built.** No model outputs or experimental performance results are included. The tracked TOFU inspection summary contains dataset metadata only.

## Repository Layout

```text
.
├── app/          # Planned interactive demo
├── docs/         # Dataset provenance, compute assessment, and baseline instructions
├── experiments/  # Frozen TOFU manifest and GradDiff experiment profiles
├── results/      # Tracked, metadata-only TOFU inspection summary
├── src/          # Data inspection, baseline evaluation, and guarded GradDiff runners
└── tests/        # Offline data, objective, integration, and runner-safety tests
```

## References

- Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., & Kolter, J. Z. (2024). [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121).
- [OpenUnlearning GradDiff implementation](https://github.com/locuslab/open-unlearning/blob/main/src/trainer/unlearn/grad_diff.py) and [TOFU reproduction settings](https://github.com/locuslab/open-unlearning/blob/main/docs/repro.md).
