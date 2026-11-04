# GradDiff experiment plan: TOFU forget01 / retain99

**Status:** Experiment plan prepared; a synthetic GradDiff prototype and its offline CPU/MPS checks are implemented. No TOFU GradDiff training, fine-tuning, official model evaluation, cloud provisioning, or cloud charges have occurred.

## Decision and experiment target

The primary experiment is OpenUnlearning GradDiff on the TOFU 01/99 protocol using the TOFU-fine-tuned Llama 3.2 1B Instruct `full` checkpoint as the model to unlearn. Use the complete 40-record `forget01` training split and the 3,960-record `retain99` training split. Use the Linux CUDA profile in [the experiment specification](../experiments/graddiff_tofu_forget01_retain99.json) for the main result. The Apple MPS file is a prototype-only adaptation design.

The method objective is

$$
\mathcal{L}_{\mathrm{GradDiff}} = -\gamma\,\mathrm{NLL}(F) +
\alpha\,\mathrm{NLL}(R), \quad \gamma=1,\;\alpha=1.
$$

OpenUnlearning computes the negative model loss on the forget batch, adds the retain loss, and supports NLL or KL retain loss. Its GradDiff config selects `gamma=1.0`, `alpha=1.0`, and `NLL`; this does not create a separate reference model for the training loss. See the [GradDiff implementation](https://github.com/locuslab/open-unlearning/blob/main/src/trainer/unlearn/grad_diff.py) and [GradDiff trainer config](https://github.com/locuslab/open-unlearning/blob/main/configs/trainer/GradDiff.yaml).

## Audit of the saved paired baseline

I read and verified the existing local artifact at `results/tofu_forget01_retain99_baseline.json` without changing it. Its SHA-256 is `1473f6abc0c42164ee55dcf27328ab0b72c9e86e12c15764b71800e7462e8a0e`. The canonical hash of the checked-in [manifest](../experiments/tofu_forget01_retain99_manifest.json) matches the result's declared hash: `6fa4af8b1b71809593443e38c2a388edcd2faef4e5db1f2bca154bca19ccee49`. The result JSON is intentionally ignored by Git; this repository records its audit metadata but does not copy or publish the historical responses.

| Audit item | Verified value |
|---|---|
| Starting checkpoint and tokenizer | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, revision `88e31200b97e4c0c04ae0d2f0b591f427046d192` |
| Local snapshot | Matching snapshot directory exists with model, tokenizer, config, and generation config. `model.safetensors` is 2,471,645,608 bytes. |
| Dataset | `locuslab/TOFU`, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`; all four local QA files match pinned SHA-256 digests and row counts. |
| Forget cases | All 40 `forget01` cases; manifest record indices and question/answer hashes match source records. |
| Retain cases | 40 unique `retain99` cases sampled from 3,960 rows by SHA-256 ranking with seed 2026; indices and question/answer hashes match. |
| Saved outputs | 40 forget and 40 retain generated responses with per-case and aggregate lexical diagnostics. No paraphrase or holdout responses are in this artifact. |
| Runtime metadata | Seed 2026; greedy decoding; max 128 generated tokens; MPS FP16; Python 3.11.18, PyTorch 2.9.1, Transformers 4.57.1, macOS arm64. |
| Forget diagnostics | Strict and normalized exact match 0.55; mean normalized token-overlap F1 0.863678. |
| Retain diagnostics | Strict exact match 0.45; normalized exact match 0.475; mean normalized token-overlap F1 0.830211. |
| Official TOFU metrics | Truth Ratio, Forget Quality, and Model Utility are explicitly `not_computed`. |

These are generated-answer lexical diagnostics, not official TOFU scores. The 40 forget prompts and the sampled retain prompts are training examples for this protocol, so the scores are in-sample behavior checks. This audit establishes neither forgetting nor retained utility.

## Data roles and evaluation leakage controls

The TOFU `forget01` and `retain99` files are training inputs for the planned run. OpenUnlearning's [`ForgetRetainDataset`](https://github.com/locuslab/open-unlearning/blob/main/src/data/unlearn.py) uses `anchor=forget`: each forget row is visited once per epoch; a retain row is sampled uniformly with replacement for each forget row. Thus the 3,960 retain rows are available for sampling but are not all traversed in an epoch. Ten epochs mean ten passes over the 40 forget rows and 40 random retain draws per epoch, subject to distributed sampler and batching behavior.

The pre-run and post-run diagnostics reuse the same manifest. This is useful for paired before/after comparisons, but it is not a held-out test: the released full model was fine-tuned on TOFU, and GradDiff directly trains on the forget01 and retain99 records being queried. A lower match rate on the same prompts can reflect wording, response variation, or over-unlearning. It does not establish that the underlying information became unrecoverable.

Use these roles and restrictions:

1. **Direct training-example diagnostics:** Evaluate all 40 `forget01` questions and the exact 40-case retained sample. Label them in-sample diagnostics. Preserve raw answers and metric definitions; do not call these a held-out test or official benchmark result.
2. **Official paraphrase evaluation:** Use all 40 paired rows in `forget01_perturbed`, asking `paraphrased_question` and scoring against `paraphrased_answer` and supplied perturbed answers. This holds out wording but repeats the target facts. It tests prompt robustness, not independent knowledge generalization.
3. **Independent paraphrase set:** Before training, have two human annotators independently write one meaning-preserving paraphrase for each forget question without seeing model outputs. Resolve discrepancies, freeze prompt and target/alias files with checksums, and do not use the set for training or checkpoint selection. These prompts test wording robustness for the same facts.
4. **Benchmark holdout:** Keep `holdout01` out of training and do not inspect post-run results until epoch 10 is fixed. The pinned split has 40 records and no normalized question overlap with forget01 or retain99. Its JSONL contains only `question` and `answer`, not author IDs, so local files do not independently prove profile disjointness by ID; document it as the official holdout split and retain that limitation.
5. **Retained utility:** Run official Model Utility, including retained TOFU and general-purpose evaluation sets, and report components with the aggregate. The retain-only model is a reference; it is not the same thing as the 40-question lexical sample.

Freeze hyperparameters before viewing any post-unlearning score. Epoch 10 is the primary checkpoint by protocol, not one selected because it looks best on forget or holdout answers. The `0.05` absolute Model Utility non-inferiority margin in the config is a project guardrail, not an official TOFU threshold. If epoch 10 fails it, report that the run failed the retained-utility control and keep the starting checkpoint as the accepted model.

## OpenUnlearning configuration, optimizer, and evaluation

The [TOFU unlearning config](https://github.com/locuslab/open-unlearning/blob/main/configs/experiment/unlearn/tofu/default.yaml) defaults to `forget10`, `retain90`, and `holdout10`; override each for this experiment. The [Llama 3.2 1B model config](https://github.com/locuslab/open-unlearning/blob/main/configs/model/Llama-3.2-1B-Instruct.yaml) selects BF16 and `flash_attention_2`. GradDiff inherits the [fine-tuning trainer config](https://github.com/locuslab/open-unlearning/blob/main/configs/trainer/finetune.yaml): learning rate `1e-5`, 10 epochs, 8 examples per device, accumulation 4, `paged_adamw_32bit`, and gradient checkpointing off. The concrete [TOFU script](https://github.com/locuslab/open-unlearning/blob/main/scripts/tofu_unlearn.sh) uses 4 examples per device, accumulation 4, gradient checkpointing on, and two GPUs (effective global batch 32). The [reproduction notes](https://github.com/locuslab/open-unlearning/blob/main/docs/repro.md) report two 48 GB L40S GPUs with DeepSpeed ZeRO-3 and caution that changing distributed setup can change results.

The canonical profile uses that concrete two-GPU script setup, 10 epochs, `learning_rate=1e-5`, `warmup_epochs=1`, `weight_decay=0.01`, and GradDiff `gamma=alpha=1`. Project seed 2026 matches the paired manifest and explicitly overrides the upstream Trainer seed of 0. Use the released checkpoint's tokenizer and immutable revision in both training and evaluation.

Use a separate Python 3.11 environment. The reviewed upstream requirements pin PyTorch 2.9.1, Transformers 5.5.4, Datasets 3.0.1, Accelerate 1.13.0, bitsandbytes 0.49.2, DeepSpeed 0.15.4, Hydra, and other packages. The quickstart separately installs FlashAttention 2.8.3; its example wheel is for CUDA 12.8, Linux x86_64, and CPython 3.11. The existing `.venv-ml` has Transformers 4.57.1 and is only an inference-smoke environment.

The source review used live OpenUnlearning `main` links. An immutable upstream commit SHA was **not captured** in this preparation. Before any run, clone the source, record the exact commit, confirm the reviewed files, and archive the resolved Hydra config, dependency lock or container digest, dataset/model revisions, and evaluation-log revision. Do not execute against moving `main`.

## Local synthetic GradDiff prototype

The small local prototype lives in [`src/graddiff.py`](../src/graddiff.py) and [`src/graddiff_train.py`](../src/graddiff_train.py); its offline tests are in [`tests/test_graddiff.py`](../tests/test_graddiff.py). The CLI intentionally supports only the built-in synthetic fixture. It does not import Transformers, resolve Hugging Face assets, read TOFU files, or load a checkpoint.

The loss helper implements causal next-token cross entropy by shifting logits and labels once, ignoring `-100` labels, and taking the mean over valid target tokens. It computes the forget and retain means independently, matching the usual Hugging Face causal-LM `.loss` reduction used by OpenUnlearning's handler, then returns `-gamma * forget_nll + alpha * retain_nll`. The optimizer minimizes this scalar, so the forget contribution is gradient ascent on forget NLL and the retain contribution is gradient descent on retain NLL. `gamma` and `alpha` are finite nonnegative coefficients, defaulting to the reviewed official values of 1.0.

The trainer checks finite inputs and losses, parameter gradients, model parameters, and floating optimizer state. The built-in fixture uses two independent parameter rows so the expected directions are controlled: forget NLL increases, retain NLL decreases, and the minimized objective decreases after each AdamW step. Run it with:

```sh
python -m src.graddiff_train --dry-run --device auto --steps 3
```

`auto` selects MPS only when PyTorch reports that MPS is built and available in that process; otherwise it selects CPU. `--device mps` fails immediately if unavailable. There is no retry on CPU after an MPS execution error. The prototype rejects `WORLD_SIZE` other than 1 and has no distributed-training path. The report records the device actually selected, seed, losses, and that no checkpoint or external data was loaded.

### Validation scope and deviations

The automated tests check shifted-token mean reduction (including ignored targets), separate forget/retain means, the official coefficient signs, non-finite rejection, the synthetic AdamW update, device selection, single-process enforcement, and CLI behavior. CPU unit tests and dry runs pass. The MPS synthetic dry run also passes, and a tiny-fixture CPU/MPS comparison of scalar losses, gradients, and a Torch AdamW update passes the proposed FP32 tolerances. These checks establish local objective behavior and CPU/MPS consistency on a tiny fixture; they do not establish direct implementation equivalence against a pinned OpenUnlearning revision.

This prototype is deliberately much smaller than OpenUnlearning. It does not implement the upstream trainer, Hugging Face model/tokenizer loading, TOFU chat formatting and label masks, padding and attention-mask behavior, batching/retain sampler, distributed sampler semantics, accumulation and scheduler, gradient checkpointing, bitsandbytes `paged_adamw_32bit`, or checkpoint/evaluation flow. It uses a tiny synthetic model and plain Torch AdamW (weight decay disabled in the isolated dry run). Thus the formula and the standard mean-NLL normalization are matched, but the training pipeline and optimizer trajectory are not scientifically equivalent to the upstream run.

The Apple adaptation profile records the proposed full-model deviations from upstream: FP32 rather than BF16, PyTorch SDPA rather than FlashAttention 2, Torch AdamW rather than bitsandbytes paged AdamW, one MPS process rather than two CUDA processes with ZeRO-3, and microbatch 1 with 32-step accumulation. The proposed FP32 parameter/gradient/Adam-state floor is an arithmetic estimate, not a measured peak. MPS-specific full-model memory, throughput, determinism, and stability remain unknown. Any MPS runtime error is surfaced rather than hidden by a device fallback.

The following are still required before a limited real-model pilot: pin the exact OpenUnlearning commit; compare official and local losses, every parameter gradient, and one update on the same tiny causal model and fixed inputs; verify actual tokenizer, prompt template, target masking, padding, batching, and data sampling; then do a memory-only model load and one explicitly bounded training step with rollback to the untouched starting checkpoint. Do not interpret success on the synthetic fixture as evidence that a 1.2B model fits or that TOFU unlearning has occurred. The current prototype is **not yet ready for a limited real-model experiment** because those official-pipeline compatibility checks remain incomplete.

The official [TOFU evaluation config](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu.yaml) enables Truth Ratio, Forget Quality, forget QA probability and ROUGE, Model Utility, privacy leakage, and extraction-strength metrics. Truth Ratio compares normalized likelihood of the paraphrased correct answer with structurally similar perturbed answers; Forget Quality is the KS-test p-value comparing forget-side Truth Ratio distributions with the retain-only reference; Model Utility aggregates the official retain, Real Authors, and World Facts evaluations. Forget Quality requires matching retain-only evaluation logs; use official OpenUnlearning logs only after verifying provenance, or recompute using the pinned retain99 checkpoint and evaluator. The upstream README documents `python setup_data.py --eval` for obtaining evaluation logs. The saved lexical baseline is not a substitute.

OpenUnlearning documents that evaluation during unlearning is supported only for one accelerator process; for the distributed run, save checkpoints and evaluate after training. Save each epoch's model weights and score epochs 0, 1, 5, and 10 after training. Epoch 10 is primary; earlier scores are a trajectory, not a selection mechanism. Keep holdout results sealed until the fixed primary result is set. Keep the initial checkpoint and historical baseline untouched. Model-only checkpoints support evaluation and rollback to a valid weight snapshot but cannot resume the exact optimizer trajectory.

## Hardware recommendation and limits

**Recommended execution:** Linux x86_64 CUDA with two NVIDIA L40S GPUs, 48 GB VRAM each, Accelerate, and DeepSpeed ZeRO-3. This best matches OpenUnlearning's reported reproduction setup. The inventoried Mac has MPS but no CUDA. A paid cloud GPU is recommended for the official-faithful run if no equivalent CUDA system is already available. A cloud GPU is not a mathematical requirement for GradDiff; it is needed to follow this upstream hardware/backend path on the current host. No provider has been selected, no instance has been started, and no cloud charge has been incurred.

PyTorch supports GPU training through [MPS on Apple Silicon](https://docs.pytorch.org/docs/main/notes/mps.html), but the upstream attention, optimizer, and ZeRO-3 launch are CUDA/Linux-specific. The current [bitsandbytes accelerator matrix](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/README.md#accelerator-support) includes some Apple Silicon features but excludes its paged optimizer on Metal. The MPS prototype config records the deviations: FP32, PyTorch SDPA, `torch.optim.AdamW`, one MPS process, microbatch 1 with 32-step accumulation, and gradient checkpointing. The loss remains GradDiff, but this is a separate port result because it changes precision, kernels, optimizer, distributed sampling, and execution.

The local weight file is 2,471,645,608 bytes. A rough training-state floor is about 15 GB for roughly 1.2B BF16 parameters, gradients, and two FP32 Adam moments, before activations, temporary buffers, and macOS. The MPS FP32 adaptation has a rough 19.2 GB state floor under the same parameter-count assumption, before activations. These are arithmetic estimates, not peak-memory measurements. Full-model MPS memory, runtime, and stability on this Mac remain unverified. Ten model-only checkpoints are estimated at about 25 GB from the observed file size; 40 GB persistent disk is budgeted for checkpoints, logs, and caches. That is a planning allowance, not a measured output size.

The local tests establish the objective formula, mean-loss normalization, gradient contribution signs, and controlled Torch AdamW behavior on a tiny synthetic model. The tiny-fixture CPU-versus-MPS loss, gradient, and Torch AdamW update comparison passes its proposed tolerances. The direct comparison with the official handler, paged-optimizer comparison, tokenizer/label-mask check, retain-sampling check, and effective-batch comparison have not run. Passing those checks on a tiny fixture would establish only objective/update agreement there, not full-model fit, speed, deterministic kernels, or CUDA benchmark equivalence.

## Commands reserved for a later approved execution

The JSON file is a validated record of planned settings, not a Hydra config that OpenUnlearning loads directly. After freezing source, model, data, and evaluation-log pins, explicitly override the 01/99 splits, tokenizer revision, seed, checkpointing, and save/evaluation behavior. For example:

```sh
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default trainer=GradDiff \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  model.model_args.revision=88e31200b97e4c0c04ae0d2f0b591f427046d192 \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  model.tokenizer_args.revision=88e31200b97e4c0c04ae0d2f0b591f427046d192 \
  forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
  data.forget.TOFU_QA_forget.args.hf_args.revision=324592d84ae4f482ac7249b9285c2ecdb53e3a68 \
  data.retain.TOFU_QA_retain.args.hf_args.revision=324592d84ae4f482ac7249b9285c2ecdb53e3a68 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json \
  task_name=tofu_Llama-3.2-1B-Instruct_forget01_GradDiff_seed2026 \
  trainer.method_args.gamma=1.0 trainer.method_args.alpha=1.0 \
  trainer.method_args.retain_loss_type=NLL \
  trainer.args.seed=2026 trainer.args.data_seed=2026 \
  trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
  trainer.args.warmup_epochs=1.0 trainer.args.weight_decay=0.01 \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.gradient_checkpointing=true \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.do_eval=false trainer.args.eval_on_start=false \
  trainer.args.eval_strategy=no \
  trainer.args.save_strategy=epoch trainer.args.save_only_model=true \
  trainer.args.save_total_limit=11
```

This is an unexecuted example. Validate every override against the frozen upstream Hydra config first. In particular, confirm the pinned TOFU revision reaches both dataset loaders and the framework applies the intended 512-token max length. After training, evaluate the initial, epoch-1, epoch-5, and epoch-10 weights with the official evaluator using `forget_split=forget01`, `holdout_split=holdout01`, and validated retain99 logs. Write outputs to new paths; never overwrite historical result artifacts.
