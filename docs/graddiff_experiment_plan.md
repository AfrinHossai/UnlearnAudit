# GradDiff experiment plan: TOFU forget01 / retain99

**Status:** Experiment plan prepared; the synthetic GradDiff prototype, read-only real-model integration, and guarded local real-model runner are implemented. No TOFU GradDiff optimizer update, fine-tuning, official model evaluation, cloud provisioning, or cloud charges have occurred.

## Decision and experiment target

The primary experiment is OpenUnlearning GradDiff on the TOFU 01/99 protocol using the TOFU-fine-tuned Llama 3.2 1B Instruct `full` checkpoint as the model to unlearn. Use the complete 40-record `forget01` training split and the 3,960-record `retain99` training split. Use the Linux CUDA profile in [the experiment specification](../experiments/graddiff_tofu_forget01_retain99.json) for the main result. The Apple MPS file is a prototype-only adaptation design.

The method objective is

$$
\mathcal{L}_{\mathrm{GradDiff}} = -\gamma\,\mathrm{NLL}(F) +
\alpha\,\mathrm{NLL}(R), \quad \gamma=1,\;\alpha=1.
$$

OpenUnlearning computes the negative model loss on the forget batch, adds the retain loss, and supports NLL or KL retain loss. Its GradDiff config selects `gamma=1.0`, `alpha=1.0`, and `NLL`; this does not create a separate reference model for the training loss. The reviewed GradDiff source is pinned at commit `a456aa277be14b836248e42b13b3befe47bb03f4` in the [experiment profile](../experiments/graddiff_tofu_forget01_retain99.json).

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

The [TOFU unlearning config](https://github.com/locuslab/open-unlearning/blob/a456aa277be14b836248e42b13b3befe47bb03f4/configs/experiment/unlearn/tofu/default.yaml) defaults to `forget10`, `retain90`, and `holdout10`; override each for this experiment. The [Llama 3.2 1B model config](https://github.com/locuslab/open-unlearning/blob/a456aa277be14b836248e42b13b3befe47bb03f4/configs/model/Llama-3.2-1B-Instruct.yaml) selects BF16 and `flash_attention_2`. GradDiff inherits the [fine-tuning trainer config](https://github.com/locuslab/open-unlearning/blob/a456aa277be14b836248e42b13b3befe47bb03f4/configs/trainer/finetune.yaml): learning rate `1e-5`, 10 epochs, 8 examples per device, accumulation 4, `paged_adamw_32bit`, and gradient checkpointing off. The concrete [TOFU script](https://github.com/locuslab/open-unlearning/blob/a456aa277be14b836248e42b13b3befe47bb03f4/scripts/tofu_unlearn.sh) uses 4 examples per device, accumulation 4, gradient checkpointing on, and two GPUs (effective global batch 32). The [reproduction notes](https://github.com/locuslab/open-unlearning/blob/a456aa277be14b836248e42b13b3befe47bb03f4/docs/repro.md) report two 48 GB L40S GPUs with DeepSpeed ZeRO-3 and caution that changing distributed setup can change results.

The canonical profile uses that concrete two-GPU script setup, 10 epochs, `learning_rate=1e-5`, the pinned Transformers default `warmup_ratio=0.0`, `weight_decay=0.0`, and GradDiff `gamma=alpha=1`. The pinned TOFU script does not override weight decay or warmup, so the base trainer defaults apply. Project seed 2026 matches the paired manifest and explicitly overrides the upstream Trainer seed of 0. Use the released checkpoint's tokenizer and immutable revision in both training and evaluation.

Use a separate Python 3.11 environment. At the pinned OpenUnlearning commit, `requirements.txt` specifies PyTorch 2.4.1, Transformers 4.51.3, Datasets 3.0.1, Accelerate 0.34.2, bitsandbytes 0.44.1, and DeepSpeed 0.15.4, among other packages. The quickstart separately installs FlashAttention 2.8.3; its example wheel is for CUDA 12.8, Linux x86_64, and CPython 3.11. The existing `.venv-ml` has PyTorch 2.9.1 and Transformers 4.57.1, so synthetic tests there do not verify behavior under the upstream dependency set.

The GradDiff handler file history identifies `a456aa277be14b836248e42b13b3befe47bb03f4` as its last modifying commit. That exact SHA and permalinks are now recorded in the experiment profile. The synthetic source-equation test transcribes the handler's loss arithmetic at that SHA; it does not import the OpenUnlearning package or execute its Trainer. Before a scientific run, check out the pinned tree, verify all experiment/data/model/evaluator configs, and archive the resolved Hydra config plus environment lock or container digest. Do not execute against moving `main`.

## Local synthetic GradDiff prototype

The small local prototype lives in [`src/graddiff.py`](../src/graddiff.py) and [`src/graddiff_train.py`](../src/graddiff_train.py); its offline tests are in [`tests/test_graddiff.py`](../tests/test_graddiff.py). The CLI intentionally supports only the built-in synthetic fixture. It does not import Transformers, resolve Hugging Face assets, read TOFU files, or load a checkpoint.

The loss helper implements causal next-token cross entropy by shifting logits and labels once, ignoring `-100` labels, and taking the mean over valid target tokens. It computes the forget and retain means independently, matching the usual Hugging Face causal-LM `.loss` reduction used by OpenUnlearning's handler, then returns `-gamma * forget_nll + alpha * retain_nll`. The optimizer minimizes this scalar, so the forget contribution is gradient ascent on forget NLL and the retain contribution is gradient descent on retain NLL. `gamma` and `alpha` are finite nonnegative coefficients, defaulting to the reviewed official values of 1.0.

The trainer checks finite inputs and losses, parameter gradients, model parameters, and floating optimizer state. The built-in fixture uses two independent parameter rows so the expected directions are controlled: forget NLL increases, retain NLL decreases, and the minimized objective decreases after each AdamW step. Run it with:

```sh
python -m src.graddiff_train --dry-run --device auto --steps 3
```

`auto` selects MPS only when PyTorch reports that MPS is built and available in that process; otherwise it selects CPU. `--device mps` fails immediately if unavailable. There is no retry on CPU after an MPS execution error. The prototype rejects `WORLD_SIZE` other than 1 and has no distributed-training path. The report records the device actually selected, seed, losses, and that no checkpoint or external data was loaded.

### Validation scope and deviations

The automated tests check shifted-token mean reduction (including ignored targets), separate forget/retain means, the official coefficient signs, non-finite rejection, the synthetic AdamW update, device selection, single-process enforcement, CLI behavior, fixture-based model integration, and the new runner's update gate, clipping, checkpoint safety, OOM response, and log contents. A deterministic test transcribes the pinned Transformers causal-LM loss and OpenUnlearning handler formulas, then compares local and reference scalar losses and input-logit gradients on fixed logits and masked labels. The real cached tokenizer and checksum-pinned forget/retain data were also used for tokenization-only preparation; model weights were not loaded. The current restricted-process run passed 64 tests with one MPS-only test skipped because MPS is unavailable in that process. An earlier normal-Mac run passed the prior 56-test suite, including the tiny MPS test. Synthetic CPU/MPS runs pass, including the tiny-fixture loss, gradient, and Torch AdamW update comparison. These checks do not execute the official Trainer or establish full pipeline equivalence. See the [read-only real-model smoke plan](graddiff_model_integration_smoke.md), which has not yet been executed.

The synthetic training-step prototype remains separate from the read-only real-model integration and the new local runner. [`src/graddiff_integration.py`](../src/graddiff_integration.py) applies the checkpoint tokenizer's system/user/assistant template, masks prompt tokens, and right-pads small batches. [`src/graddiff_model_smoke.py`](../src/graddiff_model_smoke.py) is a local-cache-only command that compares model and local losses with one forget and one retain batch; it contains no optimizer or backward path and has not been run with the full checkpoint. [`src/graddiff_real_train.py`](../src/graddiff_real_train.py) adds a guarded single-device update path. It is not the OpenUnlearning Trainer and does not implement the official collator/sampler, distributed sampler semantics, gradient accumulation, scheduler, or bitsandbytes `paged_adamw_32bit`. The local loss ignores `-100` targets and explicitly shifts causal labels. Both the synthetic step and the new real runner use Torch AdamW, but their small/local training loops and optimizer trajectories are not scientifically equivalent to the upstream run.

The Apple adaptation profile records the proposed full-model deviations from upstream: FP32 rather than BF16, PyTorch SDPA rather than FlashAttention 2, Torch AdamW rather than bitsandbytes paged AdamW, one MPS process rather than two CUDA processes with ZeRO-3, and microbatch 1 with 32-step accumulation. The proposed FP32 parameter/gradient/Adam-state floor is an arithmetic estimate, not a measured peak. MPS-specific full-model memory, throughput, determinism, and stability remain unknown. Any MPS runtime error is surfaced rather than hidden by a device fallback.

The tokenizer/template boundary, answer/EOS target mask, right-padding behavior, pinned split checksums, and one-record-per-split batch preparation have been checked with the real cached tokenizer and lightweight fixtures. The new runner defaults to validation: it reads local checkpoint config/tokenizer metadata and checksum-pinned split files, prepares one answer-only forget/retain pair, and does not deserialize model weights or create an optimizer. Weight updates require `--apply-updates`, an explicit `--device mps` or `--device cpu`, the exact `--confirm-model-revision`, and a new report path under Git-ignored `outputs/`. The default maximum is one optimizer step; batch size is fixed at one; the per-invocation cap is 40 steps, one pass over forget01. It records the immutable model revision, dataset checksums, split row indices (without QA text), loss components, hyperparameters, memory snapshots, and actual device. A checkpoint can be requested only under ignored `outputs/`; existing paths and the original TOFU checkpoint are rejected.

The local update path loads the model in FP32, uses PyTorch SDPA, enables gradient checkpointing, and uses `torch.optim.AdamW(foreach=False, fused=False)` with weight decay zero. It computes forget and retain losses in sequence and backpropagates each contribution before the next forward, checks finite losses and gradients, clips the combined gradient norm (default 1.0), then steps. This keeps only one model forward's logits live at a time. It uses no autocast, scheduler, or gradient accumulation. Forget rows are traversed in order and retain rows are sampled uniformly with the configured seed. MPS errors are surfaced; the runner never retries on CPU. Reports omit question/answer text and environment variables. Checkpoint and report destinations must be new paths under ignored `outputs/`.

Run metadata-only/tokenization validation in the existing local environment with:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_real_train --dry-run
```

After the full-model read-only smoke succeeds on the physical Mac and current memory pressure has been checked, the exact command for a single carefully monitored MPS optimizer step is:

```sh
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  ./.venv-ml/bin/python -m src.graddiff_real_train \
  --apply-updates --device mps \
  --confirm-model-revision 88e31200b97e4c0c04ae0d2f0b591f427046d192 \
  --max-steps 1 --seed 2026 --learning-rate 1e-5 \
  --gamma 1.0 --alpha 1.0 --max-grad-norm 1.0 \
  --max-sequence-length 512 \
  --report-path outputs/graddiff-step1-mps-2026-11-03.json \
  --save-checkpoint outputs/graddiff-step1-mps-2026-11-03
```

For an intentional CPU run, change only `--device mps` to `--device cpu`; the program does not automatically choose CPU when MPS is unavailable. The local Python environment is Torch 2.9.1 and Transformers 4.57.1; the reviewed upstream pins are Torch 2.4.1 and Transformers 4.51.3. The runner's FP32 parameter/gradient/two-moment AdamW floor is estimated at 19,773,030,400 bytes (18.42 GiB). The documented 27.1–31.1 GiB planning envelope includes a non-measured 8–12 GiB runtime reserve and selected live tensors; it is not a measured peak or a guarantee that current free unified memory is sufficient. The runner has not loaded the real checkpoint, run its full-model forward/backward, exercised AdamW on MPS at model scale, or measured peak memory. **It is not yet cleared for even one real-model optimizer step.** First run the read-only full-model loss/memory smoke and verify the available memory on the physical Mac. The pinned Trainer/collator, upstream optimizer trajectory, accumulation/scheduler, and CUDA distributed behavior remain unverified; they block benchmark-reproduction claims even if a local pilot later succeeds.

The official [TOFU evaluation config](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu.yaml) enables Truth Ratio, Forget Quality, forget QA probability and ROUGE, Model Utility, privacy leakage, and extraction-strength metrics. Truth Ratio compares normalized likelihood of the paraphrased correct answer with structurally similar perturbed answers; Forget Quality is the KS-test p-value comparing forget-side Truth Ratio distributions with the retain-only reference; Model Utility aggregates the official retain, Real Authors, and World Facts evaluations. Forget Quality requires matching retain-only evaluation logs; use official OpenUnlearning logs only after verifying provenance, or recompute using the pinned retain99 checkpoint and evaluator. The upstream README documents `python setup_data.py --eval` for obtaining evaluation logs. The saved lexical baseline is not a substitute.

OpenUnlearning documents that evaluation during unlearning is supported only for one accelerator process; for the distributed run, save checkpoints and evaluate after training. Save each epoch's model weights and score epochs 0, 1, 5, and 10 after training. Epoch 10 is primary; earlier scores are a trajectory, not a selection mechanism. Keep holdout results sealed until the fixed primary result is set. Keep the initial checkpoint and historical baseline untouched. Model-only checkpoints support evaluation and rollback to a valid weight snapshot but cannot resume the exact optimizer trajectory.

## Hardware recommendation and limits

**Recommended execution:** Linux x86_64 CUDA with two NVIDIA L40S GPUs, 48 GB VRAM each, Accelerate, and DeepSpeed ZeRO-3. This best matches OpenUnlearning's reported reproduction setup. The inventoried Mac has MPS but no CUDA. A paid cloud GPU is recommended for the official-faithful run if no equivalent CUDA system is already available. A cloud GPU is not a mathematical requirement for GradDiff; it is needed to follow this upstream hardware/backend path on the current host. No provider has been selected, no instance has been started, and no cloud charge has been incurred.

PyTorch supports GPU training through [MPS on Apple Silicon](https://docs.pytorch.org/docs/main/notes/mps.html), but the upstream attention, optimizer, and ZeRO-3 launch are CUDA/Linux-specific. The current [bitsandbytes accelerator matrix](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/README.md#accelerator-support) includes some Apple Silicon features but excludes its paged optimizer on Metal. The MPS prototype config records the deviations: FP32, PyTorch SDPA, `torch.optim.AdamW`, one MPS process, microbatch 1 with 32-step accumulation, and gradient checkpointing. The loss remains GradDiff, but this is a separate port result because it changes precision, kernels, optimizer, distributed sampling, and execution.

The checkpoint's BF16 weights occupy 2,471,645,608 bytes on disk. Its model config describes 16 layers, hidden size 2,048, intermediate size 8,192, vocabulary size 128,256, and tied input/output embeddings. Counting those matrices and norms gives an architecture-derived estimate of 1,235,814,400 parameters. FP32 weights (4.94 GB), gradients (4.94 GB), and two FP32 Adam moments (9.89 GB) yield a 19.77 GB (18.42 GiB) persistent-state floor. This excludes optimizer metadata, activations, logits, temporary copies, allocator caches, and macOS/apps. At batch 1 and sequence length 512 in FP32, the two forget/retain logits are about 501 MiB in total; two sets of per-layer checkpoint-boundary activations are about 136 MiB. Layer workspaces are additional; if an SDPA backend materializes one 32-head 512x512 attention matrix, that single tensor is 32 MiB. These are calculated tensor footprints, not a peak-memory profile. `os.sysconf` reports 51,539,607,552 bytes of total physical memory on this host (48 GiB); about 29.58 GiB remains after the state floor before activations and device/OS overhead. The MPS profile adds an explicit 8–12 GiB planning reserve for host/device runtime overhead; it is not a measurement. Including that reserve and the listed 0.68 GiB selected live tensors gives a rough 27.1–31.1 GiB planning envelope, leaving about 16.9–20.9 GiB of total capacity for uncounted activations and transients on paper. Current available memory and full-model MPS allocation and speed remain unmeasured. The 10-checkpoint and disk estimates in the profile are planning estimates, not measured outputs.

The local tests establish the objective formula, mean-loss normalization, gradient contribution signs, controlled Torch AdamW behavior on a tiny synthetic model, and read-only integration behavior on lightweight fixtures. The actual cached tokenizer produces stable answer-only labels on the selected pinned data examples. The direct runtime comparison with the official handler, paged-optimizer comparison, accumulation/scheduler comparison, full-model read-only forward, and model-sized memory profile have not run. Passing fixture checks establishes only the tested tokenization and local loss behavior; it does not establish full-model fit, speed, deterministic kernels, or CUDA benchmark equivalence.

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
  trainer.args.warmup_ratio=0.0 trainer.args.weight_decay=0.0 \
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
