# Machine-unlearning compute feasibility

**Assessment date:** 2026-11-03
**Scope:** Inspection of this Mac and OpenUnlearning, a separate project-local environment for a no-checkpoint inference smoke test, and preparation of an offline-tested TOFU evaluator. The five-question baseline was generated later and is documented in [baseline findings](baseline_findings.md). The paired 01/99 manifest is now prepared but has not been evaluated. This task did not download model weights, run model inference, or perform unlearning.

## Decision

This Mac can run PyTorch and a tiny Transformers model on its MPS GPU when the process has normal access to Metal. The same checks run inside the restricted command sandbox report MPS unavailable; the sandboxed process sees zero devices. Outside the sandbox, both project-local Python 3.11 and Anaconda PyTorch 2.9.1 see one MPS device. The root cause is sandbox GPU isolation, not a Mac, macOS, Python-architecture, or Conda conflict.

The local environment is ready for small-model inference and synthetic MPS checks. This does not establish that the full OpenUnlearning GradDiff recipe runs on Mac: its configured attention backend and reported distributed setup target CUDA/Linux. For an unmodified, comparable run, use a Linux cloud instance with NVIDIA GPUs; the project's reported reproduction used two 48 GB L40S GPUs with DeepSpeed ZeRO-3. Full-parameter training feasibility on this Mac remains untested.

## Mac and Python inventory

| Item | Observed |
|---|---|
| Mac | MacBook Pro, model identifier Mac16,8 |
| Chip | Apple M4 Pro |
| GPU | 16-core built-in M4 Pro GPU; macOS reports Metal supported |
| Memory | 48 GB unified memory |
| macOS | 26.6.2, build 25G83 |
| Architecture | arm64 |
| Free disk | 238 GiB on the data volume at inspection time |

Python runtimes found:

| Runtime | Version | Relevant packages |
|---|---:|---|
| Anaconda base, `/opt/anaconda3/bin/python` (the `python` command) | 3.13.5 | PyTorch 2.9.1 and Transformers 4.57.1 |
| Miniconda base, `/opt/miniconda3/bin/python` | 3.13.5 | No PyTorch, Transformers, or MLX detected |
| python.org framework, `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (the `python3` command) | 3.13.7 | No PyTorch, Transformers, or MLX detected |
| User-local, `~/.local/bin/python3.11` | 3.11.18 | No PyTorch, Transformers, or MLX detected |
| Apple Command Line Tools, `/usr/bin/python3` | 3.9.6 | No PyTorch, Transformers, or MLX detected |

The active Anaconda interpreter is arm64 and has PyTorch 2.9.1 and Transformers 4.57.1. MLX is not installed in the checked Python environments. The repository's original `requirements.txt` remains standard-library-only; the new machine-learning smoke dependencies are kept separately in `requirements-ml.txt` and its lock file.

## Verified project-local smoke environment

Python 3.11.18 was available at `/Users/afrinhossain/.local/bin/python3.11`; Python 3.12 was not found on `PATH`, in the standard Python framework, Homebrew, Conda, or user-local locations, and pyenv is not installed. The existing `.venv` is CPython 3.13.7 with pip only and remains the TOFU-inspection environment. A separate ignored `.venv-ml` was created with Python 3.11.18. Direct dependencies are pinned to PyTorch 2.9.1 and Transformers 4.57.1 in [`requirements-ml.txt`](../requirements-ml.txt); [`requirements-ml.lock`](../requirements-ml.lock) records the complete resolved package set, including transitive versions and the pip 26.2.2 / setuptools 79.0.1 bootstrap tools. `pip check` reported no broken requirements.

The repeatable diagnostic is [`scripts/diagnose_ml_environment.py`](../scripts/diagnose_ml_environment.py). It reports Python, macOS, MPS/CUDA support and selected environment variables; chooses MPS when PyTorch says it is available, otherwise CPU; then runs a 4×4 synthetic tensor operation and inference with a randomly initialized one-layer GPT-2 model. It does not load weights or access Hugging Face.

| Test context | MPS result | Smoke result |
|---|---|---|
| Restricted command sandbox, Anaconda and `.venv-ml` | Built `true`; macOS 15+ check `true`; low-level runtime probe `false`; device count `0`; available `false` | `.venv-ml` used CPU; tensor computation and random GPT-2 inference passed; logits shape `(1, 4, 64)` |
| Outside the restricted sandbox, same `.venv-ml` and a clean process environment | Built `true`; macOS 15+ check `true`; low-level runtime probe `true`; device count `1`; available `true` | MPS tensor computation and random GPT-2 inference passed; logits shape `(1, 4, 64)` |
| Outside the restricted sandbox, existing Anaconda Python 3.13.5 | Built `true`; MPS available `true`; device count `1` | MPS availability check passed |

The sandbox process had `CONDA_DEFAULT_ENV=base` and `CONDA_PREFIX=/opt/anaconda3`; `PYTHONHOME`, `PYTHONPATH`, PyTorch MPS overrides, and DYLD library overrides were unset. Clearing the Conda variables and other overrides did not make MPS visible inside the sandbox. Outside the sandbox, MPS was available in both the new venv and Anaconda, ruling out the Anaconda package and these environment variables as the cause. No system settings or global Python installations were changed. To use MPS, run the diagnostic from a normal Terminal process with `.venv-ml/bin/python`; the restricted sandbox cannot access the Metal device and will correctly fall back to CPU.

Recreate the local environment from the repository root with:

```sh
python3.11 -m venv .venv-ml
./.venv-ml/bin/python -m pip install --no-cache-dir -r requirements-ml.lock
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONPATH \
  ./.venv-ml/bin/python scripts/diagnose_ml_environment.py
```

This lightweight environment is for device and inference smoke checks only. It is not the full OpenUnlearning dependency set, whose current requirements pin Transformers 5.5.4.

## OpenUnlearning, TOFU, and GradDiff

The [official OpenUnlearning README](https://github.com/locuslab/open-unlearning) documents Python 3.11 for its quickstart, and [setup.py](https://raw.githubusercontent.com/locuslab/open-unlearning/main/setup.py) declares Python `>=3.11`. The current [requirements file](https://raw.githubusercontent.com/locuslab/open-unlearning/main/requirements.txt) pins PyTorch 2.9.1 and Transformers 5.5.4, among other packages. PyTorch matches the active environment, but its Transformers 4.57.1 does not match the pinned version. Python 3.13 satisfies the declared minimum, though the project quickstart explicitly creates Python 3.11; use a separate Python 3.11 environment if following the upstream setup later.

The upstream quickstart separately installs `flash-attn==2.8.3`; its example prebuilt wheel is for CUDA 12.8, Linux x86_64, and CPython 3.11. OpenUnlearning's [Llama 3.2 1B config](https://raw.githubusercontent.com/locuslab/open-unlearning/main/configs/model/Llama-3.2-1B-Instruct.yaml) selects `flash_attention_2` and bfloat16. Its [default fine-tuning trainer config](https://raw.githubusercontent.com/locuslab/open-unlearning/main/configs/trainer/finetune.yaml), inherited by GradDiff, uses per-device batch size 8, gradient accumulation 4, bfloat16, `paged_adamw_32bit`, no gradient checkpointing, and 10 epochs. The [TOFU baseline script](https://raw.githubusercontent.com/locuslab/open-unlearning/main/scripts/tofu_unlearn.sh) launches two GPUs through Accelerate/DeepSpeed and overrides the per-device batch to 4, accumulation to 4, and gradient checkpointing on; that gives effective batch size 32. The [reproduction notes](https://raw.githubusercontent.com/locuslab/open-unlearning/main/docs/repro.md) also report two 48 GB L40S GPUs and DeepSpeed ZeRO-3, though their batch-size line differs from the baseline script. Follow the concrete script settings and record them for the proposed run.

The TOFU default experiment uses the `forget10`, `retain90`, and `holdout10` splits. Its configured starting model is `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`. OpenUnlearning describes its `full` checkpoint as the TOFU-finetuned target model, so it has already learned from the TOFU facts before unlearning begins. The [checkpoint is a 1B-parameter BF16 model](https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full) and its `model.safetensors` file is listed as **2.47 GB**. This is preferable to starting from the general Llama instruct model: the base instruct model has not undergone the TOFU fine-tuning whose influence the unlearning run is meant to remove.

OpenUnlearning supports [GradDiff](https://raw.githubusercontent.com/locuslab/open-unlearning/main/src/trainer/unlearn/grad_diff.py), which trains against the forget set while adding a retain-set loss. Its [GradDiff config](https://raw.githubusercontent.com/locuslab/open-unlearning/main/configs/trainer/GradDiff.yaml) sets `gamma=1`, `alpha=1`, and retain loss type `NLL`. The TOFU experiment config defaults to GradAscent, so an OpenUnlearning GradDiff run must explicitly set `trainer=GradDiff`.

## Compatibility and resource requirements

| Area | Assessment |
|---|---|
| PyTorch on the Mac GPU | PyTorch's [MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html) distinguishes a build compiled with MPS from MPS being available at runtime. Both checked PyTorch 2.9.1 environments report MPS available outside the restricted sandbox; inside it, the MPS runtime probe is false and the device count is zero. |
| CUDA-specific pieces | The configured `flash_attention_2` path and example FlashAttention wheel target CUDA/Linux and do not run as-is on Apple MPS. The reported distributed recipe uses Accelerate with DeepSpeed ZeRO-3, for which OpenUnlearning documents no MPS setup. The requirements pin bitsandbytes 0.49.2; [recent bitsandbytes releases have some Apple Silicon support](https://github.com/bitsandbytes-foundation/bitsandbytes/releases), but do not assume the pinned `paged_adamw_32bit` path works on this Mac without checking it. A local port would need MPS-compatible attention and optimizer choices, smaller microbatches, and likely gradient checkpointing. |
| MLX | Not present, and it is a separate Apple Silicon framework rather than a drop-in device backend for OpenUnlearning's Hugging Face Trainer workflow. |
| Memory | The 2.47 GB BF16 weight file is only the inference/checkpoint footprint. A rough full-training floor is about 15 GB for weights, gradients, and two FP32 Adam moments for roughly 1.2B parameters, before activations, temporary tensors, evaluation, and macOS. Plan for the process to use tens of GB; actual peak depends heavily on batch size, sequence length, attention backend, and checkpointing. The upstream default microbatch of 8 without checkpointing is not a prudent first MPS attempt on 48 GB shared with macOS. |
| Disk and downloads | The full TOFU-trained checkpoint is 2.47 GB. Allow another roughly 2.5 GB for a saved unlearned checkpoint and reserve at least 10 GB for model cache, output, logs, and temporary files. The Mac currently has 238 GiB free. A separate retain-only checkpoint, if later needed as a reference, would add another model-sized download. |

The memory figures above are planning estimates, not measurements from this Mac. They follow from the checkpoint size and the storage required by full-parameter training; the OpenUnlearning reproduction page reports its actual hardware but does not give a minimum-memory requirement for a single machine.

## Local baseline evaluation status

The repository contains an offline-testable evaluator for the initial five-question run and a paired evaluator for all 40 pinned `forget01` questions plus a deterministic 40-question sample from `retain99`. The initial run uses OpenUnlearning's official `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` checkpoint at immutable revision `88e31200b97e4c0c04ae0d2f0b591f427046d192`. The paired case manifest and exact command are documented in [paired TOFU evaluation](paired_tofu_evaluation.md). The paired evaluation has not run.

The evaluator loads the bundled tokenizer from the same checkpoint revision (not the separately hosted `meta-llama` tokenizer), applies its chat template with OpenUnlearning's fixed system prompt/date, and uses greedy generation. On MPS it casts the BF16 checkpoint to FP16; on CPU it casts to FP32. The weight file itself is about 2.47 GB; FP32 model weights require roughly twice the FP16/BF16 storage, with additional memory for model loading, activations, and KV cache. The Mac's 48 GB unified memory should leave room for a small inference sample, but exact peak memory on this checkpoint has not been measured. The initial five-question baseline was run, but the paired 80-question direct-QA diagnostic has not. Neither result computes the official TOFU truth-ratio, Forget Quality, or Model Utility protocol.

See [`tofu_baseline.md`](tofu_baseline.md) for the original configuration, [`baseline_findings.md`](baseline_findings.md) for the five-question audit, and [`paired_tofu_evaluation.md`](paired_tofu_evaluation.md) for the frozen manifest, official metric plan, and command to run the expanded diagnostic from a normal macOS Terminal.

## Recommended unlearning experiment after the baseline

Run the expanded `forget01`/`retain99` pre-unlearning diagnostic from [`paired_tofu_evaluation.md`](paired_tofu_evaluation.md), then run the official TOFU evaluation for the same split pair before interpreting any GradDiff result. The official Forget Quality metric also needs a compatible retain99-only reference checkpoint or a separately retrained retain99 oracle and its evaluation logs. For a closest match to the published OpenUnlearning training setup, use a Linux cloud environment with two 48 GB NVIDIA GPUs and the upstream Python 3.11 environment. Explicitly override the upstream split defaults to `forget01`, `retain99`, and `holdout01`; do not compare a 01/99 baseline with a 10/90 unlearning run. The paired text diagnostics do not substitute for official Truth Ratio, Forget Quality, or Model Utility.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default trainer=GradDiff \
  forget_split=forget01 retain_split=retain99 holdout_split=holdout01 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.gradient_checkpointing=true \
  task_name=llama32_1b_tofu_forget01_graddiff
```

The model path in the default TOFU config already points to `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`. The cloud run will still need to download that 2.47 GB checkpoint and the official evaluation logs (the framework documents `python setup_data.py --eval` for fetching these logs); the pinned TOFU QA files used by the paired diagnostic are already in the ignored local `.cache/tofu/` directory. A local MPS pilot can be considered later after adapting the CUDA-specific framework settings and separately measuring training memory. This assessment does not change the repository's stated GA objective or claim that any model has been unlearned.
