# TOFU dataset provenance and inspection

## Dataset and split

TOFU (Task of Fictitious Unlearning) is a benchmark for studying targeted unlearning in language models. Its profiles and question-answer pairs describe fictional authors; the benchmark authors report generating the synthetic biographies with GPT-4. TOFU defines forget and retain sets so that future model experiments can compare behavior on information intended to be forgotten with behavior that should be retained.

This repository inspects the `forget01` split. TOFU describes it as the 1% forget set: two author profiles out of the benchmark's 200 profiles. The pinned source contains 40 question-answer records. The inspection code does not assume the file's schema from its name: it reads the records, reports the observed fields and value types, and then validates that each record has non-empty string `question` and `answer` fields.

## Source, version, and checksum

The dataset is downloaded directly from the official Hugging Face dataset repository maintained by Locus Lab:

- Dataset card: [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)
- Pinned source revision: [`324592d84ae4f482ac7249b9285c2ecdb53e3a68`](https://huggingface.co/datasets/locuslab/TOFU/tree/324592d84ae4f482ac7249b9285c2ecdb53e3a68)
- Source file: [`forget01.json`](https://huggingface.co/datasets/locuslab/TOFU/resolve/324592d84ae4f482ac7249b9285c2ecdb53e3a68/forget01.json)
- SHA-256: `d0047332bd27a489c552659831f4ffff080e2b9b6e063c7c951ac4de9d494a63`

Although the source file has a `.json` extension, its contents are newline-delimited JSON (one object per line). The script saves it under `.cache/tofu/`, verifies the pinned checksum, and reuses the cached file only when that checksum matches. No model checkpoints are part of this pipeline. The Hugging Face dataset card lists the dataset license as MIT; consult the source card and paper for full details.

The inspection summary at `results/tofu_forget01_inspection.json` stores the repository ID, split, pinned revision, checksum, record count, observed schema, and validation status. It deliberately excludes the question and answer text. Downloaded files, caches, and virtual environments are ignored by Git.

## Attribution

Please cite the benchmark authors when using TOFU:

> Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. “TOFU: A Task of Fictitious Unlearning for LLMs.” 2024. [arXiv:2401.06121](https://arxiv.org/abs/2401.06121).

The [official TOFU project page](https://locuslab.github.io/tofu/) and [Locus Lab GitHub repository](https://github.com/locuslab/tofu) provide the benchmark description, evaluation protocol, and implementation context.

## Inspection is not machine unlearning

This pipeline performs dataset I/O, schema discovery, field validation, and example display only. It does not load or modify model weights, train or fine-tune a model, run an unlearning method, or measure model utility or forgetting. Displaying an answer from the forget set says nothing about whether a model has memorized or forgotten that answer. Any later unlearning claim requires a separately specified model, method, comparison, and evaluation protocol; this dataset inspection is not evidence of successful unlearning.
