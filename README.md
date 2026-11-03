# UnlearnGuard: Privacy Leakage After Machine Unlearning

**Project:** `UnlearnGuard-Privacy-Leakage-Research`  
**Status:** TOFU dataset inspection pipeline implemented; model unlearning experiments remain planned.

> **Research status:** The repository now downloads and inspects the official TOFU `forget01` dataset split. No model has been trained, fine-tuned, evaluated, or unlearned here, and no unlearning results or claims are reported.

## Motivation

Large language models can reproduce information associated with their training data. Machine unlearning aims to reduce a model's reliance on selected data after training, while preserving useful behavior on data that should remain. A low score on a standard question-answering prompt alone does not establish that the target information is no longer recoverable: answers may vary across prompts, and unlearning can also damage unrelated capabilities.

This project is planned as a reproducible study of that privacy-utility trade-off. It will use TOFU (Task of Fictitious Unlearning), a benchmark built around synthetic fictional-author profiles and designated forget and retain data. Its controlled setup makes it possible to study targeted forgetting without using real people's private records. The benchmark also provides evaluations for both forget quality and retained model utility. [Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*](https://arxiv.org/abs/2401.06121).

## Objective

Evaluate whether one established unlearning method—**gradient ascent (GA) on the forget set**—reduces the recoverability of TOFU target answers, and measure the effect on retained knowledge. GA is selected as a well-described baseline used in prior LLM unlearning work; it is not presented here as a recommended or privacy-guaranteeing method. [Yao et al., *Machine Unlearning of Pre-trained Large Language Models*](https://aclanthology.org/2024.acl-long.457/).

The planned study will address:

1. How does GA change target-answer recovery on TOFU forget-set questions?
2. Do the results hold for a fixed set of paraphrased or reformulated prompts?
3. What changes in forgetting coincide with changes in utility on TOFU retain and related evaluation sets?

## Planned Methodology

Every item in this section is **planned work**. No models have been trained or evaluated in this repository.

1. **Pin the benchmark and model.** Use the TOFU data and its designated forget/retain protocol. Before running experiments, select and record a compatible TOFU-finetuned checkpoint and immutable dataset/model revisions. Publish configuration files, data provenance, and checksums where licensing permits.
2. **Run one unlearning technique.** Apply gradient ascent to the forget examples using a documented optimizer, learning rate, batch size, number of steps, and random seed. Keep the method fixed across comparisons; record each configuration and checkpoint identifier.
3. **Compare reference models.** Evaluate the pre-unlearning checkpoint, the GA-unlearned checkpoint, and—where feasible under the compute budget—a retain-only reference trained without the forget set. The retain-only model provides a useful benchmark for the TOFU forget-quality comparison.
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

See [docs/compute_feasibility.md](docs/compute_feasibility.md) for a read-only assessment of local hardware and Python support for machine-unlearning experiments, OpenUnlearning compatibility, and a recommended next experiment.

This pipeline only reads the benchmark data. It does not load a model or perform any training, fine-tuning, evaluation, or machine unlearning. The inspected records are not evidence that any model has forgotten anything.

## Future Interactive Demo

An interactive application is planned after the experiment pipeline is implemented. It is intended to let a user inspect the fixed evaluation prompts, compare pre-unlearning and GA-unlearned responses, and explore target-answer recovery and utility metrics across the forget and retain sets. The interface should expose model, dataset, and run configuration so that displayed comparisons can be traced to saved experiment artifacts.

**The demo has not been built.** No model outputs or experimental performance results are included. The tracked TOFU inspection summary contains dataset metadata only.

## Repository Layout

```text
.
├── app/          # Planned interactive demo
├── docs/         # Dataset provenance and reproducibility documentation
├── experiments/  # Planned run configurations and experiment records
├── results/      # Tracked, metadata-only TOFU inspection summary
├── src/          # Dataset inspection code; model experiments remain planned
└── tests/        # Standard-library tests for inspection and validation
```

## References

- Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., & Kolter, J. Z. (2024). [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121).
- Yao, Y., et al. (2024). [Machine Unlearning of Pre-trained Large Language Models](https://aclanthology.org/2024.acl-long.457/).
