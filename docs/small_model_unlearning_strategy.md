# Smaller-model unlearning strategy for Apple Silicon

**Task 21 assessment, 2026-11-03. Research and documentation only. No model weights downloaded, instantiated, trained or modified. No cloud resources used.**

## Recommendation and execution status

Prepare a **separate SmolLM2-135M controlled fact-acquisition → full-parameter GradDiff experiment**, using 40 pinned TOFU forget records and a deterministic 120-record retain subset. First establish that the model learned the target facts; only then attempt forgetting with retained-utility controls. This is a TOFU-subset experiment, not an official TOFU reproduction. It offers the smallest verified candidate here, a standard Llama-class implementation already familiar to our fixtures, and enough room to keep FP32 and full-parameter updates rather than introducing quantization or adapters.

**The recommendation is conditional, not clearance to execute.** The base model is publicly released but is not cached here and is not a verified TOFU memorization checkpoint. Its ability to learn these particular records within a practical budget is an assumption to test. The next implementation task must add a separate pinned model/encoder profile and fact-acquisition stage; the existing Llama runner is deliberately incompatible with a substituted model ID or parameter count.

The existing **32 GiB SGD / 42 GiB AdamW pre-load guards, 6 GiB continuing reserve, 20% system memory-free criterion, and device/output protections remain unchanged**. Retain them for initial small-model implementation and rehearsals; do not use a model-size estimate as permission to lower them. Task 20's last snapshot was **16.31 GiB free + speculative**, below 32 GiB, with 48 GiB total, a 95% macOS free indicator and zero swap. That is historical, not a fresh reading. Under the present guards runtime remains **NO-GO** until resources are safely reclaimed and fresh readings pass. Any future model-specific threshold would require its own explicit review and measurements; none is approved or implemented here. A smaller working set reduces memory risk once admitted; it does not itself satisfy admission.

The original `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` revision `88e31200b97e4c0c04ae0d2f0b591f427046d192`, its configs, manifests, generated baselines and results stay the reference track. No performance number from a new backbone may replace them.

## Verified availability versus unverified suitability

### Official base-model releases

These are publisher-maintained public model pages, not community reuploads. Availability/licensing comes from cards and metadata; MPS operation and TOFU knowledge have not been tested in this repository.

| Candidate | Availability and size evidence | Architecture / license | TOFU knowledge status |
|---|---|---|---|
| **HuggingFaceTB/SmolLM2-135M** | [Official card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M); [Hub metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M) reports public, ungated and **134,515,008** safetensors parameters | [Config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json): LlamaForCausalLM, 30 layers, hidden 576, vocabulary 49,152, tied embeddings. Card: Apache-2.0 | Base pretraining release, not a TOFU-trained checkpoint. Must demonstrate acquisition |
| **HuggingFaceTB/SmolLM2-360M** | [Official card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M); approximately 360M | [Config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json): LlamaForCausalLM, 32 layers, hidden 960, vocabulary 49,152, tied embeddings. Architecture-derived count **361,821,120**, not a loaded-model count. Apache-2.0 | No suitable TOFU checkpoint verified in this review |
| **EleutherAI/pythia-160m** | [Official card](https://huggingface.co/EleutherAI/pythia-160m) explicitly lists **162,322,944** total parameters, not just the 85M non-embedding count | GPT-NeoX, 12 layers, hidden 768; Apache-2.0. Publisher provides intermediate pretraining checkpoints | Generic research backbone; its pretraining steps are not TOFU fact-acquisition checkpoints |
| **Qwen/Qwen2.5-0.5B-Instruct** | [Official card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) states about 0.49B parameters | [Config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json): Qwen2ForCausalLM, 24 layers, hidden 896, vocabulary 151,936. **494,032,768** is an architecture-derived estimate including QKV biases. Apache-2.0 | Instruction tuning does not establish knowledge of TOFU authors; see the author-released adapter lead below |

For the selected 135M model, Hub metadata resolved `main` to **`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`** during this review. Record that revision as a candidate pin, then revalidate the commit's config/tokenizer and weight-file checksums before any authorized download/use. A current metadata response is not an authenticated local weight audit. Config `main` URLs above aid inspection but must not be execution pins. Other candidates' execution pins remain to be resolved if selected. No tokenizer or model was loaded in this task.

### Does a suitably TOFU-fine-tuned small checkpoint already exist?

The [OpenUnlearning TOFU collection](https://huggingface.co/collections/open-unlearning/tofu-new-models) contains full/retain variants starting with Llama 3.2 1B and larger families. It does not establish a sub-billion full/retain pair. Searches of official publisher pages and targeted TOFU/SmolLM2/Pythia/Qwen references did **not** verify a ready-to-use dense, sub-billion TOFU full/retain pair matching our protocol. This is a bounded search result, not a claim that none exists anywhere.

There is an important positive lead: [Waseem and Fawaz's MRL 2026 paper](https://aclanthology.org/2026.mrl-1.43/) and its [author repository](https://github.com/Waseem177/forgotten-in-msa) describe a **TOFU-trained Qwen2.5-0.5B-Instruct anchor adapter** and advertise `anchor_v1` plus unlearned adapters in the `checkpoints-v4` release. That is an author-reported TOFU fine-tuning artifact, not a Qwen-publisher or OpenUnlearning dense checkpoint. The release asset endpoint could not be retrieved in this review: payload availability, checksums, immutable base pin, exact split provenance and adapter contents remain unverified. Therefore it is **not a validated ready-to-run starting checkpoint here**.

Its reported training/evaluation backbone is 4-bit NF4 with LoRA. The authors explicitly warn that applying the same adapters to an unquantized base changes the measured result. Our Torch/MPS lock has no validated equivalent NF4 training/evaluation path; substituting FP32 would be a new experiment. The artifact could save acquisition work after a separate provenance and backend audit, but does not currently offer a lower-risk route than a small fresh FP32 acquisition experiment. Published results are not empirical evidence for our Mac or proposed settings. This assessment uses the paper's base/anchor/retain-control lesson rather than importing its claimed effects.

Thus, **a small TOFU adapter release is advertised, but an appropriately pinned, accessible, MPS-validated starting checkpoint has not been established**. The recommended experiment includes acquisition rather than pretending the selected base already learned TOFU. If the release is later verified and selected, freeze a different protocol rather than silently substituting it.

## Comparison of feasible experimental approaches

All four approaches below create an explicit acquired-knowledge checkpoint and a matched retain-only reference before unlearning. Each is feasible **on paper** on a 48 GiB Apple Silicon host once the unchanged runtime guards pass; none has a measured MPS training result here. Standard Transformers model classes are a promising integration route, not evidence of successful MPS kernels. Use explicit `mps`, FP32, SDPA where supported, `use_cache=False`, verified non-reentrant checkpointing and no implicit CPU fallback. [PyTorch's MPS documentation](https://docs.pytorch.org/docs/2.9/notes/mps.html) establishes the backend, not compatibility of every model/operation.

| Approach | MPS integration / effort | Experimental validity | Reproducibility | Decision |
|---|---|---|---|---|
| **SmolLM2-135M, controlled TOFU subset, full-parameter acquisition + GradDiff** | Lowest estimated state footprint; Llama-class checkpointing matches the existing tiny fixture family. Still needs a new model pin/count and plain-QA encoder. Moderate implementation work | Most direct small experiment for our loss; requires measured acquisition and utility gates. Weak capacity may prevent meaningful paraphrase recall | Candidate SHA is known; all data selections, schedules and own reference checkpoints can be pinned | **Recommended first experiment**, with acquisition failure treated as a valid negative result |
| **SmolLM2-360M, same experiment** | Same Llama-class integration but more state/workspaces. Similar code effort | Additional capacity might help acquisition, but this is untested; cannot claim superior forgetting | Same deterministic protocol, new revision/count/baselines | Preregistered escalation only if the 135M acquisition stage fails; separate result, not cherry-picked replacement |
| **Pythia-160M, same experiment** | Small state; GPT-NeoX needs a new architecture validator/format/checkpointing audit. Higher effort than SmolLM2 | Strong research provenance and pretraining history, but no assumed TOFU knowledge | Exact total count and pretraining steps are documented; select one immutable final revision | Useful independent replication after the first result |
| **Qwen2.5-0.5B-Instruct, fresh FP32 acquisition then full-parameter GradDiff** | Larger vocabulary/logits and state; requires Qwen2 validator and chat-mask audit. No NF4 dependency for this fresh route | Instruction formatting may help generation, but acquisition still must be measured. Published adapter does not validate fresh dense FP32 training | Official base card/config; own full/retain checkpoints needed. Resolve immutable revision | Higher-memory contingency; author-adapter replication is a distinct, higher-backend-effort alternative |

### Memory estimates, not measurements

Let `P` be the counts above. FP32 weights = `4P` bytes, gradients = `4P`, momentum-free SGD per-parameter optimizer state = 0, AdamW moments = `8P`. Units below are GiB (`2^30` bytes). The model-count metadata is verified/derived as labelled; **every allocation, overhead and peak below is estimated**. Full-parameter training is assumed throughout.

Batch one, maximum sequence length 512, no KV cache, sequential forget/retain graphs and checkpointing are assumed. Workspace allowances include activations, logits/loss, recomputation, loader/allocator uncertainty during normal computation, and backend temporaries. They are generous planning choices, not proven upper bounds. CPU/MPS load copies and checkpoint serialization need separate measurement. One model is resident at a time.

| Backbone | FP32 weights | Gradients | SGD optimizer state | AdamW moments | Persistent SGD / AdamW | Activation/workspace allowance | SGD working-peak plan | AdamW working-peak plan | Host envelope for AdamW (+16 GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M | 0.501 | 0.501 | 0 | 1.002 | 1.002 / 2.004 | 2–4 | 3.00–5.00 | 4.00–6.00 | 20.00–22.00 |
| SmolLM2-360M | 1.348 | 1.348 | 0 | 2.696 | 2.696 / 5.392 | 3–5 | 5.70–7.70 | 8.39–10.39 | 24.39–26.39 |
| Pythia-160M | 0.605 | 0.605 | 0 | 1.209 | 1.209 / 2.419 | 2–4 | 3.21–5.21 | 4.42–6.42 | 20.42–22.42 |
| Qwen2.5-0.5B | 1.840 | 1.840 | 0 | 3.681 | 3.681 / 7.362 | 4–6 | 7.68–9.68 | 11.36–13.36 | 27.36–29.36 |

Host envelopes add the prior assessment's 12 GiB OS/other-process reserve and 4 GiB uncertainty; they are not readings of currently available memory. The selected experiment starts with SGD for both acquisition and unlearning to keep optimizer behavior and the 32 GiB admission policy consistent. Its corresponding host planning envelope is approximately **19–21 GiB**, versus 33.21–37.21 GiB for the existing Llama SGD design. AdamW is shown as an acquisition alternative only; it would retain its 42 GiB admission requirement unless separately reviewed. Smaller arithmetic does not authorize changing either guard.

A single FP32 logits tensor at length 512 is 96 MiB for SmolLM2's vocabulary, versus approximately 297 MiB for Qwen. FP32 loss tensors/copies, saved activations and temporary allocations remain additional. Nominal model names and on-disk BF16/F16 size must not be used as total training-memory estimates. No throughput, wall time or forgetting rate is predicted.

## Recommended controlled experiment

### 1. Freeze the acquisition problem before fitting

Use `locuslab/TOFU` revision **`324592d84ae4f482ac7249b9285c2ecdb53e3a68`**, already licensed and checksum-pinned in [the data documentation](data.md). Proposed training set: all **40 forget01** records plus **120 retain99** records selected by the existing stable SHA-256 ranking algorithm with seed 2026. Reuse file hashes, not the existing Llama manifest/output filenames; write a separately versioned small-model manifest later. The retain pool for this experiment is those 120 rows, not all 3,960. This changes the TOFU protocol and must be labelled prominently.

Before training, freeze record indices, exact QA hashes, aliases/key-fact scoring rules, tokenization, train order, official forget01 paraphrase alignments, and held-out wording. Use the official forget/retain division; do not infer new author IDs from row order or claim the sampled retain rows represent a fixed number of authors. Keep holdout01 out of fitting and model selection. No target facts should appear in an inference prompt; do not supply their biographies as retrieval context.

For the base SmolLM2 model, propose the deterministic format `Question: {question}\nAnswer:\n{answer}` with an answer-ending EOS. This is a new format, not the Llama instruct chat template. Require an exact tokenized prompt-prefix boundary, mask prompt/padding labels to `-100`, right-pad with an explicitly validated existing token ID, and reject truncation or uncertain boundary alignment. The tokenizer has not been loaded here: EOS/pad settings and prefix behavior must be checked rather than inferred. Do not resize embeddings or invent a chat template merely to pass today's validator.

Use a small, disjoint synthetic development fact set to choose and freeze the acquisition learning rate and epoch budget before fitting the target subset. The proposed target cap is **20 epochs / 3,200 single-example acquisition steps** (160 rows × 20); that is a research budget, not a runtime prediction or an implemented command. Start with FP32 full-parameter SGD, zero momentum/weight decay, no scheduler, clipping 1.0, and explicit checkpointing. Learning rate remains a calibration prerequisite: the Llama SGD default must not be assumed suitable for this backbone or for memorization. Preserve original weights in a separate immutable cache and use new ignored outputs for all derived models.

### 2. Establish learned facts before allowing an unlearning claim

Measure the untouched base `M0` without context, then train an acquired-knowledge checkpoint `M_all` on the fixed 160 rows. The acquisition checkpoint is fixed by the preregistered budget, not selected as the best held-out forget result.

Proposed admission gates for the *scientific experiment*: at least **80% frozen key-fact recovery on both forget and retain direct questions**, at least **50 percentage-point gain on forget recovery over M0**, lower answer NLL on both splits, and at least **50% forget recovery on frozen unseen paraphrases**. Record whole-answer normalized token F1 and exact match as well. These are proposed project thresholds, not established model performance or official TOFU standards. Aliases must be specified from the source answers before model outputs are seen.

If M0 already knows too many targets to establish acquisition, or M_all fails the gates, report contamination/acquisition failure and stop. Do not redefine the forget set as only whichever rows happened to be memorized. The 360M model is a separately declared follow-up, not a covert substitute. For a completely controlled fallback, design a new versioned synthetic fact corpus with seeded random entity–attribute assignments and independently written paraphrases, then measure M0 ignorance again. That would be a synthetic-memory study rather than TOFU and must have separate results.

### 3. Create the counterfactual and matched controls

Train `M_R` from the **same original M0 and seed**, using only the 120 retain rows, with the same per-retain-example exposure count and acquisition settings. This creates our own retain-only reference for this backbone/subset. It is exposure-matched, not identical in total step count: at 20 epochs M_all has 3,200 steps and M_R 2,400. Document that difference and constant learning-rate policy. The released Llama retain99 checkpoint/logs cannot serve as this model's counterfactual.

Starting independently from M_all, prepare (a) GradDiff, (b) a retain-only continued-training control with identical retain draws/update count/optimizer and clipping, and (c) the unchanged M_all for a no-op evaluation control. Evaluate one model at a time. Do not chain these controls from each other or keep an extra reference model resident for the NLL GradDiff objective.

### 4. Attempt forgetting only after acquisition and memory evidence

Preserve the validated objective `-gamma*mean_NLL(F) + alpha*mean_NLL(R)` with **gamma=alpha=1**, independent shifted-token means and full-parameter FP32 gradients. Proposed pilot: **one fixed pass over the 40 forget rows**, each paired with a seeded uniform draw from the 120-row retain pool; batch one, no accumulation, clipping 1.0, zero-momentum SGD, and seed 2026. Calibrate and freeze the unlearning learning rate on disjoint development facts first. One pass is a minimal feasibility test, not an expectation of success; no effect must be reported as a negative result. A longer schedule requires a new preregistered protocol, not tuning after test inspection.

Suggested descriptive success gates: at least **0.25 absolute reduction in forget key-fact recovery** on both direct and held-out paraphrase questions compared with M_all; increased forget answer NLL; greater forget-recovery reduction than the retain-only continued-training control; retain recovery and mean token-F1 drops each no greater than **0.05**, with retain NLL worsening no more than **0.1 nats per supervised token**. Report proximity to M_R on forget behavior alongside utility; no single scalar certifies deletion. These thresholds are proposals to freeze before execution, not guarantees of achievable effects.

Include held-out wording for a fixed 40-case retain diagnostic subset, authored and frozen independently of outputs. Check degeneration, refusals, empty answers and unrelated repeated output; reduced recall caused by global collapse is failure. Preserve a small fixed general-text likelihood suite or validated official Real Authors/World Facts inputs to detect broader drift. Report that suite separately, with its data license/pins, rather than relabelling it official Model Utility. Inspect holdout01 only after the final checkpoint is fixed, and never use it to tune.

The two forget author profiles and 40 correlated questions provide limited statistical generality; do not treat all questions as independent subjects or infer broad privacy guarantees. Start with one seed as a feasibility case; preregister additional seeds before claiming a stable result. A different model/template/subset and self-trained M_R do not produce directly comparable official TOFU Forget Quality or Model Utility scores. Any future official-score claim requires exact evaluator compatibility and the appropriate matched reference, not reuse of Llama logs.

## Implementation and safety prerequisites

1. **A separate model profile and acquisition mode:** today's runner pins the Llama repository, expected weight-file size and 1,235,814,400-parameter count. Keep those protections intact. Add an explicitly allowlisted SmolLM2 base/tokenizer revision, its own file checksums/architecture count and validated plain-QA collation. Do not point the Llama runner at a different directory or remove count checks. Implement M0 evaluation, acquisition and retain-only controls before any scientific unlearning claim.
2. **No memory exception:** keep current 32 GiB SGD / 42 GiB AdamW admission thresholds for initial implementations and rehearsals, at least 48 GiB total capacity, the 6 GiB reserve, current pressure/RSS/MPS telemetry and no CPU fallback. Task 20's 16.31 GiB snapshot fails. Reclaim resources manually and take fresh readings; no cache clearing, process killing, allocator-limit changes or cloud launch is authorized. There is no automatic small-model waiver.
3. **Staged validation in a later authorized task:** first metadata/tokenizer-only checks; then tiny fixtures of the actual architecture and masking; then physical read-only loading/loss parity; then backward-only rehearsal at relevant maximum lengths. Keep cache disabled and verify non-reentrant checkpoint activation/RNG handling, finite losses/gradients, all intended parameters trainable and unchanged weights in backward-only mode. Missing MPS support is a stop, not an implicit device switch.
4. **Authorized asset acquisition:** the candidate pin is verified only as metadata. Check access and licenses, freeze tokenizer/config/weight digests and resolve any publisher revisions before explicitly authorizing model download. No model weights or research adapter archive were downloaded here. Existing project-local dependencies may support the architecture, but no packages were installed or upgraded and compatibility is untested.
5. **Scientific freeze and artifact isolation:** select learning rates on development facts; freeze data/aliases, schedules, M_R/control construction, decoding/scoring and pass/fail gates. Store all new outputs under separately named Git-ignored local paths and refuse overwrite/symlink escapes. Log only metadata/indices/hashes/aggregate measurements, not secrets. Do not modify Llama model files, profiles or baseline JSONs. Obtain separate authorization for any real parameter update.

## Provenance, licensing and checks

The four official cards list **Apache-2.0** for their model releases; preserve applicable license and notice files. A LlamaForCausalLM architecture class does not make SmolLM2 the Meta Llama-3.2 checkpoint or transfer the latter's release metadata to it. TOFU's [official dataset card](https://huggingface.co/datasets/locuslab/TOFU) lists MIT; preserve its attribution. Confirm separate licenses/checksums for any third-party adapter or translated dataset before reuse. No adapter-availability assertion in this document substitutes for a payload audit.

Repository inspection found only the existing Llama snapshot in the project cache. Reference baseline SHA-256 values recorded before this task's edits:

- Paired 40/40 baseline: `1473f6abc0c42164ee55dcf27328ab0b72c9e86e12c15764b71800e7462e8a0e`.
- Initial five-case baseline: `f46e928a6b8384f86bf7294ca034b7fa3c386d510991921e49822b0e6ce2aa59`.

Completed offline checks: 39 local Markdown links resolve; all four experiment JSON files parse; architecture-derived counts and the memory arithmetic agree; all 17 tracked source/configuration files match the pre-task commit; and both baseline hashes above are unchanged. Diff whitespace checks pass. No runtime tests, model loading, backward passes or optimizer steps are part of this assessment. The recommendation establishes a concrete controlled experiment to prepare; it does not establish current memory readiness, memorization, forgetting, MPS compatibility, throughput or retained utility.
