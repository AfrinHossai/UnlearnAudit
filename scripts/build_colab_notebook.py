"""Regenerate the output-free Task 35 notebook. No notebook cells are executed."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(value):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(value).strip().splitlines(True)})


def code(value):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": textwrap.dedent(value).strip().splitlines(True)})


markdown("""
    # SmolLM2-135M / TOFU: staged Colab CUDA workflow (Task 35)
    **Prepared, never run. No acquisition, unlearning or results are claimed.**

    Start with **0 — initial environment check** only. Do not use **Run all**.
    Every later action requires typing its exact stage name in a fresh prompt.
    No cell starts a later stage. Training remains blocked by missing reviewed
    calibration evidence and key-fact annotations; null rates are intentional.

    [Full instructions and recovery rules](https://github.com/AfrinHossai/UnlearnGuard-Privacy-Leakage-Research/blob/main/docs/smollm2_colab_workflow.md).
    Opening this notebook does not allocate a GPU. You control runtime connection.
    Check your account's subscription/compute-unit information yourself: Python
    cannot establish whether your session is free. Never accept an upgrade, paid
    compute purchase, GCP/Enterprise runtime or dedicated VM for this workflow.
    GPU availability and duration are not guaranteed.
""")
markdown("""
    ## 0 — Initial environment check (the first safe action)
    Run this cell alone. It reads version/device information; no package install,
    checkout, Drive mount, model weights, optimizer, training or evaluation.
    You may run it on a CPU session first. Missing CUDA means **STOP** before
    model stages, not permission to fall back to CPU.
""")
code("""
    import importlib.metadata
    import json
    import platform
    import shutil
    import subprocess
    import sys

    initial = {"python": platform.python_version(), "platform": platform.platform(),
               "billing_status": "UNKNOWN: verify your own Colab account and resource panel",
               "gpu_available": False}
    for package in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "psutil"):
        try:
            initial[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            initial[package] = "missing"
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                                 "--format=csv,noheader"], capture_output=True, text=True)
        initial["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    try:
        import torch
        initial["torch_cuda_build"] = torch.version.cuda
        initial["gpu_available"] = torch.cuda.is_available()
    except Exception as error:
        initial["torch_import_error"] = type(error).__name__ + ": " + str(error)
    print(json.dumps(initial, indent=2))
    print("STOP HERE. Review the report. No model has been loaded or trained.")
""")
markdown("""
    ## 1 — Configuration and explicit stage helper (definitions only)
    In **Runtime → Change runtime type**, select a GPU only after checking your
    plan/resource details. GPU type is assigned subject to availability. No GPU
    is assumed. The isolated profile supports Linux Python 3.11/3.12 and Torch
    2.9.1+cu126. A changed Colab image may require a separately reviewed profile.

    Set `WORKFLOW_COMMIT` to the full 40-character Git SHA containing this
    workflow (copy from GitHub's commit page). Never use `main` as a run identity.
    Set a new `RUN_ID` for each experiment; reuse it only to recover that run.
    `PROTOCOL` must point to your later, independently reviewed external file.
""")
code("""
    from pathlib import Path
    import os
    import re

    WORKFLOW_COMMIT = ""  # required immutable commit containing Task 35
    RUN_ID = "smollm2-colab-seed2026-001"
    REPO = Path("/content/UnlearnGuard-Privacy-Leakage-Research")
    VENV = Path("/content/unlearnguard-colab-env")
    PYTHON = VENV / "bin/python"
    RUN_DIR = Path("/content/drive/MyDrive/UnlearnGuard-runs") / RUN_ID
    PROTOCOL = Path("/content/drive/MyDrive/UnlearnGuard-protocol/frozen_protocol.json")

    def authorize(name):
        if input(f"Run only '{name}'? Type exactly {name}, or leave blank to skip: ").strip() != name:
            print("Skipped. No action taken.")
            return False
        return True

    def run_stage(name):
        if not authorize(name):
            return
        if not PYTHON.is_file():
            raise RuntimeError("STOP: prepare the isolated environment first")
        resume = False
        if name in ("acquire", "reference", "retain-control", "unlearn"):
            resume = input("Fresh attempt: leave blank. Recover committed steps: type resume: ").strip() == "resume"
        command = [str(PYTHON), "-m", "src.colab_workflow", "--stage", name, "--confirm", name,
                   "--run-dir", str(RUN_DIR)]
        if name == "freeze":
            command += ["--protocol", str(PROTOCOL)]
        if resume:
            command += ["--resume"]
        runtime_env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="2026",
                           TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
        subprocess.run(command, cwd=REPO, env=runtime_env, check=True)
""")
markdown("""
    ## 2 — Checkout only (network; no model or dataset)
    Reconnect: rerun definitions with the **same commit** and **same run ID**,
    then this cell. A different/dirty checkout is rejected, never reset.
""")
code("""
    if authorize("checkout"):
        if not re.fullmatch(r"[0-9a-f]{40}", WORKFLOW_COMMIT):
            raise ValueError("STOP: provide the immutable workflow commit first")
        if not REPO.exists():
            subprocess.run(["git", "clone", "--no-checkout",
                            "https://github.com/AfrinHossai/UnlearnGuard-Privacy-Leakage-Research.git", str(REPO)], check=True)
            subprocess.run(["git", "checkout", "--detach", WORKFLOW_COMMIT], cwd=REPO, check=True)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if actual != WORKFLOW_COMMIT or subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True):
            raise RuntimeError("STOP: checkout differs; inspect it or use a new directory")
        print("Verified workflow commit:", actual)
""")
markdown("""
    ## 3 — Isolated dependencies only (network; no weights)
    Installs CUDA wheels into a fresh venv, preserving Colab's preinstalled
    Torch/vision/audio stack. Do not install `requirements-ml.lock` here.
    No CUDA toolkit/driver or paid service is installed. An incompatible driver
    or unavailable wheel is a stop; do not silently change pins.
""")
code("""
    if authorize("dependencies"):
        if sys.platform != "linux" or sys.version_info[:2] not in ((3, 11), (3, 12)):
            raise RuntimeError("STOP: this profile requires Linux Python 3.11/3.12")
        if not (REPO / "requirements-colab.txt").is_file():
            raise RuntimeError("STOP: checkout the workflow first")
        if not PYTHON.exists():
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "torch==2.9.1",
                        "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(REPO / "requirements-colab.txt")], check=True)
        if (RUN_DIR / "runtime.json").is_file():
            # Recovery: mount Drive before this cell to restore resolved transitive pins.
            recovered_lock = Path("/content/unlearnguard-resume-lock.txt")
            recovered_lock.write_text(json.loads((RUN_DIR / "runtime.json").read_text())["pip_freeze"])
            subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(recovered_lock)], check=True)
        subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
        print("Dependencies installed. Run the separate environment stage next.")
""")
markdown("""
    ## 4 — Strict isolated environment / CUDA verification only
    Checks exact core pins, dependency consistency, driver, GPU visibility,
    architecture import and a tiny CUDA matmul. No weights or optimizer.
    Passing is compatibility evidence only; it does not prove training fit.
""")
code('run_stage("environment")')
markdown("""
    ## 5 — Persistent storage only
    Mount your own Drive. Files remain private in your Drive unless you share
    them. A mount authorizes notebook access to Drive: review Google's prompt.
    Never store secrets in notebook outputs. Runtime disk `/content` is volatile.
    Use one connected writer per run. Check Drive quota separately; filesystem
    free-space readings may not represent your account's quota.
""")
code("""
    if authorize("storage"):
        from google.colab import drive
        drive.mount("/content/drive")
        print("Persistent run location:", RUN_DIR)
""")
markdown("""
    ## 6 — Verify official pinned TOFU data only
    This stage may download five checksum-pinned JSON files. Reproduces the
    entire existing manifest and overlap checks. No weights or scoring.
""")
code('run_stage("data")')
markdown("""
    ## 7 — Freeze reviewed protocol (expected STOP until dependencies exist)
    Prepare an external copy of `experiments/smollm2_135m_colab_protocol.template.json`.
    Missing key-fact aliases and reviewed development-only rate calibration are
    **unresolved dependencies**, not successful work. This task supplies neither
    calibration results nor invented annotations. See the documentation for the
    required evidence schema. Freeze all aliases and diagnostic rules before M0
    outputs; never select rates on TOFU scores or lower gates to force progress.
    An unchanged frozen protocol can be verified again after a disconnect.
""")
code('run_stage("freeze")')

stages = [
    ("8 — Pinned model loading and read-only parity", "load",
     "Explicitly downloads the pinned SmolLM2 weights (~269 MB) only here; verifies file hashes before deserializing. Checks one answer-only loss and saves immutable M0. No backward or updates."),
    ("9 — Base-model pre-acquisition evaluation (M0)", "baseline",
     "Greedy generation and answer-only NLL on the five separate nonsealed sets and utility probes. No training. Direct cases are intentionally training/evaluation reuse for later recall diagnostics."),
    ("10 — Controlled fact acquisition (M_all)", "acquire",
     "Explicit weight updates: fixed 20 epochs, 3,200 SGD steps, batch 1, no accumulation, FP32, max 512 tokens. Frozen calibrated rate only. Does not run evaluation or any later stage."),
    ("11 — Independent retain-only acquisition (M_R)", "reference",
     "Starts from the same immutable M0, never M_all. Fixed 20 epochs / 2,400 steps over 120 retain cases. This is an exposure-matched subset reference, not an official TOFU retain oracle."),
    ("12 — Pre-unlearning evaluation (M_all)", "pre",
     "Same fixed sets and decoding as M0. Acquisition gates compare literal key-fact recovery and NLL with baseline before any branch updates. Failed acquisition stops the experiment; no target tuning."),
    ("13 — Retain reference evaluation (M_R)", "reference-eval",
     "Same fixed diagnostics on independently acquired M_R. No weight updates."),
    ("14 — No-op control", "noop",
     "Reevaluates unchanged M_all; exact summary agreement with pre is required. No weight updates."),
    ("15 — Matched retain-only continuation", "retain-control",
     "Starts independently from fixed M_all. Forty SGD steps using the manifest's exact retain draws and the same unlearning rate/clipping/budget."),
    ("16 — Retain continuation evaluation", "retain-control-eval",
     "Separately evaluates the matched continuation checkpoint. Required before GradDiff."),
    ("17 — GradDiff unlearning branch", "unlearn",
     "Starts independently from M_all. Forty paired steps minimizing -gamma*NLL(forget)+alpha*NLL(retain), gamma=alpha=1. Sequential backwards, separate means, one clipped SGD step per pair. No post evaluation is launched."),
    ("18 — Post-unlearning forget/retain evaluation and comparison", "post",
     "Evaluates the fixed GradDiff checkpoint using identical sets/decoding. Saves separate forget/retain summaries, after-minus-before, distance to M_R and matched continuation comparison. Passing descriptive gates is not proof of erasure or an official benchmark score. Failed gates must be reported."),
]
for title, name, description in stages:
    markdown(f"## {title}\n{description}\n\nRun this cell only when you intend this stage. It never launches a subsequent stage.")
    code(f'run_stage("{name}")')

markdown("""
    ## Recovery and honest completion
    A stage is complete only with its verified `STAGE.complete.json` and hashed
    final artifact. A log line or a `running_not_completed` attempt is not proof.
    Runtime termination may leave a stale running attempt: without a completion
    marker it remains incomplete. For interrupted training rerun this notebook's
    definitions, checkout, dependencies, environment and storage; use the same
    commit, Drive run ID and exact environment. Then run **only the interrupted
    stage**, typing `resume`. Model, optimizer, RNG and next-step cursor recover
    from the newest intact checkpoint; uncommitted steps replay. Changed hardware,
    drivers or packages stop exact resume. Do not relabel a new environment as the
    same run. Evaluations restart from their fixed checkpoint after interruption.

    Two committed checkpoint generations per stage are retained; partial writes
    never count. Every 40 updates is a recovery boundary. Account quota failures,
    missing telemetry, nonfinite loss/gradients/weights, insufficient reserves,
    owner `RUN_DIR/STOP` file or 30-minute stage invocation budget stop progress.
    Keep the original base and completed branch checkpoints. Read the full
    recovery instructions before removing anything. **Do not keep a runtime
    alive with anti-idle scripts.** Nothing in this notebook guarantees free
    resources, memory fit, acquisition or unlearning success.
""")

for index, cell in enumerate(cells):
    cell["id"] = f"task35-{index:03d}"
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"colab": {"name": "smollm2_135m_tofu_colab.ipynb"},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}
if __name__ == "__main__":
    (ROOT / "notebooks/smollm2_135m_tofu_colab.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
