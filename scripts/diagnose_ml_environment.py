#!/usr/bin/env python3
"""Report PyTorch device support and run a tiny no-download inference smoke test."""

from __future__ import annotations

import json
import os
import platform
import sys

import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel


def _low_level_mps_probe() -> bool | None:
    """Call PyTorch's private runtime probe when this version exposes it."""
    probe = getattr(torch._C, "_mps_is_available", None)
    if probe is None:
        return None
    try:
        return bool(probe())
    except Exception:
        return None


def _mps_device_count() -> int | None:
    try:
        return int(torch.mps.device_count())
    except Exception:
        return None


def _environment_diagnostics() -> dict[str, object]:
    names = (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
    )
    return {name: os.environ.get(name) for name in names if name in os.environ}


def main() -> None:
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    macos_compatible = bool(torch.backends.mps.is_macos_or_newer(15, 0))
    low_level_probe = _low_level_mps_probe()
    mps_count = _mps_device_count()
    device = torch.device("mps" if mps_available else "cpu")

    if mps_available:
        mps_reason = "PyTorch MPS runtime reports an available device."
    elif not mps_built:
        mps_reason = "This PyTorch binary was not built with MPS support."
    elif not macos_compatible:
        mps_reason = "The running macOS version is below PyTorch's MPS minimum."
    elif platform.machine() != "arm64":
        mps_reason = "This Python process is not running as native arm64."
    else:
        mps_reason = (
            "MPS is compiled in and OS/architecture checks pass, but PyTorch's "
            "runtime probe does not detect an MPS device. The public checks do "
            "not identify the lower-level cause."
        )

    torch.manual_seed(0)
    values = torch.arange(1, 17, dtype=torch.float32, device=device).reshape(4, 4)
    product = values @ values.T
    if device.type == "mps":
        torch.mps.synchronize()
    assert product.shape == (4, 4)
    assert bool(torch.isfinite(product).all().item())

    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = GPT2LMHeadModel(config).to(device)
    model.eval()
    input_ids = torch.tensor([[1, 5, 9, 2]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids).logits
    if device.type == "mps":
        torch.mps.synchronize()
    assert logits.shape == (1, 4, 64)
    assert bool(torch.isfinite(logits).all().item())

    report = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_architecture": platform.machine(),
        "macos_version": platform.mac_ver()[0] or None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torch_mps_built": mps_built,
        "torch_mps_available": mps_available,
        "torch_mps_low_level_probe": low_level_probe,
        "torch_mps_device_count": mps_count,
        "torch_mps_macos_15_compatible": macos_compatible,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "mps_limitation": mps_reason,
        "environment_overrides": _environment_diagnostics(),
        "selected_device": str(device),
        "synthetic_tensor_shape": list(product.shape),
        "synthetic_tensor_result": product.tolist(),
        "random_gpt2_config": {
            "vocab_size": config.vocab_size,
            "n_positions": config.n_positions,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
        },
        "random_gpt2_logits_shape": list(logits.shape),
        "random_gpt2_logits_finite": bool(torch.isfinite(logits).all().item()),
        "checkpoint_or_dataset_downloaded": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
