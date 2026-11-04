"""Ordinary, unmerged PyTorch LoRA. No loader, optimizer, network or PEFT dependency.

Imported only by explicit execution or synthetic diagnostics, never the plan CLI.
Native artifact keys are original module paths plus lora_A / lora_B.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def target_shapes():
    return {f"model.layers.{i}.self_attn.{target}": (output, 576)
            for i in range(30) for target, output in (("q_proj", 576), ("v_proj", 192))}


class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) B A x; random A, zero B; never merges W0."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: int = 8):
        super().__init__()
        if (type(base) is not nn.Linear or base.bias is not None
                or base.weight.dtype != torch.float32 or base.weight.device.type != "cpu"
                or type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1):
            raise ValueError("LoRA initialization requires a bias-free CPU FP32 Linear")
        if base.weight.grad is not None:
            raise ValueError("preexisting base gradient")
        self.base_layer = base.requires_grad_(False)
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base_layer(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def inject_qv(model):
    """Prevalidate all 60 exact paths before mutating a single module."""
    expected = target_shapes()
    observed = {name: module for name, module in model.named_modules()
                if name.endswith((".q_proj", ".v_proj"))}
    if set(observed) != set(expected):
        raise RuntimeError("LoRA target-module coverage mismatch")
    for name, shape in expected.items():
        module = observed[name]
        if (type(module) is not nn.Linear or tuple(module.weight.shape) != shape
                or module.bias is not None or module.weight.dtype != torch.float32
                or module.weight.device.type != "cpu" or module.weight.grad is not None):
            raise RuntimeError(f"unsupported target: {name}")
    model.requires_grad_(False)
    wrappers = {}
    for name in expected:
        wrapper = LoRALinear(observed[name])
        wrapper.train(model.training)
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, wrapper)
        wrappers[name] = wrapper
    return wrappers


def adapter_parameters(wrappers):
    return {f"{name}.{suffix}": getattr(module, suffix)
            for name, module in sorted(wrappers.items()) for suffix in ("lora_A", "lora_B")}
