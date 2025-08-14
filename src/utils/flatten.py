from __future__ import annotations
import torch
from torch import nn
from typing import Iterable

def parameters_to_vector(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])

def vector_to_parameters(vec: torch.Tensor, params: Iterable[torch.Tensor]) -> None:
    pointer = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(vec[pointer:pointer+numel].view_as(p))
        pointer += numel

def grads_to_vector(params: Iterable[torch.Tensor]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.reshape(-1))
    return torch.cat(grads)

def add_inplace(model: nn.Module, delta: torch.Tensor, alpha: float = 1.0) -> None:
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.add_(alpha * delta[pointer:pointer+numel].view_as(p))
        pointer += numel
