from __future__ import annotations
import torch
from torch import nn

def hvp(loss_fn, model: nn.Module, batch, v: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    loss = loss_fn(model, batch)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=True, retain_graph=True)
    grad_dot = 0.0
    pointer = 0
    for p, g in zip(model.parameters(), grads):
        numel = p.numel()
        v_part = v[pointer:pointer+numel].view_as(p)
        grad_dot = grad_dot + (g * v_part).sum()
        pointer += numel
    Hv_parts = torch.autograd.grad(grad_dot, list(model.parameters()),
                                   retain_graph=False, create_graph=create_graph)
    Hv = torch.cat([h.reshape(-1) for h in Hv_parts])
    return Hv.detach() if not create_graph else Hv
