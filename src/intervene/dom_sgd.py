from __future__ import annotations
import torch
from ..utils.flatten import grads_to_vector, add_inplace

def dom_sgd_step(model, loss_fn, batch, V, lr: float):
    model.zero_grad(set_to_none=True)
    L0 = float(loss_fn(model, batch).item())
    loss_fn(model, batch).backward()
    g_flat = grads_to_vector(model.parameters())
    P_S = lambda x: V @ (V.t() @ x)
    delta = -lr * P_S(g_flat)
    add_inplace(model, delta, alpha=1.0)
    L1 = float(loss_fn(model, batch).item())
    return L1, L0

def bulk_sgd_step(model, loss_fn, batch, V, lr: float):
    model.zero_grad(set_to_none=True)
    L0 = float(loss_fn(model, batch).item())
    loss_fn(model, batch).backward()
    g_flat = grads_to_vector(model.parameters())
    P_B = lambda x: x - V @ (V.t() @ x)
    delta = -lr * P_B(g_flat)
    add_inplace(model, delta, alpha=1.0)
    L1 = float(loss_fn(model, batch).item())
    return L1, L0
