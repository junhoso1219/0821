from __future__ import annotations
import torch
from typing import List

def signal_ps_grad_sq(grad_flat: torch.Tensor, V: torch.Tensor) -> float:
    coeffs = V.t() @ grad_flat
    return float((coeffs * coeffs).sum().item())

def noise_trace_ps_sigma(grad_flats: List[torch.Tensor], V: torch.Tensor) -> float:
    if len(grad_flats) < 2:
        raise ValueError("Need at least 2 gradient samples to estimate covariance trace.")
    G = torch.stack(grad_flats, dim=1)           # (dim, M)
    coeffs = V.t() @ G                           # (k, M)
    M = coeffs.shape[1]
    mean = coeffs.mean(dim=1, keepdim=True)
    centered = coeffs - mean
    var = (centered.pow(2).sum(dim=1) / (M - 1.0))
    return float(var.sum().item())

def r_and_threshold(eta: float, mu: float, signal_sq: float, noise_trace: float) -> tuple[float, float]:
    r = signal_sq / max(noise_trace, 1e-12)
    r_th = (eta * mu) / max(1e-12, (2.0 - eta * mu))
    return float(r), float(r_th)
