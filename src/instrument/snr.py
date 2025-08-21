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

def r_and_threshold(eta: float, mu: float, signal_sq: float, noise_trace: float) -> tuple[float, float, int]:
    """
    Returns (r, r_th, mask_applicable)
    - r = signal_sq / tr(P_S Sigma)
    - mu is clipped at 0.0 for theoretical consistency
    - if mu_clip <= 0 or eta*mu_clip >= 2: theorem not applicable -> r_th = +inf, mask = 0
    - else r_th = (eta * mu_clip) / (2 - eta * mu_clip), mask = 1
    """
    r = signal_sq / max(noise_trace, 1e-12)
    mu_clip = max(0.0, float(mu))
    eta_mu = eta * mu_clip
    if mu_clip <= 0.0 or eta_mu >= 2.0:
        return float(r), float("inf"), 0
    r_th = eta_mu / max(1e-12, (2.0 - eta_mu))
    return float(r), float(r_th), 1
