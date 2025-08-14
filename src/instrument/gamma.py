from __future__ import annotations
import torch
from .subspace import projector
def estimate_gamma(apply_H, V: torch.Tensor, iters: int = 50, tol: float = 1e-5) -> float:
    device = V.device; dim = V.shape[0]
    P_S = projector(V); P_B = lambda x: x - P_S(x)
    x = P_S(torch.randn(dim, device=device)); x = x / (x.norm() + 1e-12)
    last_val = None
    for _ in range(iters):
        y = P_S(apply_H(P_B(apply_H(P_S(x)))))
        lam = (x @ y).item()
        nrm = y.norm() + 1e-12; x = y / nrm
        if last_val is not None and abs(lam - last_val) < tol * max(1.0, abs(lam)): break
        last_val = lam
    lam = max(0.0, (x @ P_S(apply_H(P_B(apply_H(P_S(x)))))).item())
    gamma = (lam ** 0.5)
    return float(gamma)
