from __future__ import annotations
import torch
def power_max_eig(apply_H, dim: int, iters: int = 50, tol: float = 1e-5, device="cpu", seed: int = 0):
    g = torch.Generator(device=device); g.manual_seed(seed)
    v = torch.randn(dim, generator=g, device=device); v = v / (v.norm() + 1e-12)
    last = None
    for _ in range(iters):
        w = apply_H(v); lam = (v @ w).item()
        v = w / (w.norm() + 1e-12)
        if last is not None and abs(lam - last) < tol * max(1.0, abs(lam)): break
        last = lam
    lam = (v @ apply_H(v)).item()
    return float(lam), v
