from __future__ import annotations
import torch

def topk_power(apply_H, dim: int, k: int = 5, iters: int = 50, tol: float = 1e-5,
               device: str | torch.device = "cpu", seed: int | None = 42):
    g = torch.Generator(device=device)
    if seed is not None: g.manual_seed(seed)
    V = []; eigvals = []
    for j in range(k):
        v = torch.randn(dim, generator=g, device=device)
        if V:
            P = torch.stack(V, dim=1); v = v - P @ (P.t() @ v)
        v = v / (v.norm() + 1e-12)
        last_val = None
        for _ in range(iters):
            w = apply_H(v)
            if V: P = torch.stack(V, dim=1); w = w - P @ (P.t() @ w)
            lam = (v @ w).item()
            nrm = w.norm() + 1e-12
            v_next = w / nrm
            if last_val is not None and abs(lam - last_val) < tol * max(1.0, abs(lam)): break
            v = v_next; last_val = lam
        w = apply_H(v)
        if V: P = torch.stack(V, dim=1); w = w - P @ (P.t() @ w)
        lam = (v @ w).item()
        V.append(v / (v.norm() + 1e-12)); eigvals.append(lam)
    V = torch.stack(V, dim=1)
    return torch.tensor(eigvals, device=device), V
