from __future__ import annotations
import torch

def projector(V: torch.Tensor):
    def P(x: torch.Tensor) -> torch.Tensor:
        return V @ (V.t() @ x)
    return P

def projector_bulk(V: torch.Tensor, dim: int):
    def P_B(x: torch.Tensor) -> torch.Tensor:
        return x - V @ (V.t() @ x)
    return P_B

def mix_operator(V: torch.Tensor, dim: int, alpha: float):
    P_S = projector(V); P_B = projector_bulk(V, dim)
    def P_alpha(x: torch.Tensor) -> torch.Tensor:
        return alpha * P_S(x) + (1 - alpha) * P_B(x)
    return P_alpha
