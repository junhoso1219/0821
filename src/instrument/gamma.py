from __future__ import annotations
import math
import torch
from torch import Tensor


def _project_S(x: Tensor, V: Tensor) -> Tensor:
    return V @ (V.t() @ x)


def _project_B(x: Tensor, V: Tensor) -> Tensor:
    return x - _project_S(x, V)


@torch.no_grad()
def gamma_power(
    hvp_fn,               # function: Tensor[D] -> Tensor[D], returns H @ vec
    V: Tensor,            # [D,k], orthonormal basis of S
    iters: int = 10,
    tol: float = 1e-6,
    device: torch.device | None = None,
):
    """
    Estimate gamma = || P_S H P_B ||_2 via power iteration on
    T = P_S H P_B P_B H P_S. Cost per iter ~ 2 HVPs.

    Returns (gamma, lam, iters_used)
    """
    D, _k = V.shape
    device = device or V.device
    z = torch.randn(D, device=device)
    z = _project_S(z, V)
    z = z / (z.norm() + 1e-12)
    lam_prev = 0.0

    used = 0
    for t in range(max(1, iters)):
        used = t + 1
        s = _project_S(z, V)
        hs = hvp_fn(s)
        pb_hs = _project_B(hs, V)
        h_pb_hs = hvp_fn(pb_hs)
        y = _project_S(h_pb_hs, V)

        lam = float(torch.dot(z, y))
        yn = y.norm() + 1e-12
        z = y / yn
        # update current estimate BEFORE potential break to avoid returning stale value
        if abs(lam - lam_prev) <= tol * max(1.0, abs(lam_prev)):
            lam_prev = lam
            break
        lam_prev = lam

    lam = max(lam_prev, 0.0)
    gamma = lam ** 0.5
    return float(gamma), float(lam), used


@torch.no_grad()
def principal_angle_max(V_ref: Tensor, V_hat: Tensor) -> float:
    """
    Max principal angle ε [radians] between subspaces span(V_ref) and span(V_hat).
    Both V_ref and V_hat should have orthonormal columns.
    """
    # Guard against shape issues
    if V_ref is None or V_hat is None:
        return 0.0
    try:
        M = V_hat.t() @ V_ref
        svals = torch.linalg.svdvals(M)
        s_min = torch.clamp(svals.min(), 0.0, 1.0)
        eps = float(torch.arccos(s_min))
        if not math.isfinite(eps):
            return 0.0
        return eps
    except Exception:
        return 0.0


def mu_eff_gamma_k1(mu_S: float, eps: float, gamma: float) -> float:
    """
    Conservative lower bound (k=1 case):
    v^T H v >= mu_S * cos^2(eps) - gamma * sin(2*eps), clipped at 0.
    """
    val = mu_S * (math.cos(eps) ** 2) - gamma * math.sin(2.0 * eps)
    return max(val, 0.0)


def corrected_threshold(eta: float, mu_eff: float) -> float:
    """
    r_th = (eta * mu_eff) / (2 - eta * mu_eff); if eta*mu_eff>=2 -> +inf, if mu_eff<=0 -> +inf
    """
    if mu_eff <= 0.0:
        return float("inf")
    eta_mu = eta * mu_eff
    if eta_mu >= 2.0:
        return float("inf")
    denom = max(1e-12, 2.0 - eta_mu)
    return float(eta_mu / denom)
