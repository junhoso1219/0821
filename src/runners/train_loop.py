from __future__ import annotations
import argparse, torch
from torch import nn
from ..utils.seed import set_seed
from ..intervene.dom_sgd import dom_sgd_step, bulk_sgd_step
from ..instrument.snr import signal_ps_grad_sq, noise_trace_ps_sigma, r_and_threshold

class Quadratic(nn.Module):
    def __init__(self, mu=100.0, lam=1.0, bias=0.0, dim_s=1, dim_b=1, device="cpu"):
        super().__init__()
        self.dim_s, self.dim_b = dim_s, dim_b
        self.theta = nn.Parameter(torch.zeros(dim_s + dim_b, device=device))
        self.mu, self.lam, self.bias = mu, lam, bias
        self.device = device
    def forward(self, x):
        s = self.theta[:self.dim_s]; b = self.theta[self.dim_s:]
        loss = 0.5*self.mu*(s@s) + 0.5*self.lam*(b@b) + self.bias * b.sum()
        return loss
def quadratic_loss(model, batch): return model(batch)

def run_demo_quadratic(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    dim_s, dim_b = 1, 1
    model = Quadratic(mu=args.mu, lam=args.lam, bias=args.bias, dim_s=dim_s, dim_b=dim_b, device=device)
    model.theta.data[:] = torch.tensor([args.s0, args.b0], device=device)
    dim = dim_s + dim_b
    V = torch.zeros(dim, 1, device=device); V[0,0] = 1.0
    lr = args.lr
    def grad_sample():
        g_true = torch.tensor([args.mu*model.theta[0].item(), args.lam*model.theta[1].item()+args.bias], device=device)
        g_noise = torch.randn_like(g_true) * torch.tensor([args.s_noise, args.b_noise], device=device)
        return g_true + g_noise
    grads = [grad_sample() for _ in range(args.M)]
    signal = signal_ps_grad_sq(grad_sample(), V)
    noise_tr = noise_trace_ps_sigma(grads, V)
    r, r_th = r_and_threshold(lr, args.mu, signal, noise_tr)
    batch = None
    L1_dom, L0_dom = dom_sgd_step(model, quadratic_loss, batch, V, lr)
    dom_delta = L1_dom - L0_dom
    model.theta.data[:] = torch.tensor([args.s0, args.b0], device=device)
    L1_bulk, L0_bulk = bulk_sgd_step(model, quadratic_loss, batch, V, lr)
    bulk_delta = L1_bulk - L0_bulk
    print(f"[Quadratic demo] r={r:.4g}, r_th={r_th:.4g}, ΔL_dom={dom_delta:.4g}, ΔL_bulk={bulk_delta:.4g}")
    print("Expectation: if r<=r_th, Dom‑SGD non‑descent; Bulk typically descends when bias drives B.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--demo_quadratic", type=int, default=0)
    ap.add_argument("--mu", type=float, default=100.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--s0", type=float, default=0.01)
    ap.add_argument("--b0", type=float, default=0.0)
    ap.add_argument("--s_noise", type=float, default=1.0)
    ap.add_argument("--b_noise", type=float, default=0.05)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()
    if args.demo_quadratic:
        return run_demo_quadratic(args)
    print(">> Use `src/runners/train_cifar.py` for CIFAR‑10/ResNet‑18 demo.")
    return 0
if __name__ == "__main__":
    main()
