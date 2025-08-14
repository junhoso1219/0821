from __future__ import annotations
import argparse, os, time, math, json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ..datasets.cifar10 import get_cifar10_loaders
from ..models.resnet_cifar import ResNet18CIFAR
from ..utils.seed import set_seed
from ..utils.io import CSVLogger
from ..utils.flatten import grads_to_vector, add_inplace
from ..instrument.hvp import hvp
from ..instrument.lanczos import topk_power
from ..instrument.snr import noise_trace_ps_sigma, r_and_threshold
from ..eos.sharpness import power_max_eig

def cross_entropy_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    x, y = batch
    logits = model(x)
    return nn.functional.cross_entropy(logits, y)

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); n, c = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); pred = logits.argmax(dim=1)
        c += (pred == y).sum().item(); n += y.numel()
    model.train(); return c / max(1, n)

def flat_dim(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _get_grad_flat(model, batch, loss_fn):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch); loss.backward()
    from ..utils.flatten import grads_to_vector
    return grads_to_vector(model.parameters()).detach()

def _clone_params(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])

def _restore_params(model: nn.Module, vec: torch.Tensor) -> None:
    with torch.no_grad():
        pointer = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(vec[pointer:pointer+numel].view_as(p)); pointer += numel

def _delta_loss_after_step(model, loss_fn, batch, delta_vec: torch.Tensor) -> float:
    from ..utils.flatten import add_inplace
    theta0 = _clone_params(model); add_inplace(model, delta_vec, alpha=1.0)
    L1 = float(loss_fn(model, batch).item()); _restore_params(model, theta0)
    L0 = float(loss_fn(model, batch).item()); return L1 - L0

def eval_delta_multi_batch(model, loss_fn, base_batch, delta_vec: torch.Tensor, loader: DataLoader, M: int, device):
    deltas = [ _delta_loss_after_step(model, loss_fn, base_batch, delta_vec) ]
    it = iter(loader)
    for _ in range(max(0, M-1)):
        try: xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        deltas.append(_delta_loss_after_step(model, loss_fn, (xb, yb), delta_vec))
    return float(sum(deltas)/len(deltas))

def compute_best_c_scale(records, c_grid):
    import numpy as np
    arr = np.array(records, dtype=float)  # (N,3): (r, r_th, deltaL_dom)
    r = arr[:,0]; rth = arr[:,1]; ddom = arr[:,2]
    y_true = (ddom >= 0.0).astype(int)
    best = {"c": 1.0, "f1": -1.0, "prec":0.0,"rec":0.0,"tp":0,"fp":0,"fn":0,"tn":0}
    for c in c_grid:
        y_pred = (r <= c * rth).astype(int)
        tp = int(((y_pred==1)&(y_true==1)).sum())
        tn = int(((y_pred==0)&(y_true==0)).sum())
        fp = int(((y_pred==1)&(y_true==0)).sum())
        fn = int(((y_pred==0)&(y_true==1)).sum())
        prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn)
        f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        if f1 > best["f1"]:
            best = {"c":c,"f1":f1,"prec":prec,"rec":rec,"tp":tp,"fp":fp,"fn":fn,"tn":tn}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--eig_freq", type=int, default=200)
    ap.add_argument("--noise_M", type=int, default=8)
    ap.add_argument("--ema", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--logdir", type=str, default="results")
    ap.add_argument("--cpu", action="store_true")
    # New toggles
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--skip_intervene", action="store_true")
    ap.add_argument("--skip_eos", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--dummy_data", action="store_true")
    ap.add_argument("--dummy_size", type=int, default=1024)
    ap.add_argument("--rth_scale", type=float, default=1.0)
    ap.add_argument("--auto_rth_scale", action="store_true")
    ap.add_argument("--cv_warmup_frac", type=float, default=0.0)
    ap.add_argument("--cv_warmup_steps", type=int, default=0)
    ap.add_argument("--c_grid", type=str, default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--eval_M", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = get_cifar10_loaders(args.data, args.batch_size, args.workers,
                                                    aug=True, dummy=args.dummy_data, dummy_size=args.dummy_size, seed=args.seed)
    model = ResNet18CIFAR().to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    run_dir = os.path.join(args.logdir, time.strftime("%Y%m%d-%H%M%S")); os.makedirs(run_dir, exist_ok=True)
    logger = CSVLogger(os.path.join(run_dir, "metrics.csv"),
        fieldnames=["step","epoch","batch","loss","acc","r","r_th","r_th_eff","mu","ps_grad_sq","tr_ps_sigma",
                    "deltaL_dom","deltaL_bulk","deltaL_full","lambda_max","two_over_lr","trigger","cstar"])

    dim = flat_dim(model)
    # init V random orthonormal
    V = torch.randn(dim, args.k, device=device)
    for j in range(args.k):
        v = V[:,j]
        for i in range(j):
            v = v - (V[:,i] @ v) * V[:,i]
        V[:,j] = v / (v.norm() + 1e-12)
    mu = 0.0; step = 0; ema_r = None

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.cv_warmup_steps or int(args.cv_warmup_frac * total_steps)
    auto_records = []; cstar = None
    c_grid = [float(x) for x in args.c_grid.split(",")] if args.c_grid else [1.0]

    def current_rth_scale():
        if args.auto_rth_scale and cstar is not None: return cstar
        return args.rth_scale

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device); batch = (x, y)

            if args.eig_freq > 0 and step % args.eig_freq == 0:
                def apply_H(v_flat: torch.Tensor) -> torch.Tensor:
                    return hvp(cross_entropy_loss, model, batch, v_flat, create_graph=False)
                eigvals, V = topk_power(apply_H, dim=dim, k=args.k, iters=50, tol=1e-3, device=device, seed=args.seed+step)
                mu = float(eigvals[min(args.k-1, len(eigvals)-1)].item())

            # gradient & signal
            model.zero_grad(set_to_none=True)
            loss = cross_entropy_loss(model, batch); loss.backward()
            from ..utils.flatten import grads_to_vector
            grad_flat = grads_to_vector(model.parameters()).detach()
            coeffs = V.t() @ grad_flat; ps_grad_sq = float((coeffs*coeffs).sum().item())

            # noise trace
            grad_samples = [grad_flat]; it = iter(train_loader)
            for _ in range(args.noise_M - 1):
                try: xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader); xb, yb = next(it)
                xb, yb = xb.to(device), yb.to(device)
                g = _get_grad_flat(model, (xb, yb), cross_entropy_loss); grad_samples.append(g)
            tr_ps_sigma = float(noise_trace_ps_sigma(grad_samples, V))

            r, r_th = r_and_threshold(args.lr, mu, ps_grad_sq, tr_ps_sigma)
            if ema_r is None: ema_r = r
            else: ema_r = args.ema * ema_r + (1 - args.ema) * r

            rth_scale_used = current_rth_scale()
            r_th_eff = rth_scale_used * r_th
            trigger = int(ema_r <= r_th_eff and (args.eig_freq<=0 or args.lr * mu < 2.0))

            # virtual ΔL (multi-batch average)
            if not args.skip_intervene:
                model.zero_grad(set_to_none=True)
                loss_now = cross_entropy_loss(model, batch); loss_now.backward()
                g_cur = grads_to_vector(model.parameters()).detach()
                P_S = lambda x: V @ (V.t() @ x); P_B = lambda x: x - V @ (V.t() @ x)
                delta_dom  = -args.lr * P_S(g_cur)
                delta_bulk = -args.lr * P_B(g_cur)
                delta_full = -args.lr * g_cur
                evalM = max(1, args.eval_M)
                deltaL_dom  = eval_delta_multi_batch(model, cross_entropy_loss, batch, delta_dom,  train_loader, evalM, device)
                deltaL_bulk = eval_delta_multi_batch(model, cross_entropy_loss, batch, delta_bulk, train_loader, evalM, device)
                deltaL_full = eval_delta_multi_batch(model, cross_entropy_loss, batch, delta_full,  train_loader, evalM, device)
            else:
                deltaL_dom = float("nan"); deltaL_bulk = float("nan"); deltaL_full = float("nan")

            # warmup for c*
            if args.auto_rth_scale and step < warmup_steps and not math.isnan(deltaL_dom):
                auto_records.append((ema_r, r_th, deltaL_dom))
                if step + 1 == warmup_steps and len(auto_records) >= 10:
                    best = compute_best_c_scale(auto_records, c_grid)
                    cstar = best["c"]
                    with open(os.path.join(run_dir, "cstar.json"), "w", encoding="utf-8") as f:
                        json.dump(best, f, indent=2)
                    print(f"[Auto r_th scale] selected c*={cstar:.3g} with F1={best['f1']:.3f}")

            # EoS
            if not args.skip_eos and args.eig_freq>0:
                lam_max, _ = power_max_eig(lambda v: hvp(cross_entropy_loss, model, batch, v),
                                           dim=dim, iters=30, tol=1e-3, device=device, seed=args.seed+42)
                two_over_lr = 2.0 / args.lr
            else:
                lam_max, two_over_lr = float("nan"), float("nan")

            logger.log({
                "step": step, "epoch": epoch, "batch": batch_idx,
                "loss": float(loss.item()), "acc": float(-1.0),
                "r": float(ema_r), "r_th": float(r_th), "r_th_eff": float(r_th_eff), "mu": float(mu),
                "ps_grad_sq": float(ps_grad_sq), "tr_ps_sigma": float(tr_ps_sigma),
                "deltaL_dom": float(deltaL_dom), "deltaL_bulk": float(deltaL_bulk), "deltaL_full": float(deltaL_full),
                "lambda_max": float(lam_max), "two_over_lr": float(two_over_lr), "trigger": int(trigger),
                "cstar": float(cstar if cstar is not None else float("nan")),
            })

            # train step
            opt.zero_grad(set_to_none=True)
            loss = cross_entropy_loss(model, batch); loss.backward(); opt.step()

            step += 1
            if args.max_steps > 0 and step >= args.max_steps: break

        if not args.skip_eval:
            acc = accuracy(model, test_loader, device)
            print(f"[epoch {epoch}] test acc={acc:.4f}  (logs: {run_dir})")
        if args.max_steps > 0 and step >= args.max_steps: break

if __name__ == "__main__":
    main()
