from __future__ import annotations
import argparse, os, csv, math
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def ffloat(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def pr_auc_norm_from_subset(z: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if z.size == 0:
        return float("nan"), float("nan")
    # Build c grid from z subset
    zf = z[np.isfinite(z)]
    if zf.size == 0:
        return float("nan"), float("nan")
    qmin = np.nanquantile(zf, 0.001)
    qmax = np.nanquantile(zf, 0.999)
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
        qmin, qmax = float(np.nanmin(zf)), float(np.nanmax(zf))
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
        qmin, qmax = 1e-4, 10.0
    c_grid = np.geomspace(max(1e-8, qmin), max(qmin * (1 + 1e-6), qmax), num=200)
    precs, recs = [], []
    yb = y.astype(int)
    for c in c_grid:
        yhat = (z <= c).astype(int)
        TP = int(((yhat == 1) & (yb == 1)).sum()); FP = int(((yhat == 1) & (yb == 0)).sum())
        FN = int(((yhat == 0) & (yb == 1)).sum())
        prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
        precs.append(prec); recs.append(rec)
    order = np.argsort(recs)
    pr_auc = float(np.trapezoid(np.array(precs)[order], np.array(recs)[order]))
    p = float(np.mean(yb)) if yb.size > 0 else float("nan")
    norm = float((pr_auc - p) / max(1e-12, 1.0 - p)) if np.isfinite(p) else float("nan")
    return pr_auc, norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--bins_x", type=int, default=6)
    ap.add_argument("--bins_y", type=int, default=6)
    ap.add_argument("--var_x", type=str, default="gap_mu", help="gap_mu or eps_current")
    ap.add_argument("--var_y", type=str, default="eps_current", help="eps_current or gap_mu")
    args = ap.parse_args()

    rows = load_rows(args.metrics)
    outdir = args.outdir or os.path.dirname(args.metrics)
    os.makedirs(outdir, exist_ok=True)

    r = np.array([ffloat(r.get("r", "nan")) for r in rows], dtype=float)
    r_th = np.array([ffloat(r.get("r_th_eff", r.get("r_th", "nan"))) for r in rows], dtype=float)
    mu = np.array([ffloat(r.get("mu", "nan")) for r in rows], dtype=float)
    eps_cur = np.array([ffloat(r.get("eps_current", "nan")) for r in rows], dtype=float)
    lam = np.array([ffloat(r.get("lambda_max", "nan")) for r in rows], dtype=float)
    two = np.array([ffloat(r.get("two_over_lr", "nan")) for r in rows], dtype=float)
    ddom = np.array([ffloat(r.get("deltaL_dom", "nan")) for r in rows], dtype=float)

    base_mask = np.isfinite(r) & np.isfinite(r_th) & np.isfinite(ddom)
    mask_eos = np.isfinite(lam) & np.isfinite(two) & (lam < two)
    mask = base_mask & mask_eos
    r, r_th, mu, eps_cur, lam, two, ddom = r[mask], r_th[mask], mu[mask], eps_cur[mask], lam[mask], two[mask], ddom[mask]

    z = r / np.maximum(r_th, 1e-12)
    y = (ddom >= 0.0).astype(int)

    gap_mu = (two - lam) * np.maximum(mu, 0.0)
    if args.var_x == "gap_mu":
        X = gap_mu
    else:
        X = eps_cur
    if args.var_y == "gap_mu":
        Y = gap_mu
    else:
        Y = eps_cur

    # Define bins by quantiles for robust partitioning
    def qbins(v: np.ndarray, nb: int) -> np.ndarray:
        vfin = v[np.isfinite(v)]
        qs = np.linspace(0.0, 1.0, nb + 1)
        edges = np.quantile(vfin, qs) if vfin.size > 0 else np.linspace(0, 1, nb + 1)
        return np.unique(edges)

    edges_x = qbins(X, max(2, int(args.bins_x)))
    edges_y = qbins(Y, max(2, int(args.bins_y)))

    heat = np.full((len(edges_y) - 1, len(edges_x) - 1), np.nan, dtype=float)
    for iy in range(len(edges_y) - 1):
        ylo, yhi = edges_y[iy], edges_y[iy + 1]
        my = (Y >= ylo) & (Y < yhi) if iy < len(edges_y) - 2 else (Y >= ylo) & (Y <= yhi)
        for ix in range(len(edges_x) - 1):
            xlo, xhi = edges_x[ix], edges_x[ix + 1]
            mx = (X >= xlo) & (X < xhi) if ix < len(edges_x) - 2 else (X >= xlo) & (X <= xhi)
            m = my & mx
            if np.any(m):
                _, norm = pr_auc_norm_from_subset(z[m], y[m])
                heat[iy, ix] = norm

    plt.figure(figsize=(6, 5))
    im = plt.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, label="normAUPRC")
    plt.xlabel(args.var_x)
    plt.ylabel(args.var_y)
    plt.title("Conditional normAUPRC heatmap")
    out_png = os.path.join(outdir, f"conditional_heatmap_{args.var_x}_vs_{args.var_y}.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()

    out_md = os.path.join(outdir, f"conditional_heatmap_{args.var_x}_vs_{args.var_y}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("### Conditional normAUPRC heatmap\n\n")
        f.write(f"- X: {args.var_x}, Y: {args.var_y}\n\n")
        f.write(f"![heatmap]({out_png})\n")
    print("WROTE", out_png)
    print("WROTE", out_md)

if __name__ == "__main__":
    main()


