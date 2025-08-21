from __future__ import annotations
import argparse, os, csv, math
import numpy as np

def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def ffloat(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def pr_auc_from_scores(score: np.ndarray, y: np.ndarray, points: int = 600):
    m = np.isfinite(score) & np.isfinite(y)
    score = score[m]; y = y[m].astype(int)
    if score.size == 0:
        return float("nan"), float("nan")
    lo = float(np.nanquantile(score, 0.001)); hi = float(np.nanquantile(score, 0.999))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(score)), float(np.nanmax(score))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    grid = np.linspace(lo, hi, num=max(2, points))
    precs, recs = [], []
    for t in grid:
        yhat = (score >= t).astype(int)  # higher -> positive
        TP = int(((yhat==1)&(y==1)).sum()); FP = int(((yhat==1)&(y==0)).sum())
        FN = int(((yhat==0)&(y==1)).sum())
        prec = TP / max(1, TP+FP); rec = TP / max(1, TP+FN)
        precs.append(prec); recs.append(rec)
    order = np.argsort(recs)
    pr_auc = float(np.trapezoid(np.array(precs)[order], np.array(recs)[order]))
    p = float(np.mean(y)) if y.size>0 else float("nan")
    norm = float((pr_auc - p) / max(1e-12, 1.0 - p)) if np.isfinite(p) else float("nan")
    return pr_auc, norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="holdout/metrics.csv")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--use_margin", action="store_true", help="Use (lambda_max - two_over_lr) instead of lambda_max")
    args = ap.parse_args()

    rows = load_rows(args.metrics)
    lam = np.array([ffloat(r.get("lambda_max","nan")) for r in rows], dtype=float)
    two = np.array([ffloat(r.get("two_over_lr","nan")) for r in rows], dtype=float)
    ddom = np.array([ffloat(r.get("deltaL_dom","nan")) for r in rows], dtype=float)
    y = (ddom >= 0.0).astype(int)
    score = lam - two if args.use_margin else lam
    pr, norm = pr_auc_from_scores(score, y)

    outdir = os.path.dirname(args.metrics)
    outp = args.out or os.path.join(outdir, "sam_baseline_proxy.md")
    with open(outp, "w", encoding="utf-8") as f:
        f.write("### SAM 1-step proxy baseline (hold-out)\n\n")
        f.write(f"- score: {'lambda_max - 2/lr' if args.use_margin else 'lambda_max'}\n")
        f.write(f"- AUPRC: {pr:.4f}\n\n")
        f.write(f"- normAUPRC: {norm:.4f}\n")
    print("WROTE", outp)

if __name__ == "__main__":
    main()


