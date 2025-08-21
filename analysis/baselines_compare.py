from __future__ import annotations
import argparse, os, csv, json, math
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

def pr_auc_from_scores(score: np.ndarray, y: np.ndarray):
    # Build threshold grid over score quantiles
    m = np.isfinite(score) & np.isfinite(y)
    score = score[m]; y = y[m].astype(int)
    if score.size == 0:
        return float("nan"), float("nan"), (np.array([]), np.array([]))
    lo = float(np.nanquantile(score, 0.001)); hi = float(np.nanquantile(score, 0.999))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(score)), float(np.nanmax(score))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    grid = np.linspace(lo, hi, num=600)
    precs, recs = [], []
    for t in grid:
        yhat = (score >= t).astype(int)  # higher score => more positive
        TP = int(((yhat==1)&(y==1)).sum()); FP = int(((yhat==1)&(y==0)).sum())
        FN = int(((yhat==0)&(y==1)).sum())
        prec = TP / max(1, TP+FP); rec = TP / max(1, TP+FN)
        precs.append(prec); recs.append(rec)
    order = np.argsort(recs)
    pr_auc = float(np.trapezoid(np.array(precs)[order], np.array(recs)[order]))
    p = float(np.mean(y)) if y.size>0 else float("nan")
    norm = float((pr_auc - p) / max(1e-12, 1.0 - p)) if np.isfinite(p) else float("nan")
    return pr_auc, norm, (np.array(recs), np.array(precs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="metrics.csv (holdout 권장)")
    ap.add_argument("--out", type=str, default=None, help="결과 md 경로 (기본: metrics와 같은 디렉토리)")
    args = ap.parse_args()

    rows = load_rows(args.metrics)
    r = np.array([ffloat(r["r"]) for r in rows], dtype=float)
    rth = np.array([ffloat(r.get("r_th_eff", r.get("r_th", "nan"))) for r in rows], dtype=float)
    ddom = np.array([ffloat(r["deltaL_dom"]) for r in rows], dtype=float)
    lam = np.array([ffloat(r.get("lambda_max","nan")) for r in rows], dtype=float)
    two = np.array([ffloat(r.get("two_over_lr","nan")) for r in rows], dtype=float)
    psg = np.array([ffloat(r.get("ps_grad_sq","nan")) for r in rows], dtype=float)
    g2  = np.array([ffloat(r.get("grad_norm_sq","nan")) for r in rows], dtype=float)
    trS = np.array([ffloat(r.get("tr_sigma_full","nan")) for r in rows], dtype=float)
    dfull = np.array([ffloat(r.get("deltaL_full","nan")) for r in rows], dtype=float)

    y = (ddom >= 0.0).astype(int)
    mask = np.isfinite(r) & np.isfinite(rth) & np.isfinite(ddom)
    # Scores
    z = r / np.maximum(rth, 1e-12)
    score_z = -z  # smaller z => positive
    score_eos = (lam - two)  # closer/over EoS => positive
    score_psg = -psg  # larger PS grad -> negative evidence
    score_r = -r  # smaller r => positive
    score_lam = lam  # sharpness proxy
    score_g   = -g2  # larger |g| -> negative evidence
    score_rfull = -(g2 / np.maximum(trS, 1e-12))  # smaller r_full => positive
    score_sam1 = dfull  # SAM 1-step sharpness proxy: loss increment after one full step

    metrics = []
    for name, score in (
        ("z_eff", score_z),
        ("eos_margin", score_eos),
        ("ps_grad", score_psg),
        ("r_only", score_r),
        ("lam_max", score_lam),
        ("grad_norm", score_g),
        ("r_full", score_rfull),
        ("sam1_full", score_sam1),
    ):
        pr, norm, _ = pr_auc_from_scores(score[mask], y[mask])
        metrics.append((name, pr, norm))

    outdir = os.path.dirname(args.metrics)
    outp = args.out or os.path.join(outdir, "baselines_compare.md")
    lines = []
    lines.append("### Baselines comparison (hold-out)")
    lines.append("")
    lines.append("| score | AUPRC | normAUPRC |")
    lines.append("|---|---:|---:|")
    for name, pr, norm in metrics:
        lines.append(f"| {name} | {pr:.4f} | {norm:.4f} |")
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("WROTE", outp)

if __name__ == "__main__":
    main()


