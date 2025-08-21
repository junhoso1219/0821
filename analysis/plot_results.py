from __future__ import annotations
import argparse, os, csv, math, json
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    def f(v):
        try: return float(v)
        except: return float("nan")
    for r in rows:
        for k in list(r.keys()):
            if k in ("epoch","batch","step","trigger"):
                r[k] = float(r[k])
            else:
                r[k] = f(r[k])
    return rows

def confusion_stats(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    TP = int(((y_pred==1)&(y_true==1)).sum())
    TN = int(((y_pred==0)&(y_true==0)).sum())
    FP = int(((y_pred==1)&(y_true==0)).sum())
    FN = int(((y_pred==0)&(y_true==1)).sum())
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    acc  = (TP+TN)/max(1,(TP+TN+FP+FN))
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return dict(TP=TP,TN=TN,FP=FP,FN=FN,precision=prec,recall=rec,accuracy=acc,F1=f1)

def pr_roc_from_c_sweep(z, y_true, c_grid):
    y_true = np.asarray(y_true).astype(int)
    P = y_true.sum(); N = len(y_true) - P
    precs, recs, tprs, fprs = [], [], [], []
    for c in c_grid:
        y_pred = (z <= c).astype(int)
        s = confusion_stats(y_true, y_pred)
        precs.append(s["precision"]); recs.append(s["recall"])
        tpr = s["recall"]
        fpr = (s["FP"] / max(1, N))
        tprs.append(tpr); fprs.append(fpr)
    pr_order = np.argsort(recs)
    roc_order = np.argsort(fprs)
    pr_auc = float(np.trapezoid(np.array(precs)[pr_order], np.array(recs)[pr_order]))
    roc_auc = float(np.trapezoid(np.array(tprs)[roc_order], np.array(fprs)[roc_order]))
    return (np.array(recs), np.array(precs), pr_auc), (np.array(fprs), np.array(tprs), roc_auc)

def _rank01(x: np.ndarray) -> np.ndarray:
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks / (n + 1.0)

def _plot_reliability_rank(outdir: str, score: np.ndarray, y_true: np.ndarray, bins: int = 20) -> None:
    rank = _rank01(score)
    qs = np.linspace(0.0, 1.0, max(2, bins) + 1)
    edges = np.quantile(rank, qs)
    edges = np.unique(edges)
    xs, ys = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (rank >= lo) & (rank < hi) if i < len(edges) - 2 else (rank >= lo) & (rank <= hi)
        if not np.any(m):
            continue
        xs.append(rank[m].mean())
        ys.append(y_true[m].mean())
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    if xs:
        plt.plot(xs, ys, marker="o")
    plt.xlabel("mean rank (0–1)")
    plt.ylabel("empirical P(ΔL_dom ≥ 0)")
    plt.title(f"Reliability (rank), bins={bins}")
    plt.savefig(os.path.join(outdir, "reliability_rank.png"), dpi=160, bbox_inches="tight"); plt.close()

def _plot_prg(outdir: str, recs: np.ndarray, precs: np.ndarray, prevalence: float) -> float:
    eps = 1e-12
    P = np.clip(precs, eps, 1.0)
    R = np.clip(recs, eps, 1.0)
    denom = max(eps, 1.0 - prevalence)
    PG = (P - prevalence) / (denom * P)
    RG = (R - prevalence) / (denom * R)
    m = np.isfinite(PG) & np.isfinite(RG)
    RGm, PGm = RG[m], PG[m]
    auprg = float(np.trapezoid(PGm[np.argsort(RGm)], np.sort(RGm))) if RGm.size > 1 else float("nan")
    plt.figure()
    plt.plot(RGm, PGm, marker=".", linewidth=1)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Recall Gain (RG)")
    plt.ylabel("Precision Gain (PG)")
    plt.title(f"PRG curve (p={prevalence:.3f}, AUPRG≈{auprg:.4f})")
    plt.savefig(os.path.join(outdir, "prg_curve.png"), dpi=160, bbox_inches="tight"); plt.close()
    return auprg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--use_eff", action="store_true", help="Use r_th_eff instead of r_th")
    ap.add_argument("--c_grid", type=str, default="0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0")
    ap.add_argument("--auto_grid", action="store_true", help="Auto-generate threshold grid from z distribution")
    ap.add_argument("--grid_points", type=int, default=200, help="Number of points for auto grid")
    ap.add_argument("--grid_qmin", type=float, default=0.01, help="Lower quantile for auto grid")
    ap.add_argument("--grid_qmax", type=float, default=0.99, help="Upper quantile for auto grid")
    ap.add_argument("--logspace", action="store_true", help="Use logarithmic spacing for auto grid when z>0")
    ap.add_argument("--drop_eos", action="store_true", help="Drop steps in edge-of-stability (lambda_max >= 2/lr)")
    ap.add_argument("--plot_reliability", action="store_true", help="Plot rank-based reliability diagram")
    ap.add_argument("--reliability_bins", type=int, default=20, help="Number of bins for reliability diagram and ECE computation")
    ap.add_argument("--plot_prg", action="store_true", help="Plot Precision-Recall-Gain curve")
    args = ap.parse_args()

    rows = load_metrics(args.metrics)
    outdir = args.outdir or os.path.dirname(args.metrics)
    os.makedirs(outdir, exist_ok=True)

    r      = np.array([r["r"] for r in rows], dtype=float)
    r_th   = np.array([r["r_th_eff"] if args.use_eff and not math.isnan(r["r_th_eff"]) else r["r_th"] for r in rows], dtype=float)
    ddom   = np.array([r["deltaL_dom"] for r in rows], dtype=float)
    lam    = np.array([r["lambda_max"] for r in rows], dtype=float)
    twoolr = np.array([r["two_over_lr"] for r in rows], dtype=float)
    maskA  = np.array([r.get("mask_applicable", 1.0) for r in rows], dtype=float)
    base_mask = np.isfinite(r) & np.isfinite(r_th) & np.isfinite(ddom) & (maskA > 0.5)
    if args.drop_eos:
        eos_mask = np.isfinite(lam) & np.isfinite(twoolr) & (lam < twoolr)
        mask = base_mask & eos_mask
    else:
        mask = base_mask
    r, r_th, ddom = r[mask], r_th[mask], ddom[mask]

    y_pred = (r <= r_th).astype(int)
    y_true = (ddom >= 0.0).astype(int)
    cm = confusion_stats(y_true, y_pred)

    plt.figure(); plt.scatter(r, r_th, s=6, alpha=0.6)
    plt.xlabel("r = ||P_S∇L||^2 / tr(P_S Σ) (EMA)")
    plt.ylabel("r_th" + (" (eff)" if args.use_eff else ""))
    plt.title("SNR vs Threshold")
    plt.savefig(os.path.join(outdir, "scatter_r_vs_rth.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure(); plt.hist(ddom[~np.isnan(ddom)], bins=80)
    plt.xlabel("ΔL_dom"); plt.ylabel("count"); plt.title("Distribution of ΔL_dom")
    plt.savefig(os.path.join(outdir, "hist_deltaL_dom.png"), dpi=160, bbox_inches="tight"); plt.close()

    # Compute score z = r / r_th
    z = r / np.maximum(r_th, 1e-12)
    z_finite = z[np.isfinite(z)]
    # Build threshold grid
    if args.auto_grid and z_finite.size > 0:
        qmin = float(np.nanquantile(z_finite, max(0.0, min(1.0, args.grid_qmin))))
        qmax = float(np.nanquantile(z_finite, max(0.0, min(1.0, args.grid_qmax))))
        if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
            qmin, qmax = float(np.nanmin(z_finite)), float(np.nanmax(z_finite))
        # Fallback to a broad, sane range if the span is too narrow
        if not np.isfinite(qmin) or not np.isfinite(qmax) or (qmax - qmin) <= 1e-6:
            qmin, qmax = 1e-4, 10.0
        # Ensure strictly positive for logspace
        if args.logspace:
            eps = 1e-12
            lo = max(qmin, eps)
            hi = max(qmax, lo * (1.0 + 1e-6))
            c_grid = np.geomspace(lo, hi, num=max(2, args.grid_points))
        else:
            c_grid = np.linspace(qmin, qmax, num=max(2, args.grid_points))
        c_grid = c_grid.tolist()
    else:
        c_grid = [float(x) for x in args.c_grid.split(",")]
    y_true_bin = (ddom >= 0.0).astype(int)
    (recs, precs, pr_auc), (fprs, tprs, roc_auc) = pr_roc_from_c_sweep(z, y_true_bin, c_grid)
    # Prevalence and normalized AUPRC
    prevalence = float(np.mean(y_true_bin)) if y_true_bin.size > 0 else float("nan")
    denom = (1.0 - prevalence) if np.isfinite(prevalence) else float("nan")
    if denom is None or (isinstance(denom, float) and (denom <= 0 or not np.isfinite(denom))):
        normalized_auprc = float("nan")
    else:
        normalized_auprc = float((pr_auc - prevalence) / max(1e-12, denom))

    plt.figure(); plt.plot(recs, precs, marker=".", linewidth=1)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curve (AUC={pr_auc:.3f})"); plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure(); plt.plot(fprs, tprs, marker=".", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR (Recall)"); plt.title(f"ROC curve (AUC={roc_auc:.3f})"); plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure(); plt.axis('off')
    text = f"Confusion Matrix @c=1 ({'eff' if args.use_eff else 'base'})\\n\\n" \
           f"TP={cm['TP']}  FP={cm['FP']}\\nFN={cm['FN']}  TN={cm['TN']}\\n" \
           f"Acc={cm['accuracy']:.3f}  Prec={cm['precision']:.3f}  Rec={cm['recall']:.3f}  F1={cm['F1']:.3f}"
    plt.text(0.05, 0.5, text, fontsize=12)
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=160, bbox_inches="tight"); plt.close()

    # Optional diagnostics
    # score higher => positive; here we take score = -z so that larger is more positive
    score = -z
    if args.plot_reliability:
        _plot_reliability_rank(outdir, score, y_true_bin, bins=max(5, int(args.reliability_bins)))
    auprg = float("nan")
    if args.plot_prg:
        auprg = _plot_prg(outdir, recs, precs, prevalence)

    # Rank-based Expected Calibration Error (ECE)
    def _ece_rank(score_vec: np.ndarray, y_true_vec: np.ndarray, bins: int = 20) -> float:
        if score_vec.size == 0:
            return float("nan")
        rank = _rank01(score_vec)
        qs = np.linspace(0.0, 1.0, max(2, bins) + 1)
        edges = np.quantile(rank, qs)
        edges = np.unique(edges)
        n = rank.size
        total = 0.0
        weight = 0
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (rank >= lo) & (rank < hi) if i < len(edges) - 2 else (rank >= lo) & (rank <= hi)
            cnt = int(np.sum(m))
            if cnt <= 0:
                continue
            mean_rank = float(np.mean(rank[m]))
            mean_pos = float(np.mean(y_true_vec[m]))
            total += (cnt / max(1, n)) * abs(mean_pos - mean_rank)
            weight += cnt
        return float(total) if weight > 0 else float("nan")

    ece_rank = _ece_rank(score, y_true_bin, bins=max(5, int(args.reliability_bins)))

    with open(os.path.join(outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "cm_at_c1": cm,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "prevalence": prevalence,
            "normalized_auprc": normalized_auprc,
            "auprg": auprg,
            "ece_rank": ece_rank,
            "c_grid": c_grid
        }, f, indent=2)

    print(f"Saved plots to {outdir}")
    print(f"CM@c=1: {cm}")
    print(f"PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")

if __name__ == "__main__":
    main()
