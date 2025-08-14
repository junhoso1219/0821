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
    pr_auc = float(np.trapz(np.array(precs)[pr_order], np.array(recs)[pr_order]))
    roc_auc = float(np.trapz(np.array(tprs)[roc_order], np.array(fprs)[roc_order]))
    return (np.array(recs), np.array(precs), pr_auc), (np.array(fprs), np.array(tprs), roc_auc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--use_eff", action="store_true", help="Use r_th_eff instead of r_th")
    ap.add_argument("--c_grid", type=str, default="0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0")
    args = ap.parse_args()

    rows = load_metrics(args.metrics)
    outdir = args.outdir or os.path.dirname(args.metrics)
    os.makedirs(outdir, exist_ok=True)

    r      = np.array([r["r"] for r in rows], dtype=float)
    r_th   = np.array([r["r_th_eff"] if args.use_eff and not math.isnan(r["r_th_eff"]) else r["r_th"] for r in rows], dtype=float)
    ddom   = np.array([r["deltaL_dom"] for r in rows], dtype=float)
    mask = (~np.isnan(r)) & (~np.isnan(r_th)) & (~np.isnan(ddom))
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

    c_grid = [float(x) for x in args.c_grid.split(",")]
    z = r / np.maximum(r_th, 1e-12)
    (recs, precs, pr_auc), (fprs, tprs, roc_auc) = pr_roc_from_c_sweep(z, (ddom >= 0.0).astype(int), c_grid)

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

    with open(os.path.join(outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"cm_at_c1": cm, "pr_auc": pr_auc, "roc_auc": roc_auc, "c_grid": c_grid}, f, indent=2)

    print(f"Saved plots to {outdir}")
    print(f"CM@c=1: {cm}")
    print(f"PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")

if __name__ == "__main__":
    main()
