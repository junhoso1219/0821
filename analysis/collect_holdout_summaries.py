from __future__ import annotations
import os, re, json, csv, argparse
from typing import Dict, List, Tuple

RESULTS_DIR = "results"

def read_meta(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return meta

def list_result_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        return []
    dirs = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p) and re.match(r"^20\d{6}-\d{6}$", name):
            dirs.append(p)
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs

def load_summary(run_dir: str) -> Dict[str, float] | None:
    hold = os.path.join(run_dir, "holdout")
    summ = os.path.join(hold, "metrics_summary.json")
    if not os.path.isfile(summ):
        return None
    try:
        with open(summ, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "pr_auc": float(data.get("pr_auc", float("nan"))),
            "roc_auc": float(data.get("roc_auc", float("nan"))),
            "prevalence": float(data.get("prevalence", float("nan"))),
            "normalized_auprc": float(data.get("normalized_auprc", float("nan"))),
            "auprg": float(data.get("auprg", float("nan"))),
            "ece_rank": float(data.get("ece_rank", float("nan")))
        }
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default=os.path.join(RESULTS_DIR, "holdout_summary.csv"))
    ap.add_argument("--out_md", type=str, default=os.path.join(RESULTS_DIR, "holdout_summary.md"))
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    for d in list_result_dirs(RESULTS_DIR):
        meta = read_meta(os.path.join(d, "meta.txt"))
        summ = load_summary(d)
        if summ is None:
            continue
        row = {
            "run_dir": d,
            "seed": meta.get("seed", ""),
            "variant": meta.get("variant", ""),
            "k": meta.get("k", ""),
            "gamma_iters": meta.get("gamma_iters", ""),
            "gamma_freq": meta.get("gamma_freq", ""),
            "pr_auc": f"{summ['pr_auc']:.4f}",
            "roc_auc": f"{summ['roc_auc']:.4f}",
            "prevalence": f"{summ['prevalence']:.4f}",
            "normalized_auprc": f"{summ['normalized_auprc']:.4f}",
            "auprg": f"{summ['auprg']:.4f}",
            "ece_rank": f"{summ['ece_rank']:.4f}",
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader(); w.writerows(rows)

    # Markdown summary with simple group means
    def group_mean(key: str, filt) -> float:
        vals = [float(r[key]) for r in rows if filt(r)]
        return sum(vals) / len(vals) if vals else float("nan")

    seeds6 = {"1001","2002","3003","4004","5005","6006"}
    seeds9 = seeds6 | {"7007","8008","9009"}

    def is_on(r): return r.get("variant") == "on"
    def is_off(r): return r.get("variant") == "off"

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("### Hold‑out summaries (per run)\n\n")
        f.write(f"CSV: {args.out_csv}\n\n")
        f.write("| seed | variant | k | giters | gfreq | normAUPRC | AUPRC | AUPRG | ROC | prev | ece_rank | run |\n")
        f.write("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['seed']} | {r['variant']} | {r.get('k','')} | {r['gamma_iters']} | {r['gamma_freq']} | {r['normalized_auprc']} | {r['pr_auc']} | {r['auprg']} | {r['roc_auc']} | {r['prevalence']} | {r['ece_rank']} | {r['run_dir']} |\n")

        f.write("\n---\n\n### Group means (normAUPRC)\n\n")
        f.write(f"- 6×2 on: {group_mean('normalized_auprc', lambda r: is_on(r) and r['seed'] in seeds6):.4f}\n")
        f.write(f"- 6×2 off: {group_mean('normalized_auprc', lambda r: is_off(r) and r['seed'] in seeds6):.4f}\n")
        f.write(f"- 9×2 on: {group_mean('normalized_auprc', lambda r: is_on(r) and r['seed'] in seeds9):.4f}\n")
        f.write(f"- 9×2 off: {group_mean('normalized_auprc', lambda r: is_off(r) and r['seed'] in seeds9):.4f}\n")

    print("WROTE", args.out_csv)
    print("WROTE", args.out_md)

if __name__ == "__main__":
    main()


