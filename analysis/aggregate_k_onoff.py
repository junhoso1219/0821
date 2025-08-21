from __future__ import annotations
import argparse, os, re, json
from typing import Dict, List

RESULTS_DIR = "results"

def read_meta(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "=" in ln:
                    k, v = ln.split("=", 1)
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
    summ = os.path.join(run_dir, "holdout", "metrics_summary.json")
    if not os.path.isfile(summ):
        return None
    try:
        with open(summ, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "normalized_auprc": float(data.get("normalized_auprc", float("nan")))
        }
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "k_onoff_summary.md"))
    args = ap.parse_args()

    # Collect rows
    rows: List[Dict[str, str]] = []
    for d in list_result_dirs(RESULTS_DIR):
        meta = read_meta(os.path.join(d, "meta.txt"))
        if not meta:
            continue
        k = meta.get("k"); var = meta.get("variant"); seed = meta.get("seed")
        if k is None or var not in ("on", "off") or seed is None:
            continue
        summ = load_summary(d)
        if summ is None:
            continue
        rows.append({
            "run": d,
            "k": k,
            "variant": var,
            "seed": seed,
            "normalized_auprc": summ["normalized_auprc"],
        })

    def mean(xs: List[float]) -> float:
        return sum(xs)/len(xs) if xs else float("nan")

    # Aggregate by k and variant
    ks = sorted({r["k"] for r in rows}, key=lambda x: int(x))
    out_lines = ["### k × on/off summary (hold‑out, normAUPRC)", "", "| k | on mean | off mean | diff(on−off) | n_on | n_off |", "|---:|---:|---:|---:|---:|---:|"]
    for k in ks:
        on_vals = [r["normalized_auprc"] for r in rows if r["k"] == k and r["variant"] == "on"]
        off_vals = [r["normalized_auprc"] for r in rows if r["k"] == k and r["variant"] == "off"]
        mo = mean(on_vals); mf = mean(off_vals)
        diff = (mo - mf) if (mo==mo and mf==mf) else float("nan")
        out_lines.append(f"| {k} | {mo:.4f} | {mf:.4f} | {diff:+.4f} | {len(on_vals)} | {len(off_vals)} |")
    out_lines.append("\n---\n\n")
    out_lines.append("### Per-run values")
    for r in rows:
        out_lines.append(f"- k={r['k']} {r['variant']} seed={r['seed']}: normAUPRC={r['normalized_auprc']:.4f} | {r['run']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


