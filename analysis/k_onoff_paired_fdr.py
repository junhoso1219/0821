from __future__ import annotations
import argparse, os, re, json, random
from typing import Dict, List, Tuple

RESULTS_DIR = "results"

def read_meta(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    meta[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return meta

def list_result_dirs(base: str) -> List[str]:
    if not os.path.isdir(base): return []
    dirs = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p) and re.match(r"^20\d{6}-\d{6}$", name):
            dirs.append(p)
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs

def load_norm_auprc(run_dir: str) -> float | None:
    summ = os.path.join(run_dir, "holdout", "metrics_summary.json")
    if not os.path.isfile(summ): return None
    try:
        with open(summ, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("normalized_auprc"))
    except Exception:
        return None

def collect_by_k() -> Dict[str, Dict[str, Dict[str, float]]]:
    # return: k -> seed -> {on: val, off: val}
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for d in list_result_dirs(RESULTS_DIR):
        meta = read_meta(os.path.join(d, "meta.txt"))
        k = meta.get("k"); var = meta.get("variant"); seed = meta.get("seed")
        if not k or var not in ("on","off") or not seed: continue
        val = load_norm_auprc(d)
        if val is None: continue
        table.setdefault(k, {}).setdefault(seed, {})[var] = val
    return table

def paired_bootstrap_diff(vals_on: List[float], vals_off: List[float], B: int = 10000, seed: int = 13) -> Tuple[float, Tuple[float, float], float]:
    assert len(vals_on) == len(vals_off) and len(vals_on) > 0
    diffs = [a - b for a, b in zip(vals_on, vals_off)]
    mean_diff = sum(diffs) / len(diffs)
    random.seed(seed)
    n = len(diffs)
    boots = []
    for _ in range(B):
        idxs = [random.randrange(n) for __ in range(n)]
        boots.append(sum(diffs[i] for i in idxs) / n)
    boots.sort()
    lo = boots[int(0.025 * B)]; hi = boots[int(0.975 * B)]
    # two-sided p-value estimated as fraction of bootstrap means with opposite sign or more extreme
    if mean_diff >= 0:
        extreme = sum(1 for b in boots if b <= 0.0)
    else:
        extreme = sum(1 for b in boots if b >= 0.0)
    p = max(1.0 / B, extreme / B) * 2.0
    p = min(1.0, p)
    return mean_diff, (lo, hi), p

def bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(reversed(order), start=1):
        # iterate from largest to smallest for monotonicity
        j = order[m - rank]
        val = pvals[j] * m / (m - rank + 1)
        q[j] = min(1.0, max(val, prev))
        prev = q[j]
    # enforce monotone nondecreasing with sorted p
    # simple pass already ensures via prev tracking
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "k_onoff_paired_fdr.md"))
    ap.add_argument("--B", type=int, default=20000)
    args = ap.parse_args()

    table = collect_by_k()
    rows = []
    pvals = []
    ks = sorted(table.keys(), key=lambda x: int(x))
    stats_by_k = {}
    for k in ks:
        seeds = sorted(table[k].keys(), key=lambda x: int(x))
        vals_on, vals_off = [], []
        for s in seeds:
            pair = table[k][s]
            if "on" in pair and "off" in pair:
                vals_on.append(pair["on"])
                vals_off.append(pair["off"])
        if len(vals_on) == 0:
            continue
        mean_diff, ci, p = paired_bootstrap_diff(vals_on, vals_off, B=args.B)
        stats_by_k[k] = (mean_diff, ci, p, len(vals_on))
        pvals.append(p)
    # BH-FDR
    qvals = bh_fdr(pvals)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("### k × on/off paired bootstrap with BH-FDR (hold‑out, normAUPRC)\n\n")
        f.write("| k | n_pairs | mean(on−off) | 95% CI | p_boot | q_BH |\n")
        f.write("|---:|---:|---:|:---:|---:|---:|\n")
        i = 0
        for k in ks:
            if k not in stats_by_k: continue
            mean_diff, ci, p, n = stats_by_k[k]
            q = qvals[i]; i += 1
            f.write(f"| {k} | {n} | {mean_diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {q:.4f} |\n")
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


