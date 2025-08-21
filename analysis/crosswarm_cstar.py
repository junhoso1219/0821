from __future__ import annotations
import os, re, json, csv, argparse
from typing import Dict, List, Tuple
import math

RESULTS_DIR = "results"

def list_result_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        return []
    dirs: List[str] = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p) and re.match(r"^20\d{6}-\d{6}$", name):
            dirs.append(p)
    dirs.sort(key=lambda p: os.path.getmtime(p))
    return dirs

def read_meta(d: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        with open(os.path.join(d, "meta.txt"), "r", encoding="utf-8") as f:
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

def ffloat(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def load_holdout_rows(run_dir: str) -> List[Dict[str, str]]:
    hp = os.path.join(run_dir, "holdout", "metrics.csv")
    if not os.path.isfile(hp):
        return []
    with open(hp, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_full_rows(run_dir: str) -> List[Dict[str, str]]:
    fp = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(fp):
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def compute_cstar_from_warmup(run_dir: str, warmup_steps: int = 80) -> float:
    rows = load_full_rows(run_dir)
    if not rows:
        return float("nan")
    import numpy as np
    steps = np.array([ffloat(rw.get("step", "nan")) for rw in rows], dtype=float)
    r = np.array([ffloat(rw.get("r", "nan")) for rw in rows], dtype=float)
    r_th = np.array([ffloat(rw.get("r_th", "nan")) for rw in rows], dtype=float)
    r_th_gamma_eff = np.array([ffloat(rw.get("r_th_gamma_eff", "nan")) for rw in rows], dtype=float)
    ddom = np.array([ffloat(rw.get("deltaL_dom", "nan")) for rw in rows], dtype=float)
    mask = np.isfinite(steps) & (steps < float(warmup_steps)) & np.isfinite(r) & np.isfinite(r_th) & np.isfinite(ddom)
    r, r_th, r_th_gamma_eff, ddom = r[mask], r_th[mask], r_th_gamma_eff[mask], ddom[mask]
    if r.size < 5:
        return float("nan")
    base_th = np.where(np.isfinite(r_th_gamma_eff), r_th_gamma_eff, r_th)
    z = r / np.maximum(base_th, 1e-12)
    y = (ddom >= 0.0).astype(int)
    zf = z[np.isfinite(z)]
    if zf.size == 0:
        return float("nan")
    qmin = float(np.quantile(zf, 0.01)); qmax = float(np.quantile(zf, 0.99))
    if not (math.isfinite(qmin) and math.isfinite(qmax)) or qmax <= qmin:
        qmin, qmax = float(np.nanmin(zf)), float(np.nanmax(zf))
    if not (math.isfinite(qmin) and math.isfinite(qmax)) or qmax <= qmin:
        return float("nan")
    c_grid = np.geomspace(max(1e-8, qmin), max(qmin * (1 + 1e-6), qmax), num=200)
    best_f1, best_c = -1.0, float("nan")
    for c in c_grid:
        yhat = (z <= c).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / max(1e-12, (prec + rec))) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_c = f1, float(c)
    return best_c

def compute_point_metrics(rows: List[Dict[str, str]], cstar: float) -> Dict[str, float]:
    if not rows or not (cstar > 0 and math.isfinite(cstar)):
        return {"tp": math.nan, "fp": math.nan, "fn": math.nan, "tn": math.nan, "prec": math.nan, "rec": math.nan, "f1": math.nan, "prev": math.nan}
    import numpy as np
    r = np.array([ffloat(rw.get("r", "nan")) for rw in rows], dtype=float)
    r_th_eff = np.array([ffloat(rw.get("r_th_eff", rw.get("r_th", "nan"))) for rw in rows], dtype=float)
    ddom = np.array([ffloat(rw.get("deltaL_dom", "nan")) for rw in rows], dtype=float)
    lam = np.array([ffloat(rw.get("lambda_max", "nan")) for rw in rows], dtype=float)
    two = np.array([ffloat(rw.get("two_over_lr", "nan")) for rw in rows], dtype=float)
    mask = np.isfinite(r) & np.isfinite(r_th_eff) & np.isfinite(ddom) & np.isfinite(lam) & np.isfinite(two) & (lam < two)
    r = r[mask]; r_th_eff = r_th_eff[mask]; ddom = ddom[mask]
    if r.size == 0:
        return {"tp": math.nan, "fp": math.nan, "fn": math.nan, "tn": math.nan, "prec": math.nan, "rec": math.nan, "f1": math.nan, "prev": math.nan}
    z = r / np.maximum(r_th_eff, 1e-12)
    y = (ddom >= 0.0).astype(int)
    yhat = (z <= cstar).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec / max(1e-12, (prec + rec))) if (prec + rec) > 0 else 0.0
    prev = float(np.mean(y))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "prec": prec, "rec": rec, "f1": f1, "prev": prev}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--variant", type=str, default=None, help="on/off or None for both")
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "crosswarm_cstar_k1_onoff.md"))
    args = ap.parse_args()

    runs: List[Tuple[str, Dict[str, str]]] = []
    for d in list_result_dirs(RESULTS_DIR):
        meta = read_meta(d)
        if not meta:
            continue
        if str(meta.get("k", "")) != str(args.k):
            continue
        if args.variant and meta.get("variant", "") != args.variant:
            continue
        # cstar.json is stored at run root when auto_rth_scale is used
        cstar_path = os.path.join(d, "cstar.json")
        has_holdout = os.path.isfile(os.path.join(d, "holdout", "metrics.csv"))
        if not has_holdout:
            continue
        runs.append((d, meta))

    # group by variant
    by_var: Dict[str, List[Tuple[str, Dict[str, str]]]] = {"on": [], "off": []}
    for d, m in runs:
        v = m.get("variant", "")
        if v in by_var:
            by_var[v].append((d, m))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"### Cross-warmup c* evaluation (k={args.k})\n\n")
        f.write("- 각 소스 런의 c*를 타 시드 런 hold-out에 적용하여 (prec, rec, F1) 보고\n\n")
        for variant in ([args.variant] if args.variant else ["on", "off"]):
            if variant not in by_var or not by_var[variant]:
                continue
            f.write(f"#### variant={variant}\n\n")
            f.write("| src_seed | tgt_seed | c* | prec | rec | F1 | prev | src | tgt |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
            # prepare cstar per source (from cstar.json or fallback warmup estimation)
            src_info: List[Tuple[str, str, float]] = []  # (seed, dir, cstar)
            for d, m in by_var[variant]:
                seed = m.get("seed", "")
                cstar_path = os.path.join(d, "cstar.json")
                cval: float = float("nan")
                try:
                    with open(cstar_path, "r", encoding="utf-8") as cf:
                        cj = json.load(cf)
                        cval = float(cj.get("cstar", float("nan")))
                except Exception:
                    # fallback: estimate from warmup window
                    cval = compute_cstar_from_warmup(d, warmup_steps=80)
                src_info.append((seed, d, cval))

            for src_seed, src_dir, cval in src_info:
                # apply to all targets with same variant, different seed
                for tgt_dir, tgt_meta in [(d, m) for d, m in by_var[variant] if m.get("seed", "") != src_seed]:
                    tgt_seed = tgt_meta.get("seed", "")
                    rows = load_holdout_rows(tgt_dir)
                    metrics = compute_point_metrics(rows, cval)
                    f.write(
                        f"| {src_seed} | {tgt_seed} | {cval:.4g} | {metrics['prec']:.3f} | {metrics['rec']:.3f} | {metrics['f1']:.3f} | {metrics['prev']:.3f} | {src_dir} | {tgt_dir} |\n"
                    )
            f.write("\n")
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


