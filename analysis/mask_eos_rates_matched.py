from __future__ import annotations
import argparse, os, re, csv, math
from typing import Dict, List, Optional

RESULTS_DIR = "results"

def read_meta(meta_path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
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

def find_run(seed: int, variant: str, prefer_giters: Optional[int]) -> Optional[str]:
    for d in list_result_dirs(RESULTS_DIR):
        meta = read_meta(os.path.join(d, "meta.txt"))
        if not meta:
            continue
        if meta.get("variant", "") != variant:
            continue
        try:
            sd = int(meta.get("seed", "-1"))
        except Exception:
            sd = -1
        if sd != int(seed):
            continue
        if not os.path.isfile(os.path.join(d, "holdout", "metrics.csv")):
            continue
        giters_val = meta.get("gamma_iters")
        if prefer_giters is None:
            return d
        try:
            gi = int(giters_val) if giters_val is not None else None
        except Exception:
            gi = None
        if variant == "on":
            if gi is None and prefer_giters == 20:
                return d
            if gi == prefer_giters:
                return d
            else:
                continue
        else:
            if gi is None:
                return d
    return None

def ffloat(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def compute_rates(metrics_csv: str) -> Dict[str, float]:
    total = 0
    cnt_app = 0
    cnt_eos_keep = 0
    cnt_both = 0
    with open(metrics_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            lam = ffloat(r.get("lambda_max", "nan"))
            two = ffloat(r.get("two_over_lr", "nan"))
            app = ffloat(r.get("mask_applicable", "nan"))
            if not (math.isfinite(lam) and math.isfinite(two) and math.isfinite(app)):
                continue
            total += 1
            is_app = app > 0.5
            is_eos_keep = lam < two
            if is_app:
                cnt_app += 1
            if is_eos_keep:
                cnt_eos_keep += 1
            if is_app and is_eos_keep:
                cnt_both += 1
    if total == 0:
        return {"applicable_rate": float("nan"), "eos_keep_rate": float("nan"), "combined_rate": float("nan"), "n": 0}
    return {
        "applicable_rate": cnt_app / total,
        "eos_keep_rate": cnt_eos_keep / total,
        "combined_rate": cnt_both / total,
        "n": total,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1001,2002,3003,4004,5005,6006")
    ap.add_argument("--prefer_giters", type=int, default=20)
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "mask_eos_rates_6x2.md"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    vals = {"on": [], "off": []}
    lines: List[str] = []
    for sd in seeds:
        d_on = find_run(sd, "on", args.prefer_giters)
        d_off = find_run(sd, "off", None)
        for var, d in (("on", d_on), ("off", d_off)):
            if not d:
                continue
            hold = os.path.join(d, "holdout", "metrics.csv")
            rates = compute_rates(hold)
            vals[var].append(rates)
            lines.append(f"- seed={sd}, {var}: app={rates['applicable_rate']:.3f}, eos_keep={rates['eos_keep_rate']:.3f}, both={rates['combined_rate']:.3f} | {d}")

    def mean(key: str, var: str) -> float:
        xs = [v[key] for v in vals[var] if v["n"] > 0]
        return (sum(xs) / len(xs)) if xs else float("nan")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("### Mask/EoS rates (hold‑out, 6×2 matched)\n\n")
        f.write("| variant | applicable_rate | eos_keep_rate | combined_rate | n_runs |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for var in ("on", "off"):
            f.write(f"| {var} | {mean('applicable_rate', var):.4f} | {mean('eos_keep_rate', var):.4f} | {mean('combined_rate', var):.4f} | {len(vals[var])} |\n")
        f.write("\n---\n\n### Per-seed\n\n")
        for ln in lines:
            f.write(ln + "\n")
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


