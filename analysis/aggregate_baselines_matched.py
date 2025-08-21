from __future__ import annotations
import argparse, os, re
from typing import Dict, List, Optional, Tuple

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
        hold_metrics = os.path.join(d, "holdout", "metrics.csv")
        if not os.path.isfile(hold_metrics):
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

def parse_baselines_md(path: str) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        for ln in lines:
            if not ln.startswith("|") or ln.startswith("|---"):
                continue
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if len(parts) < 3:
                continue
            if parts[0] == "score":
                continue
            score = parts[0]
            try:
                auprc = float(parts[1])
            except Exception:
                auprc = float("nan")
            try:
                norm = float(parts[2])
            except Exception:
                norm = float("nan")
            metrics[score] = {"AUPRC": auprc, "normAUPRC": norm}
    except FileNotFoundError:
        pass
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1001,2002,3003,4004,5005,6006")
    ap.add_argument("--prefer_giters", type=int, default=20)
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "baselines_compare_6x2.md"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    scores = ["z_eff","eos_margin","ps_grad","r_only","lam_max","grad_norm","r_full","sam1_full"]

    vals_on: Dict[str, List[float]] = {s: [] for s in scores}
    vals_off: Dict[str, List[float]] = {s: [] for s in scores}

    rows: List[str] = []
    for sd in seeds:
        d_on = find_run(sd, "on", args.prefer_giters)
        d_off = find_run(sd, "off", None)
        if not d_on or not d_off:
            continue
        bl_on = parse_baselines_md(os.path.join(d_on, "holdout", "baselines_compare.md"))
        bl_off = parse_baselines_md(os.path.join(d_off, "holdout", "baselines_compare.md"))
        parts = [f"seed={sd}"]
        for sc in scores:
            vo = bl_on.get(sc, {}).get("normAUPRC", float("nan"))
            vf = bl_off.get(sc, {}).get("normAUPRC", float("nan"))
            if not (vo != vo or vf != vf):
                vals_on[sc].append(vo)
                vals_off[sc].append(vf)
            parts.append(f"{sc}: on={vo:.4f}, off={vf:.4f}")
        rows.append("; ".join(parts))

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("### Baselines compare (normAUPRC means, 6×2 matched)\n\n")
        f.write("| score | on mean | off mean | on−off | n |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for sc in scores:
            mo = mean(vals_on[sc]); mf = mean(vals_off[sc])
            n = min(len(vals_on[sc]), len(vals_off[sc]))
            diff = (mo - mf) if (mo==mo and mf==mf) else float("nan")
            f.write(f"| {sc} | {mo:.4f} | {mf:.4f} | {diff:+.4f} | {n} |\n")
        f.write("\n---\n\n")
        f.write("### Per-seed values (normAUPRC)\n\n")
        for ln in rows:
            f.write(f"- {ln}\n")
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


