from __future__ import annotations
import argparse, os, re, json, subprocess, time, math, random
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

def find_run(seed: int, variant: str, prefer_giters: Optional[int], required_k: Optional[int] = None, dataset_filter: Optional[str] = None) -> Optional[str]:
    for d in list_result_dirs(RESULTS_DIR):
        meta = read_meta(os.path.join(d, "meta.txt"))
        if not meta:
            continue
        if meta.get("variant", "").strip() != variant:
            continue
        if dataset_filter is not None and meta.get("dataset") != dataset_filter:
            continue
        # Optional k filter
        if required_k is not None:
            try:
                k_val = int(meta.get("k", "-1"))
            except Exception:
                k_val = -1
            if k_val != int(required_k):
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
            # No constraint
            return d
        try:
            gi = int(giters_val) if giters_val is not None else None
        except Exception:
            gi = None
        # on: prefer exact match if present, otherwise accept if None (off runs have None)
        if variant == "on":
            if gi is None and prefer_giters is not None:
                # Accept legacy meta without giters only if prefer is 20 (original 6x2 runs)
                if prefer_giters == 20:
                    return d
                else:
                    continue
            if gi == prefer_giters:
                return d
            else:
                continue
        else:  # off
            # off runs have no gamma_iters; accept
            if gi is None:
                return d
    return None

def ensure_metrics_summary(hold_dir: str) -> Dict[str, float]:
    summ_path = os.path.join(hold_dir, "metrics_summary.json")
    if not os.path.isfile(summ_path):
        metrics_path = os.path.join(hold_dir, "metrics.csv")
        cmd = [
            "python3", "analysis/plot_results.py", "--metrics", metrics_path,
            "--outdir", hold_dir, "--use_eff", "--auto_grid",
            "--grid_points", "600", "--grid_qmin", "0.001", "--grid_qmax", "0.999",
            "--logspace", "--drop_eos", "--plot_reliability", "--plot_prg"
        ]
        subprocess.run(cmd, check=True)
    with open(summ_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "pr_auc": float(data.get("pr_auc", float("nan"))),
        "roc_auc": float(data.get("roc_auc", float("nan"))),
        "prevalence": float(data.get("prevalence", float("nan"))),
        "normalized_auprc": float(data.get("normalized_auprc", float("nan"))),
        "auprg": float(data.get("auprg", float("nan")))
    }

def paired_bootstrap_diffs(values_on: List[float], values_off: List[float], B: int = 10000, seed: int = 1337) -> Tuple[float, Tuple[float, float]]:
    assert len(values_on) == len(values_off) and len(values_on) > 0
    random.seed(seed)
    diffs = [on - off for on, off in zip(values_on, values_off)]
    mean_diff = sum(diffs) / len(diffs)
    n = len(diffs)
    boots = []
    for _ in range(B):
        idxs = [random.randrange(n) for __ in range(n)]
        boots.append(sum(diffs[i] for i in idxs) / n)
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    return mean_diff, (lo, hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="1001,2002,3003,4004,5005,6006")
    ap.add_argument("--prefer_giters", type=int, default=20, help="Prefer on-runs with this gamma_iters; None to ignore")
    ap.add_argument("--k_filter", type=int, default=-1, help="If >=0, require runs to have meta k equal to this value")
    ap.add_argument("--dataset_filter", type=str, default=None, help="If set, require runs to have this dataset in meta.txt")
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "onoff_paired_bootstrap_6x2.md"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    prefer = args.prefer_giters
    if prefer < 0:
        prefer = None  # type: ignore
    k_required: Optional[int] = None if args.k_filter < 0 else int(args.k_filter)
    dataset_required: Optional[str] = args.dataset_filter

    pairs = []
    rows_md: List[str] = []
    vals_on: List[float] = []
    vals_off: List[float] = []

    for sd in seeds:
        d_on = find_run(sd, "on", prefer, k_required, dataset_required)
        d_off = find_run(sd, "off", None, k_required, dataset_required)
        if not d_on or not d_off:
            continue
        hold_on = os.path.join(d_on, "holdout")
        hold_off = os.path.join(d_off, "holdout")
        # strict meta consistency check when available
        meta_on = read_meta(os.path.join(d_on, "meta.txt"))
        meta_off = read_meta(os.path.join(d_off, "meta.txt"))
        required_same_keys = [
            "dataset", "k", "eig_freq", "eval_M", "noise_M", "max_steps"
        ]
        inconsistent = []
        for ksame in required_same_keys:
            von = meta_on.get(ksame)
            voff = meta_off.get(ksame)
            if von is not None and voff is not None and von != voff:
                inconsistent.append((ksame, von, voff))
        if inconsistent:
            # skip this pair but tell the user why
            print("SKIP pair due to meta mismatch:", sd, d_on, d_off, inconsistent)
            continue
        summ_on = ensure_metrics_summary(hold_on)
        summ_off = ensure_metrics_summary(hold_off)
        na_on = summ_on.get("normalized_auprc", float("nan"))
        na_off = summ_off.get("normalized_auprc", float("nan"))
        if not (math.isfinite(na_on) and math.isfinite(na_off)):
            continue
        pairs.append((sd, d_on, d_off, na_on, na_off))
        vals_on.append(na_on)
        vals_off.append(na_off)
        rows_md.append(f"| {sd} | {na_on:.4f} | {na_off:.4f} | {na_on - na_off:+.4f} | {d_on} | {d_off} |")

    if not pairs:
        print("No matched pairs found.")
        return

    mean_diff, (ci_lo, ci_hi) = paired_bootstrap_diffs(vals_on, vals_off)
    mean_on = sum(vals_on) / len(vals_on)
    mean_off = sum(vals_off) / len(vals_off)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        title = "### On−Off 페어드 부트스트랩 (hold‑out, normAUPRC)\n\n"
        f.write(title)
        f.write(f"- 페어 수: {len(pairs)}\n")
        f.write(f"- on 평균: {mean_on:.4f}, off 평균: {mean_off:.4f}\n")
        f.write(f"- on−off 평균차: {mean_diff:+.4f} (95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])\n\n")
        f.write("| seed | on normAUPRC | off normAUPRC | diff(on−off) | on dir | off dir |\n")
        f.write("|---:|---:|---:|---:|---|---|\n")
        for row in rows_md:
            f.write(row + "\n")
        # identical-diff warning
        if any(abs((on - off)) < 1e-6 for (_, _, _, on, off) in pairs):
            f.write("\n> Note: 일부 페어에서 on/off 차이가 ~0으로 관측되었습니다. 매칭/메타를 확인하세요.\n")
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


