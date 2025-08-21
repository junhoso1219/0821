from __future__ import annotations
import os
import csv
import math
import argparse
import random
from typing import List, Dict, Tuple


def read_meta(meta_path: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if not os.path.isfile(meta_path):
        return info
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
    except Exception:
        return {}
    return info


def load_metrics_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def is_finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def filter_analysis_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for r in rows:
        # mask_applicable
        mask_applicable = to_float(r.get("mask_applicable", "1"))
        if mask_applicable < 0.5:
            continue
        # EoS drop: lambda_max < two_over_lr if both available
        lam = to_float(r.get("lambda_max", "nan"))
        tol = to_float(r.get("two_over_lr", "nan"))
        if math.isfinite(lam) and math.isfinite(tol):
            if not (lam < tol):
                continue
        filtered.append(r)
    return filtered


def compute_labels_and_scores(rows: List[Dict[str, str]]) -> Tuple[List[int], List[float]]:
    labels: List[int] = []
    scores: List[float] = []
    for r in rows:
        # label: ΔL_dom >= 0
        delta_dom = to_float(r.get("deltaL_dom", "nan"))
        if not math.isfinite(delta_dom):
            continue
        y = 1 if (delta_dom >= 0.0) else 0
        # score: z = r / r_th_eff (prefer gamma_eff if finite)
        r_val = to_float(r.get("r", "nan"))
        r_th_gamma_eff = to_float(r.get("r_th_gamma_eff", "nan"))
        r_th_eff = to_float(r.get("r_th_eff", "nan"))
        r_th = to_float(r.get("r_th", "nan"))
        denom = float("nan")
        if is_finite_positive(r_th_gamma_eff):
            denom = r_th_gamma_eff
        elif is_finite_positive(r_th_eff):
            denom = r_th_eff
        elif is_finite_positive(r_th):
            denom = r_th
        if not (math.isfinite(r_val) and is_finite_positive(denom)):
            continue
        z = r_val / denom
        if not math.isfinite(z):
            continue
        labels.append(y)
        scores.append(z)
    return labels, scores


def precision_recall_auc(labels: List[int], scores: List[float]) -> float:
    # Manual PR-AUC without sklearn. Sort by descending score.
    n = len(labels)
    if n == 0:
        return float("nan")
    paired = list(zip(scores, labels))
    paired.sort(key=lambda t: t[0], reverse=True)
    total_pos = sum(l for _, l in paired)
    total_neg = n - total_pos
    if total_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_score = None
    precisions: List[float] = []
    recalls: List[float] = []
    for s, y in paired:
        if prev_score is not None and s != prev_score:
            if (tp + fp) > 0:
                precisions.append(tp / (tp + fp))
                recalls.append(tp / total_pos)
        if y == 1:
            tp += 1
        else:
            fp += 1
        prev_score = s
    if (tp + fp) > 0:
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos)
    # Add (recall=0, precision=pos_ratio) anchor
    precisions = [precisions[0] if precisions else (total_pos / n)] + precisions
    recalls = [0.0] + recalls
    # Trapezoidal integration in recall domain
    auc = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        auc += dr * (precisions[i] + precisions[i - 1]) * 0.5
    return auc


def normalized_auprc(labels: List[int], scores: List[float]) -> float:
    n = len(labels)
    if n == 0:
        return float("nan")
    p = sum(labels) / n
    auc = precision_recall_auc(labels, scores)
    if not math.isfinite(auc):
        return float("nan")
    if p >= 1.0:
        return 0.0
    return (auc - p) / (1.0 - p)


def circular_block_indices(n: int, block_len: int) -> List[int]:
    if n <= 0 or block_len <= 0:
        return []
    num_blocks = math.ceil(n / block_len)
    idx: List[int] = []
    for _ in range(num_blocks):
        start = random.randrange(n)
        for b in range(block_len):
            idx.append((start + b) % n)
    return idx[: n]


def paired_block_bootstrap_diff(on_rows: List[Dict[str, str]], off_rows: List[Dict[str, str]], block_len: int, B: int) -> Tuple[float, List[float]]:
    # Align by index using the minimum length after filtering
    on_f = filter_analysis_rows(on_rows)
    off_f = filter_analysis_rows(off_rows)
    n = min(len(on_f), len(off_f))
    if n == 0:
        return float("nan"), []
    on_f = on_f[: n]
    off_f = off_f[: n]

    # Pre-extract labels/scores sequences per index
    on_labels, on_scores = compute_labels_and_scores(on_f)
    off_labels, off_scores = compute_labels_and_scores(off_f)
    m = min(len(on_labels), len(off_labels))
    on_labels = on_labels[: m]
    on_scores = on_scores[: m]
    off_labels = off_labels[: m]
    off_scores = off_scores[: m]
    if m == 0:
        return float("nan"), []

    # Observed diff
    obs_on = normalized_auprc(on_labels, on_scores)
    obs_off = normalized_auprc(off_labels, off_scores)
    obs_diff = obs_on - obs_off

    diffs: List[float] = []
    for _ in range(B):
        idx = circular_block_indices(m, block_len)
        y_on = [on_labels[i] for i in idx]
        s_on = [on_scores[i] for i in idx]
        y_off = [off_labels[i] for i in idx]
        s_off = [off_scores[i] for i in idx]
        d = normalized_auprc(y_on, s_on) - normalized_auprc(y_off, s_off)
        diffs.append(d)
    return obs_diff, diffs


def percentile_ci(samples: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    xs = sorted(samples)
    lo = xs[int(math.floor((alpha / 2.0) * (len(xs) - 1)))]
    hi = xs[int(math.ceil((1.0 - alpha / 2.0) * (len(xs) - 1)))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/elicer/project_0814_2/results")
    ap.add_argument("--dataset_filter", type=str, default="cifar100")
    ap.add_argument("--k_filter", type=int, default=1)
    ap.add_argument("--block_len", type=int, default=10)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # Find runs and pair by seed
    run_dirs = [os.path.join(args.root, d) for d in os.listdir(args.root) if d.startswith("20")]
    seed_to_runs: Dict[str, Dict[str, str]] = {}
    for d in sorted(run_dirs):
        meta = read_meta(os.path.join(d, "meta.txt"))
        if not meta:
            continue
        if meta.get("dataset") != args.dataset_filter:
            continue
        if str(meta.get("k")) != str(args.k_filter):
            continue
        variant = meta.get("variant")
        seed = meta.get("seed")
        if variant not in ("on", "off") or not seed:
            continue
        if not os.path.isfile(os.path.join(d, "holdout", "metrics.csv")):
            continue
        seed_to_runs.setdefault(seed, {})[variant] = d

    pairs: List[Tuple[str, str, str]] = []  # (seed, on_dir, off_dir)
    for seed, m in seed_to_runs.items():
        if "on" in m and "off" in m:
            pairs.append((seed, m["on"], m["off"]))

    results: List[Tuple[str, float, float, float]] = []  # seed, obs_diff, ci_lo, ci_hi
    boot_all: List[float] = []
    for seed, on_dir, off_dir in pairs:
        on_rows = load_metrics_rows(os.path.join(on_dir, "holdout", "metrics.csv"))
        off_rows = load_metrics_rows(os.path.join(off_dir, "holdout", "metrics.csv"))
        obs_diff, diffs = paired_block_bootstrap_diff(on_rows, off_rows, args.block_len, args.B)
        ci_lo, ci_hi = percentile_ci(diffs, 0.05)
        results.append((seed, obs_diff, ci_lo, ci_hi))
        boot_all.append(obs_diff)

    # Aggregate paired bootstrap across seeds: mean diff of observed
    mean_obs = float("nan")
    if results:
        mean_obs = sum(r[1] for r in results) / len(results)

    # Write markdown
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("### Moving-block paired bootstrap (hold-out, normAUPRC)\n\n")
        f.write(f"dataset={args.dataset_filter}, k={args.k_filter}, pairs={len(pairs)}, block_len={args.block_len}, B={args.B}\n\n")
        f.write(f"- mean observed diff(on−off): {mean_obs:.4f}\n\n")
        f.write("| seed | obs diff | 95% CI (block bootstrap) | on dir | off dir |\n")
        f.write("|---:|---:|:---:|---|---|\n")
        for seed, obs, lo, hi in results:
            on_dir = seed_to_runs[seed]["on"]
            off_dir = seed_to_runs[seed]["off"]
            f.write(f"| {seed} | {obs:.4f} | [{lo:.4f}, {hi:.4f}] | {on_dir} | {off_dir} |\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()


