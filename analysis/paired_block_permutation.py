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
        mask_applicable = to_float(r.get("mask_applicable", "1"))
        if mask_applicable < 0.5:
            continue
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
        delta_dom = to_float(r.get("deltaL_dom", "nan"))
        if not math.isfinite(delta_dom):
            continue
        y = 1 if (delta_dom >= 0.0) else 0
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
    precisions = [precisions[0] if precisions else (total_pos / n)] + precisions
    recalls = [0.0] + recalls
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


def per_seed_diff(on_rows: List[Dict[str, str]], off_rows: List[Dict[str, str]]) -> float:
    on_f = filter_analysis_rows(on_rows)
    off_f = filter_analysis_rows(off_rows)
    n = min(len(on_f), len(off_f))
    if n == 0:
        return float("nan")
    on_f = on_f[: n]
    off_f = off_f[: n]
    on_y, on_s = compute_labels_and_scores(on_f)
    off_y, off_s = compute_labels_and_scores(off_f)
    m = min(len(on_y), len(off_y))
    if m == 0:
        return float("nan")
    on_y, on_s = on_y[: m], on_s[: m]
    off_y, off_s = off_y[: m], off_s[: m]
    return normalized_auprc(on_y, on_s) - normalized_auprc(off_y, off_s)


def paired_sign_permutation_pvalue(diffs: List[float], R: int) -> float:
    diffs = [d for d in diffs if math.isfinite(d)]
    if not diffs:
        return float("nan")
    obs = sum(diffs) / len(diffs)
    count = 0
    for _ in range(R):
        # random sign flip per pair
        s = 0.0
        for d in diffs:
            if random.random() < 0.5:
                s += d
            else:
                s += -d
        mean_perm = s / len(diffs)
        if abs(mean_perm) >= abs(obs):
            count += 1
    return (count + 1) / (R + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/elicer/project_0814_2/results")
    ap.add_argument("--dataset_filter", type=str, default="cifar100")
    ap.add_argument("--k_filter", type=int, default=1)
    ap.add_argument("--R", type=int, default=20000)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

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

    pairs: List[Tuple[str, str, str]] = []
    for seed, m in seed_to_runs.items():
        if "on" in m and "off" in m:
            pairs.append((seed, m["on"], m["off"]))

    diffs: List[Tuple[str, float]] = []
    for seed, on_dir, off_dir in pairs:
        on_rows = load_metrics_rows(os.path.join(on_dir, "holdout", "metrics.csv"))
        off_rows = load_metrics_rows(os.path.join(off_dir, "holdout", "metrics.csv"))
        d = per_seed_diff(on_rows, off_rows)
        diffs.append((seed, d))

    mean_obs = float("nan")
    if diffs:
        mean_obs = sum(d for _, d in diffs if math.isfinite(d)) / len([1 for _, d in diffs if math.isfinite(d)])
    p_val = paired_sign_permutation_pvalue([d for _, d in diffs], args.R)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("### Paired sign permutation test (hold-out, normAUPRC)\n\n")
        f.write(f"dataset={args.dataset_filter}, k={args.k_filter}, pairs={len(pairs)}, R={args.R}\n\n")
        f.write(f"- mean observed diff(on−off): {mean_obs:.4f}\n")
        f.write(f"- p_perm(two-sided, sign-flip): {p_val:.4f}\n\n")
        f.write("| seed | diff(on−off) | on dir | off dir |\n")
        f.write("|---:|---:|---|---|\n")
        for seed, d in diffs:
            on_dir = seed_to_runs[seed]["on"]
            off_dir = seed_to_runs[seed]["off"]
            f.write(f"| {seed} | {d:.4f} | {on_dir} | {off_dir} |\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()


