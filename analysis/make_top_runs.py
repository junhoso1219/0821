from __future__ import annotations
import csv, os, argparse
from typing import List, Dict

def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    for r in rows:
        r["normalized_auprc_f"] = to_float(r.get("normalized_auprc", "nan"))

    on_rows = [r for r in rows if r.get("variant") == "on"]
    off_rows = [r for r in rows if r.get("variant") == "off"]
    on_rows.sort(key=lambda r: r["normalized_auprc_f"], reverse=True)
    off_rows.sort(key=lambda r: r["normalized_auprc_f"], reverse=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("### Top runs by normalized AUPRC (hold‑out)\n\n")
        f.write(f"Source CSV: {args.csv}\n\n")
        def write_table(title: str, items: List[Dict[str, str]]):
            f.write(f"#### {title}\n\n")
            f.write("| rank | seed | giters | gfreq | normAUPRC | AUPRC | AUPRG | ROC | prev | ece_rank | run |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for i, r in enumerate(items[: args.k]):
                f.write(
                    f"| {i+1} | {r.get('seed','')} | {r.get('gamma_iters','')} | {r.get('gamma_freq','')} | {r.get('normalized_auprc','')} | {r.get('pr_auc','')} | {r.get('auprg','')} | {r.get('roc_auc','')} | {r.get('prevalence','')} | {r.get('ece_rank','')} | {r.get('run_dir','')} |\n"
                )
            f.write("\n")

        write_table("on", on_rows)
        write_table("off", off_rows)
    print("WROTE", args.out)

if __name__ == "__main__":
    main()


