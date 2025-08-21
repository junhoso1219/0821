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
    ap.add_argument("--k", type=int, default=3)
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
        f.write("### 시각 요약: Top-{} on/off (hold-out)\n\n".format(args.k))
        f.write("원본 CSV: {}\n\n".format(args.csv))

        def write_block(title: str, items: List[Dict[str, str]]):
            f.write("#### {}\n\n".format(title))
            for i, r in enumerate(items[: args.k]):
                run_dir = r.get("run_dir", "")
                seed = r.get("seed", "")
                na = r.get("normalized_auprc", "")
                hold = os.path.join(run_dir, "holdout")
                pr = os.path.join(hold, "pr_curve.png")
                prg = os.path.join(hold, "prg_curve.png")
                f.write("- seed={} | normAUPRC={} | run={}\n\n".format(seed, na, run_dir))
                if os.path.isfile(pr):
                    f.write("![]({})\n\n".format(pr))
                if os.path.isfile(prg):
                    f.write("![]({})\n\n".format(prg))
            f.write("\n")

        write_block("on (정규화 AUPRC 상위)", on_rows)
        write_block("off (정규화 AUPRC 상위)", off_rows)

    print("WROTE", args.out)

if __name__ == "__main__":
    main()


