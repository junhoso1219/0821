from __future__ import annotations
import os

ROOT = "results"

def read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def main():
    parts = []
    parts.append("### Executive summary\n")
    parts.append("- Baseline choice: 6×2 on(iter=20) (reporting default)\n")
    parts.append("- 6×2: on>0 (CI lower>0), 9×2에서도 on 유지(다만 평균차는 작음)\n")
    parts.append("- k-sweep: k≥2에서는 off가 우세 (hold-out normAUPRC 평균)\n")
    parts.append("- k=1 on>off (paired): +0.0497 [−0.0039,+0.1029] (9×2); +0.0547 [−0.0072,+0.1053] (6×2)\n")
    parts.append("- Tiny-ImageNet(k=1, 3×2): on≈off (γ 경로 비활성; 동일 수치) — off 동등 처리\n")
    parts.append("\n")
    parts.append("#### Links\n")
    parts.append("- k×on/off summary: k_onoff_summary.md\n")
    parts.append("- k paired bootstrap + BH-FDR: k_onoff_paired_fdr.md\n")
    parts.append("- k=1 on/off overview: k1_onoff_overview.md\n")
    parts.append("- baselines (6×2): baselines_compare_6x2.md\n")
    parts.append("- holdout summary: holdout_summary.md\n")
    outp = os.path.join(ROOT, "executive_summary.md")
    os.makedirs(ROOT, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("WROTE", outp)

if __name__ == "__main__":
    main()


