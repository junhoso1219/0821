### Executive summary

- Baseline choice: 6×2 on(iter=20) (reporting default)

- 6×2: on>0 (CI lower>0), 9×2에서도 on 유지(다만 평균차는 작음)

- k-sweep: k≥2에서는 off가 우세 (hold-out normAUPRC 평균)

- k=1 on>off (paired): +0.0497 [−0.0039,+0.1029] (9×2); +0.0547 [−0.0072,+0.1053] (6×2)

- Tiny-ImageNet(k=1, 3×2): on≈off (γ 경로 비활성; 동일 수치) — off 동등 처리



#### Links

- k×on/off summary: k_onoff_summary.md

- k paired bootstrap + BH-FDR: k_onoff_paired_fdr.md

- k=1 on/off overview: k1_onoff_overview.md

- baselines (6×2): baselines_compare_6x2.md

- holdout summary: holdout_summary.md
