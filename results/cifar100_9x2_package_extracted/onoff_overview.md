### γ on/off 개요 (hold‑out, normalized AUPRC)

본 표는 시드 매칭된 on/off 비교의 페어드 부트스트랩 결과를 요약합니다.

| set | pairs | on mean | off mean | diff(on−off) | 95% CI | source |
|---|---:|---:|---:|---:|:---:|---|
| 6×2 | 6 | 0.0210 | 0.0360 | −0.0150 | [−0.0971, +0.0641] | `results/onoff_paired_bootstrap_6x2.md` |
| 9×2 | 9 | 0.0618 | 0.0340 | +0.0278 | [−0.0036, +0.0619] | `results/onoff_paired_bootstrap_9x2.md` |
| CIFAR‑100 9×2 (k=1) | 6 | 0.0417 | 0.0275 | +0.0142 | [−0.0639, +0.0896] | `results/onoff_paired_bootstrap_cifar100_k1_9x2.md` |

참고(블록 부트스트랩, 평균 CI):
- 6×2: on(iter=20) 0.1000 [0.0055, 0.1890], off 0.0360 [−0.0095, 0.0796] — `results/auprc_ci_block_matched.md`
- 9×2: on(iter=20) 0.0817 [0.0184, 0.1483], off 0.0368 [−0.0283, 0.1119] — `results/auprc_ci_block_matched_9x2.md`
- 9×2(best): on(iters=40,freq=60) 0.0585 [0.0088, 0.1115], off 0.0368 [−0.0283, 0.1119] — `results/auprc_ci_block_matched_best_9x2.md`


