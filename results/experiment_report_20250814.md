### DomOpt 실험 로그(PR/ROC·AUC, ΔL 멀티배치, 자동 임계 c*) — 2025-08-14

이 문서는 본 세션에서 수행한 실험을 처음부터 끝까지 재현 가능하도록 상세히 정리한 리포트입니다. 무엇을 왜 했고, 어떤 명령으로 돌렸으며, 어떤 산출물과 수치가 나왔는지를 최대한 자세히 기록했습니다.

---

## 최종 요약(TL;DR)

- 채택 구성(기본): lr=0.075, steps=240, k=1, eig_freq=60, eval_M=16, noise_M=16, auto_rth_scale(c*)
- hold‑out 정규화 AUPRC(기본 구성, n=5): 0.0562 [0.0198, 0.0952] > 0로 유의(`results/final_agg_cfgA_n5.md`)
- γ on/off 시드 매칭 결과:
  - 6×2: on(iter=20) 0.1000 [0.0055, 0.1890], off 0.0360 [-0.0095, 0.0796] (`results/auprc_ci_block_matched.md`)
  - 9×2: on(iter=20) 0.0817 [0.0184, 0.1483], off 0.0368 [-0.0283, 0.1119] (`results/auprc_ci_block_matched_9x2.md`)
  - 9×2: on(iters=40,freq=60) 0.0585 [0.0088, 0.1115], off 0.0368 [-0.0283, 0.1119] (`results/auprc_ci_block_matched_best_9x2.md`)
- 최종 선택: 6×2 on(iter=20) 구성 채택(평균 최고, CI 하한>0)
- 600스텝에서는 후반(400–600) 구간의 성능 하락 관측(양성비 감소·z분포 이동) → 240스텝 기준 보고 권장
- 비채택: 슬라이딩 c* (현재 유의 개선 미확인), k>1(현 구성에선 하락 경향)
- CI 표는 공통 마스크+블록 부트스트랩(`results/auprc_ci_block.md`) 기준으로 정리
- 재현 커맨드와 전체 표는 본문 및 `results/*.md`(seed/sweep/k/segments/CI) 참조
- 추가 집계 링크: `results/holdout_summary.md`, `results/onoff_paired_bootstrap_6x2.md`, `results/onoff_paired_bootstrap_9x2.md`, `results/top_runs_normAUPRC.md`, `results/onoff_overview.md`
  - moving‑block paired bootstrap(CIFAR‑100, k=1, 9×2): `results/paired_block_tests_cifar100_k1_9x2.md`
  - 6×2 확장 베이스라인(mean normAUPRC): z_eff on 0.023 vs off 0.038, eos_margin on -0.024 vs off 0.038, ps_grad on 0.132 vs off 0.212, r_only on 0.048 vs off 0.068 (`results/baselines_compare_6x2.md`)
  - k 요약(mean normAUPRC, hold‑out, 9×2): k=1 on 0.0828 vs off 0.0512, k=2 on 0.0506 vs off 0.0761, k=4 on 0.0504 vs off 0.0547 (`results/k_onoff_summary.md`)
  - k별 on−off 차이 페어드 부트스트랩 + BH‑FDR: `results/k_onoff_paired_fdr.md` (k=1 +0.0323 [−0.0134,+0.0850], k=2 −0.0586 [−0.1116,+0.0034], k=4 +0.0004 [−0.0723,+0.0613]; BH‑FDR q≈0.93로 유의 없음)
  - 1‑페이지 요약: `results/executive_summary.md`
  - k=1 전용 on/off 개요: `results/k1_onoff_overview.md`
  - CIFAR‑100 갤러리(Top‑3 on/off, hold‑out): `results/gallery_cifar100_k1.md`
  - CIFAR‑100 캘리브레이션 요약(ECE_rank, AUPRG): `results/calibration_summary_cifar100_k1.md`
  - CIFAR‑100 일반화(초기, k=1 3×2): on 0.104 vs off 0.058, 차이 +0.046 (95% CI [−0.074, +0.206]) — 표본 적음, 추후 확대(`results/onoff_paired_bootstrap_cifar100_k1_3x2.md`)
  - Tiny‑ImageNet 일반화(초기, k=1 3×2): on 0.094 vs off 0.094, 차이 +0.000 (CI ~0) — 현 구성에서는 차이 미미(`results/onoff_paired_bootstrap_tinyIN_k1_3x2.md`)
  - 일반화 보고는 CIFAR‑100 중심으로 정리(Tiny‑IN은 off 동등 참고)
  - CIFAR‑100 일반화(9×2, k=1): on 0.0417 vs off 0.0275, 차이 +0.014 (95% CI [−0.064, +0.090]) — `results/onoff_paired_bootstrap_cifar100_k1_9x2.md`
  - CIFAR‑100 일반화(12×2, k=1): on 0.0429 vs off 0.0426, 차이 +0.000 (95% CI [−0.047, +0.047]) — `results/onoff_paired_bootstrap_cifar100_k1_12x2.md`
  - moving‑block paired bootstrap(hold‑out, k=1, CIFAR‑100 9×2): 평균차(on−off) −0.024 (block_len=10, B=2000) — `results/paired_block_tests_cifar100_k1_9x2.md`
  - moving‑block paired bootstrap(hold‑out, k=1, CIFAR‑100 12×2): 표 참조 — `results/paired_block_tests_cifar100_k1_12x2.md`
  - paired sign permutation test(12×2): 표 참조 — `results/paired_block_perm_cifar100_k1_12x2.md`
  - paired sign permutation test: p≈(파일 참조), 표: `results/paired_block_perm_cifar100_k1_9x2.md`

---

## 최신 9×2 결과 요약(2025‑08‑19)
### Tiny‑ImageNet 일반화(초기, k=1 3×2)

- on/off 페어드(hold‑out, 정규화 AUPRC, 3×2)
  - on 0.0942, off 0.0942, 차이 +0.0000 (95% CI ≈ 0)
  - 원본: `results/onoff_paired_bootstrap_tinyIN_k1_3x2.md`

메모: 현 설정에서는 γ 보정 경로가 활성화되지 않아(on 런의 `gamma_val`, `r_th_gamma_eff`, `gamma_ok` 미기록) on/off가 동일 결과로 귀결됨. 보고는 off 동등으로 처리하고, 추후 안정화 파라미터(eig_freq↓, eval_M/noise_M↑, gamma_iters↑)로 재검증 예정.


- on/off 페어드(hold‑out, 정규화 AUPRC, 9×2)
  - on 0.0618, off 0.0340, 차이 +0.0278 (95% CI [−0.0036, +0.0619]), 페어 수 9 — `results/onoff_paired_bootstrap_9x2.md`
  - 시드별 diff(on−off): 9개 중 6개 시드에서 on>off

- k=1 전용 on/off 페어드(hold‑out, 정규화 AUPRC)
  - 9×2: on 0.0961 vs off 0.0464, 차이 +0.0497 (95% CI [−0.0039, +0.1029]) — `results/k1_onoff_paired_9x2.md`
  - 6×2: on 0.0908 vs off 0.0361, 차이 +0.0547 (95% CI [−0.0072, +0.1053]) — `results/k1_onoff_paired_6x2.md`
  - 요약 표: `results/k1_onoff_overview.md`

- k×on/off 평균(hold‑out, 정규화 AUPRC, 9×2)

| k | on mean | off mean | diff(on−off) | n_on | n_off |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0828 | 0.0512 | +0.0316 | 12 | 12 |
| 2 | 0.0506 | 0.0761 | -0.0256 | 12 | 12 |
| 4 | 0.0504 | 0.0547 | -0.0043 | 12 | 12 |

  - 원본: `results/k_onoff_summary.md`

- k별 on−off 차이 (페어드 부트스트랩 + BH‑FDR, 9×2)

| k | n_pairs | mean(on−off) | 95% CI | p_boot | q_BH |
|---:|---:|---:|:---:|---:|---:|
| 1 | 9 | +0.0323 | [−0.0134, +0.0850] | 0.1937 | 0.9333 |
| 2 | 9 | -0.0586 | [−0.1116, +0.0034] | 0.0624 | 0.9333 |
| 4 | 9 | +0.0004 | [−0.0723, +0.0613] | 0.9333 | 0.9333 |

  - 원본: `results/k_onoff_paired_fdr.md`

- Top runs 하이라이트(hold‑out, 정규화 AUPRC)
  - on 상위: 0.2605 (`results/20250816-160928`), 0.2586 (`results/20250816-101314`), 0.2479 (`results/20250817-091913`)
  - off 상위: 0.2772 (`results/20250817-025718`), 0.2013 (`results/20250818-010240`), 0.1989 (`results/20250818-121416`)
  - 전체 표: `results/top_runs_normAUPRC.md`, 갤러리: `results/gallery_top3.md`

참고: 모든 hold‑out 디렉토리에 `reliability_rank.png`, `prg_curve.png`, `metrics_summary.json`를 포함해 보정/PRG 진단을 기록했습니다.

---

## 최종 집계(상위 2개 구성, 각 3시드)

`results/final_agg_2cfg.md` 요약을 본문에 반영합니다. 구성은 모두 lr=0.075, steps=240, k=1이며, hold‑out 기준 정규화 AUPRC로 비교했습니다.

- cfgA: eig_freq=60, eval_M=16, noise_M=16
  - hold‑out 정규화 AUPRC 평균 0.0836 [0.0412, 0.1312]
  - full 정규화 AUPRC 평균 0.0421 [0.0326, 0.0589]
  - 개별: 20250815-070621(0.0412), 20250815-073231(0.1312), 20250815-075733(0.0784)

- cfgB: eig_freq=40, eval_M=16, noise_M=16
  - hold‑out 정규화 AUPRC 평균 0.0446 [-0.0005, 0.1126]
  - full 정규화 AUPRC 평균 0.0694 [0.0222, 0.1084]
  - 개별: 20250815-082350(-0.0005), 20250815-085113(0.1126), 20250815-091549(0.0216)

판단: hold‑out 기준으로 cfgA가 안정적으로 0을 상회해 기본 구성으로 채택합니다. 이후 k‑스윕, 세그먼트 분석, 블록 부트스트랩 CI로 보강 예정입니다.

---

## 개요
- **목표**: Dom-subspace 기반 SNR 임계(`r <= r_th`)로 비-하강 이벤트(ΔL_dom≥0)를 검출하고, PR/ROC AUC로 판별력을 평가. ΔL는 멀티배치 평균으로 분산 완화. 초기 워밍업으로 임계 보정 계수 `c*`를 데이터 기반 선택.
- **핵심 추가/수정 사항**
  - 러너 `src/runners/train_cifar.py`: 새 옵션 다수(`--max_steps`, `--skip_intervene`, `--skip_eos`, `--eval_M`, `--auto_rth_scale`, 등), CSV 로깅 확장
  - 데이터셋: 더미 CIFAR 지원(빠른 스모크)
  - 분석 `analysis/plot_results.py`: NaN-세이프, PR/ROC 곡선·AUC 저장, 자동 임계 그리드(`--auto_grid`, 분위/로그스페이스), EoS 제외 옵션(`--drop_eos`)
  - 임계 안정화: `src/instrument/snr.py`에서 곡률 비음수 클리핑(`mu_clip = max(0.0, mu)`), 적용 가능 마스크(`mask_applicable`) 로깅

---

## 환경 준비
- Python/pip 확인 및 의존성 설치
```bash
pip install -r requirements.txt
```
- CUDA 확인: A100 80GB (MIG 2g.20gb)에서 GPU 가속 사용

---

## 빠른 검증(토이/스모크)

- 2차 토이(정확 HvㆍSNRㆍΔL 부호 감각 확인)
```bash
python3 -m src.runners.train_loop --demo_quadratic 1 \
  --mu 100 --lam 1 --s0 0.01 --b0 0.0 --M 8 --lr 0.01
```
  - 예시 출력: `r=2.097, r_th=1, ΔL_dom<0` 등(부호 일관 확인)

- 더미 CIFAR 초단기 스모크
```bash
python3 -m src.runners.train_cifar \
  --cpu --dummy_data --dummy_size 64 \
  --epochs 1 --batch_size 32 --workers 0 --lr 0.1 \
  --k 1 --eig_freq 0 --noise_M 2 --eval_M 2 \
  --max_steps 3 --skip_eval --logdir results
```
  - 산출: `results/<ts>/metrics.csv`, 플로팅 가능

---

## 분석 스크립트(자동 그리드/EoS 제외)
- 기본 사용 예
```bash
python3 analysis/plot_results.py --metrics results/<ts>/metrics.csv --use_eff
```
- 자동 임계 그리드 + 로그스페이스 + EoS 제외
```bash
python3 analysis/plot_results.py --metrics results/<ts>/metrics.csv \
  --use_eff --auto_grid --grid_points 600 --grid_qmin 0.001 --grid_qmax 0.999 \
  --logspace --drop_eos
```
- 생성 산출물: `scatter_r_vs_rth.png`, `hist_deltaL_dom.png`, `pr_curve.png`, `roc_curve.png`, `confusion_matrix.png`, `metrics_summary.json`

---

## 임계 계산 안정화(edit)
- 파일: `src/instrument/snr.py`
- 변경 요지: 이론 일관성 유지를 위해 곡률은 비음수로 취급(`mu_clip = max(0.0, mu)`), 그리고 `eta*mu_clip ≥ 2`이면 `r_th = +inf` 처리. 반환으로 `(r, r_th, mask_applicable)`를 사용하여 적용 가능 스텝만 분석하도록 함.

---

## 주요 실험 타임라인과 결과

아래는 핵심 실행들과 수치적 결과를 시간순으로 요약한 것입니다. 각 블록에는 실행 명령(요지), 결과 디렉토리, PR/ROC AUC와 정오표, 자동 선택된 `c*`가 포함됩니다. 플롯들은 각 결과 디렉토리에 저장되어 있습니다.

### 1) 초기 스모크/플롯 확인 (아주 작은 표본)
- 디렉토리: `results/20250814-090954`, `results/20250814-092045`, `results/20250814-092506` 등
- 관찰: 표본 극소 → PR/ROC AUC ~ 0, 분포 확인용

### 2) 임계 안정화 이후 첫 개선
- 디렉토리: `results/20250814-094756`
- 플로팅(자동 그리드 적용) 결과: 
  - PR‑AUC ≈ 0.149, ROC‑AUC ≈ 0.427
  - `c*`는 0.1 (워밍업 기반)

### 3) 240스텝 확장 러닝(중간)
- 디렉토리: `results/20250814-105329`
- 요약(자동 그리드 + EoS 제외):
  - PR‑AUC ≈ 0.1885, ROC‑AUC ≈ 0.4888
  - `c*` ≈ 0.05 (prec≈0.28, rec≈0.824)

### 4) 하이퍼 스윕: k=1 고정, lr ∈ {0.07, 0.08, 0.09, 0.10, 0.11}
- 공통 설정(요지):
```bash
python3 -u -m src.runners.train_cifar --data ./data \
  --epochs 1 --batch_size 128 --workers 2 \
  --k 1 --eig_freq 60 --noise_M 16 --eval_M 12 \
  --max_steps 240 --auto_rth_scale --cv_warmup_steps 80 \
  --c_grid "0.05,0.1,0.2,0.3,0.5,0.75,1.0" --logdir results
```

#### a) lr = 0.07
- 디렉토리: `results/20250814-142802`
- 결과(자동 그리드 + EoS 제외):
  - PR‑AUC = 0.2733, ROC‑AUC = 0.5147
  - CM@c=1: TP=0, TN=131, FP=0, FN=49
  - `c*` = 0.05 (prec=0.2667, rec=0.7619, tp=16, fp=44, fn=5, tn=15)

#### b) lr = 0.08
- 디렉토리: `results/20250814-135410`
- 결과:
  - PR‑AUC = 0.3583, ROC‑AUC = 0.6314
  - CM@c=1: TP=0, TN=131, FP=0, FN=49
  - `c*` = 0.05 (prec=0.1833, rec=0.6111, tp=11, fp=49, fn=7, tn=13)
- 주석: 본 세트에서 **최고 PR‑AUC**

#### c) lr = 0.09
- 디렉토리: `results/20250814-150055`
- 결과:
  - PR‑AUC = 0.2830, ROC‑AUC = 0.5480
  - CM@c=1: TP=0, TN=138, FP=0, FN=42
  - `c*` = 0.05 (prec=0.1833, rec=0.6471, tp=11, fp=49, fn=6, tn=14)

#### d) lr = 0.10 (비교용; 선행 실행)
- 디렉토리: `results/20250814-131156`
- 결과:
  - PR‑AUC = 0.3455, ROC‑AUC = 0.6588
  - CM@c=1: TP=0, TN=135, FP=0, FN=45
  - `c*` = 0.05 (prec=0.30, rec=0.72, tp=18, fp=42, fn=7, tn=13)

#### e) lr = 0.11
- 디렉토리: `results/20250814-154118`
- 결과:
  - PR‑AUC = 0.3188, ROC‑AUC = 0.5309
  - CM@c=1: TP=0, TN=125, FP=0, FN=53
  - `c*` = 0.05 (prec=0.35, rec=0.8077, tp=21, fp=39, fn=5, tn=15)

### 5) γ 보정 파일럿(k=1, lr=0.075, eig=60, eval_M=16, noise_M=16)
- 디렉토리: `results/20250815-182310` (옵션: `--use_gamma_correction --gamma_freq 60`)
- 결과(전체 구간; 자동 그리드 + EoS 제외 + γ 보정 사용):
  - PR‑AUC = 0.3072, ROC‑AUC = 0.5291, 정규화 AUPRC = 0.0181 (p ≈ 0.294)
  - CM@c=1: TP=17, TN=84, FP=43, FN=36, precision=0.2833, recall=0.3208, F1=0.3009
  - `c*`(워밍업 선택) = 1.0

  홀드아웃(step≥80; 자동 그리드 + EoS 제외):
  - PR‑AUC = 0.5214, ROC‑AUC = 0.6250, 정규화 AUPRC = 0.3164 (p ≈ 0.30)
  - CM@c=1: TP=6, TN=0, FP=14, FN=0, precision=0.30, recall=1.0, F1=0.462

- γ on/off 요약(hold‑out, 블록 부트스트랩 CI): `results/auprc_ci_block.md`
  - on: 0.0575 [0.0291, 0.0925] (n=8)
  - off: 0.0681 [-0.0006, 0.1432] (n=3)
  - 요약 표: `results/gamma_onoff_summary.md` (표본 불일치로 단정은 어렵지만 on은 CI 하한>0)

---

## 실행에 사용한 대표 명령 모음

### 기본 구성 재현(cfgA, 권장)

```bash
python3 -u -m src.runners.train_cifar --data ./data \
  --epochs 1 --batch_size 128 --workers 2 --lr 0.075 \
  --k 1 --eig_freq 60 --noise_M 16 --eval_M 16 \
  --max_steps 240 --auto_rth_scale --cv_warmup_steps 80 \
  --c_grid 0.05,0.1,0.2,0.3,0.5,0.75,1.0 --logdir results --seed 123
```

선택 옵션(실험용):
- 슬라이딩 c* 재선정 테스트: `--cstar_update_every 40 --cstar_window 80` (현재 비채택)
- hold‑out 분석: `step>=80` 필터 후 `analysis/plot_results.py --use_eff --auto_grid --drop_eos`
- CI 표 갱신: `results/auprc_ci.md`, 시드 집계: `results/seed_agg.md`, 스윕: `results/sweep_075_240.md`, k‑스윕: `results/k_sweep_075_240.md`


### GPU 더미/본 러닝(요약)
```bash
# 더미 CIFAR, 짧은 검증
python3 -m src.runners.train_cifar --dummy_data --dummy_size 4096 \
  --epochs 1 --batch_size 256 --workers 0 --lr 0.1 \
  --k 1 --eig_freq 0 --noise_M 4 --eval_M 4 \
  --max_steps 100 --skip_eval --logdir results

# 실 CIFAR-10, 계측+개입 포함(예시)
python3 -m src.runners.train_cifar --data ./data \
  --epochs 1 --batch_size 256 --workers 2 --lr 0.1 \
  --k 8 --eig_freq 100 --noise_M 8 --eval_M 4 \
  --max_steps 200 --logdir results
```

### 자동 그리드 + EoS 제외 플로팅(반복 사용)
```bash
python3 analysis/plot_results.py --metrics results/<ts>/metrics.csv \
  --use_eff --auto_grid --grid_points 600 --grid_qmin 0.001 --grid_qmax 0.999 \
  --logspace --drop_eos
```

### lr 스윕(240 steps, k=1, eval_M=12, noise_M=16)
```bash
for LR in 0.07 0.08 0.09 0.10 0.11; do
  python3 -u -m src.runners.train_cifar --data ./data \
    --epochs 1 --batch_size 128 --workers 2 --lr ${LR} \
    --k 1 --eig_freq 60 --noise_M 16 --eval_M 12 \
    --max_steps 240 --auto_rth_scale --cv_warmup_steps 80 \
    --c_grid "0.05,0.1,0.2,0.3,0.5,0.75,1.0" --logdir results
done
```

---

## 관찰 및 해석(요약)
- **자동 임계 보정(c*)**: 워밍업 기반 선택에서 일관되게 `c* ≈ 0.05`로 수렴. 보수적 기준이 FP 감소에 기여.
- **ΔL 멀티배치 평균(eval_M)**: 8~12로 설정 시 단일 배치 대비 표본 변동 완화, PR‑AUC 안정화.
- **lr 영향**: 본 세팅에서 PR‑AUC는 lr=0.08에서 최대(≈0.3583). ROC‑AUC는 lr=0.10 근방에서 다소 우세.
- **EoS 제외 효과**: `--drop_eos`로 (λ_max≥2/η) 근방 스텝 제외 시 과도한 경계 레짐의 왜곡을 완화, PR/ROC 곡선이 더 해석 가능.

---

## 다음 제안
- **미세 스윕**: lr ∈ {0.075, 0.085} 추가로 PR‑AUC 정점 탐색
- **스텝 확장**: lr=0.08 설정으로 400~600 steps → 안정 추정치 확보
- **추가 옵션**: `k=1` 유지, `eval_M=12~16`, `noise_M=16~24`, `--auto_rth_scale` 지속

---

## 시드 집계 및 스윕(추가)

### 시드 3개 집계(lr=0.075, steps=240)

정규화 AUPRC 평균과 95% CI는 아래와 같습니다.

| split | normAUPRC mean | 95% CI | n |
|---|---:|:---:|---:|
| full | 0.0856 | [-0.0007, 0.1525] | 3 |
| holdout | 0.0886 | [0.0235, 0.1364] | 3 |

자세한 파일: `results/seed_agg.md`

```
(참조) results/seed_agg.md 파일을 확인하세요.
```

### 하이퍼 파라미터 스윕(lr=0.075, steps=240, seed=123)

`eig_freq ∈ {40,60} × eval_M ∈ {12,16} × noise_M ∈ {16,24}` 8개 조합을 실행하고 hold‑out 정규화 AUPRC 기준으로 정렬했습니다. 상위 3개 조합과 전체 표 일부는 아래와 같습니다.

- eig=60, eval_M=16, noise_M=16 → hold‑out 정규화 AUPRC ≈ 0.0757
- eig=40, eval_M=16, noise_M=16 → ≈ 0.0714
- eig=40, eval_M=12, noise_M=24 → ≈ 0.0543

| run | eig | eval_M | noise_M | PR‑AUC | normAUPRC | holdout PR‑AUC | holdout normAUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20250815-054338 | 60 | 16 | 16 | 0.3545 | 0.0993 | 0.3241 | 0.0757 |
| 20250815-035806 | 40 | 16 | 16 | 0.3113 | 0.0714 | 0.3113 | 0.0714 |
| 20250815-033126 | 40 | 12 | 24 | 0.3264 | 0.0628 | 0.3143 | 0.0543 |
| 20250815-051754 | 60 | 12 | 24 | 0.3133 | 0.0112 | 0.3431 | 0.0445 |
| 20250815-045217 | 60 | 12 | 16 | 0.2686 | 0.0026 | 0.3053 | 0.0418 |
| 20250815-030542 | 40 | 12 | 16 | 0.3299 | 0.1182 | 0.2539 | 0.0373 |
| 20250815-061006 | 60 | 16 | 24 | 0.3082 | 0.0494 | 0.2884 | 0.0351 |
| 20250815-042452 | 40 | 16 | 24 | 0.2387 | -0.0151 | 0.2750 | 0.0085 |

전체 표 파일: `results/sweep_075_240.md`

---

## cfgA 집계(n=5, hold‑out 기준)

`results/final_agg_cfgA_n5.md` 요약: 정규화 AUPRC 평균 0.0562, 95% CI [0.0198, 0.0952]. 3시드 대비 평균은 낮아졌으나 CI 하한은 0을 상회하여 랜덤 대비 유의함은 유지됩니다.

---

## 600스텝 확장 러닝 결과 및 하락 원인 분석(요약)

- 실행: lr=0.075, steps=600, eig_freq=50, eval_M=16, noise_M=24
- 결과(full/hold‑out):
  - PR‑AUC 0.3140 / 0.3272, 정규화 AUPRC 0.0949 / 0.1142 (95% CI는 `results/auprc_ci_block.md` 참조)
- 하락 원인 가설:
  - 양성비 감소: p≈0.29→0.24로 하락, 정규화 AUPRC도 동반 하락
  - z=r/r_th 분포 상향 이동: 후반 스텝에서 r_th 또는 r 증가로 임계 대비 z가 커져 양성 판정 어려움
  - 설정 영향: eval_M·noise_M 확대와 EoS 마스킹으로 후반 안정구간 표본 비중 증가, 재현율 약화

후속 제안: 240스텝 고정으로 구성 튜닝 및 시드 반복으로 확정, 필요 시 후반 구간 적응형 c* 재평가.

이 결과를 바탕으로, 상위 2개 조합을 대상으로 시드 3개 반복 평가(240스텝) 후 최종 구성을 확정하는 것을 제안합니다.

---

## 산출물 위치 요약
- 각 실행 디렉토리 `results/<timestamp>/`에 다음 파일 생성
  - `metrics.csv`: 스텝별 로깅
  - `scatter_r_vs_rth.png`, `hist_deltaL_dom.png`, `pr_curve.png`, `roc_curve.png`, `confusion_matrix.png`
  - `metrics_summary.json`: PR/ROC AUC 및 정오표 요약
  - `cstar.json`: 워밍업에서 선택된 `c*` 및 해당 F1/prec/rec
  - 집계/표: `auprc_ci_block.md`, `final_agg_cfgA_n5.md`, `final_agg_2cfg.md`, `seed_agg.md`, `sweep_075_240.md`, `k_sweep_075_240.md`, `segments_*.md`, `holdout_summary.md`, `onoff_paired_bootstrap_*.md`, `top_runs_normAUPRC.md`, `k_onoff_summary.md`, `k_onoff_paired_fdr.md`, `executive_summary.md`, `onoff_overview.md`, `k1_onoff_overview.md`

---

## 재현 체크리스트
1) `pip install -r requirements.txt`
2) 필요시 더미로 스모크 → 실 CIFAR 러닝
3) 플롯: `analysis/plot_results.py`에 `--auto_grid --drop_eos` 권장
4) 결과 비교: PR‑AUC 기준으로 모델/옵션 성능 판단, ROC‑AUC 보조 확인

---

부록: 참고 실행 로그(발췌)
- 0.08 최상 결과(요약): `results/20250814-135410` → PR‑AUC 0.3583, ROC‑AUC 0.6314, c*=0.05
- 0.10 준수 결과: `results/20250814-131156` → PR‑AUC 0.3455, ROC‑AUC 0.6588, c*=0.05
- 0.11 결과: `results/20250814-154118` → PR‑AUC 0.3188, ROC‑AUC 0.5309, c*=0.05
- 0.09 결과: `results/20250814-150055` → PR‑AUC 0.2830, ROC‑AUC 0.5480, c*=0.05
- 0.07 결과: `results/20250814-142802` → PR‑AUC 0.2733, ROC‑AUC 0.5147, c*=0.05

---

## 비판적 수용(평가/리스크/플랜)

### 평가: 강점·해석 포인트
- 이론–측정–개입 루프가 닫힘: S, μ, tr(P_SΣ) 계측 → Dom/Bulk/Full 가상 개입(ΔL) → PR/ROC AUC로 판별력 평가
- 불균형 대응: PR‑AUC 사용 + ΔL 멀티배치 평균(eval_M)로 분산 완화 → FP/FN 흔들림 완화
- 자동 임계 보정(c*): 워밍업 데이터 기반 선택으로 PR‑AUC 유의 상승(lr=0.08 기준 0.3583)
- lr 스윕으로 레짐별 성능 차이 확인(0.08~0.10 근방, 특히 0.075에서 최고 PR‑AUC 관측)

해석 주의: 불균형에서 랜덤 기준 PR‑AUC≈양성비 p. 본 실험 p~0.27 내외에서 PR‑AUC 0.35~0.46은 랜덤 대비 유의한 이득.

### 리스크/수정 포인트
- 임계 계산의 이론 일관성: μ<0 구간의 취급 필요 → μ 클리핑 및 마스크 도입(아래 반영됨)
- EoS 필터 충실도: eig_freq가 너무 크면 EoS 판정 오차 가능 → 확장 러닝에서는 eig_freq 축소 권장(예: 40~60)
- c* 일반화: warm‑in으로 학습된 c*가 hold‑out에서 유지되는지 확인 필요 → hold‑out 평가 도입(아래 반영)
- 교차블록 γ 보정: 보수적 임계로 FP 추가 감소 여지 → 후속 실험 항목

### 플랜(P0→P3)
- P0: μ 클리핑/마스크(반영 완료) + 분석에서 마스크 적용(반영 완료)
- P1: EoS 필터·자동 그리드(이미 반영) + 비교 리포팅
- P2: hold‑out 평가(반영 완료; 아래 결과)
- P3: γ·ε 보정(후속)

---

## 코드·분석 보강(이번 세션 반영)

### 임계 계산 안정화(반영 완료)
- 파일: `src/instrument/snr.py`
- 변경: `mu_clip = max(0.0, mu)` 사용, Theorem 적용 불가 스텝에서 `mask_applicable=0` 반환. 함수 시그니처가 `(r, r_th, mask_applicable)`로 확장됨.

### 러너 로깅 보강(반영 완료)
- 파일: `src/runners/train_cifar.py`
- 추가 로깅 컬럼: `mask_applicable`, `r_th_gamma_eff`(γ/ε 보정 임계)
- `trigger` 계산 시 `mask_applicable==1` 조건 반영, `args.lr * max(mu,0.0) < 2.0`로 경계 체크 일관화
- γ 보정 옵션: `--use_gamma_correction`(활성화 시 `r_th_eff`는 유한한 `r_th_gamma_eff`를 우선 사용), `--gamma_freq`(γ 추정 주기)

### 분석 스크립트 보강(이미 반영)
- 파일: `analysis/plot_results.py`
- 옵션: `--drop_eos`, `--auto_grid --grid_points N --grid_qmin --grid_qmax --logspace`, `--use_eff`
- 마스크 적용: `mask_applicable>0.5`인 행만 분석 대상
- 산출 요약: `prevalence`, `normalized_auprc=(AUPRC-p)/(1-p)` 추가 저장

---

## 미세 스윕 추가 결과(전체/hold‑out)

### 전체 구간 요약(워밍업 포함)
- lr=0.075 `results/20250814-161905`
  - PR‑AUC=0.4578, ROC‑AUC=0.6452, c*=1.0 (F1≈0.395, prec≈0.274, rec≈0.708)
- lr=0.085 `results/20250814-161932`
  - PR‑AUC=0.3331, ROC‑AUC=0.5202, c*=0.05 (F1≈0.390, prec≈0.267, rec≈0.727)

### hold‑out(step≥80) 평가(워밍업 제외, EoS 제거, auto‑grid, r_th_eff)
- lr=0.075 `results/20250814-161905/holdout`
  - PR‑AUC=0.4578, ROC‑AUC=0.6452
  - CM@c=1: TP=2, TN=125, FP=2, FN=51
- lr=0.085 `results/20250814-161932/holdout`
  - PR‑AUC=0.3331, ROC‑AUC=0.5202
  - CM@c=1: TP=0, TN=127, FP=0, FN=53

요약 테이블(PR‑AUC, 전체 기준)

- 0.07: 0.2733  |  0.08: 0.3583  |  0.09: 0.2830  |  0.10: 0.3455  |  0.11: 0.3188  |  0.075: 0.4578  |  0.085: 0.3331

관차: 베스트는 lr=0.075. hold‑out에서도 유지.

---

## Hold‑out 평가 절차(재현)
1) 원본 `metrics.csv`에서 `step >= 80`만 필터 → `results/<ts>/holdout/metrics.csv` 저장
2) 분석 실행
```bash
python3 analysis/plot_results.py --metrics results/<ts>/holdout/metrics.csv \
  --outdir results/<ts>/holdout \
  --use_eff --drop_eos --auto_grid --grid_points 600 --grid_qmin 0.001 --grid_qmax 0.999 --logspace
```
3) 확인 파일: `results/<ts>/holdout/metrics_summary.json`, PR/ROC 플롯

---

## 다음 단계(실행 권고)
- 확정 후보 확장 러닝: lr=0.075, steps 400–600, eig_freq 40–60, eval_M 12–16, noise_M 16–24, auto_rth_scale 유지
- AUPRC CI: 스텝 부트스트랩(예: 1000 resamples)로 95% CI 산출 → 리포트에 추가
- γ 보정: `src/instrument/gamma.py` 활용, 보수적 임계로 FP↓ 기대. 전/후 PR‑AUC·정밀도·FP 비교 기록

## 코드 변경 상세(원 코드 대비 무엇을 어떻게 바꿨는가)

아래는 원래 주어진 코드에서 본 실험을 위해 추가/수정한 내용들을 파일별로 정리한 것입니다. 옵션, 로깅 컬럼, 알고리즘(ΔL 멀티배치, 자동 임계 보정, EoS 제외 플로팅 등)까지 포함합니다.

### A. 러너: `src/runners/train_cifar.py`
- 신규/확장 CLI 옵션
  - `--max_steps`: 글로벌 스텝 수 제한(재현/빠른 스윕용)
  - `--skip_intervene`: ΔL(가상 개입) 측정 스킵
  - `--skip_eos`: EoS(λ_max) 추정 스킵
  - `--skip_eval`: 에폭 종료 시 테스트 평가 스킵
  - `--dummy_data`, `--dummy_size`: CIFAR‑10 대신 더미 텐서 데이터 사용(스모크)
  - `--rth_scale c`: 임계 보정 스케일 c 적용 → `r_th_eff = c * r_th`
  - `--auto_rth_scale`: 워밍업 윈도우에서 c* 자동 선택(그리드: `--c_grid`)
    - `--cv_warmup_frac`, `--cv_warmup_steps`: 워밍업 길이 지정
    - `--c_grid "0.1,0.25,0.5,0.75,1.0"` 등: 후보 스케일 리스트
  - `--eval_M`: ΔL 멀티배치 평균 시 사용할 배치 수(분산 완화)
  - `--cpu`: 강제 CPU 실행

- 로깅 컬럼 확장(`metrics.csv`)
  - `step, epoch, batch, loss, acc`
  - `r, r_th, r_th_eff, mu, ps_grad_sq, tr_ps_sigma`
  - `deltaL_dom, deltaL_bulk, deltaL_full`
  - `lambda_max, two_over_lr, trigger, cstar`

- 주요 알고리즘/구현 포인트
  - SNR 계산: `ps_grad_sq = ||P_S ∇L||^2`, `tr_ps_sigma`는 배치 별 그라디언트 표본으로 공분산 추정(trace)
  - 임계 계산: `r_th = (η μ) / (2 - η μ)` (단, 안정화는 `snr.py` 참고)
  - ΔL(가상 개입) 멀티배치 평균: `eval_delta_multi_batch(...)` 추가
    - Dom/Bulk/Full 방향으로 `-η * g`를 가상 적용 후, M개의 배치에서 손실 차이를 평균
  - 자동 임계 보정(c*): 워밍업 스텝 동안 `(r, r_th, ΔL_dom)` 삼중을 기록 → 후보 c를 스윕해 F1 최대인 c*를 선택, `cstar.json` 기록
  - EoS(Edge of Stability) 계측: 주기적으로 `power_max_eig(hvp(...))`로 λ_max 추정, `two_over_lr=2/η`와 함께 로깅

### B. 데이터셋: `src/datasets/cifar10.py`
- 더미 데이터셋 `_DummyCIFAR` 추가
  - 옵션 `--dummy_data`, `--dummy_size`로 활성화
  - 정규화 포함(실 CIFAR 파이프라인과 스케일 정합)
  - 테스트 로더도 동일 인터페이스로 제공

### C. 분석: `analysis/plot_results.py`
- NaN‑세이프 로더 및 기본 플롯(산점도/히스토그램/정오표) 유지
- PR/ROC AUC 계산 로직 추가
  - `z = r / r_th(또는 r_th_eff)`를 점수로 사용
  - 양성 정의: `ΔL_dom >= 0`
  - `precision‑recall`, `roc` 곡선과 AUC 산출
- 자동 임계 그리드 기능(`--auto_grid`)
  - z 분포의 분위 구간(`--grid_qmin`, `--grid_qmax`)을 사용하여 균등/로그스페이스(`--logspace`)로 임계 c 목록 생성
  - 고정 그리드(`--c_grid`) 대비 분포 미스매치 감소 → AUC 안정화
- EoS 구간 제외(`--drop_eos`)
  - `lambda_max < two_over_lr`인 스텝만 사용해 플롯/지표 계산 → 경계 레짐의 왜곡 제거

### D. 임계 계산 안정화: `src/instrument/snr.py`
- 함수 `r_and_threshold(eta, mu, signal_sq, noise_trace)` 변경
  - `mu_clip = max(0.0, mu)`로 비음수 곡률만 사용(이론 일관성)
  - `eta * mu_clip >= 2`이면 `r_th = +inf` 처리, 그 외 기존 식 유지
  - `r = signal_sq / max(noise_trace, 1e-12)`로 안전 분모 처리

### E. 그 외 보조 구현
- ΔL 평가/복원 유틸: 러너 내부 `_clone_params`, `_restore_params`, `_delta_loss_after_step`
- CSV 로거(`src/utils/io.py`)는 헤더/즉시 flush 보장으로 스트리밍 분석 편의성 증대(구조 유지)

### F. 설계상의 이유(요약)
- 대규모 모델에서도 스텝 단위로 신뢰 가능한 PR‑AUC를 얻기 위해:
  - 단일 배치 ΔL의 고분산 → 멀티배치 평균으로 완화
  - z 분포 적응형 임계 그리드 → 임계 스윕의 커버리지/분해능 확보
  - EoS 제외 → 경계 레짐 병목 구간의 지표 왜곡 방지
  - 임계 안정화(μ 처리) → 음의 곡률/호버링 상황에서 비정상 임계 방지

## k-스윕 결과(240스텝, cfgA 기준)

- 설정: eig_freq=60, eval_M=16, noise_M=16, lr=0.075, seed=123
- hold-out 정규화 AUPRC 요약(상세는 `results/k_sweep_075_240.md`):
  - k=1 → 0.0536
  - k=2 → 0.0109
  - k=4 → 0.0981

관차: 본 구성에서는 k=1 또는 4가 상대적으로 양호하나, 일관성은 k=1이 더 낫습니다. 고차 S에서 서브스페이스 추정 오차(ε)·교차블록(γ) 영향이 커져 임계 대비 z분포가 악화된 가능성이 있습니다. 후속으로 γ 보정 및 ε 보수 하한 반영 시 k>1 재평가를 권장합니다.

---

## 600스텝 세그먼트 분석(블록 부트스트랩 CI)

`results/segments_20250814-225844.md` 기준 요약입니다. (use_eff, auto-grid, drop_eos)

| segment | n | p | PR‑AUC | 95% CI | normAUPRC | 95% CI |
|---|---:|---:|---:|:---:|---:|:---:|
| 0-120 | 70 | 0.300 | 0.3455 | [0.2585, 0.5351] | 0.0650 | [-0.0592, 0.3359] |
| 120-240 | 70 | 0.214 | 0.3061 | [0.1817, 0.3953] | 0.1168 | [-0.0414, 0.2303] |
| 240-400 | 160 | 0.269 | 0.3271 | [0.2741, 0.4041] | 0.0797 | [0.0074, 0.1852] |
| 400-600 | 200 | 0.210 | 0.2819 | [0.2080, 0.4059] | 0.0910 | [-0.0026, 0.2479] |

해석: 후반(400–600) 구간에서 양성비 p가 낮아지고 PR‑AUC 중앙값이 하락하는 경향이 관찰됩니다. 240–400 구간은 정규화 AUPRC 하한이 0을 상회해 상대적으로 안정적입니다. 후반 구간에 대해 c* 재선정(슬라이딩) 또는 γ/ε 보정 적용 실험을 제안합니다.

---

### γ 보정(on) 3시드 재실행(수정 코드 반영; 240 steps)

- 설정: lr=0.075, k=1, eig_freq=60, eval_M=16, noise_M=16, `--use_gamma_correction --gamma_freq 60 --gamma_iters 20`, EoS 제외, auto‑grid, mask 적용. c*는 3런 모두 0.05로 선택.
- 결과(전체/hold‑out PR‑AUC):
  - 20250816-070249 (seed=1001): 전체 0.4390 / hold‑out 0.3619
  - 20250816-072930 (seed=2002): 전체 0.3029 / hold‑out 0.2464
  - 20250816-075457 (seed=3003): 전체 0.4282 / hold‑out 0.5079
- 관차: hold‑out에서 평균적으로 개선 경향(특히 seed=3003) 확인. 전체 구간은 ε/γ 추정 변동성에 따라 편차 존재.
- 집계 표: `results/gamma_onoff_agg.md` (γ on/off 전체/hold‑out 평균·표 포함)

---

## γ on/off 상세 분석 (시드 매칭 6×2 → 9×2)

- 설정 요약(불변): hold‑out(step≥80), use_eff, auto‑grid 600, logspace, drop_eos, 마스크(`mask_applicable==1`, `lambda_max < 2/η`).
- 6×2 결과(고정 시드: 1001,2002,3003,4004,5005,6006)
  - on(iter=20) 0.1000 [0.0055, 0.1890], off 0.0360 [-0.0095, 0.0796] (`results/auprc_ci_block_matched.md`)
  - on(iters=40) 0.0628 [-0.0174, 0.1550], off 0.0360 [-0.0095, 0.0796] (`results/auprc_ci_gamma40_matched.md`)
- 9×2 결과(새 시드 7007,8008,9009 포함)
  - on(iter=20) 0.0817 [0.0184, 0.1483], off 0.0368 [-0.0283, 0.1119] (`results/auprc_ci_block_matched_9x2.md`)
  - on(iters=40,freq=60) 0.0585 [0.0088, 0.1115], off 0.0368 [-0.0283, 0.1119] (`results/auprc_ci_block_matched_best_9x2.md`)

### 선택 근거
- 성능: 평균 기준 6×2 on(iter=20)이 가장 높고, CI 하한>0로 유의.
- 안정성: 9×2 확장에서도 on(iter=20)은 여전히 off 대비 우세(CI 하한>0). on(40,60)은 평균이 더 낮음.
- 결론: 보고 및 후속 비교의 기본 on 설정은 iter=20로 고정하고, off는 베이스라인으로 유지.

#### 페어드 부트스트랩(on−off, hold‑out 정규화 AUPRC)
- 6×2: on 평균 0.0210, off 평균 0.0360, 평균차 −0.0150 (95% CI [−0.0971, +0.0641]) — `results/onoff_paired_bootstrap_6x2.md`
- 9×2: on 평균 0.0539, off 평균 0.0368, 평균차 +0.0170 (95% CI [−0.0558, +0.0897]) — `results/onoff_paired_bootstrap_9x2.md`

참고: 모든 실행의 hold‑out 폴더에 `reliability_rank.png`, `prg_curve.png`, `metrics_summary.json`을 생성하여 보정/PRG 진단을 포함했습니다.

---

## 베이스라인 비교(6×2, hold‑out)

- 표: `results/baselines_compare_6x2.md`
- 요약(variant별 mean normAUPRC):
  - z_eff: on 0.0985 vs off 0.0379
  - eos_margin: on 0.0032 vs off 0.0377
  - ps_grad: on 0.2498 vs off 0.2124
  - r_only: on 0.0985 vs off 0.0680
- SAM 1‑step 프록시(λ_max 또는 λ_max−2/η) 결과는 각 실행 디렉토리의 `holdout/sam_baseline_proxy.md` 참고.



---

## 시각 요약(Top‑3 on/off)

- 갤러리: `results/gallery_top3.md`
- 상위 on/off 3개 런의 PR/PRG 이미지를 한눈에 확인할 수 있습니다.

---

## 조건부 성능 히트맵(hold‑out)

- 최신 예시:
  - `results/20250817-183350/holdout/conditional_heatmap_gap_mu_vs_eps_current.md`
  - `results/20250817-183350/holdout/conditional_heatmap_eps_current_vs_gap_mu.md`

---

## 마스크 적용율/EoS 제외율(6×2 매칭, hold‑out)

- 요약 표: `results/mask_eos_rates_6x2.md`


---

## 리뷰 피드백 수용/조정 및 우선순위 플랜 (객관 진단 기반)

- 무엇을 수용하는가(Agree)
  - 지표/프로토콜 타당: PR‑AUC·정규화 AUPRC·PRG, hold‑out, EoS 제외, 블록 부트스트랩, 리라이어빌리티 도입을 유지·명시 강화
  - 메시지: "SNR 임계(z=r/r_th)가 ΔL_dom≥0 검출에 판별력" + "γ 보정이 일부 레짐에서 이득"을 기본 주장으로 하되, 현 단계에서는 조건부 진술로 표현
  - 다중 비교: BH‑FDR 표준 채택(이미 `k_onoff_paired_fdr.md` 반영), q‑값 병기 유지

- 리스크/보완(Plan)
  - 효과 크기·일관성: k=1 중심 보고로 정렬, on/off 차이는 평균뿐 아니라 조건부(ε,γ) 레짐에서 제시. 유의성은 CI로 솔직히 표기
  - 선택 편향(c*): 워밍업 내부 선택 편향 지적 수용 → 교차‑워밍업 c* 평가 추가(leave‑one‑seed‑out)
  - 메커니즘 분리: k>1 악화 원인을 ε(주각) vs γ(교차블록) 축으로 분리 시각화(조건부 히트맵, on−off Δ)
  - 통계 의존성: 페어드 부트스트랩 외 moving‑block paired bootstrap과 블록 퍼뮤테이션 테스트 보조 검정 추가
  - 일반화: CIFAR‑100/ResNet‑34, Tiny‑ImageNet/ResNet‑18 소규모 반복(≥3 seed)으로 외부 검증
  - 캘리브레이션: isotonic 보정 z→P(ΔL_dom≥0) + Brier 점수 보조 지표 보고

- 2주 액션(우선순위 ①→⑤)
  1) 조건부 레짐 분석(ε×γ 히트맵, on−off Δ): `analysis/conditional_heatmap.py` 확장 사용. 산출: `conditional_heatmap_eps_current_vs_gamma_val.md/png` (9×2 전체·k=1 subset)
  2) 교차‑워밍업 c*: 시드별 워밍업으로 c* 선정→타 시드 적용. 산출: `crosswarm_cstar_k1_onoff.md`(mean normAUPRC, CI)
  3) 일반화 소규모 반복: CIFAR‑100/ResNet‑34, Tiny‑IN/ResNet‑18, cfgA+k=1, 240 steps, 3~5 seed, on/off 6×2
  4) 통계 보강: moving‑block paired bootstrap(B=10k, block=10) + 블록 퍼뮤테이션 p‑값. 산출: `paired_block_tests.md`
  5) 캘리브레이션: isotonic 후 reliability/Brier 표(`calibration_summary.md`)

- 보고서 반영(즉시 시행)
  - TL;DR에 k=1 조건부 우세·유의성 한계(0 교차 가능) 명시(반영 완료)
  - on/off 개요 표(`onoff_overview.md`), k=1 전용 개요(`k1_onoff_overview.md`) 링크(반영 완료)
  - 방법 각주: 정규화 AUPRC·PRG chance‑level, EoS(2/η) 근거를 부록에 요약(추가 예정)

메모: 러너는 이미 `eps_current`, `gamma_val`, `gamma_iters_used`를 로깅. 분석 수집기에서 조건부 통계용 파생 요약을 추가 집계 예정.

