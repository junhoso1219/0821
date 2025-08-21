# domopt-exp — Dom‑OPT Predict→Intervene→Verify

**What's new in this drop**  
- Runner options: `--max_steps`, `--skip_intervene`, `--skip_eos`, `--skip_eval`, `--dummy_data`, `--dummy_size`  
- Threshold controls: `--rth_scale`, `--auto_rth_scale`, `--cv_warmup_frac|--cv_warmup_steps`, `--c_grid`  
- ΔL variance reduction: `--eval_M` (multi‑batch average)  
- Analysis: **NaN‑safe**, **PR/ROC curves & AUC** (c‑sweep over `z=r/r_th`), `--use_eff` to use corrected threshold `r_th_eff`

Quick smoke:
```bash
pip install -r requirements.txt
python -m src.runners.train_loop --demo_quadratic 1
# CIFAR-10 with dummy tensors (no download), fast check
python -m src.runners.train_cifar --dummy_data --max_steps 5 --eig_freq 0 --skip_intervene --skip_eos --skip_eval
```

Data setup:
- We do NOT version `data/` in git. Prepare datasets locally at `data/`:
  - CIFAR‑10/100: will be auto‑downloaded by runners when `--data ./data --dataset cifar10|cifar100`
  - Tiny‑ImageNet‑200: place as `data/tiny-imagenet-200/` or run:
    ```bash
    mkdir -p data && cd data
    wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O tiny-imagenet-200.zip
    unzip -q -o tiny-imagenet-200.zip
    ```
  - Custom: put under `data/<name>/...` and pass `--data ./data --dataset <name>` if supported.

Recommended runs:
```bash
# CIFAR-10 baseline (k=1, 240 steps)
python -m src.runners.train_cifar --data ./data --dataset cifar10 \
  --epochs 1 --batch_size 128 --workers 2 --lr 0.075 \
  --k 1 --eig_freq 60 --noise_M 8 --eval_M 8 \
  --max_steps 240 --auto_rth_scale --cv_warmup_steps 80 \
  --c_grid 0.05,0.1,0.2,0.3,0.5,0.75,1.0 --logdir results --seed 1001

# Hold-out analysis (step>=80), EoS drop, auto-grid
python analysis/plot_results.py --metrics results/<ts>/metrics.csv --use_eff --auto_grid --drop_eos
awk 'BEGIN{FS=","} NR==1{for(i=1;i<=NF;i++){if($i=="step") s=i} print; next} NR>1{if($s>=80) print $0}' \
  results/<ts>/metrics.csv > results/<ts>/holdout/metrics.csv
python analysis/plot_results.py --metrics results/<ts>/holdout/metrics.csv --outdir results/<ts>/holdout \
  --use_eff --auto_grid --drop_eos --plot_reliability --plot_prg
```
