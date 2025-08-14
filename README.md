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
