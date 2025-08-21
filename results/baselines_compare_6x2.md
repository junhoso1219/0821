### Baselines compare (normAUPRC means, 6×2 matched)

| score | on mean | off mean | on−off | n |
|---|---:|---:|---:|---:|
| z_eff | 0.0230 | 0.0379 | -0.0149 | 6 |
| eos_margin | -0.0235 | 0.0377 | -0.0612 | 6 |
| ps_grad | 0.1324 | 0.2124 | -0.0800 | 6 |
| r_only | 0.0476 | 0.0680 | -0.0204 | 6 |
| lam_max | -0.0235 | 0.0377 | -0.0612 | 6 |
| grad_norm | nan | nan | +nan | 0 |
| r_full | nan | nan | +nan | 0 |
| sam1_full | 0.0917 | 0.1287 | -0.0370 | 6 |

---

### Per-seed values (normAUPRC)

- seed=1001; z_eff: on=0.0185, off=-0.0235; eos_margin: on=0.0130, off=0.0992; ps_grad: on=0.0035, off=0.2987; r_only: on=0.0157, off=0.0655; lam_max: on=0.0130, off=0.0992; grad_norm: on=nan, off=nan; r_full: on=nan, off=nan; sam1_full: on=0.0742, off=0.0915
- seed=2002; z_eff: on=-0.0650, off=0.0869; eos_margin: on=-0.0066, off=-0.0790; ps_grad: on=0.3799, off=0.2497; r_only: on=-0.1444, off=0.0370; lam_max: on=-0.0066, off=-0.0790; grad_norm: on=nan, off=nan; r_full: on=nan, off=nan; sam1_full: on=0.0272, off=0.0686
- seed=3003; z_eff: on=0.0802, off=-0.0359; eos_margin: on=-0.0976, off=-0.0070; ps_grad: on=0.0024, off=0.1720; r_only: on=0.0948, off=0.0411; lam_max: on=-0.0976, off=-0.0070; grad_norm: on=nan, off=nan; r_full: on=nan, off=nan; sam1_full: on=-0.0413, off=0.0722
- seed=4004; z_eff: on=0.1387, off=0.0512; eos_margin: on=-0.0771, off=0.0561; ps_grad: on=0.1927, off=0.2888; r_only: on=0.1152, off=0.1341; lam_max: on=-0.0771, off=0.0561; grad_norm: on=nan, off=nan; r_full: on=nan, off=nan; sam1_full: on=0.1412, off=0.2713
- seed=5005; z_eff: on=0.0001, off=0.0349; eos_margin: on=0.0585, off=0.1217; ps_grad: on=0.0395, off=0.3094; r_only: on=0.0787, off=0.1152; lam_max: on=0.0585, off=0.1217; grad_norm: on=nan, off=nan; r_full: on=nan, off=nan; sam1_full: on=0.3347, off=0.0648
- seed=6006; z_eff: on=-0.0346, off=0.1138; eos_margin: on=-0.0314, off=0.0352; ps_grad: on=0.1763, off=-0.0444; r_only: on=0.1256, off=0.0149; lam_max: on=-0.0314, off=0.0352; grad_norm: on=nan, off=nan; r_full: on=nan, off=nan; sam1_full: on=0.0145, off=0.2038
