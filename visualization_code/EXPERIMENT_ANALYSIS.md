# Experiment Results Analysis

## Summary of Experiments Conducted

### Experiment 1: Incremental Data Study
- **Runs**: 149/150 successful
- **Best Configuration**: CNN + 25% 2018 data + `lin_bal_stab` penalty
- **Best Test R²**: 0.4438

### Experiment 2: Comprehensive Architecture Comparison
- **Best Configuration**: CNN + RMSprop + no penalty
- **Best Test R²**: 0.467

### Experiment 3: Penalty Ablation Study
- **16 configurations** tested (all combinations of 4 penalty types)
- **Key finding**: Clear tradeoff between balance and prediction

---

## Key Findings

### 1. Causal Validity vs Predictive Accuracy Tradeoff

| Optimization Goal | Best Config | Outcome R² | Balance Score |
|-------------------|-------------|------------|---------------|
| Prediction | None/ci_penalty | 0.467-0.38 | 0.93-0.95 (poor) |
| Causal Balance | balancing | 0.34 | 0.63 (good) |

### 2. Treatment Balance AUC Results (Ideal = 0.5)

Configurations sorted by balance (best first):
```
~ balancing                               : 0.315 +/- 0.012
~ linearization_balancing_ci_penalty      : 0.345 +/- 0.053
~ linearization_balancing_ci_penalty_stability: 0.354 +/- 0.053
~ linearization_balancing                 : 0.356 +/- 0.053
~ linearization_balancing_stability       : 0.363 +/- 0.020
+ baseline                                : 0.464 +/- 0.026
+ linearization                           : 0.518 +/- 0.004
```

Configurations marked with `~` achieve balance AUC < 0.4 (closer to ideal 0.5).

### 3. Outcome Prediction R² (Higher = Better)

```
ci_penalty_stability    : 0.382 +/- 0.005
ci_penalty              : 0.379 +/- 0.005
stability               : 0.373 +/- 0.014
baseline                : 0.370 +/- 0.016
balancing               : 0.342 +/- 0.000
```

### 4. Penalty Effect Summary

| Penalty | Effect on R² | Effect on Balance |
|---------|--------------|-------------------|
| **Balancing** | -8-10% | Strong improvement (0.93 → 0.63) |
| **Linearization** | -2-3% | Slight degradation |
| **CI Penalty** | +1-2% | Neutral |
| **Stability** | +0-1% | Slight improvement |

---

## Comparison to Initial Results

Initial finding: CNN + 75% 2018 + `lin_bal` + seed=42

The systematic experiments show this was likely an artifact of:
1. Single seed variability
2. Limited penalty configurations tested
3. The "best" changes depending on optimization criterion (prediction vs balance)

---

## Recommendations

### For Causal Mediation Analysis
Use `balancing` or `linearization_balancing_ci_penalty`:
- Accept R² ~ 0.33
- Achieve balance score 0.63-0.69 (vs 0.93 baseline)
- Treatment effect estimates will be less confounded

### For Pure Prediction
Use `none` or `ci_penalty_stability`:
- Achieve R² ~ 0.38-0.47
- Balance is poor but irrelevant for prediction task

### For Publication
Report both:
1. Predictive baseline (no penalties): R² = 0.467
2. Causally-valid model (balancing): R² = 0.342, Balance = 0.63
3. Show the tradeoff curve

---

## Outstanding Items

1. **1 failed run** from Experiment 1 - need to identify which configuration
2. **Data files** should be committed to `visualizations/incremental_data_experiment/data/` (output directory)
3. **Figures** should be generated and committed

---

## Raw Results Tables

### Ablation Study Full Results

| config_name | outcome_r2_mean | outcome_r2_std | balance_score | mediator_r2 | effective_dim |
|-------------|-----------------|----------------|---------------|-------------|---------------|
| balancing | 0.3422 | 0.0002 | 0.6299 | 0.4018 | 3.0 |
| balancing_ci_penalty | 0.3190 | 0.0138 | 0.7319 | 0.4348 | 3.0 |
| balancing_ci_penalty_stability | 0.3313 | 0.0284 | 0.7417 | 0.4326 | 3.0 |
| balancing_stability | 0.3291 | 0.0123 | 0.7343 | 0.4197 | 3.5 |
| baseline | 0.3702 | 0.0164 | 0.9279 | 0.5060 | 4.5 |
| ci_penalty | 0.3793 | 0.0050 | 0.9463 | 0.5010 | 4.5 |
| ci_penalty_stability | 0.3816 | 0.0053 | 0.9504 | 0.4953 | 4.5 |
| linearization | 0.3612 | 0.0206 | 0.9636 | 0.4283 | 4.0 |
| linearization_balancing | 0.3276 | 0.0149 | 0.7120 | 0.4006 | 3.0 |
| linearization_balancing_ci_penalty | 0.3308 | 0.0108 | 0.6891 | 0.3750 | 3.0 |
| linearization_balancing_ci_penalty_stability | 0.3210 | 0.0146 | 0.7083 | 0.4124 | 3.0 |
| linearization_balancing_stability | 0.3184 | 0.0237 | 0.7255 | 0.3764 | 3.0 |
| linearization_ci_penalty | 0.3674 | 0.0035 | 0.9633 | 0.4597 | 4.0 |
| linearization_ci_penalty_stability | 0.3771 | 0.0067 | 0.9342 | 0.4648 | 4.0 |
| linearization_stability | 0.3560 | 0.0174 | 0.9792 | 0.4278 | 4.0 |
| stability | 0.3729 | 0.0137 | 0.8474 | 0.5104 | 4.5 |
