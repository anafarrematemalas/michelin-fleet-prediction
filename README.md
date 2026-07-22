# Michelin Fleet Prediction — EM Lyon Data Science Mission

Predict `total_vehicles` for 254 Turkey vehicle configurations using synthetic data augmentation and gradient boosting.
**Best result: WMSE = 176.6M** (vs 1,005M baseline — **82% improvement**).

> Under the supervision of Michelin and emlyon Business School.

---

## Problem Statement

**Task**: Reconstruct Turkey's complete vehicle fleet across all (maker × segment × energy × body_style × age) configurations.

**Challenge**: 73 of 254 test configurations (28.7%) are entirely absent from Turkey's training data — a cold-start problem that pure regression cannot solve. The model must estimate vehicle counts for combinations it has never seen in Turkey.

**Input**:
- 11 European countries, 91,763 training rows
- 254 Turkey test rows with a `baseline_total_vehicles` estimate to improve upon

**Output**: One `total_vehicles` prediction per test row.

---

## Metrics

### WMSE — Weighted Mean Squared Error *(competition metric)*

Each squared error is weighted by the true vehicle count. Predicting a popular model correctly matters far more than a rare one.

```
WMSE = Σ(y_true · (y_true − y_pred)²) / Σ(y_true)
```

**Example**: Config A has 50,000 vehicles, Config B has 100 vehicles.
Both have a 10% prediction error:
- Config A contributes: `50,000 × 5,000² = 1.25 × 10¹²`
- Config B contributes: `100 × 10² = 10,000`

Config A is penalized **125 million times more**. The model must get mass-market configurations right.

---

### TVD — Total Variation Distance *(synthetic data quality metric)*

Measures how different two probability distributions are. Range [0, 1] — lower is better.

```
TVD(P, Q) = 0.5 × Σ |P(k) − Q(k)|   for all categories k
```

> All TVD values are computed with **vehicle-weighted marginals**: a config with 50k vehicles
> contributes 500× more to P(energy=DIES) than a config with 100 vehicles.

---

### TSTR — Train on Synthetic, Test on Real *(generator utility metric)*

A model is trained on synthetic data and tested on real data.
If TSTR F1 ≈ TRTR F1 (Train on Real, Test on Real), the synthetic data is a viable substitute for the real data.

---

### Propensity Score — AUC *(generator realism metric)*

A binary classifier tries to distinguish real rows from synthetic rows.
- **AUC ≈ 0.5** → indistinguishable from real (ideal generator)
- **AUC ≈ 1.0** → synthetic data is easily detected as fake

---

## Results Summary

| Rank | Notebook | Method | WMSE | vs Baseline |
|------|----------|--------|------|-------------|
| **1** | `09` | **LightGBM — improved pipeline** | **176.6M** | **−82%** |
| 2 | `09` | Ensemble (optimised weights, 0.8 LGB) | 182.0M | −82% |
| 3 | `09` | XGBoost (tuned, Bayesian optimisation) | 221.8M | −78% |
| 4 | `07` | TVAE augmented XGBoost (multi-country) | 559M | −44% |
| 5 | `02` | Gaussian Copula augmented XGBoost | 570M | −43% |
| 6 | `01` | XGBoost (900 trees, no augmentation) | 357M | −65% |
| — | — | Heuristic baseline | 1,005M | reference |
| — | `08` | kNN collaborative filtering | 1,080M | +7% (worse) |

---

## Prediction Pipelines

### Parts 1–3 — Augmented XGBoost (`02`, `07`)

The first three pipelines test whether synthetic data generation can fill cold-start gaps and improve WMSE.

**Target engineering** (common to all parts):
```
reference_total = mean_share(combo_key) × country_total   [6-col key, training data only]
t_i = log1p(total_vehicles / reference_total)             [clipped at 5]
pred = expm1(t̂_i) × reference_total                       [inverse transform]
```

**Sample weights**: `s_i = total_vehicles` — aligns training loss with WMSE by penalising errors on high-volume configs.

| Strategy | Generator | Synthetic rows | Cold-start coverage | WMSE |
|----------|-----------|---------------|---------------------|------|
| Part 1 (multi-country) | TVAE-v7  | 9,366  | 19 / 73 | 559M |
| Part 1 (multi-country) | CTGAN-v7 | 37,187 | 68 / 73 | 670M |
| Part 2 (Turkey-only)   | TVAE     | 100K   | 50 / 73 | 640M |
| Part 2 (Turkey-only)   | CTGAN    | 100K   | 44 / 73 | 713M |
| Conditional            | GC-v2    | 7,200  | 72 / 73 | 570M |

> **GC paradox**: Gaussian Copula fails distributional quality tests (propensity AUC = 0.85) yet achieves near-TVAE prediction accuracy (570M vs 559M). It uses *conditional sampling* — generating exactly one count per missing config rather than replicating the full distribution — which is sufficient for targeted cold-start imputation.

---

### Part 4 — Improved Pipeline (`09`)

Five structural improvements over Parts 1–3, achieving **176.6M WMSE (−82%)**.

| # | Improvement | Effect |
|---|-------------|--------|
| 1 | **Consistent reference normalisation** | `reference_total` computed from training data only, joined to both train and test — eliminates distribution mismatch |
| 2 | **Country similarity-weighted sample weights** | `s_i = total_vehicles × σ_c` where σ_c is cosine similarity of country c's energy mix to Turkey |
| 3 | **Additional country-level features** | `ev_pct`, `lpg_pct`, `log_country_total`, `cosine_sim_turkey`, interaction features (`energy×segment`, `maker×energy`) |
| 4 | **Bayesian hyperparameter optimisation** | 30 trials, TPE sampler over XGBoost search space |
| 5 | **Multi-model evaluation** | XGBoost, CatBoost, LightGBM — LightGBM wins with native categorical handling |

**Country similarity formula**:
```
σ_c = (v_c · v_TR) / (‖v_c‖ × ‖v_TR‖)
v_{c,e} = fleet share of energy type e in country c
```
Scale-invariant: captures energy-mix composition, not market size.
Croatia σ = 0.9999, Spain σ = 0.9997 → nearly same weight as Turkish rows.
Belarus, Bosnia σ ≈ 0.50 → substantially downweighted.

**Why LightGBM beats XGBoost**: One-hot encoding 7 categorical columns produces a 91,763 × 3,847 sparse matrix. LightGBM uses native integer-coded categoricals with histogram splitting — no OHE needed. This alone closes a 45M WMSE gap (221.8M → 176.6M).

**Why augmentation hurts in Part 4**: Croatia and Spain (σ ≥ 0.999) already provide Turkey-relevant signal via similarity weighting. Synthetic rows add noise to the high-volume configurations that drive WMSE. Augmentation is most valuable when the base model is poorly calibrated to the target market.

**Part 4 detailed results**:

| Model | Augmentation | WMSE | vs Baseline |
|-------|-------------|------|-------------|
| **LightGBM** | **None** | **176.6M** | **−82%** |
| Ensemble (optimised) | None | 182.0M | −82% |
| XGBoost (tuned) | None | 221.8M | −78% |
| CatBoost | None | 281.6M | −72% |
| XGBoost (tuned) | + TVAE | 236.8M | −76% |
| XGBoost (tuned) | + GC | 244.0M | −76% |
| XGBoost (tuned) | + CTGAN | 275.9M | −73% |

The optimised ensemble assigns weights: XGBoost 0.10, CatBoost 0.10, LightGBM 0.80.

**Cross-validation note**: GroupKFold CV (groups = country) is **not** a reliable proxy for Turkey test WMSE. Fold WMSEs range from 232M to 325,970M; the Turkey fold alone yields 325,970M vs actual 221.8M. All reported WMSEs are evaluated directly on the 254-row Turkey test set.

---

## Generator Quality Evaluation

### Statistical Evaluation (`05_eval_statistical.ipynb`) — CTGAN-v6, TVAE-v6, GC-v2

| Generator | Avg. Cat. TVD | Age KS | Count KS | Config share MAE |
|-----------|--------------|--------|----------|-----------------|
| CTGAN-v6  | **0.380** ✓  | POOR (0.33) | OK (0.18)   | **0.000099** ✓ |
| TVAE-v6   | 0.384        | POOR (0.30) | POOR (0.27) | 0.000104       |
| GC-v2     | 0.609        | GOOD (0.07) | POOR (0.51) | 0.000116       |

Note: TVAE shows posterior collapse on `code_age` (Cramér's V = 6.37 >> 1.0 — catastrophic distortion).
CTGAN is the best-balanced generator overall; GC replicates age well but fails on count distributions.

Metrics: **TVD** (vehicle-weighted marginals), **KS test** (continuous columns), **Chi-square** + **Cramér's V** (categorical), **config-level share MAE**.

---

### ML Utility Evaluation (`06_eval_ml_utility.ipynb`) — CTGAN-v7, TVAE-v7, GC-v2

| Generator | TSTR F1 gap | Propensity AUC | Verdict |
|-----------|------------|----------------|---------|
| CTGAN-v7  | +0.01      | 0.511          | ✓ excellent |
| TVAE-v7   | +0.02      | 0.528          | ✓ excellent |
| GC-v2     | −0.12      | 0.853          | ✗ poor  |

CTGAN and TVAE trained on multi-country data slightly *outperform* real Turkey training data on TSTR
(+0.01 / +0.02 gap) because they encode fleet patterns from all 11 countries.
GC fails both tests: ELEC precision = 0.03 (over-generates diesel), body_style distribution distorted.

> Despite failing ML utility tests, GC achieves competitive WMSE (570M) via conditional sampling —
> the evaluation mismatch reflects different objectives (full distribution vs targeted slot imputation).

---

## Repository Structure

```
├── README.md
├── data/                                          # not tracked (Michelin confidential)
│   ├── EM_LYON_train_set_20260206.csv             # 91,763 rows — 11 countries
│   ├── EM_LYON_test_set_20260206.csv              # 254 rows — Turkey only
│   ├── gc_synth_turkey_v2.csv                     # 7,200 GC synthetic rows
│   ├── ctgan_synth_turkey_v7.csv                  # 37,187 CTGAN synthetic rows
│   ├── tvae_synth_turkey_v7.csv                   # 9,366 TVAE synthetic rows
│   └── turkey_predictions_nb09.csv                # final predictions (LightGBM, 176.6M)
├── notebooks/
│   ├── 00_eda.ipynb                               # exploratory data analysis (10 figures)
│   ├── 01_xgboost_land_rover_augmentation.ipynb   # XGBoost no augmentation (WMSE 357M)
│   ├── 02_gaussian_copula_rf_pipeline.ipynb       # GC fills missing configs → XGBoost (WMSE 570M)
│   ├── 03_ctgan_tvae_synthesis.ipynb              # CTGAN / TVAE v4 + vehicle-weighted reweighting
│   ├── 04_gaussian_copula_synthesis.ipynb         # Gaussian Copula synthesis
│   ├── 05_eval_statistical.ipynb                  # TVD / KS / Chi² statistical evaluation
│   ├── 06_eval_ml_utility.ipynb                   # TSTR + propensity score — 3 generators
│   ├── 07_ctgan_augmented_xgboost_v5.ipynb        # CTGAN/TVAE augmented XGBoost (TVAE best: 559M)
│   ├── 08_knn_baseline_corrected.ipynb            # kNN collaborative filtering (1,080M)
│   └── 09_improved_xgboost.ipynb                  # improved pipeline — LightGBM best (176.6M) ★
└── report/
    ├── report_approach_b.tex                      # full LaTeX report
    ├── presentation_plan.pdf                      # 4-speaker presentation guide (25 min)
    └── figures/                                   # EDA + model figures
```

---

## Data

| File | Rows | Description |
|------|------|-------------|
| `EM_LYON_train_set_20260206.csv` | 91,763 | 11 European countries — historical fleet data |
| `EM_LYON_test_set_20260206.csv` | 254 | Turkey test configurations to predict |

**Features**: `country_iso`, `car_maker_name`, `car_segment_name`, `energy` (13 types), `body_style` (10 types), `code_age` (6 buckets)
**Target**: `total_vehicles` — highly right-skewed (median 39, max 2.25M)

> Raw data files are excluded from this repository (competition confidentiality).

---

## Setup

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost optuna sdv matplotlib scipy
```

Key libraries:
- `sdv` — CTGAN, TVAE, GaussianCopulaSynthesizer
- `xgboost` / `lightgbm` / `catboost` — gradient boosting
- `optuna` — Bayesian hyperparameter optimisation
- `scipy` — KS test, Chi-square test
