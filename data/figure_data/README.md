
This directory contains the processed and generated data used for figures in [Gaussian Process Emulation for Exploring Complex Infectious Disease Models](https://www.medrxiv.org/content/10.1101/2024.11.28.24318136v2). It includes sensitivity analyses  results, Gaussian Process (GP) model predictions, and training metrics for three model outputs: maximum incidence (`imax`), epidemic duration (`duration`), and outbreak probability (`outbreak-probability` or `establishment`).  

---

## Directory Overview

```

figure_data/
├── heatmap_alphaRest/
│   ├── GP/
│   │   ├── predictions_outbreak-probability_heatmap_alphaRest_0.tsv
│   │   ├── predictions_outbreak-probability_heatmap_alphaRest_1.tsv
│   │   └── ... (0–19)
│   └── IBM/
│       └── IBM-simulations_outbreak-probability_heatmap_alphaRest.tsv
├── predictions_duration_round15_snap7_heatmap.tsv
├── predictions_duration_round15_snap7_test.tsv
├── predictions_imax_round15_snap3_heatmap.tsv
├── predictions_imax_round15_snap3_test.tsv
├── predictions_outbreak-probability_round15_snap5_heatmap.tsv
├── predictions_outbreak-probability_round15_snap5_test.tsv
├── sobol_indices_duration.tsv
├── sobol_indices_imax.tsv
├── sobol_indices_outbreak-probability.tsv
├── sobol_indices_outbreak-probability_subdomain.tsv
├── training_progress_duration.tsv
├── training_progress_imax.tsv
└── training_progress_outbreak-probability.tsv

```

- **heatmap_alphaRest/**: Contains heatmap data showing predicted outbreak probability as a function of `alphaRest` and other parameters.  
  - `GP/`: GP predictions.  
  - `IBM/`: Individual-based model simulations.

- **predictions_*_heatmap.tsv**: Model predictions for all parameter combinations used in heatmap panels of the sensitivity analysis plots. 
- **predictions_*_test.tsv**: Predictions for the held-out test set.  
- **sobol_indices_*.tsv**: Sobol sensitivity indices from global sensitivity analysis using the three GPs.  
- **training_progress_*.tsv**: Training metrics (loss, RMSE) for GP models over rounds/snapshots.

---

## File Columns

### Prediction Files (`predictions_*_*.tsv`)

| Column | Description |
|--------|-------------|
| `simRound` | Training round number (0 = initial round, 1–15 = active learning rounds). Only meaningful for training datasets; included in test datasets for format consistency. |
| `simID` | Identifier for a specific parameter combination within a round. |
| `alphaRest` | **Average infectivity** – baseline probability of infection per day. |
| `alphaAmp` | **Seasonality strength** – scaling factor controlling seasonal variation in infection probability. |
| `alphaShift` | **First case timing** – timing of the first case relative to seasonal peak in infection probability. |
| `infTicksCount` | **Infectious period** – average number of days an individual remains infectious. |
| `avgVisitsCount` | **Average mobility** – average number of daily visits per individual. |
| `pVisits` | **Mobility skewness** – parameter controlling variability of daily visits. |
| `propSocialVisits` | **Social structure** – probability that a visit occurs within a family cluster. |
| `locPerSGCount` | **Family cluster size** – number of locations per family cluster. |
| `mean` or `pred` | Predicted outcome (imax, duration, or outbreak probability). |
| `lower` | Lower bound of CI. |
| `upper` | Upper bound of CI. |
| `maxIncidence` | Highest proportion of infectious individuals seen in any timestep (for `maxIncidence` model). |
| `epidemicSize` | Final epidemic size (not used in manuscript). |
| `duration` | Duration of the epidemic (log-transformed). |
| `sd_maxIncidence` | Standard deviation of maximum incidence across simulation runs. |
| `sd_epidemicSize` | Standard deviation of epidemic size. |
| `sd_duration` | Standard deviation of epidemic duration. |
| `model` | Model type (`imax`, `duration`, `establishment` (= `outbreak probability`)). |

### Sobol Indices Files (`sobol_indices_*.tsv`)

| Column | Description |
|--------|-------------|
| `var1`, `var2` | Input parameters for sensitivity analysis. |
| `effect` | Type of effect (`First Order`, `Total Effects`, `Second Order`). |
| `sobol` | Sobol sensitivity index. |
| `ci` | 95% confidence interval for the Sobol index. |
| `n` | Number of samples used in calculation. |
| Additional columns (subdomain) | Parameter ranges used in subdomain analysis. |

### Training Progress Files (`training_progress_*.tsv`)

| Column | Description |
|--------|-------------|
| `ModelType` | Epidemic outcome (`imax`, `duration`, `establishment` (=`outbreak probability`)). |
| `Round` | Training round. |
| `Snapshot` | Snapshot within round. |
| `TrainingLoss` | Loss function value on training data. |
| `TrainingRMSE` | Root mean squared error on training data. |
| `ValidationRMSE` | Root mean squared error on validation data. |

