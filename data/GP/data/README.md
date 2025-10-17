# GP Training and Test Data

This folder contains the datasets used for training and testing the Gaussian Process (GP) models described in the manuscript. 
These datasets include IBM simulation outputs of different epidemiological metrics under a range of input parameter combinations.

---

## Contents

| Filename | Description |
| -------- | ----------- |
| `sim-training-maxIncidence-round15.txt` | Training data for **maximum incidence** after 1 initial & 15 active training rounds. |
| `sim-training-duration-round15.txt` | Training data for **epidemic duration** (log-transformed) after 1 initial & 15 active training rounds. |
| `sim-training-establishment-round15.txt` | Training data for **outbreak probability** after 1 initial & 15 active training rounds |
| `DD-AML-test-LHS-10000-condSim-logDuration.txt` | Test dataset for **maximum incidence** and **duration**, generated using Latin Hypercube Sampling (LHS) with 10,000 points where epidemic outbreaks occurred |
| `DD-AML-test-LHS-10000-establishment.txt` | Test dataset for **outbreak probability**, using 10,000 LHS points. |

---

## Data Format

All files are tab-separated text files. The columns generally include:

| Column | Description |
| ------ | ----------- |
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
| `maxIncidence` | Highest proportion of infectious individuals seen in any timestep (for `maxIncidence` model). |
| `epidemicSize` | Final epidemic size (not used in manuscript). |
| `duration` | Duration of the epidemic (log-transformed). |
| `sd_maxIncidence` | Standard deviation of maximum incidence across simulation runs. |
| `sd_epidemicSize` | Standard deviation of epidemic size. |
| `sd_duration` | Standard deviation of epidemic duration. |
| `establishment` | Outbreak probability |

> **Note:** Only relevant outcome columns are used for each GP model:  
> - `maxIncidence` GP uses `maxIncidence` and `sd_maxIncidence`.  
> - `duration` GP uses `duration` (log-transformed) and `sd_duration`.  
> - `establishment` GP uses `establishment` (outbreak probability) with no standard deviation.  

---

## Input Parameter Ranges

| Code Variable       | Manuscript Equivalent          | Range       |
| ------------------ | ----------------------- | ----------- |
| `alphaRest`        | Average infectivity     | [0, 0.03]   |
| `alphaAmp`         | Seasonality strength    | [0, 1]      |
| `alphaShift`       | First case timing       | [0, 1]      |
| `infTicksCount`    | Infectious period       | [4, 6]      |
| `avgVisitsCount`   | Average mobility        | [1, 5]      |
| `pVisits`          | Mobility skewness       | [0.05, 0.95]|
| `propSocialVisits` | Social structure        | [0, 1]      |
| `locPerSGCount`    | Family cluster size     | [1, 20]     |

These are the **GP model input parameters** and correspond to the columns in the data files.

---

## Usage Notes

- **Training datasets:** used to fit the GP models for `maxIncidence`, `duration`, and `establishment`.  
- **Test datasets:** used to validate GP predictions, often generated via Latin Hypercube Sampling (LHS).  
- `simRound` is important for training data bookkeeping, but is ignored in test datasets.  
- All datasets are in tab-separated format for easy loading in Python (`pandas.read_csv(..., sep='\t')`) or R.
