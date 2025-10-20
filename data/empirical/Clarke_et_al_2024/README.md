# Clarke et al. (2024) – Dengue Incidence Data (Colombia)

This folder contains the empirical dengue incidence data used in the [preprint](https://www.medrxiv.org/content/10.1101/2024.11.28.24318136v2).
The dataset is provided as a **zipped CSV file** containing weekly dengue case counts at the municipality (Admin-2) level for Colombia between January 1st, 2007, to December 31st, 2019


##  Data Overview

Each record includes the following key fields:

| Column name | Description |
|--------------|-------------|
| `adm_0_name` | Country name (always `COLOMBIA`) |
| `adm_1_name` | Department name (Admin-1) |
| `adm_2_name` | Municipality name (Admin-2) |
| `full_name`  | Concatenated administrative names (`Country, Department, Municipality`) |
| `ISO_A0`     | ISO country code |
| `FAO_GAUL_code` | GAUL (FAO) administrative code used for spatial referencing |
| `RNE_iso_code` | Regional administrative code (ISO format) |
| `IBGE_code`  | Administrative identifier (if available) |
| `calendar_start_date` / `calendar_end_date` | Start and end dates of the reporting week |
| `Year`       | Calendar year |
| `dengue_total` | Number of dengue cases reported during that week |
| `case_definition_standardised` | Definition used for case classification (e.g., `Total`) |
| `S_res` / `T_res` | Spatial and temporal resolution (`Admin2`, `Week`) |
| `UUID` | Unique data record identifier (OpenDengue ID) |

---

## File Contents

- `Highest temporal resolution data_COLOMBIA_20021229_20191231.csv.zip`  
  Compressed CSV file with the full dataset.

After unzipping:

```r
dengue_incidence <- read.csv("Highest temporal resolution data_COLOMBIA_20021229_20191231.csv", header = TRUE)
dengue_incidence <- dengue_incidence[dengue_incidence$T_res == "Week", ] # Keep only weekly data


