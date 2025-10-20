
# Empirical Data

This directory contains the empirical datasets used in the revised manuscript to link dengue incidence data (Clarke et al., 2024) with environmental and demographic data (Siraj et al., 2018).  

It also includes identifier mapping used to connect municipality-level data across sources.

---

## Directory Structure

```

data/
└── empirical/
├──   Clarke_et_al_2024/       # OpenDengue incidence data (zipped CSV files)
├──   Siraj_et_al_2018/        # Environmental and demographic data
└── linkIDs.txt              # Municipality ID linkage table (OpenDengue ↔ OCHA)

```

---

## Files and Content

### 1. `Clarke_et_al_2024/`
Contains the dengue incidence data from **Clarke et al. (2024)** (*OpenDengue* project).  
Data are provided in zipped CSV format and include weekly dengue cases at the municipality (admin-2) level across Colombia.

### 2. `Siraj_et_al_2018/`
Contains environmental and demographic indicators at the municipality level, originally published by **Siraj et al. (2018)**.  
Data include variables such as population size, urbanization, mean travel time, and economic indicators.

### 3. `IDLinks.txt`
Provides the linkage between OpenDengue municipality identifiers (GAUL codes) and OCHA administrative identifiers (ADM2_PCODE).  
This file was generated through the documented workflow in `EmpData-Link.Rmd`

| Column | Description |
|---------|-------------|
| `adm_1_name` | Department name (OpenDengue) |
| `adm_2_name` | Municipality name (OpenDengue) |
| `full_name` | Full hierarchical name string |
| `FAO_GAUL_code` | OpenDengue GAUL administrative code |
| `dengue_ID` | Internal identifier |
| `adm2_std` | Standardized municipality name (lowercase, ASCII) |
| `adm1_std` | Standardized department name (lowercase, ASCII) |
| `ocha_ID` | OCHA municipality code (`ADM2_PCODE`) |

---
