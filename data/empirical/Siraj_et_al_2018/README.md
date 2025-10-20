# Siraj et al. (2018) – Municipality-Level Environmental and Demographic Data

This folder contains environmental, demographic, and entomological datasets originally published in **Siraj et al. (2018)** and used in the accompanying [preprint](https://www.medrxiv.org/content/10.1101/2024.11.28.24318136v2).

The data provide municipality-level attributes (e.g., population, economic activity, travel time, and *Aedes aegypti* suitability) for Colombia and are used to complement the dengue incidence data from **Clarke et al. (2024)** (*OpenDengue*).

---


## Data Overview

### `municip_aggregate_non_ts.csv`

Static (non–time-series) dataset containing demographic and socioeconomic attributes for each municipality.

| Column name       | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| `ID_ESPACIA`      | Unique spatial identifier for each municipality                          |
| `NOM_MUNICI`      | Municipality name                                                        |
| `Wpop2015`        | Estimated total population in 2015                                       |
| `WpopBirths2015`  | Estimated number of births in 2015                                       |
| `UrbanPop`        | Estimated urban population                                               |
| `MeanGCP_2005USD` | Mean Gross Cell Product (GCP) per municipality in 2005 USD               |
| `MeanTraveltime`  | Mean travel time (in minutes) to the nearest major city or health center |

Example to load:

```r
env_data <- read.csv("municip_aggregate_non_ts.csv")
head(env_data)
```

---

### `municip_Ae_aegypti_weeks_weighted.csv`

Weekly *Aedes aegypti* abundance probability per municipality.
Each column (`week1`–`week52`) corresponds to the mean modeled *Aedes aegypti* abundance probability for the respective week.
Values range between **0 and 1**, with higher values indicating higher probability for *Aedes aegypti*. 

| Column name      | Description                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| `ID_ESPACIA`     | Unique spatial identifier for each municipality                              |
| `NOM_MUNICI`     | Municipality name                                                            |
| `week1`–`week52` | Mean *Aedes aegypti* abundance probability per week                              |

Example to load:

```r
aedes_data <- read.csv("municip_Ae_aegypti_weeks_weighted.csv")
head(aedes_data[, 1:10])  # display ID and first few weeks
```

---





