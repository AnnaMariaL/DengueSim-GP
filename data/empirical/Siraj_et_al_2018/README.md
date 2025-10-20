# Siraj et al. (2018) – Municipality-Level Environmental and Demographic Data

This folder contains environmental and demographic data originally published in **Siraj et al. (2018)** and used in the following [preprint](https://www.medrxiv.org/content/10.1101/2024.11.28.24318136v2). 

The dataset provides municipality-level characteristics (economic activity, travel time, etc.) for Colombia and is used to complement the OpenDengue incidence data from Clarke *et al.* (2024). 

---

## Data Overview

### `municip_aggregate_non_ts.csv`

Static (non–time-series) dataset with municipality-level attributes.

| Column name | Description |
|--------------|-------------|
| `ID_ESPACIA` | Unique spatial identifier for each municipality |
| `NOM_MUNICI` | Municipality name |
| `Wpop2015` | Estimated total population in 2015 |
| `WpopBirths2015` | Estimated number of births in 2015 |
| `UrbanPop` | Urban population estimate |
| `MeanGCP_2005USD` | Mean Gross Cell Product (GCP) per municipality in 2005 USD |
| `MeanTraveltime` | Mean travel time (in minutes) to the nearest major city or health center |


Example to load:
```r
env_data <- read.csv("municip_aggregate_non_ts.csv")
head(env_data)


