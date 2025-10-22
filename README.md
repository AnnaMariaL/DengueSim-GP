# DengueSim-GP
The repository accompanies the manuscript *Gaussian Process emulation for exploring complex infectious disease models* which is currently available in [preprint](https://www.medrxiv.org/content/10.1101/2024.11.28.24318136v2).

---


## Repository Structure

| File                      | Description                                                                                                                                                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`src/SIR_gp.py`**           | Core implementation of the Gaussian Process emulator class that emulates the individual-based-model [DengueSim](https://github.com/AnnaMariaL/DengueSim). Automatically detects and uses GPU acceleration if available (via `torch.cuda.is_available()`); otherwise, it defaults to CPU computation. |
| **`src/GP-demo.ipynb`**       | Jupyter notebook illustrating the principles behind GP emulation, including model training, prediction, sensitivity analysis, and comparison with true model outputs. |
| **`src/gp_emulator_env.yml`** | Conda environment specification for GP usage.|
| **`src/EmpData-Link.Rmd`**  | R Markdown workflow used to link municipality-level data. |
| **`src/EmpData-Detect.Rmd`** | R Markdown workflow for identifying epidemic periods from dengue incidence time series and exporting detected epidemic intervals (see `OpenDengue*.txt` outputs under `data/empirical/`). 
| **`src/EmpData-Calibrate.ipynb`** | Jupyter notebook for municipality-specific calibration of the Gaussian Process emulator to empirical dengue outbreak data, identifying optimal parameter combinations that best reproduce observed maximum incidences (see `*tsv` outputs under `data/parameter_exploration`) |
| **`data/`** | Directory containing simulation and empirical datasets used for emulator validation and calibration. |
| **`data/empirical/`** | Contains real-world dengue incidence, environmental, and demographic data used for empirical analyses and data linkage. See below for details.  |
| **`data/parameter_exploration/`** | Contains output .tsv files from parameter exploration using the maximum incidence GP emulator. |


---

### `data/empirical/` Directory Overview

| Subdirectory / File | Description |
| -------------------- | ------------ |
| **`Clarke_et_al_2024/`** | Contains *OpenDengue* incidence data from Clarke *et al.* (2024). |
| **`Siraj_et_al_2018/`** | Contains environmental and demographic indicators at the municipality level, originally published by Siraj *et al.* (2018). |
| **`linkIDs.txt`** | Linkage table connecting dengue incidence data with environmental and demographic indicators at the municipality level (generated with `../src/EmpData-Link.Rmd`)  |
| **`OpenDengue_detected_epidemics.txt`**      | Summary of all detected dengue epidemics by municipality, including timing (`t`, `duration`) and peak incidence (`imax`).                                                 |
| **`OpenDengue_detected_epidemics_full.txt`** | Full version of the detected epidemic dataset, containing start and end dates (`xmin`, `xmax`), municipality codes, epidemic IDs, thresholds, durations, and peak values. |
---

## Environment Setup

To run the notebooks, clone this repository and create a Conda environment using the provided file:

```bash
# Clone repository
git clone https://github.com/DengueSim-GP/DengueSim-GP.git
cd DengueSim-GP

# Create and activate environment
conda env create --file gp_emulator_env.yml
conda activate gp_emulator_env

# Launch Jupyter Notebook
```

---

## Gaussian Process Implementation Overview

The `src/SIR_gp.py` script defines a **Gaussian Process (GP) class** using `gpytorch`. It includes:

* **Training and prediction** routines with automatic data handling.

* **Hardware flexibility**:
  The implementation checks for GPU availability:

  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  ```

  This ensures the code will utilize GPU acceleration when available (greatly improving training time), but remains fully functional on CPU.
---

## Demonstration Notebook (`src/GP-demo.ipynb`)

The Jupyter notebook provides a guided walk-through of:

1. Set up: Imports, Data Paths, and Parameter Space
2. Loading the Gaussian Process emulator
3. Evaluating GP performance
4. Sensitivity Analysis with the GP
5. Predictions with the GP
6. Sampling additional points based on GP predictions

The notebook includes detailed markdown explanations and inline comments to make the workflow accessible to newcomers in GP-based emulation.

---

## Learning Resources

To learn more about Gaussian Processes and how they’re implemented in `gpytorch`, check out the following resources:

* [GPyTorch Documentation](https://gpytorch.ai/)
* [GPyTorch Tutorials on GitHub](https://github.com/cornellius-gp/gpytorch/tree/main/examples)
* [Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*](http://www.gaussianprocess.org/gpml/)
* [Emukit Framework for Bayesian Optimization and Emulation](https://emukit.github.io/)

