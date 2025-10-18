# DengueSim-GP
The repository accompanies the manuscript *Gaussian Process emulation for exploring complex infectious disease models* which is currently available in [preprint](https://www.medrxiv.org/content/10.1101/2024.11.28.24318136v2).

---

## Repository Structure

| File                      | Description                                                                                                                                                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`src/SIR_gp.py`**           | Core implementation of the Gaussian Process emulator class that emulates the individual-based-model [DengueSim](https://github.com/AnnaMariaL/DengueSim). Automatically detects and uses GPU acceleration if available (via `torch.cuda.is_available()`); otherwise, it defaults to CPU computation. |
| **`src/GP-demo.ipynb`**       | Jupyter notebook illustrating the principles behind GP emulation, including model training, prediction, sensitivity analysis, and comparison with true model outputs. 
| **`src/gp_emulator_env.yml`** | Conda environment specification for GP usage.                                                                                                                                                            |
| **`data/`**               | Directory containing example input/output data for model training and validation (if applicable). TBD                                                                                                                                                       |

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

