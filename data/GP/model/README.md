# Pre-trained GP Models

This directory contains the **pre-trained Gaussian Process (GP) models** used in the manuscript.

## Files

| Filename | Description |
| -------- | ----------- |
| `maxIncidence-round15-snap3.pth` | GP model trained to predict maximum incidence. |
| `duration-round15-snap7.pth`     | GP model trained to predict epidemic duration (log-transformed). |
| `establishment-round15-snap5.pth` | GP model trained to predict outbreak probability (establishment). |

## Usage

These `.pth` files can be loaded in Python using PyTorch:

```python
from SIR_gp import * #class implementation of the GP

# Example: load the maxIncidence GP model
gp = SIR_GP(training_data="sim-training-maxIncidence-round15.txt", model_type="maxIncidence")
gp.load(filename="GP/model/maxIncidence_model.pth")
````
