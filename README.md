# Trajectory Distribution Prediction Evaluation with Energy Score

## Overview
This repository contains codes that reproduce our experiments and illustrations used in the paper.
- `propriety_showcase.py`: corresponds to the "Propriety Showcase" from the paper.
- `samplesize_effect_and_sensitivity_study.py`:  corresponds to the two experiments from the paper.
  - "Effect of Sample Size"
  - "Sensitivity Study"
- `bernoulli_simulation.py`: corresponds to the "Bernoulli simulation" from the paper's appendix.
- `real-data experiment`: real-data experiment with ETH/UCY dataset.

Illustrations/Demonstrations:
- `first_illustration.py`: illustration of the first page of the paper
- `es_variations_demonstration.py`: demonstrations of ESS and EST

Other:
- `metrics.py`: implementation of energy score and other metrics used in the paper


## Setup

```Python 3.8.5``` is used for this project. 

```bash
# For conda environments
conda env create -f environment.yml

# For pip
pip install -r requirements.txt
