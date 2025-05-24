# PT-Symmetric Quaternionic Model for SPARC Galaxy Rotation Curves

This repository contains the Python code and analysis pipeline used to fit galaxy rotation curves from the SPARC dataset with the proposed PT-Symmetric Quaternionic model, alongside ΛCDM and MOND models, as presented in the paper:


## Overview

The primary script `src/sparc_fitter_cli.py` performs MCMC fitting for each galaxy in the SPARC dataset. It calculates goodness-of-fit statistics (χ², reduced χ², AIC, BIC) and generates comparison plots.

An interactive version of this analysis is also available as a Jupyter Notebook in the `notebooks/` directory, which can be run directly in Google Colab.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ice91/PT_Quaternionic_Galaxy_Fits/blob/main/notebooks/quaternionic_sparc_analysis.ipynb)

## Dependencies

The required Python libraries are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt

python -m src.fit_rotation_curves \
    --data-path dataset/sparc_processed_nohead.txt \
    --outdir results/ \
    --nwalkers 2 \
    --steps 10