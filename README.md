# OUSD Absence Project

This repository contains code, notebooks, and utilities for exploring features and training models aimed at predicting student absence / chronic absenteeism in the Oakland Unified School District (OUSD). Data files are not included in this repository — see "Expected data" below.

## Repository structure
- script/
  - experiments.py — main experiment runner (training, CV, evaluation)
  - pca_utils.py — PCA utilities (fit/transform/save components)
- experiment/ — interactive Jupyter notebooks for step-by-step analysis and model development:
  - Feature engineering, PCA, logistic regression, random forest, gradient boosting, and combined experiments.
- requirements.txt — Python package dependencies
- gitignore.txt — ignore rules and suggestions

## Expected data (NOT included)
Place your dataset files locally and update paths in scripts/notebooks. Typical expected filenames (examples only):
- evaldata_cleaned.csv — cleaned input data (student records, attendance, demographic and neighborhood features)
- engineered_features.csv — features produced by preprocessing/feature engineering
- pca_features.csv — PCA-transformed feature set

Note: The repository intentionally does not include CSVs. 

## Setup

1. Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running experiments

- Notebooks (interactive)
  - Launch Jupyter Lab or Notebook and open files under experiment/ to run cells sequentially for reproducible analysis:
    ```sh
    jupyter lab
    ```
- Scripted runs
  - Run the main experiment pipeline (edit file paths and parameters inside script/experiments.py before running):
    ```sh
    python script/experiments.py
    ```

## Scripts overview

- script/pca_utils.py
  - Utility functions to fit PCA, transform datasets, and persist components/variance explained.
- script/experiments.py
  - Orchestrates data loading, preprocessing (or expects preprocessed inputs), model training, cross-validation, and metric output. Consider converting hard-coded values to CLI args or a config file.


## Data privacy & license

- Ensure any student or personally identifiable information is handled according to applicable privacy policies before sharing results.
- Add a LICENSE file if you plan to publish code publicly.

## Contributing

- Open issues for reproducibility, bugs, or experiment improvements.
- Add new notebooks or scripts under experiment/ or script/ and update this README with brief descriptions.

...existing code...
```// filepath: /Users/lyc/Documents/GitHub/ousd-absence-project/README.md
# ousd-absence-project
...existing code...

# OUSD Absence Project

This repository contains code, notebooks, and utilities for exploring features and training models aimed at predicting student absence / chronic absenteeism in the Oakland Unified School District (OUSD). Data files are not included in this repository — see "Expected data" below.

## Repository structure
- script/
  - experiments.py — main experiment runner (training, CV, evaluation)
  - pca_utils.py — PCA utilities (fit/transform/save components)
- experiment/ — interactive Jupyter notebooks for step-by-step analysis and model development:
  - Feature engineering, PCA, logistic regression, random forest, gradient boosting, and combined experiments.
- requirements.txt — Python package dependencies
- gitignore.txt — ignore rules and suggestions

## Expected data (NOT included)
Place your dataset files locally and update paths in scripts/notebooks. Typical expected filenames (examples only):
- evaldata_cleaned.csv — cleaned input data (student records, attendance, demographic and neighborhood features)
- engineered_features.csv — features produced by preprocessing/feature engineering
- pca_features.csv — PCA-transformed feature set

Note: The repository intentionally does not include CSVs. Add your own datasets into a data/ directory or modify the paths in script/experiments.py and the notebooks.

## Setup

1. Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running experiments

- Notebooks (interactive)
  - Launch Jupyter Lab or Notebook and open files under experiment/ to run cells sequentially for reproducible analysis:
    ```sh
    jupyter lab
    ```
- Scripted runs
  - Run the main experiment pipeline (edit file paths and parameters inside script/experiments.py before running):
    ```sh
    python script/experiments.py
    ```

## Scripts overview

- script/pca_utils.py
  - Utility functions to fit PCA, transform datasets, and persist components/variance explained.
- script/experiments.py
  - Orchestrates data loading, preprocessing (or expects preprocessed inputs), model training, cross-validation, and metric output. Consider converting hard-coded values to CLI args or a config file.

## Reproducibility & tips

- Set random seeds in notebooks and scripts (numpy, pandas, scikit-learn) to reproduce results.
- Use a config or argparse for experiment parameters and file paths to avoid editing code.
- For quick iteration, sample a smaller subset of data before running full model training.

## Suggested next steps

- Add a small config.yaml or JSON for experiment parameters (paths, model hyperparameters, CV settings).
- Add unit tests for key utilities (pca_utils functions) and a smoke test for script/experiments.py using a tiny synthetic dataset.
- Document the evaluation protocol (metrics, thresholds, train/test split) in a short RESULTS.md or the final notebook.

## Data privacy & license

- Ensure any student or personally identifiable information is handled according to applicable privacy policies before sharing results.
- Add a LICENSE file if you plan to publish code publicly.

## Contributing

- Open issues for reproducibility, bugs, or experiment improvements.
- Add new notebooks or scripts under experiment/ or script/ and update this README with brief descriptions.

