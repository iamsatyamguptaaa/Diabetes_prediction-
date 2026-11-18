# Diabetes Prediction — Educational Machine Learning Project

This repository contains a reproducible analysis notebook and a Gradio application
for an educational diabetes prediction demonstration.

## Project contents
- `data/diabetes.csv` — supplied tabular dataset (768 rows, 8 predictors + Outcome).
- `notebooks/diabetes_analysis.ipynb` — EDA, leakage-safe preprocessing, evaluation,
  and model export.
- `app.py` — Gradio interface with input validation and model-estimated probability.
- `diabetes_model.pkl` — fitted scikit-learn pipeline used by the app.
- `requirements.txt` — runtime dependencies.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

For the notebook, install notebook/analysis packages:
```bash
pip install pandas matplotlib seaborn jupyter
jupyter notebook notebooks/diabetes_analysis.ipynb
```

## Method
The notebook performs EDA, reports zero-coded values in selected medical
measurements, replaces those zeros with missing values, and fits median
imputation inside a scikit-learn pipeline. A stratified holdout split is made
before model fitting. The Random Forest uses class weights instead of
oversampling the full dataset before splitting, preventing that leakage route.
The notebook reports accuracy, precision, recall, F1, ROC-AUC, and a confusion
matrix. It also includes optional stratified 5-fold cross-validation.

## Data and attribution
The files were supplied by a project collaborator. Confirm the dataset's
redistribution terms and retain the original dataset/model/project attribution
required by its source before publishing. Do not imply that this project or
dataset is exclusively your original work unless that is accurate.

## Important limitation
This is a coursework/educational demonstration, not a medical device or
diagnostic tool. Do not use its output for clinical decisions. The dataset is
small and may not represent other populations.
