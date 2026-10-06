# Lending Club P2P Loan Default Prediction

**UE24CS352A Machine Learning Mini-Project — Team A-15 (Sec-A), Problem Statement 39**

Team: Aman Priyansh (PES1UG24AM030) and Addanki Harsha Rajeswari (PES1UG24AM014)

Paper reproduced: Peiqian Li and Gao Han, *LendingClub Loan Default and Profitability Prediction*, Stanford CS229, 2018.

## Notebooks
- `01_Baseline_Reproduction.ipynb` — rebuilds the paper (data, three classifiers, six regressors, strategy).
- `02_Improvements.ipynb` — ablation study (leak removal, time split, XGBoost, tuning), plus a live demo.

## How to run
Open each notebook in Google Colab. Runtime → Change runtime type → **A100 GPU** (L4 also works) → Runtime → Run all. Both notebooks download the Kaggle dataset by themselves.

Approximate time on A100: about 20 minutes per notebook.
