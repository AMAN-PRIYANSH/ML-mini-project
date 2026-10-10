# Lending Club P2P Loan Default Prediction: Reproduction and Improvements

**UE24CS352A Machine Learning, Mini-Project. Team A-15 (Section A), Problem Statement 39**

| Name | SRN |
|---|---|
| Aman Priyansh | PES1UG24AM030 |
| Addanki Harsha Rajeswari | PES1UG24AM014 |

Paper reproduced: Peiqian Li and Gao Han, *LendingClub Loan Default and Profitability Prediction*, Stanford CS229, 2018.
[Report](https://cs229.stanford.edu/proj2018/report/69.pdf) · [Poster](https://cs229.stanford.edu/proj2018/poster/69.pdf)

---

## What this project does

1. **Reproduces the paper (baseline).** It predicts whether a Lending Club loan will be fully paid (classification), predicts its yearly return (regression), and rebuilds the paper's "invest in the top loans" strategy.
2. **Improves it.** It finds and removes **data leakage** (the borrower's credit score *after* the loan ended), tests on **future loans** (train 2012–2014, test 2015), adds **XGBoost**, and tunes it on a **validation set**. Each change is measured separately, as an ablation study.

## Repository layout

```
01_Baseline_Reproduction.ipynb   Part 1: rebuild the paper (data, 3 classifiers, 6 regressors, strategy, comparison)
02_Improvements.ipynb            Part 2: ablation S1/S2/S3, XGBoost, tuning, strategy re-test, charts, live demo
requirements.txt                 Python packages (all pre-installed on Google Colab except gradio)
results/                         Result tables as CSV (copied from the notebook outputs)
figures/                         Charts used in the write-up and slides
docs/                            Two-page write-up (PDF)
```

## How to set up and run

The notebooks are written for **Google Colab** and need no local setup.

1. Open a notebook in Colab (File → Upload notebook, or open it from GitHub).
2. Runtime → Change runtime type → **A100 GPU** (L4 also works) → Save.
3. Runtime → **Run all**.

Both notebooks download the dataset themselves from Kaggle ([`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club), about 1.6 GB) using `kagglehub`. No files need to be uploaded.

| Notebook | Approximate run time (Colab A100) |
|---|---|
| `01_Baseline_Reproduction.ipynb` | about 20 minutes |
| `02_Improvements.ipynb` | about 15–20 minutes (plus about 1 minute for the demo) |

To run outside Colab: `pip install -r requirements.txt`. You need an NVIDIA GPU for the XGBoost cells (`device='cuda'`) and about 40 GB of RAM.

The **live demo** is the last cell of `02_Improvements.ipynb`. It starts a small Gradio web page with a public link: type a loan number from the 2015 test set and see the honest model's predicted chance of default next to what really happened.

## Results

### Baseline: paper vs our reproduction (test set)

| | Paper | Ours |
|---|---|---|
| Logistic Regression, weighted precision / recall / F1 | 0.90 / 0.88 / 0.88 | 0.90 / 0.87 / 0.88 |
| Neural Network, weighted precision / recall / F1 | 0.89 / 0.89 / 0.89 | 0.88 / 0.89 / 0.88 |
| Random Forest, weighted precision / recall / F1 | 0.89 / 0.89 / 0.89 | 0.89 / 0.89 / 0.89 |
| Ridge regression, R² | 0.238 | 0.217 |
| Neural Network regressor, R² | 0.306 | 0.252 |
| Random Forest regressor (depth 10), R² | 0.315 | 0.281 |
| Investment strategy | ~1.7% of loans → ~15% a year | 1.34% of loans → 15.2% a year |

### Improvements: ablation (test set, ROC-AUC)

| Setting | Logistic Regression | Random Forest | XGBoost | XGBoost tuned |
|---|---|---|---|---|
| S1: paper setup (leaky columns kept, random split) | 0.935 | 0.934 | 0.941 | – |
| S2: leaky columns removed, random split | 0.721 | 0.721 | 0.733 | – |
| S3: leaky columns removed, train 2012–14 / test 2015 | 0.733 | 0.731 | 0.743 | **0.744** |

### Investment strategy under the honest setup (test set, true yearly return)

| Setup and model | Top 1.7% of loans | Top 5% | Top 10% | All loans |
|---|---|---|---|---|
| S1 paper setup: Random Forest | 15.1% | 13.2% | 11.9% | 0.0% |
| S3 honest: Random Forest | 3.1% | 2.7% | 2.3% | −1.6% |
| S3 honest: XGBoost | 3.8% | 3.7% | 3.4% | −1.6% |

**Main finding:** most of the paper's accuracy and its "15% a year" strategy come from data leakage. With honest inputs and testing on future loans, the best model reaches ROC-AUC 0.744, and its strategy earns about 3.8% a year. That is still about 5 points better than investing in every loan.

## Deviations from the original paper

We tried to change as little as possible. Every change is listed here.

1. **Dataset source.** LendingClub's own download page no longer exists, so we use the Kaggle copy (2007–2018) filtered with the paper's rules (issued 2012–2015, final status only). It gives 786,820 loans vs the paper's 745,529, because the copy is newer and more loans have finished since.
2. **Missing cleaned-data file.** The authors' code reads `data_clean.csv`, which is not in their zip. We rebuilt it from the rules in the paper.
3. **Missing Neural Network classifier code.** It is not in the zip, so we rebuilt it from the settings in the paper (10-10-5-3, sigmoid, Adam, batch 200, learning rate 0.001, L2 0.0001).
4. **Paper vs code mismatch.** The authors' `mlp_regressor.py` uses L2 penalty 0.008, but the paper says 0.0001. We followed the paper.
5. **Library updates.** These are small changes so the 2018 code runs on current pandas (percent fields, date parsing), plus a memory-efficient one-hot encoding and standardisation. The maths is unchanged.
6. **Fixed random seed** (seed = 0) for the 70/30 split. The paper used none, so its exact split cannot be recovered.
7. **Random Forest regressor on the GPU.** The paper used scikit-learn's `RandomForestRegressor` on CPU, which takes hours per forest here. We used XGBoost's random-forest mode (`XGBRFRegressor`) with the same 100 trees and depths 4/8/10, and all features per split. It adds 128-bin histogram splitting (**binning**), and each tree uses a 63.2% sample without replacement. We first tried NVIDIA cuML's `RandomForestRegressor`, but in the Colab image we used its predictions on unseen rows were wrong (test R² −2.26, caused by a Treelite version mismatch), so we did not use it.

## Acknowledgements

The original code is by Peiqian Li and Gao Han (linked from their report). The data is Lending Club's public loan data via Kaggle.
