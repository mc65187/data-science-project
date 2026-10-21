# Medical Insurance Cost Analysis

A complete end-to-end data science practice project: wrangling, exploring and
building regression models to predict individual medical insurance charges.

## Problem / Research question

Medical insurance charges vary a lot from person to person. Using the 7-attribute
insurance dataset (2,771 records), this project answers two questions:

1. Which factors most strongly drive medical insurance charges?
2. How accurately can regression models predict charges from these attributes?

## Dataset

- **Source:** IBM Developer Skills Network / Coursera *DA0101EN* — "Medical
  Insurance" dataset, publicly distributed for educational use.
- **Usage permissions:** public, educational dataset (no login or fee). It is
  fully anonymized and contains **no** names, addresses, IDs or any
  personally identifiable information, so it is safe to redistribute.
- **Content:** 2,771 rows × 7 columns — `age`, `gender`, `bmi`,
  `no_of_children`, `smoker`, `region`, `charges` (USD).
- The raw file contains a few `?` placeholders: 4 missing `age` values and
  7 missing `smoker` values, handled during data wrangling.

## Methods and tools used

| Stage | Tool / method |
| --- | --- |
| Data loading & wrangling | Python, pandas (missing-value imputation: mean for `age`, mode for `smoker`) |
| Exploratory data analysis | Matplotlib + Seaborn (regression plot, box plot, correlation matrix) |
| Model development | scikit-learn — `LinearRegression`, `Pipeline` (`StandardScaler` + `PolynomialFeatures` + `LinearRegression`) |
| Model refinement | `train_test_split` (20% test), Ridge regression (α = 0.1), polynomial features (degree 2) |

## Key findings

- **Smoker status is by far the strongest single predictor**: a linear model
  using only `smoker` achieves R² ≈ **0.62**, and smokers pay ~$23,805 more on
  average than non-smokers at the same other attributes.
- The full-feature linear model reaches R² ≈ **0.75**.
- A pipeline with scaling + polynomial features reaches R² ≈ **0.85** on the
  full data.
- Best test-set performance: Ridge with degree-2 polynomial features,
  R² ≈ **0.84**, slightly below the training-fit score — a sign of mild
  overfitting that is normal for a small dataset.
- **One limitation:** the models are linear/polynomial baselines only — they do
  not capture interactions with clinical or socioeconomic context, and the
  dataset is small and coarse (regional codes 1–4, no diagnosis or lifestyle
  detail), so predictions may not generalize to other populations.

## Instructions for running the analysis

1. Create a Python 3.9+ environment and install dependencies:

   ```bash
   conda create -n ds-project python=3.9
   conda activate ds-project
   pip install -r requirements.txt
   ```

2. Launch the notebook (the dataset is already included in `data/`):

   ```bash
   jupyter notebook insurance_cost_analysis.ipynb
   ```

3. Run all cells (Kernel → Restart & Run All). The notebook:
   - loads `data/medical_insurance_dataset.csv` (with a public-URL fallback),
   - performs data wrangling and EDA,
   - fits the five models and prints each R² score,
   - saves the three key figures into `visualizations/`.

Expected outputs are already stored in the notebook, so it can also be viewed
directly on GitHub without running anything.

## Acknowledgment

Educational project based on the IBM Skills Network *DA0101EN* practice project;
analysis and documentation by Sun Jia (mc651874), University of Macau.
