# Assignment 1 — Titanic Survival Prediction

Machine learning assignment comparing SVM and Random Forest classifiers on the Titanic dataset, covering the full pipeline from data exploration to model evaluation.

## Overview

The goal is to predict passenger survival using classical ML techniques, with emphasis on preprocessing impact, feature selection, hyperparameter tuning, and cross-validated evaluation.

## Dataset

Titanic dataset loaded via `seaborn` (891 passengers, 15 features). Target variable: `survived` (binary).

## Pipeline

### 1. Data Exploration
- Class distribution, survival rates, and sex breakdown
- Cross-tabulation of survival by passenger class (1st: 63%, 2nd: 47%, 3rd: 24%)

### 2. Data Preprocessing
- **Missing values:** median imputation for `age`, mode imputation for `embarked`, dropped `deck` (77% missing)
- **Encoding:** label encoding for `sex`, one-hot encoding for `embarked`
- **Scaling:** `StandardScaler` on `age`, `fare`, `sibsp`, `parch`

### 3. Feature Selection
- Correlation heatmap to remove redundant features (`alive`, `class`, `adult_male`, `who`, `embark_town`, `alone`)
- Mutual information gain to drop low-signal features (`embarked_Q`)
- Final feature set: 8 features

### 4. Data Division
- 5-fold Stratified K-Fold cross-validation (preserves class ratio per fold)

### 5 & 6. Model Building & Tuning

| Model | Tuning | Best Parameters |
|---|---|---|
| SVM | Grid search over kernel, C, gamma | `kernel=rbf`, `C=1`, `gamma=scale` |
| Random Forest | Grid search over n\_estimators, max\_depth, min\_samples\_split | `n_estimators=300`, `max_depth=8`, `min_samples_split=2` |

### 7. Evaluation

**Tuned model comparison (cross-validated):**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| SVM | 82.60% | 80.46% | 72.22% | 76.12% |
| Random Forest | **84.62%** | **84.98%** | **72.81%** | **78.43%** |

**Impact of preprocessing:**

| Model | Before | After |
|---|---|---|
| SVM Accuracy | 67.09% | 82.60% |
| SVM F1 | 45.98% | 76.12% |
| RF Accuracy | 79.69% | 84.62% |
| RF F1 | 74.25% | 78.43% |

Random Forest outperforms SVM across all metrics. Preprocessing had the largest single impact on performance.

## Files

| File | Description |
|---|---|
| `Assignment 1.ipynb` | Main notebook with full pipeline and outputs |
| `Assignment 1 Report.md` | Written report with analysis and discussion |
| `First+Assignment.pdf` | Original assignment specification |

## Dependencies

```
pandas
seaborn
matplotlib
scikit-learn
```
