# Fake Account Detection System

> Multimodal & Graph-Based Ensemble Model for detecting fake social-media
> accounts.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## Overview

This system classifies social-media accounts as **FAKE** or **GENUINE** by
fusing ten calibrated machine-learning models, graph-centrality features, and
BERT-based text embeddings through a two-stage stacked ensemble. It ships
with Instagram Graph API integration for real-time scanning and SHAP-based
explainability that maps PCA components back to human-readable feature names.

| Metric          | Value  |
| --------------- | ------ |
| Test Accuracy   | 97.06% |
| ROC-AUC         | 0.9969 |
| F1 (fake class) | 0.9693 |

---

## Quick Start

```bash
# 1. Clone & enter the project
cd doc

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the pipeline
python main.py --no-bert      # fast (TF-IDF proxy)
python main.py                # full (downloads BERT model)

# 5. Run predictions
python main.py --predict

# 6. Demo mode (no API key needed)
python main.py --demo
```

---

## Project Structure

```
doc/
├── main.py                        # CLI entry point
├── module1_data_engineering.py     # M1: cleaning, imputation, SMOTEENN
├── module2_feature_extraction.py   # M2: metadata + BERT + PCA
├── module3_graph_construction.py   # M3: NetworkX graph + GNN
├── module4_base_models.py          # M4: 10 base classifiers
├── module5_soft_voting.py          # M5: F1-weighted soft voting
├── modules678.py                   # M6-8: graph risk, stacking, SHAP
├── modules910.py                   # M9-10: train & predict pipelines
├── module11_output.py              # M11: output formatting
├── instagram_api.py                # Instagram Graph API v19.0 client
├── pca_interpretability.py         # PCA ↔ original feature mapping
├── realtime_monitor.py             # Continuous monitoring daemon
├── requirements.txt                # Python dependencies
├── DESCRIPTION.md                  # Detailed process documentation
├── README.md                       # This file
├── fake_social_media.csv           # Dataset: social media profiles
├── fake_users.csv                  # Dataset: fake user accounts
├── real_users.csv                  # Dataset: real user accounts
└── saved_models/
    └── pipeline.pkl                # Serialised trained pipeline
```

---

## CLI Reference

### Training

```bash
python main.py                          # Train with BERT embeddings
python main.py --no-bert                # Train with TF-IDF proxy (faster)
python main.py --optuna --optuna-trials 100  # Bayesian hyperparameter tuning
python main.py --save-path model.pkl    # Custom save location
```

### Prediction

```bash
python main.py --predict                # Predict on test split
python main.py --predict --save-path model.pkl
```

### Instagram Scanning

```bash
# One-shot scan (requires API token)
python main.py --instagram user1,user2 --token YOUR_TOKEN

# Continuous monitoring every 30 minutes
python main.py --realtime user1,user2 --token YOUR_TOKEN --interval 30

# Demo mode (synthetic profiles, no token needed)
python main.py --demo
```

### Explainability

```bash
python main.py --explain-pca           # Print PCA component meanings
```

### All Flags

| Flag              | Type | Default                     | Description                         |
| ----------------- | ---- | --------------------------- | ----------------------------------- |
| `--predict`       | flag | —                           | Run prediction mode                 |
| `--optuna`        | flag | —                           | Enable Optuna tuning                |
| `--optuna-trials` | int  | 50                          | Number of Optuna trials             |
| `--no-bert`       | flag | —                           | Use TF-IDF instead of BERT          |
| `--save-path`     | str  | `saved_models/pipeline.pkl` | Pipeline artefact path              |
| `--instagram`     | str  | —                           | Comma-separated usernames           |
| `--demo`          | flag | —                           | Demo mode (synthetic profiles)      |
| `--realtime`      | str  | —                           | Usernames for continuous monitoring |
| `--interval`      | int  | 60                          | Monitoring interval (minutes)       |
| `--token`         | str  | —                           | Instagram Graph API token           |
| `--explain-pca`   | flag | —                           | PCA interpretability report         |

---

## Datasets

The system expects the following files in the project directory:

| File                                             | Description                                     |
| ------------------------------------------------ | ----------------------------------------------- |
| `fake_social_media.csv`                          | Primary dataset with profile metadata           |
| `fake_social_media_global_2.0_with_missing.xlsx` | Extended dataset with missing values            |
| `fake_users.csv`                                 | Twitter fake account dataset (cross-validation) |
| `real_users.csv`                                 | Twitter real account dataset (cross-validation) |

---

## Pipeline Architecture

```
Raw Data → Data Engineering → Feature Extraction (97 features)
         → PCA (≈77 components) → 10 Base Models (calibrated)
         → Soft Voting → OOF Stacking → Meta-Learner
         → SHAP Explanation → Output Report
```

### Models Used

| Model         | Library      | Role                        |
| ------------- | ------------ | --------------------------- |
| XGBoost       | xgboost      | Gradient boosting (primary) |
| LightGBM      | lightgbm     | Fast gradient boosting      |
| CatBoost      | catboost     | Categorical-aware boosting  |
| Random Forest | scikit-learn | Bagging ensemble            |
| Extra Trees   | scikit-learn | Randomised splits           |
| SVM (RBF)     | scikit-learn | Kernel classifier           |
| KNN           | scikit-learn | Instance-based              |
| Naïve Bayes   | scikit-learn | Probabilistic baseline      |
| MLP           | scikit-learn | Neural network              |
| AdaBoost      | scikit-learn | Adaptive boosting           |

### Stacking Strategy

1. Each model produces isotonically calibrated `P(fake)`.
2. F1-weighted soft voting fuses probabilities.
3. 5-fold OOF stacking feeds a `LogisticRegression` meta-learner.
4. SHAP TreeExplainer provides per-prediction feature attributions.
5. PCA inverse mapping translates SHAP back to original feature names.

---

## Requirements

- Python ≥ 3.11
- See `requirements.txt` for the full dependency list.

### Core Dependencies

```
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
pandas
numpy
networkx
python-louvain
shap
joblib
optuna
sentence-transformers   # optional (for real BERT)
torch-geometric         # optional (for GNN)
matplotlib              # optional (for plots)
requests                # for Instagram API
```

---

## Instagram API Setup

1. Create a [Meta Developer](https://developers.facebook.com/) app.
2. Add the **Instagram Graph API** product.
3. Generate a long-lived user access token with `instagram_basic` scope.
4. Pass the token via `--token` or set the `INSTAGRAM_ACCESS_TOKEN`
   environment variable.

```bash
export INSTAGRAM_ACCESS_TOKEN="your_token_here"
python main.py --instagram targetuser
```

---

## Detailed Documentation

See [DESCRIPTION.md](DESCRIPTION.md) for an in-depth explanation of every
module, the mathematical foundations, and the complete data-flow diagram.

---

## License

This project is provided for educational and research purposes.
