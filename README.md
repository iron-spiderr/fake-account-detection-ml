# Fake Account Detection System

> A production-grade ML system that detects fake and bot-operated social media accounts using a **10-model stacked ensemble** with SHAP-based explainability.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy: 97%](https://img.shields.io/badge/Accuracy-97.06%25-brightgreen.svg)](#results)

<p align="center">
  <img src="docs/screenshot.png" alt="Web UI Screenshot" width="700">
</p>

---

## Highlights

- **97.06% accuracy** on held-out test set (ROC-AUC: 0.9969)
- **10 calibrated classifiers** (XGBoost, LightGBM, CatBoost, RF, SVM, MLP, etc.) fused via soft voting + OOF meta-learning
- **103 engineered features** from profile metadata, text embeddings, and social graph centrality
- **SHAP explanations** with PCA inverse-mapping — every prediction is interpretable
- **Flask web UI** for live Instagram scraping, manual input, and batch scanning
- **Real-time monitoring** daemon for continuous surveillance

---

## Results

Evaluated on a held-out 20% test set (5,845 samples):

| Metric                | Value  |
| --------------------- | ------ |
| **Test Accuracy**     | 97.06% |
| **ROC-AUC**           | 0.9969 |
| **F1 (Fake class)**   | 0.9693 |
| **F1 (Genuine)**      | 0.9718 |
| **Optimal Threshold** | 0.32   |

```
Confusion Matrix:
                Predicted Genuine   Predicted Fake
Actual Genuine       2,962               88
Actual Fake             84            2,711
```

---

## Architecture

```
Raw Data ──► Data Engineering ──► Feature Extraction (103 features)
             (clean, impute,       (metadata + BERT/TF-IDF + graph)
              SMOTEENN balance)              │
                                    StandardScaler + PCA (95% var)
                                             │
                               ┌─────────────┼─────────────┐
                               │   10 Calibrated Base Models │
                               │  (XGBoost, LightGBM, CatBoost, │
                               │   RF, ET, SVM, KNN, NB, MLP, Ada)│
                               └─────────────┼─────────────┘
                                             │
                                  F1-Weighted Soft Voting
                                             │
                                  OOF Stacking + Graph Risk
                                             │
                                  Logistic Regression Meta-Learner
                                             │
                                  SHAP Explanation + Output
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-account-detection.git
cd fake-account-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py --no-bert         # Fast mode (TF-IDF proxy)
python main.py                   # Full mode (BERT embeddings)

# Launch the web UI
python -m web.app
# Open http://127.0.0.1:5000

# Or run predictions from CLI
python main.py --demo
```

---

## Project Structure

```
fake-account-detection/
│
├── main.py                          # CLI entry point (train / predict / scan)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── LICENSE
├── README.md
│
├── src/                             # Core ML pipeline (Python package)
│   ├── __init__.py
│   ├── data_engineering.py          # Data loading, cleaning, SMOTEENN balancing
│   ├── feature_extraction.py        # 103 features: metadata + BERT/TF-IDF + PCA
│   ├── graph_construction.py        # NetworkX social graph + centrality features
│   ├── base_models.py               # 10 calibrated classifiers
│   ├── soft_voting.py               # F1-weighted ensemble voting
│   ├── stacking_shap.py             # OOF stacking, meta-learner, SHAP explainer
│   ├── pipeline.py                  # End-to-end training & prediction orchestration
│   ├── output.py                    # Human-readable output formatting
│   ├── pca_interpretability.py      # SHAP ↔ original feature name mapping
│   ├── instagram_api.py             # Instagram Graph API client
│   ├── scraper.py                   # Instagram web profile scraper
│   └── realtime_monitor.py          # Continuous monitoring daemon
│
├── web/                             # Flask web application
│   ├── app.py                       # Flask server + API endpoints
│   ├── static/
│   │   ├── app.js                   # Frontend logic
│   │   └── style.css                # Dark-theme UI styles
│   └── templates/
│       └── index.html               # Single-page app template
│
├── data/                            # Training datasets
│   ├── fake_social_media.csv        # Primary dataset (3,000 profiles)
│   ├── fake_users.csv               # Twitter fake accounts (2,500)
│   ├── LIMFADD.csv                  # Instagram-style multi-class (15,000)
│   └── fake_social_media_global_2.0_with_missing.xlsx
│
├── scripts/                         # Utility scripts
│   └── generate_test_data.py        # Generate synthetic test profiles
│
├── docs/                            # Documentation
│   ├── PROJECT_DESCRIPTION.md       # Full technical documentation
│   └── PROCESS_DOCUMENT.md          # Development process notes
│
└── saved_models/                    # Generated after training (gitignored)
    └── pipeline.pkl
```

---

## Web UI

The Flask web interface provides six modes of operation:

| Tab               | Description                                      |
| ----------------- | ------------------------------------------------ |
| **Demo Scan**     | One-click demo with curated profiles             |
| **Scrape**        | Enter any public Instagram username              |
| **Self Scan**     | Scan your own account via Graph API token        |
| **Username Scan** | Batch scan multiple users via Business Discovery |
| **Manual Input**  | Enter profile fields manually (offline mode)     |
| **History**       | View all previous scan results                   |

Each result card shows: prediction label, probability bar, risk band, graph risk, SHAP feature contributions, and a natural-language explanation.

---

## CLI Reference

```bash
# Training
python main.py                                    # Train with BERT
python main.py --no-bert                          # Train with TF-IDF (faster)
python main.py --optuna --optuna-trials 100       # Bayesian hyperparameter tuning

# Inference
python main.py --demo                             # Demo with synthetic profiles
python main.py --predict                          # Predict on test split
python main.py --explain-pca                      # PCA component report

# Instagram scanning (requires API token)
python main.py --instagram user1,user2 --token TOKEN
python main.py --realtime user1,user2 --token TOKEN --interval 30
```

---

## Datasets

| File                              | Rows   | Description                                  |
| --------------------------------- | ------ | -------------------------------------------- |
| `fake_social_media.csv`           | 3,000  | Primary dataset with `is_fake` labels        |
| `fake_users.csv`                  | 2,500  | Twitter fake accounts (all label=1)          |
| `LIMFADD.csv`                     | 15,000 | Instagram-style: Bot/Scam/Spam → 1, Real → 0 |
| `...global_2.0_with_missing.xlsx` | 3,000  | Extended dataset with missing values         |

Combined: **23,416 rows** → After SMOTEENN balancing: **29,222 training samples**.

---

## Key Technologies

| Category                  | Tools                                      |
| ------------------------- | ------------------------------------------ |
| **ML Models**             | XGBoost, LightGBM, CatBoost, scikit-learn  |
| **Balancing**             | SMOTEENN (imbalanced-learn)                |
| **Text Embeddings**       | sentence-transformers / TF-IDF+SVD         |
| **Graph Analysis**        | NetworkX, Louvain community detection      |
| **Explainability**        | SHAP (TreeExplainer) + PCA inverse mapping |
| **Web Framework**         | Flask                                      |
| **Hyperparameter Tuning** | Optuna                                     |

---

## How It Works

1. **Data Engineering** — Four datasets are loaded, cleaned, adversarially normalised, imputed (MICE), and class-balanced with SMOTEENN.
2. **Feature Extraction** — 34 metadata features + 64-dim text embeddings + 5 graph centrality features = 103 features, reduced via PCA to ~20 components.
3. **Base Model Training** — 10 diverse classifiers, each wrapped in isotonic calibration for reliable probabilities.
4. **Ensemble Fusion** — F1-weighted soft voting combines base model outputs; 5-fold OOF stacking trains a Logistic Regression meta-learner.
5. **Explainability** — SHAP values computed in PCA space are projected back to original feature names via component loadings.
6. **Inference** — Saved pipeline processes any new profile in <1 second with full explanation.

For detailed technical documentation, see [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md).

---

## License

This project is provided under the [MIT License](LICENSE) for educational and research purposes.
