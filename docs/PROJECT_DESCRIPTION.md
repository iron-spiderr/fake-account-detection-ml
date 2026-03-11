# Fake Account Detection System — Full Project Description

**Project Type:** Final Year Dissertation  
**Language:** Python 3.11  
**Interface:** Flask Web Application + CLI  
**Trained Model Accuracy:** 97.06% · ROC-AUC: 0.9969 · F1: 0.9693

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Datasets](#2-datasets)
3. [System Architecture](#3-system-architecture)
4. [Module 1 — Data Engineering](#4-module-1--data-engineering)
5. [Module 2 — Feature Extraction](#5-module-2--feature-extraction)
6. [Module 3 — Graph Construction](#6-module-3--graph-construction)
7. [Module 4 — Base Model Training](#7-module-4--base-model-training)
8. [Module 5 — Soft Weighted Voting](#8-module-5--soft-weighted-voting)
9. [Module 6 — Graph Risk Scoring](#9-module-6--graph-risk-scoring)
10. [Module 7 — OOF Stacking and Meta-Learning](#10-module-7--oof-stacking-and-meta-learning)
11. [Module 8 — SHAP Explainability](#11-module-8--shap-explainability)
12. [Module 9 — Training Pipeline](#12-module-9--training-pipeline)
13. [Module 10 — Real-Time Prediction Pipeline](#13-module-10--real-time-prediction-pipeline)
14. [Module 11 — Output and Reporting](#14-module-11--output-and-reporting)
15. [Instagram Scraper](#15-instagram-scraper)
16. [Flask Web Application](#16-flask-web-application)
17. [PCA Interpretability](#17-pca-interpretability)
18. [Real-Time Monitor](#18-real-time-monitor)
19. [End-to-End Data Flow](#19-end-to-end-data-flow)
20. [Training Results](#20-training-results)
21. [Running the Project](#21-running-the-project)

---

## 1. Project Overview

This system detects fake and bot-operated social media accounts using a multi-layer machine learning ensemble. It combines:

- **Hand-crafted metadata features** derived from account statistics
- **Text embeddings** of user biographies (BERT or TF-IDF/SVD)
- **Social graph centrality features** from a NetworkX interaction graph
- **10 diverse base classifiers** each calibrated for probability output
- **Soft weighted voting** to fuse base model predictions
- **Out-of-Fold stacking** with a Logistic Regression meta-learner
- **SHAP explanations** for individual prediction transparency
- **Real-time Instagram scraping** via browser-mimic requests + instaloader fallback
- **Flask web UI** for interactive use without any command-line knowledge

The pipeline is trained once, and the result is saved to `saved_models/pipeline.pkl`. All subsequent predictions — from the web UI, API endpoints, or CLI — load this single file.

---

## 2. Datasets

Four datasets are combined during training. All are harmonised to a common schema with a binary `label` column (`1` = fake, `0` = genuine).

| File                                             | Rows   | Source                      | Label Logic                   |
| ------------------------------------------------ | ------ | --------------------------- | ----------------------------- |
| `fake_social_media.csv`                          | 3,000  | Primary dataset             | `is_fake` column (0/1)        |
| `fake_users.csv`                                 | 2,500  | Twitter raw data            | All rows are fake (label = 1) |
| `LIMFADD.csv`                                    | 15,000 | Instagram-style multi-class | Bot/Scam/Spam → 1, Real → 0   |
| `fake_social_media_global_2.0_with_missing.xlsx` | 3,000  | Global extended dataset     | Same schema as primary CSV    |

After loading and combining: **23,416 raw rows**.  
After SMOTEENN class-balancing: **29,222 rows** used for training and evaluation.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
│                                                             │
│  Datasets ──► Module 1 ──► Module 2 ──► Module 3           │
│  (4 CSVs)    Data Eng.    Features    Graph                 │
│                  │             │           │                 │
│                  └──────────────────────────► Feature        │
│                                               Matrix (PCA)  │
│                                                   │          │
│             Module 4 ─── 10 Base Models ──────────┤         │
│             Module 5 ─── Soft Voting               │         │
│             Module 7 ─── OOF Stacking ─────────────►       │
│             Module 8 ─── SHAP Explainer             │        │
│                                                     ▼        │
│                                          pipeline.pkl        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                         │
│                                                             │
│  Input ──► scraper.py ──► Module 10 ──► Module 11 ──► UI   │
│  (username)  (or manual)   Predict      Format    Flask     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Module 1 — Data Engineering

**File:** `module1_data_engineering.py`

This module is responsible for loading, cleaning, normalising, and balancing the four datasets into a single unified DataFrame.

### 1.1 Data Loading

Each dataset is loaded by a dedicated function:

- **`_load_fake_social_media(path)`** — Reads `fake_social_media.csv`. Renames `is_fake` → `label`. Converts all numeric columns (followers, following, posts, etc.) using `pd.to_numeric`. Fills missing `bio` and `username` columns with placeholders.

- **`_load_excel_global(path)`** — Reads `fake_social_media_global_2.0_with_missing.xlsx` using openpyxl. Applies the same column harmonisation as the primary CSV.

- **`_load_fake_users(path)`** — Reads `fake_users.csv` (raw Twitter data). All rows are assigned `label = 1` (fake). Maps Twitter-specific column names (`statuses_count` → `posts`, `friends_count` → `following`, etc.).

- **`_load_limfadd(path)`** — Reads `LIMFADD.csv`. Maps string labels: `Bot`, `Scam`, `Spam` → `1`; `Real` → `0`. Normalises column names to the common schema.

All four DataFrames are concatenated into one combined DataFrame. A `_source` column tracks origin.

### 1.2 Noise Removal

**`remove_noise(df)`** — Cleans the merged DataFrame:

- Deduplicates rows (keeps first occurrence)
- Removes constant columns where all values are identical (excluding `label`, `_source`, `username`, `bio`)
- Strips leading/trailing whitespace from all object-typed string columns
- Clips extreme values using **IQR-based capping**: lo = Q1 − 5×IQR, hi = Q3 + 5×IQR on every numeric column
- Resets the index

### 1.3 Adversarial Normalisation

**`adversarial_normalise(df)`** — Defends against manipulation of text fields:

- **Leet-speak decoding** — replaces `0→o`, `1→i`, `3→e`, `4→a`, `5→s`, `7→t`, `@→a`, `$→s` in usernames and bios
- **Zero-width character removal** — strips `\u200b`, `\u200c`, `\u200d`, `\ufeff`, `\u00ad` from text
- **Unicode NFKC normalisation** — collapses visually similar Unicode characters to their canonical ASCII equivalents

### 1.4 Username Entropy

**`username_entropy(username)`** / **`add_username_entropy(df)`** — Computes the **Shannon entropy** of each username's character distribution:

$$H = -\sum_{c} \frac{n_c}{N} \log_2 \frac{n_c}{N}$$

Highly random, algorithmically generated usernames (e.g. `xq7fk2m`) yield high entropy, which is a strong fake-account signal. The result is stored in a new `username_entropy` column that Module 2 includes in the feature matrix. Called by `run_data_engineering()` after adversarial normalisation.

### 1.5 Missing Value Imputation

Uses scikit-learn's `IterativeImputer` backed by `ExtraTreesRegressor`. This approach models each feature as a function of the others, cycling through multiple imputation rounds until convergence. This outperforms simple mean/median imputation for correlated features. Categorical columns receive **mode imputation** before the numeric MICE pass.

### 1.6 Class Balancing

**`balance_classes(df)`** — Applies **SMOTEENN** (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbours):

1. **SMOTE** generates synthetic minority-class samples by interpolating between existing samples in feature space.
2. **ENN** removes samples that are misclassified by their nearest neighbours, cleaning the decision boundary.

The result is a balanced dataset where both classes have approximately equal representation, reducing classifier bias.

### 1.7 Orchestration

**`run_data_engineering(..., balance=True)`** — Convenience wrapper that calls `load_datasets → remove_noise → adversarial_normalise → add_username_entropy → impute_missing → balance_classes` in sequence and returns the final DataFrame.

---

## 5. Module 2 — Feature Extraction

**File:** `module2_feature_extraction.py`

Converts the cleaned DataFrame into a numeric feature matrix ready for ML models.

### 2.1 Metadata Features (34 features)

**`extract_metadata_features(df)`** derives 34 features from account statistics:

| Feature                    | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `has_profile_pic`          | Binary 0/1                                         |
| `verified`                 | Binary 0/1                                         |
| `bio_length`               | Character length of biography                      |
| `bio_has_url`              | URL detected in bio text                           |
| `bio_spam_score`           | Count of spam keywords in bio                      |
| `followers`                | Raw follower count                                 |
| `following`                | Raw following count                                |
| `follow_diff`              | followers − following                              |
| `follower_following_ratio` | followers ÷ max(following, 1)                      |
| `followers_ratio`          | followers ÷ (followers + following + 1)            |
| `posts`                    | Number of posts                                    |
| `account_age_days`         | Days since account creation                        |
| `posts_per_day`            | posts ÷ account_age_days                           |
| `activity_rate`            | posts ÷ (followers + 1), clipped at 100            |
| `log_followers`            | log1p(followers)                                   |
| `log_posts`                | log1p(posts)                                       |
| `account_hour`             | account_age_days mod 24 (creation-hour proxy)      |
| `account_weekend`          | Binary: account_age_days mod 7 ≥ 5                 |
| `listed_ratio`             | listed_count ÷ (followers + 1)                     |
| `likes_per_post`           | favourites_count ÷ (posts + 1)                     |
| `caption_similarity_score` | Cross-post text similarity                         |
| `content_similarity_score` | Visual content similarity proxy                    |
| `follow_unfollow_rate`     | Churn signal                                       |
| `spam_comments_rate`       | Comment quality signal                             |
| `generic_comment_rate`     | Repetitive comment signal                          |
| `suspicious_links_in_bio`  | External link risk flag (clipped 0/1)              |
| `verified_low_follow`      | Binary: unverified AND < 100 followers             |
| `username_length`          | Character length of username                       |
| `username_randomness`      | Raw randomness column from dataset                 |
| `digits_count`             | Number of digit characters in username             |
| `digit_ratio`              | digits_count ÷ username_length                     |
| `special_char_count`       | Count of non-alphanumeric characters in username   |
| `repeat_char_count`        | Count of consecutive repeated characters           |
| `username_entropy`         | Shannon entropy of username character distribution |

Missing columns are filled with zeros (not errors) using the `_get_col()` helper, making inference safe even when only partial data is available.

### 2.2 Text Embeddings

**`BERTEmbedder`** (requires `sentence-transformers`):

- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Produces 384-dimensional embeddings per biography text
- Encodes in batches of 64 for efficiency

**`DummyBERTEmbedder`** (fallback, no GPU needed):

- Uses `TfidfVectorizer` (max 5,000 features) + `TruncatedSVD` (64 components)
- Produces 64-dimensional embeddings
- Gracefully returns zero vectors if all bios are empty (as is the case with the provided datasets)

### 2.3 Feature Assembly and Dimensionality Reduction

**`build_unified_features(df, embedder, graph_features, scaler, pca, fit)`**:

1. Stacks metadata (34) + text embeddings (64 or 384) + graph features (5) → typically **103 raw features**
2. Applies `StandardScaler` to zero-mean, unit-variance normalise
3. Applies `PCA` retaining **95% of variance** → reduces to approximately **20 principal components**

During training (`fit=True`), the scaler and PCA are fitted. During inference (`fit=False`), the saved scaler and PCA are applied. If inference data has fewer features than the scaler expects (e.g. graph features absent), the missing columns are padded with the **training-set feature means** (not zeros). This ensures absent features produce a neutral z-score of 0 after scaling rather than a large negative value that would bias every inference profile toward FAKE.

---

## 6. Module 3 — Graph Construction

**File:** `module3_graph_construction.py`

Models the social network structure as a graph and extracts centrality features.

### 3.1 Graph Building

**`build_interaction_graph(df)`** — Constructs an undirected `networkx.Graph`:

- Each row (user) becomes a node
- Users are bucketed into 20 logarithmic tiers by follower count
- Within each tier, up to 200 edges are sampled between similar-follower-count users
- This creates a proxy interaction graph without needing actual follower lists

### 3.2 Centrality Features

**`compute_centrality_features(G, df)`** — Computes 5 per-node features:

| Feature                  | Description                                  |
| ------------------------ | -------------------------------------------- |
| `pagerank`               | Iterative importance score (Google PageRank) |
| `betweenness_centrality` | How often a node lies on shortest paths      |
| `clustering_coefficient` | Density of connections among neighbours      |
| `degree_centrality`      | Fraction of possible connections formed      |
| `community_size_ratio`   | Node's community size ÷ graph size           |

Fake accounts tend to have low clustering coefficients (not embedded in cohesive groups), high degree centrality (mass following), and low PageRank (one-directional connections).

### 3.3 Optional GNN Embeddings

When `--gnn` is passed and `torch_geometric` is installed, a Graph Convolutional Network + Graph Attention Network produces 64-dimensional node embeddings per user, replacing the 5 scalar features with richer structural representations.

---

## 7. Module 4 — Base Model Training

**File:** `module4_base_models.py`

Trains 10 diverse classifiers, each wrapped with isotonic probability calibration.

### The 10 Base Models

| Model          | Library      | Key Parameters                   |
| -------------- | ------------ | -------------------------------- |
| XGBoost        | xgboost      | 200 trees, depth 6, lr 0.1       |
| LightGBM       | lightgbm     | 200 trees, 31 leaves, lr 0.05    |
| CatBoost       | catboost     | 200 iterations, depth 6, lr 0.05 |
| Random Forest  | scikit-learn | 200 trees                        |
| Extra Trees    | scikit-learn | 200 trees                        |
| SVM (RBF)      | scikit-learn | C=1.0, probability=True          |
| KNN            | scikit-learn | k=7 neighbours                   |
| Naive Bayes    | scikit-learn | Gaussian                         |
| MLP Neural Net | scikit-learn | 128→64 ReLU, early stopping      |
| AdaBoost       | scikit-learn | 100 estimators                   |

### Calibration

Each base model is wrapped in `CalibratedClassifierCV` with **isotonic regression** and 3-fold cross-validation. This converts raw decision scores into reliable probability estimates (Platt scaling or similar), which is essential for the soft-voting and stacking layers to function correctly.

### Individual Validation F1 Scores (Training Run)

| Model         | Val F1 |
| ------------- | ------ |
| XGBoost       | 0.9679 |
| LightGBM      | 0.9639 |
| CatBoost      | 0.9644 |
| Random Forest | 0.9647 |
| Extra Trees   | 0.9707 |
| SVM           | 0.9676 |
| KNN           | 0.9623 |
| Naive Bayes   | 0.9492 |
| MLP           | 0.9696 |
| AdaBoost      | 0.9509 |

### Optional Optuna Tuning

When `--optuna` is passed, `optuna_tune_xgboost()` runs Bayesian hyperparameter optimisation over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, and `reg_lambda`, maximising cross-validated F1 with a 3-fold `StratifiedKFold`.

---

## 8. Module 5 — Soft Weighted Voting

**File:** `module5_soft_voting.py`

Combines the calibrated probability outputs from all 10 base models into a single ensemble probability.

### How It Works

1. **Weight Computation** — For each model, compute its F1 score on the validation set. Weights are normalised so they sum to 1:

   $$w_m = \frac{F1_m}{\sum_{j} F1_j}$$

   Models with higher F1 get proportionally more influence.

2. **Soft Vote** — The ensemble probability for sample $i$ is:

   $$\hat{p}_i = \sum_{m=1}^{M} w_m \cdot p_{im}$$

   where $p_{im}$ is model $m$'s P(fake) for sample $i$.

3. **Threshold Optimisation** — Grid search over thresholds $[0.20, 0.80]$ in steps of 0.01 to find the threshold that maximises F1 on the validation set. The optimal threshold found during training was **0.32**.

---

## 9. Module 6 — Graph Risk Scoring

**File:** `modules678.py` — `compute_graph_risk()`

Produces a per-user graph risk score $\in [0, 1]$ from the social graph features.

When graph features are available:
$$\text{risk}_i = 0.5 \times \text{degree\_centrality}_i + 0.5 \times (1 - \text{community\_size\_ratio}_i)$$

High degree (mass-following behaviour) combined with low community integration is a strong fake-account signal.

When graph features are unavailable (inference on isolated profiles), a constant default risk of **0.15** is used.

---

## 10. Module 7 — OOF Stacking and Meta-Learning

**File:** `modules678.py` — `OOFStackedEnsemble`

A second-level ensemble that learns how to combine the 10 base model outputs optimally.

### Why OOF (Out-of-Fold)?

Training the meta-learner on the same data used to train the base models would cause data leakage — the base models would appear artificially accurate. OOF prevents this by ensuring every meta-training sample was predicted by base models that had never seen it.

### Process

1. **5-Fold Cross-Validation** — The training data is split into 5 folds. For each fold:
   - Base models are retrained on 4 folds
   - Predictions are made on the held-out fold
   - These predictions become the OOF meta-features for that fold's rows

2. **Meta-Feature Matrix** — The final OOF matrix is $(N, M+1)$ where $M=10$ (base models) + 1 (graph risk score).

3. **Meta-Learner** — A `LogisticRegression` (C=1.0, max_iter=1000) is fitted on the OOF meta-features. It learns which base models to trust more in which regions of the feature space.

4. **Inference** — At prediction time, the 10 base models produce probabilities as usual, graph risk is appended, and the meta-learner produces the final probability.

---

## 11. Module 8 — SHAP Explainability

**File:** `modules678.py` — `SHAPExplainer` / `DummySHAPExplainer` / `get_shap_explainer()`

Provides per-prediction explanations using SHAP (SHapley Additive exPlanations).

- **`get_shap_explainer(models, feature_names)`** selects the best available tree model in priority order: XGBoost → LightGBM → CatBoost → RandomForest → ExtraTrees. It unwraps the `CalibratedClassifierCV` wrapper to access the raw base estimator before passing it to `shap.TreeExplainer`.
- **`DummySHAPExplainer`** is used as a fallback when the `shap` library is not installed — it ranks features by absolute value magnitude as a proxy for importance.
- For each prediction, returns per-feature SHAP values indicating how much each feature pushed the probability up or down
- The top **5** contributing features (in PCA space) are mapped back to original feature names via `PCAInterpreter` and included in the human-readable explanation
- SHAP values are computed in **PCA space**, then mapped back to original 103-feature names via `PCAInterpreter.map_shap()`

---

## 12. Module 9 — Training Pipeline

**File:** `modules910.py` — `train_pipeline()`

Orchestrates all modules in sequence to produce the trained `pipeline.pkl`.

### 14-Step Process

| Step  | Action                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------- |
| 1–2   | Load all 4 datasets, clean, add username entropy, impute, balance (Module 1)                         |
| 3     | Build interaction graph and compute centrality (Module 3)                                            |
| 4–5   | Extract metadata + text + graph features, scale, PCA (Module 2)                                      |
| 6     | Stratified 80/20 train/test split; derive `fixed_meta_cols` for inference alignment                  |
| 7     | Train 10 calibrated base models; optionally replace XGBoost with Optuna-tuned variant (Module 4)     |
| 8     | Compute F1-weighted soft voting weights; grid-search optimal threshold (Module 5)                    |
| 9–10  | Generate OOF meta-features with graph risk column; train Logistic Regression meta-learner (Module 7) |
| 11–12 | Initialise SHAP explainer (best tree model); build PCA interpretation map (Module 8 + 17)            |
| 13    | Save pipeline to `saved_models/pipeline.pkl` (first save, before metrics)                            |
| 14    | Evaluate on test set; re-save pipeline with metrics appended                                         |

### Saved Pipeline Contents

The `pipeline.pkl` file contains:

```python
{
    "scaler":           StandardScaler,        # fitted on training features
    "pca":              PCA,                   # fitted PCA (103 → ~20 components)
    "bert_embedder":    DummyBERTEmbedder,     # fitted TF-IDF + SVD
    "base_models":      dict[str, Calibrated], # 10 calibrated classifiers
    "soft_voter":       SoftWeightedVoter,     # F1 weights + optimal threshold
    "meta_learner":     OOFStackedEnsemble,    # Logistic Regression meta-model
    "shap_explainer":   SHAPExplainer,         # TreeExplainer (XGBoost or best tree)
    "pca_interpreter":  PCAInterpreter,        # SHAP → original feature mapping
    "feature_names":    list[str],             # ~103 raw feature names
    "pca_feature_names": list[str],            # PCA component names (PC_0 … PC_N)
    "optimal_threshold": float,                # threshold found by grid search
    "fixed_meta_cols":  list[str],             # metadata column order for inference alignment
    "metrics":          { accuracy, roc_auc, f1_fake, f1_genuine }  # added after evaluation
}
```

---

## 13. Module 10 — Real-Time Prediction Pipeline

**File:** `modules910.py` — `predict()`

Runs inference on any new DataFrame using the saved pipeline.

### Prediction Steps

1. Load the pipeline from `saved_models/pipeline.pkl` (or accept it as a parameter)
2. Extract features using the saved scaler, PCA, and embedder (`fit=False` — transform only)
3. If inference data has fewer features than the scaler expects (e.g. graph features absent), pad with the **training-set feature means** so absent features produce a neutral z-score after scaling
4. Collect P(fake) from all 10 base models → probability matrix $(N \times 10)$
5. Soft-vote the ensemble probability using saved F1 weights
6. Append default graph risk (0.15) for unseen profiles → $(N \times 11)$ meta-feature matrix
7. Meta-learner (Logistic Regression) produces the final probability
8. Apply saved optimal threshold to produce binary label
9. Optionally compute SHAP values; map top-**5** PCA-space contributors back to original feature names via `PCAInterpreter`
10. Pass to Module 11 for output formatting

---

## 14. Module 11 — Output and Reporting

**File:** `module11_output.py`

Converts raw numeric predictions into structured, human-readable results.

### Output per Profile

| Field             | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `username`        | Account username                                                |
| `label`           | `"FAKE"` or `"GENUINE"`                                         |
| `probability`     | P(fake) ∈ [0, 1]                                                |
| `confidence`      | Formatted confidence percentage (max(p, 1−p) × 100%)            |
| `risk_band`       | `CRITICAL` (≥85%), `HIGH` (≥60%), `MEDIUM` (≥30%), `LOW` (<30%) |
| `graph_risk`      | Graph-derived risk score                                        |
| `explanation`     | Natural-language explanation string                             |
| `top_features`    | Top 5 SHAP-contributing feature names (original feature space)  |
| `top_shap_values` | List of (feature_name, SHAP_value) pairs for the top 5 features |

### Risk Bands

| Band     | Threshold      | Interpretation                                      |
| -------- | -------------- | --------------------------------------------------- |
| CRITICAL | P(fake) ≥ 0.85 | Almost certainly fake; immediate action recommended |
| HIGH     | P(fake) ≥ 0.60 | Likely fake; warrants review                        |
| MEDIUM   | P(fake) ≥ 0.30 | Suspicious; monitor                                 |
| LOW      | P(fake) < 0.30 | Likely genuine                                      |

---

## 15. Instagram Scraper

**File:** `scraper.py`

Fetches live profile data from Instagram without requiring API credentials.

### Two-Method Strategy

```
scrape_profile(username)
│
├─ Method 1: requests + browser headers (PRIMARY)
│     URL: https://www.instagram.com/api/v1/users/web_profile_info/
│     Headers: Chrome UA, X-IG-App-ID: 936619743392459
│     ↳ Separate rate-limit pool from instaloader
│     ↳ Returns immediately on 429 (no retry hang)
│
└─ Method 2: instaloader (FALLBACK)
      URL: https://i.instagram.com/api/v1/ (iPhone API)
      ↳ Only tried if Method 1 fails or returns no data
      ↳ max_connection_attempts = 1 (no 30-minute retry loop)
```

### Data Available

| Field             | Public | Private |
| ----------------- | ------ | ------- |
| Username          | ✓      | ✓       |
| Full name         | ✓      | ✓       |
| Biography         | ✓      | ✓       |
| Followers         | ✓      | ✓       |
| Following         | ✓      | ✓       |
| Post count        | ✓      | ✓       |
| Is verified       | ✓      | ✓       |
| External URL      | ✓      | ✓       |
| Business category | ✓      | ✗       |
| Post content      | ✓      | ✗       |

For private profiles, `is_partial=True` is set and a warning is included in the API response.

### Rate Limiting

When HTTP 429 is returned, `RateLimitError` is raised immediately. The Flask endpoint returns HTTP 429 with `"rate_limited": true` in the JSON. The UI displays a yellow advisory box with recovery instructions.

---

## 16. Flask Web Application

**File:** `app.py` — Templates: `templates/index.html`, `static/style.css`, `static/app.js`

A dark-themed single-page web application served by Flask.

### API Endpoints

| Endpoint                  | Method | Description                                                                                      |
| ------------------------- | ------ | ------------------------------------------------------------------------------------------------ |
| `GET /api/status`         | GET    | Pipeline ready status and session history count                                                  |
| `POST /api/demo`          | POST   | Run 5 synthetic demo profiles through the pipeline; returns accuracy summary                     |
| `POST /api/demo-test`     | POST   | Run predictions on `test_profiles.csv` with ground-truth comparison                              |
| `POST /api/scrape`        | POST   | Scrape a real Instagram username and predict                                                     |
| `POST /api/scan-self`     | POST   | Scan own account via Instagram Graph API token                                                   |
| `POST /api/scan-users`    | POST   | Scan multiple accounts via Business Discovery API                                                |
| `POST /api/manual`        | POST   | Accept manually entered profile fields, predict (uses genuine-class medians for latent features) |
| `GET /api/history`        | GET    | Return last 50 scan results from session                                                         |
| `POST /api/history/clear` | POST   | Clear session history                                                                            |

### Web UI Tabs

| Tab            | Function                                                 |
| -------------- | -------------------------------------------------------- |
| Demo Scan      | One-click demo with 5 synthetic profiles                 |
| Scrape Profile | Enter any Instagram username; live scraping + prediction |
| Self Scan      | Paste an Instagram Graph API access token                |
| Username Scan  | Comma-separated usernames via Business Discovery         |
| Manual Input   | Fill in fields manually (offline use)                    |
| History        | View all previous scan results in the current session    |

### Result Cards

Each detected profile is displayed as a card showing:

- Username and FAKE/GENUINE badge
- Probability bar (red for fake, green for genuine)
- Risk band chip
- Graph risk percentage
- Natural-language explanation
- Top contributing feature chips (with SHAP direction indicator)

Demo and demo-test responses also include a summary block with accuracy, precision, recall, and F1 computed against ground-truth labels.

---

## 17. PCA Interpretability

**File:** `pca_interpretability.py`

Bridges the gap between PCA-compressed SHAP values and human-understandable feature names.

- Stores the PCA component matrix (20 × 103)
- When SHAP produces importance scores in PCA space, `map_shap()` multiplies SHAP values by the component loadings to recover approximate importance scores in the original 103-feature space
- `component_report()` prints the top-loading original features for each principal component

---

## 18. Real-Time Monitor

**File:** `realtime_monitor.py`

A continuous monitoring daemon for production deployment scenarios.

- **`scan(profiles)`** — Batch-predicts a list of profiles
- **`start_continuous(interval_minutes)`** — Loops indefinitely, scanning on each tick
- **`generate_report()`** — Summarises detected fake accounts with timestamps
- Designed to be called from `main.py --realtime --interval 60`

---

## 19. End-to-End Data Flow

```
Input (username / manual fields / demo profiles)
          │
          ▼
scraper.py (Instagram web API → instaloader fallback)
          │
          ▼
profile_to_prediction_dict()
  → {followers, following, posts, bio, username, ...}
          │
          ▼
module2: extract_metadata_features()       [33 features]
       + DummyBERTEmbedder.embed(bio)      [64 features]
       + zero-pad graph features           [ 5 features → 0.0]
       = 103-feature raw matrix
          │
          ▼
StandardScaler.transform()   (zero-mean, unit-variance)
          │
          ▼
PCA.transform()              (103 → 20 components)
          │
          ▼
10 base models → predict_proba()  →  (N × 10) probability matrix
          │
          ▼
SoftWeightedVoter.soft_vote()     →  ensemble P(fake) per row
          │
          ▼
append graph_risk = [0.15, ...]   →  (N × 11) meta-features
          │
          ▼
OOFStackedEnsemble (LogisticRegression).predict_proba()
          │
          ▼
apply threshold 0.32              →  binary label [0, 1]
          │
          ▼
SHAPExplainer.explain()           →  top feature contributions
          │
          ▼
module11_output.format_output()   →  results DataFrame
          │
          ▼
JSON API response / UI result card
```

---

## 20. Training Results

Final evaluation on the held-out 20% test set (5,845 samples):

| Metric             | Value      |
| ------------------ | ---------- |
| Accuracy           | **97.06%** |
| ROC-AUC            | **0.9969** |
| F1 (Fake class)    | **0.9693** |
| F1 (Genuine class) | **0.9718** |
| Optimal threshold  | **0.32**   |

Confusion matrix:

```
                Predicted Genuine   Predicted Fake
Actual Genuine       2,962                88
Actual Fake             84             2,711
```

---

## 21. Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python main.py --no-bert
```

With Optuna tuning:

```bash
python main.py --no-bert --optuna --optuna-trials 50
```

### Launch the Web UI

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

### CLI Prediction

```bash
# Demo with 5 synthetic profiles
python main.py --demo

# Predict from a CSV file
python main.py --predict path/to/profiles.csv

# Explain PCA components
python main.py --explain-pca
```

### Project File Map

```
final year/
├── main.py                         CLI entry point
├── app.py                          Flask web server
├── scraper.py                      Instagram profile scraper
├── module1_data_engineering.py     Load, clean, balance data
├── module2_feature_extraction.py   Feature matrix construction
├── module3_graph_construction.py   NetworkX social graph
├── module4_base_models.py          10 calibrated classifiers
├── module5_soft_voting.py          F1-weighted soft voting
├── modules678.py                   Graph risk + OOF stacking + SHAP
├── modules910.py                   Training + prediction pipelines
├── module11_output.py              Output formatting and risk bands
├── pca_interpretability.py         SHAP → original feature mapping
├── instagram_api.py                Instagram Graph API client
├── realtime_monitor.py             Continuous monitoring daemon
├── requirements.txt                Python dependencies
├── templates/index.html            Web UI (dark theme, 6 tabs)
├── static/style.css                UI styles
├── static/app.js                   UI logic
├── saved_models/pipeline.pkl       Trained model (generated by training)
├── fake_social_media.csv           Dataset 1
├── fake_users.csv                  Dataset 2
├── LIMFADD.csv                     Dataset 3
└── fake_social_media_global_2.0_with_missing.xlsx  Dataset 4
```
