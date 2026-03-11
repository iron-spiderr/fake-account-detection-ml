# PROCESS DOCUMENT — Fake Account Detection System

> **Full Technical Walkthrough**  
> Version 3.0 — February 2026  
> Python 3.11 · scikit-learn · XGBoost · LightGBM · CatBoost · NetworkX · SHAP · Flask

---

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [System Architecture Overview](#2-system-architecture-overview)
3.  [Datasets & Data Sources](#3-datasets--data-sources)
4.  [Process 1 — Data Engineering (Module 1)](#4-process-1--data-engineering-module-1)
5.  [Process 2 — Feature Extraction (Module 2)](#5-process-2--feature-extraction-module-2)
6.  [Process 3 — Graph Construction (Module 3)](#6-process-3--graph-construction-module-3)
7.  [Process 4 — Base Model Training (Module 4)](#7-process-4--base-model-training-module-4)
8.  [Process 5 — Soft Weighted Voting (Module 5)](#8-process-5--soft-weighted-voting-module-5)
9.  [Process 6 — Graph Intelligence (Module 6)](#9-process-6--graph-intelligence-module-6)
10. [Process 7 — Meta-Learning via OOF Stacking (Module 7)](#10-process-7--meta-learning-via-oof-stacking-module-7)
11. [Process 8 — Explainability with SHAP (Module 8)](#11-process-8--explainability-with-shap-module-8)
12. [Process 9 — End-to-End Training Pipeline (Module 9)](#12-process-9--end-to-end-training-pipeline-module-9)
13. [Process 10 — Real-Time Prediction Pipeline (Module 10)](#13-process-10--real-time-prediction-pipeline-module-10)
14. [Process 11 — Output & Reporting (Module 11)](#14-process-11--output--reporting-module-11)
15. [Process 12 — Instagram API Integration](#15-process-12--instagram-api-integration)
16. [Process 13 — PCA Interpretability](#16-process-13--pca-interpretability)
17. [Process 14 — Real-Time Monitoring Daemon](#17-process-14--real-time-monitoring-daemon)
18. [Process 15 — Web Frontend (Flask Application)](#18-process-15--web-frontend-flask-application)
19. [Complete Data Flow Diagram](#19-complete-data-flow-diagram)
20. [Training vs Inference Comparison](#20-training-vs-inference-comparison)
21. [Performance Metrics & Results](#21-performance-metrics--results)
22. [File Reference](#22-file-reference)
23. [How to Run Everything](#23-how-to-run-everything)

---

## 1. Introduction

The **Fake Account Detection System** is a multimodal, graph-based ensemble machine learning pipeline designed to classify social media accounts as **FAKE** or **GENUINE**. It combines:

- **10 diverse ML classifiers** (gradient boosting, kernel methods, neural networks, instance-based)
- **BERT-based text embeddings** for bio/description analysis
- **Graph-based centrality metrics** from follower/following networks
- **Two-stage stacked ensemble** via Out-of-Fold (OOF) meta-learning
- **SHAP explainability** with PCA inverse mapping for human-readable explanations
- **Live Instagram Graph API integration** for real-time scanning
- **Flask web frontend** for browser-based interaction

### Key Performance Metrics

| Metric          | Value  |
| --------------- | ------ |
| Test Accuracy   | 97.06% |
| ROC-AUC         | 0.9969 |
| F1 (fake class) | 0.9693 |

---

## 2. System Architecture Overview

The system is organised into **11 core modules** plus supporting components. Data flows through these modules sequentially during training, and a subset is used during inference (prediction).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FAKE ACCOUNT DETECTION SYSTEM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DATA LAYER                                                        │
│  ┌──────────┐ ┌──────────────────────┐ ┌──────────┐ ┌──────────┐  │
│  │ CSV #1   │ │  Excel (missing)     │ │ Fake     │ │ Real     │  │
│  │ social   │ │  global 2.0          │ │ users    │ │ users    │  │
│  └────┬─────┘ └──────────┬───────────┘ └────┬─────┘ └────┬─────┘  │
│       └──────────────────┼──────────────────┘            │         │
│                          ▼                               ▼         │
│  PROCESSING LAYER        M1: Data Engineering                      │
│                          ▼                                         │
│                          M2: Feature Extraction (97 features)      │
│                          ▼                                         │
│              ┌───────────┴───────────┐                              │
│              ▼                       ▼                              │
│          M3: Graph              M4: 10 Base Models                 │
│              │                       │                              │
│              └───────────┬───────────┘                              │
│                          ▼                                         │
│                     M5: Soft Voting                                 │
│                          ▼                                         │
│                 M6+M7: Meta-Learner                                │
│                          ▼                                         │
│                    M8: SHAP Explain                                 │
│                          ▼                                         │
│                   M11: Output Report                               │
│                                                                     │
│  INTERFACE LAYER                                                   │
│  ┌─────────┐  ┌─────────────┐  ┌───────────────┐  ┌────────────┐  │
│  │   CLI   │  │  Instagram  │  │  Real-Time    │  │  Flask     │  │
│  │ main.py │  │  API Client │  │  Monitor      │  │  Web UI    │  │
│  └─────────┘  └─────────────┘  └───────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Datasets & Data Sources

The system consumes four datasets during training:

| #   | File                                             | Description                                                                    | Columns                                               |
| --- | ------------------------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------- |
| 1   | `fake_social_media.csv`                          | Primary dataset with social media profile metadata                             | username, bio, followers, following, posts, etc.      |
| 2   | `fake_social_media_global_2.0_with_missing.xlsx` | Extended global dataset with intentional missing values for robustness testing | Same as above with missing entries                    |
| 3   | `fake_users.csv`                                 | Twitter-based fake account dataset                                             | username, followers, following, posts, verified, etc. |
| 4   | `real_users.csv`                                 | Twitter-based real account dataset                                             | Same schema as fake_users.csv                         |

**Labels:** Each account is labelled as `1` (fake) or `0` (genuine).

**Cross-platform:** Datasets 1-2 are the primary training source; datasets 3-4 form a secondary "Twitter" set used for cross-dataset evaluation to measure transfer performance across platforms.

---

## 4. Process 1 — Data Engineering (Module 1)

**File:** `module1_data_engineering.py`  
**Purpose:** Transform raw, messy, multi-source data into a clean, balanced, analysis-ready DataFrame.

### Step-by-Step Process

#### Step 1.1 — Data Loading (`load_datasets()`)

- Read all four datasets (CSV and Excel).
- Harmonise column names across datasets (different sources use different naming conventions).
- Merge `fake_users.csv` and `real_users.csv` into a combined "Twitter" DataFrame for cross-validation.
- Assign integer labels: `1 = fake`, `0 = genuine`.

#### Step 1.2 — Noise Removal (`remove_noise()`)

- **Duplicate removal:** Drop exact duplicate rows to prevent data leakage.
- **Constant columns:** Remove columns where every value is identical (zero information gain).
- **Whitespace stripping:** Clean leading/trailing spaces from all string fields.
- **Outlier capping:** Use IQR-based filtering (5× multiplier) to cap extreme numerical values without deleting rows.

#### Step 1.3 — Adversarial Text Normalisation (`adversarial_normalise()`)

Fake accounts commonly use leet-speak and Unicode tricks to evade detection:

- **`normalise_leet()`** — Replace common substitutions:  
  `0→o`, `1→i`, `3→e`, `4→a`, `5→s`, `7→t`, `@→a`, `$→s`
- **Zero-width character removal** — Strip U+200B, U+FEFF, etc.
- **Unicode confusable collapsing** — Normalise visually similar characters (e.g., Cyrillic "а" → Latin "a").

#### Step 1.4 — Username Entropy (`username_entropy()`)

Compute the **Shannon entropy** of each username string. Random/bot-generated usernames (e.g., `xk3j7m9p2`) exhibit high entropy (>3.0 bits), while natural usernames (e.g., `john_photography`) have lower entropy. This becomes a feature in Module 2.

Formula:

$$H(x) = -\sum_{i} p(c_i) \log_2 p(c_i)$$

where $p(c_i)$ is the probability of character $c_i$ in the username.

#### Step 1.5 — Missing Value Imputation (`impute_missing()`)

- **Numeric columns:** `IterativeImputer` (MICE — Multiple Imputation by Chained Equations) using `ExtraTreesRegressor` as the estimator. This creates a multivariate regression chain where each feature is imputed based on the others.
- **Categorical columns:** Mode imputation (most frequent value).
- **Order:** Imputation runs BEFORE class balancing to avoid imputing from synthetic samples.

#### Step 1.6 — Class Balancing (`balance_classes()`)

- **Method:** SMOTEENN = SMOTE (Synthetic Minority Over-sampling) + ENN (Edited Nearest Neighbours)
- **SMOTE:** Generates synthetic minority-class (fake) samples by interpolating between nearest neighbours (k=5).
- **ENN:** Removes noisy majority-class samples that are misclassified by their 3 nearest neighbours.
- **Important:** Only applied to the training set. Test/validation data remains untouched to ensure honest evaluation.
- **Categorical handling:** Categorical columns are separated before SMOTEENN, and re-joined by index alignment afterwards.

#### Step 1.7 — Orchestration (`run_data_engineering()`)

Calls all above steps in the correct sequence and returns the final clean, balanced DataFrame.

### Why This Matters

Data engineering is the highest-impact module. Errors here propagate through every downstream component. Clean data with proper class balance is the foundation for all model accuracy.

---

## 5. Process 2 — Feature Extraction (Module 2)

**File:** `module2_feature_extraction.py`  
**Purpose:** Convert raw profile data into a unified numeric feature vector suitable for machine learning.

### Step-by-Step Process

#### Step 2.1 — Metadata Feature Engineering (`extract_metadata_features()`)

33 hand-crafted features are derived from raw profile columns:

| Category           | Features                                                                                    | Description                                    |
| ------------------ | ------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Profile**        | `has_profile_pic`, `bio_length`, `username_randomness`                                      | Basic presence/absence signals                 |
| **Network**        | `followers`, `following`, `follower_following_ratio`                                        | Social graph metrics                           |
| **Activity**       | `posts`, `posts_per_day`, `account_age_days`                                                | Temporal behaviour                             |
| **Content**        | `caption_similarity_score`, `content_similarity_score`                                      | Repetitive content detection                   |
| **Spam**           | `spam_comments_rate`, `generic_comment_rate`, `follow_unfollow_rate`                        | Bot behaviour indicators                       |
| **Suspicious**     | `suspicious_links_in_bio`, `verified`                                                       | Trust signals                                  |
| **Username**       | `username_length`, `digits_count`, `digit_ratio`, `special_char_count`, `repeat_char_count` | String-level analysis                          |
| **Derived Ratios** | `followers_ratio`, `activity_rate`, `listed_ratio`, `likes_per_post`                        | Normalised engagement metrics                  |
| **Log Transforms** | `log_followers`, `log_posts`                                                                | Reduce skewness for heavy-tailed distributions |
| **Temporal**       | `account_hour`, `account_weekend`                                                           | When was the account created                   |
| **Cross-signals**  | `verified_low_follow`, `bio_has_url`, `follow_diff`                                         | Anomaly flags                                  |

#### Step 2.2 — Text Embedding (BERT or TF-IDF Proxy)

The bio/description field is converted into a dense numeric vector:

- **Full Mode (`BERTEmbedder`):** Uses `sentence-transformers/all-MiniLM-L6-v2` to produce a **384-dimensional** embedding per bio. This captures semantic meaning — e.g., "photographer based in NYC" vs "follow me free money".
- **Fast Mode (`DummyBERTEmbedder`):** When `--no-bert` is specified, falls back to TF-IDF + TruncatedSVD producing a **64-dimensional** proxy embedding. Faster but less semantically rich.

#### Step 2.3 — Graph Embeddings

5 graph centrality features from Module 3 (PageRank, betweenness, clustering coefficient, degree centrality, community size ratio) plus optional 64-d GNN embeddings are appended.

#### Step 2.4 — Unified Feature Vector (`build_unified_features()`)

All features are concatenated into a single matrix:

```
[33 metadata] + [384 BERT or 64 TF-IDF] + [5 graph + 64 GNN] = ~97 raw features
```

This is then processed through:

1. **StandardScaler** — Zero-mean, unit-variance normalisation per feature.
2. **PCA (95% variance threshold)** — Dimensionality reduction from ~97 features to ~77 principal components, removing multicollinear redundancy while retaining 95% of the information.

#### Output Artefacts

| Artefact        | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `X_reduced`     | Shape `(N, ~77)` — the PCA-reduced feature matrix                 |
| `scaler`        | Fitted `StandardScaler` (saved for inference)                     |
| `pca`           | Fitted `PCA` object (saved for inference)                         |
| `feature_names` | Original column names before PCA (needed for SHAP interpretation) |

---

## 6. Process 3 — Graph Construction (Module 3)

**File:** `module3_graph_construction.py`  
**Purpose:** Build a social network graph and extract relationship-based features that capture suspicious community patterns.

### Step-by-Step Process

#### Step 3.1 — Build Interaction Graph (`build_interaction_graph()`)

- Create a **NetworkX graph** where each node is a user account.
- Edges represent follower/following overlaps between users in the dataset.
- This captures the social network structure that pure profile metadata misses.

#### Step 3.2 — Compute Centrality Features

For each node, compute:

| Feature                    | Formula / Description                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| **PageRank**               | Iterative importance score (same algorithm Google uses for web pages)                           |
| **Betweenness Centrality** | Fraction of shortest paths passing through this node — high values indicate broker/hub accounts |
| **Clustering Coefficient** | How interconnected a user's neighbours are — low values suggest artificial networks             |
| **Degree Centrality**      | Fraction of total possible connections — extremely high or low is suspicious                    |
| **Community Size Ratio**   | Via Louvain community detection — fake accounts tend to cluster in small, isolated communities  |

#### Step 3.3 — Optional GNN Embeddings (`FakeAccountGNN`)

If `torch_geometric` is installed:

- A 2-layer GCN (Graph Convolutional Network) + GAT (Graph Attention Network) is trained with cross-entropy on labelled nodes.
- Produces **64-dimensional** node embeddings encoding structural patterns.
- These embeddings capture non-linear graph relationships that handcrafted centrality features miss.

**Fallback:** If PyTorch Geometric is not installed, GNN embeddings are set to zero vectors, and the system relies on the 5 NetworkX centrality features.

---

## 7. Process 4 — Base Model Training (Module 4)

**File:** `module4_base_models.py`  
**Purpose:** Train 10 diverse classifiers, each producing a calibrated probability of an account being fake.

### The Model Zoo

| #   | Model             | Library      | Strengths                                                              |
| --- | ----------------- | ------------ | ---------------------------------------------------------------------- |
| 1   | **XGBoost**       | xgboost      | Gradient boosting with regularisation; handles missing values natively |
| 2   | **LightGBM**      | lightgbm     | Leaf-wise tree growth; fastest gradient booster                        |
| 3   | **CatBoost**      | catboost     | Ordered boosting; robust to overfitting                                |
| 4   | **Random Forest** | scikit-learn | Bagging ensemble; low variance                                         |
| 5   | **Extra Trees**   | scikit-learn | Extremely randomised splits; even lower variance                       |
| 6   | **SVM (RBF)**     | scikit-learn | Kernel-based; excels in high-dimensional spaces                        |
| 7   | **KNN**           | scikit-learn | Instance-based; captures local patterns                                |
| 8   | **Naive Bayes**   | scikit-learn | Probabilistic baseline; fast and simple                                |
| 9   | **MLP**           | scikit-learn | Neural network; captures non-linear interactions                       |
| 10  | **AdaBoost**      | scikit-learn | Adaptive boosting; focuses on hard examples                            |

### Why 10 Models?

**Diversity is the key to ensemble power.** Different algorithms make different kinds of errors. By combining them, the ensemble corrects individual model weaknesses:

- Tree models (XGBoost, LightGBM, CatBoost, RF, ET, AdaBoost) capture feature interactions
- SVM finds optimal hyperplanes in kernel space
- KNN leverages local neighbourhood patterns
- MLP learns non-linear representations
- Naive Bayes provides a well-calibrated probabilistic baseline

### Isotonic Calibration

After training, each model's predicted probabilities are calibrated using **isotonic regression**. Raw probabilities from many models are poorly calibrated (e.g., a model might predict 0.7, but only 50% of those cases are actually positive). Isotonic calibration maps raw scores to true probabilities, which is critical because downstream voting and stacking treat these probabilities as real values.

### Optional: Optuna Hyperparameter Tuning

`optuna_tune_xgboost()` runs **Bayesian hyperparameter search** over the XGBoost parameter space:

- Search space: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Uses 3-fold stratified cross-validation to evaluate each trial
- The winning configuration replaces the default XGBoost in the model zoo

---

## 8. Process 5 — Soft Weighted Voting (Module 5)

**File:** `module5_soft_voting.py`  
**Purpose:** Fuse the 10 calibrated probability outputs into a single ensemble prediction.

### Step-by-Step Process

#### Step 5.1 — Compute Weights (`compute_weights()`)

Each model's weight is proportional to its **F1 score** on the validation fold:

$$w_m = \frac{F1_m}{\sum_{j=1}^{M} F1_j}$$

Models with higher F1 scores have more influence on the final prediction. This is better than equal weighting because it down-weights weak models.

#### Step 5.2 — Soft Vote (`soft_vote()`)

The fused probability is a weighted average:

$$P_{\text{ensemble}} = \sum_{m=1}^{10} w_m \cdot P_m(\text{fake})$$

where $P_m$ is the calibrated probability from model $m$ and $w_m$ is its normalised F1 weight.

#### Step 5.3 — Optimal Threshold (`find_optimal_threshold()`)

Instead of using the default 0.5 threshold, the system grid-searches over thresholds from 0.20 to 0.80 (step 0.01) to find the value that maximises the F1 score on the validation set. This accounts for class imbalance and model calibration characteristics.

---

## 9. Process 6 — Graph Intelligence (Module 6)

**File:** `modules678.py`  
**Purpose:** Add network-level risk signals to the feature set.

### How It Works

`compute_graph_risk()` generates a per-user risk score based on:

- GNN prediction probability (if available)
- Community-level aggregations (what fraction of a user's community is flagged)
- Falls back to a constant default (0.15) for new/unseen users at inference time

This is a separate signal from the base model predictions — it captures "guilt by association" patterns where fake accounts cluster together in the network.

---

## 10. Process 7 — Meta-Learning via OOF Stacking (Module 7)

**File:** `modules678.py`  
**Purpose:** Combine all base model predictions + graph risk into a final prediction via a "meta-learner."

### Why Stacking?

Simple averaging (soft voting) treats model outputs as independent. In reality, models are correlated — they may all be wrong on the same hard examples. A meta-learner can learn **which model to trust in which situations**.

### Step-by-Step Process

#### Step 7.1 — Build Out-of-Fold Meta-Features (`build_meta_features_oof()`)

This is the critical step that prevents data leakage:

1. Split training data into **5 stratified folds**.
2. For each fold:
   - Train each of the 10 base models on the other 4 folds.
   - Predict on the held-out fold.
3. Concatenate held-out predictions → each training sample has a probability from each model, but **never from a model that saw it during training**.

This produces an `(N, 11)` matrix: 10 model probabilities + 1 graph risk score.

#### Step 7.2 — Train Meta-Learner (`train_meta_learner()`)

A **Logistic Regression** (`C=1.0`) is fitted on the OOF meta-features to produce the final prediction:

$$P_{\text{final}} = \sigma\left(\sum_{m=1}^{10} \beta_m P_m + \beta_{11} \cdot \text{graph\_risk} + \beta_0\right)$$

where $\sigma$ is the sigmoid function and $\beta$ values are learned weights.

### Why Logistic Regression?

- Simple enough to not overfit the stacked features (only 11 input columns).
- Its coefficients are interpretable — you can see which models the meta-learner trusts most.
- Well-calibrated probabilistic outputs.

---

## 11. Process 8 — Explainability with SHAP (Module 8)

**File:** `modules678.py`  
**Purpose:** Explain WHY a prediction was made, not just what the prediction is.

### What is SHAP?

**SHAP (SHapley Additive exPlanations)** is based on cooperative game theory. For each prediction, it computes the **marginal contribution** of every feature — how much each feature pushed the prediction toward FAKE or GENUINE.

### How It Works

1. **`SHAPExplainer`** wraps `shap.TreeExplainer` around the best tree-based model (typically XGBoost).
2. For each prediction, SHAP values are computed in **PCA space** (since the model operates on PCA components).
3. The PCA interpretability module (Process 13) then maps these back to the original feature names.

### Output Per Prediction

| Field               | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `shap_values`       | Dict mapping each PCA component to its SHAP value       |
| `top_features`      | Top-5 features by absolute SHAP contribution            |
| `top_real_features` | Top-5 original (pre-PCA) features after inverse mapping |

### Fallback

If the `shap` library is not installed, `DummySHAPExplainer` generates placeholder explanations based on feature magnitudes.

---

## 12. Process 9 — End-to-End Training Pipeline (Module 9)

**File:** `modules910.py`  
**Purpose:** Orchestrate the entire training process in a single function call.

### `train_pipeline()` — 14 Steps

| Step | Module | Action                                                               |
| ---- | ------ | -------------------------------------------------------------------- |
| 1    | M1     | Load all 4 datasets                                                  |
| 2    | M1     | Run data engineering (clean, normalise, impute, balance)             |
| 3    | M3     | Build interaction graph + compute centrality features                |
| 4    | M2     | Initialise BERT embedder (or TF-IDF fallback)                        |
| 5    | M2     | Build unified features: metadata + BERT + graph → scale → PCA        |
| 6    | -      | Train/test split (80/20, stratified by label)                        |
| 7    | M4     | Train 10 base models with isotonic calibration                       |
| 8    | M5     | Compute F1-weighted voting weights                                   |
| 9    | M7     | Generate OOF meta-features (5-fold stacking)                         |
| 10   | M7     | Train Logistic Regression meta-learner                               |
| 11   | M8     | Initialise SHAP explainer                                            |
| 12   | M8     | Create PCA interpreter for feature-name translation                  |
| 13   | -      | Save entire pipeline to `saved_models/pipeline.pkl` via joblib       |
| 14   | -      | Evaluate on test set → print accuracy, ROC-AUC, F1, confusion matrix |

### What Gets Saved

The `pipeline.pkl` file contains everything needed for inference:

```python
{
    "scaler":          fitted StandardScaler,
    "pca":             fitted PCA,
    "bert_embedder":   BERTEmbedder or DummyBERTEmbedder,
    "base_models":     dict of 10 calibrated classifiers,
    "soft_voter":      SoftWeightedVoter with learned weights,
    "meta_learner":    fitted LogisticRegression,
    "shap_explainer":  SHAPExplainer or DummySHAPExplainer,
    "feature_names":   list of original feature names,
    "pca_feature_names": list of PCA component names,
    "optimal_threshold": float (e.g., 0.42),
    "fixed_meta_cols":  list of metadata column names,
    "metrics":          dict of evaluation results,
}
```

---

## 13. Process 10 — Real-Time Prediction Pipeline (Module 10)

**File:** `modules910.py`  
**Purpose:** Run inference on new, unseen profiles using the saved pipeline.

### `predict()` — Step by Step

```
Input DataFrame (raw profile data)
    │
    ▼
Step 1: extract_metadata_features()  →  33 derived features
    │
    ▼
Step 2: build_unified_features()     →  saved scaler + PCA transform
    │                                    (NO fitting — uses saved objects)
    ▼
Step 3: get_ensemble_probabilities() →  each of 10 models predicts P(fake)
    │
    ▼
Step 4: soft_vote()                  →  F1-weighted average probability
    │
    ▼
Step 5: graph_risk                   →  default 0.15 for unseen profiles
    │
    ▼
Step 6: meta_learner.predict_proba() →  final calibrated P(fake)
    │
    ▼
Step 7: threshold comparison         →  FAKE if P ≥ threshold, else GENUINE
    │
    ▼
Step 8: SHAP explanations (optional) →  per-feature contribution scores
    │
    ▼
Step 9: format_output()              →  structured results DataFrame
```

### Key Difference from Training

During inference:

- **No SMOTEENN** (that's training-only)
- **No fitting** — only `transform()` is called on saved scaler/PCA
- **No cross-validation** — direct single-pass prediction
- **Graph risk** defaults to 0.15 for new users not in the training graph

---

## 14. Process 11 — Output & Reporting (Module 11)

**File:** `module11_output.py`  
**Purpose:** Convert raw numerical predictions into structured, human-readable reports.

### Output Fields

| Field          | Type  | Description                              |
| -------------- | ----- | ---------------------------------------- |
| `label`        | str   | "FAKE" or "GENUINE"                      |
| `probability`  | float | P(fake) ∈ [0, 1], calibrated             |
| `graph_risk`   | float | Network-level risk score ∈ [0, 1]        |
| `risk_band`    | str   | LOW / MEDIUM / HIGH / CRITICAL           |
| `confidence`   | str   | max(P, 1−P) as percentage, e.g., "87.3%" |
| `top_features` | list  | Top-5 SHAP contributions                 |
| `explanation`  | str   | Natural-language risk justification      |

### Risk Band Classification

| Band         | Probability Range | Recommended Action                   |
| ------------ | ----------------- | ------------------------------------ |
| **LOW**      | < 0.30            | No action — very likely genuine      |
| **MEDIUM**   | 0.30 – 0.60       | Flag for review — ambiguous signals  |
| **HIGH**     | 0.60 – 0.85       | Auto-flag — strong fake signals      |
| **CRITICAL** | ≥ 0.85            | Immediate action — near-certain fake |

### Explanation Generation (`generate_explanation()`)

The system produces natural-language explanations like:

> _"Prediction: FAKE (probability: 68.52%, risk: HIGH). Key factors: followers_ratio decreases fake likelihood (impact: -0.0312); username_entropy increases fake likelihood (impact: +0.0287); bio_length decreases fake likelihood (impact: -0.0195)."_

When PCA interpretability is active, explanations use real feature names instead of PCA component labels.

---

## 15. Process 12 — Instagram API Integration

**File:** `instagram_api.py`  
**Purpose:** Fetch live Instagram profile data and feed it through the detection pipeline.

### Architecture

```
┌──────────────┐       ┌──────────────────────┐       ┌──────────────┐
│  User Input  │──────▶│  InstagramAPIClient   │──────▶│  Graph API   │
│  (username   │       │                      │       │  v19.0       │
│   or token)  │       │  - Auto-detect IGAA  │       │              │
│              │       │    vs EAA tokens     │       │              │
└──────────────┘       │  - Rate limiting     │       └──────────────┘
                       │  - Error handling    │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ profile_to_dataframe()│
                       │ Convert API response  │
                       │ to pipeline format    │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  predict()            │
                       │  Run ML pipeline     │
                       └──────────────────────┘
```

### Token Types

The system supports two types of Instagram API tokens:

| Token Prefix | Type                 | API Base URL                | Capabilities                                            |
| ------------ | -------------------- | --------------------------- | ------------------------------------------------------- |
| `IGAA…`      | Instagram User Token | `graph.instagram.com/v19.0` | Own profile + limited Business Discovery                |
| `EAA…`       | Facebook User Token  | `graph.facebook.com/v19.0`  | Full Business Discovery (requires linked Facebook Page) |

**Auto-detection:** The client inspects the token prefix and automatically selects the correct API endpoint.

### Key Functions

| Function                     | What It Does                                                                |
| ---------------------------- | --------------------------------------------------------------------------- |
| `get_own_profile()`          | Fetch the authenticated user's own Instagram profile                        |
| `get_user_profile(username)` | Fetch another user's profile via Business Discovery                         |
| `get_user_media(user_id)`    | Fetch recent posts (captions, likes, comments, timestamps)                  |
| `profile_to_dataframe()`     | Convert API response into a 1-row DataFrame matching the ML pipeline schema |
| `create_demo_profiles()`     | Generate 5 realistic synthetic profiles for testing without an API token    |
| `fetch_and_analyse()`        | End-to-end: fetch profiles → transform → predict → return results           |
| `demo_analyse()`             | Same flow but with synthetic profiles                                       |

### Feature Engineering from Live API Data

The API client computes derived features in real-time during `profile_to_dataframe()`:

- **Username analysis:** length, digit count, digit ratio, special characters, repeats, Shannon entropy, randomness flag
- **Bio analysis:** length, URL detection, spam keyword scoring
- **Network ratios:** follower/following ratio
- **Activity metrics:** posts per day, account age estimation from earliest post timestamp
- **Content analysis:** caption similarity (cosine-like), spam/generic comment rate from recent media
- **Engagement:** total likes and comments across fetched posts

---

## 16. Process 13 — PCA Interpretability

**File:** `pca_interpretability.py`  
**Purpose:** Translate SHAP explanations from PCA space back to original feature space.

### The Problem

The ML models operate on PCA components (`PC_0`, `PC_1`, …, `PC_76`). SHAP explanations naturally come in PCA space. But telling a user that "PC_3 increased fake likelihood by +0.08" is meaningless.

### The Solution

Use the PCA loading matrix to map SHAP values back to original features:

$$\text{SHAP}_{\text{original}} = \text{SHAP}_{\text{PCA}} \cdot V$$

where $V \in \mathbb{R}^{K \times D}$ is the PCA components matrix (K components × D original features).

### Process

1. Extract the PCA components matrix from the saved pipeline.
2. When SHAP values are computed in PCA space, multiply by the components matrix.
3. Sum contributions per original feature across all PCA components.
4. Rank by absolute magnitude to find the most important features.
5. Map feature names to human-readable descriptions using the curated `FEATURE_DESCRIPTIONS` dictionary.

### Example Transformation

| Before (PCA space) | After (Original space)   |
| ------------------ | ------------------------ |
| PC_3: +0.082       | followers_ratio: +0.045  |
| PC_7: -0.063       | username_entropy: +0.038 |
| PC_12: +0.041      | bio_length: -0.022       |

### Feature Description Dictionary

~40 feature names are mapped to plain-English descriptions. Examples:

- `followers_ratio` → "Ratio of followers to following count"
- `username_entropy` → "Shannon entropy of the username (randomness measure)"
- `bio_has_url` → "Whether the biography contains a URL"
- `bert_12` → "Text embedding dimension 12" (auto-generated for BERT dimensions)

---

## 17. Process 14 — Real-Time Monitoring Daemon

**File:** `realtime_monitor.py`  
**Purpose:** Continuously monitor Instagram accounts on a schedule and emit alerts when suspicious accounts are detected.

### Architecture

```
┌─────────────────┐
│ RealtimeMonitor  │
│                 │     ┌──────────────┐
│  scan()     ────┼────▶│ Instagram    │
│  scan_demo()    │     │ API Client   │
│                 │     └──────────────┘
│  start_continuous()
│     └── background thread
│         └── scan every N minutes
│
│  stop()
│  get_history()  │
│  generate_report()│
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  AlertHandler    │
│                 │
│  - Create alerts │
│  - Log to console│
│  - Save to JSON  │
└──────────────────┘
```

### Alert Levels

| Level        | P(fake) Threshold | Action                                                      |
| ------------ | ----------------- | ----------------------------------------------------------- |
| **CRITICAL** | ≥ 0.85            | `logger.warning` + JSON alert file + immediate notification |
| **HIGH**     | ≥ 0.60            | `logger.warning` + JSON alert file                          |
| **MEDIUM**   | ≥ 0.30            | `logger.info` + JSON alert file                             |
| **LOW**      | < 0.30            | No alert (account is likely genuine)                        |

### Operational Modes

1. **One-shot scan:** `scan(usernames)` — fetch and analyse once, return results.
2. **Demo scan:** `scan_demo()` — same but with synthetic profiles.
3. **Continuous monitoring:** `start_continuous(usernames, interval_minutes=60)` — launches a background thread that scans every N minutes.
4. **Report generation:** `generate_report()` — creates a summary of all scans and alerts.

---

## 18. Process 15 — Web Frontend (Flask Application)

**File:** `app.py`, `templates/index.html`, `static/style.css`, `static/app.js`  
**Purpose:** Provide a browser-based interface for the detection system.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    BROWSER (Frontend)                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  index.html + style.css + app.js                    │ │
│  │                                                     │ │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │ │
│  │  │  Demo   │ │ Self     │ │ Username │ │ Manual │  │ │
│  │  │  Scan   │ │ Scan     │ │ Scan     │ │ Input  │  │ │
│  │  └────┬────┘ └────┬─────┘ └────┬─────┘ └───┬────┘  │ │
│  └───────┼───────────┼────────────┼────────────┼───────┘ │
│          │  AJAX     │            │            │         │
└──────────┼───────────┼────────────┼────────────┼─────────┘
           ▼           ▼            ▼            ▼
┌──────────────────────────────────────────────────────────┐
│                   FLASK SERVER (Backend)                   │
│                                                           │
│  POST /api/demo         →  demo_analyse()                │
│  POST /api/scan-self    →  InstagramAPIClient + predict() │
│  POST /api/scan-users   →  fetch_and_analyse()            │
│  POST /api/manual       →  profile_to_dataframe + predict()│
│  GET  /api/status       →  pipeline readiness check       │
│  GET  /api/history      →  session scan log               │
│  POST /api/history/clear→  clear session history          │
│                                                           │
│  Pipeline loaded ONCE at startup → reused for all requests│
└──────────────────────────────────────────────────────────┘
```

### Frontend Features

| Tab               | Description                                                    | Requires Token?           |
| ----------------- | -------------------------------------------------------------- | ------------------------- |
| **Demo Scan**     | Run detection on 5 simulated Instagram profiles                | No                        |
| **Self Scan**     | Scan your own authenticated Instagram account                  | Yes (IGAA or EAA)         |
| **Username Scan** | Scan other accounts by username via Business Discovery         | Yes (EAA + Facebook Page) |
| **Manual Input**  | Enter profile details by hand (username, bio, followers, etc.) | No                        |
| **History**       | View all previous scan results from the current session        | No                        |

### API Endpoints

| Endpoint             | Method | Request Body                                                                         | Response                                                |
| -------------------- | ------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `/api/status`        | GET    | —                                                                                    | `{ok, pipeline_ready, history_count}`                   |
| `/api/demo`          | POST   | —                                                                                    | `{ok, results: [{label, probability, risk_band, ...}]}` |
| `/api/scan-self`     | POST   | `{token}`                                                                            | `{ok, results: [...]}`                                  |
| `/api/scan-users`    | POST   | `{token, usernames}`                                                                 | `{ok, results: [...]}`                                  |
| `/api/manual`        | POST   | `{username, name, biography, followers, following, posts, website, has_profile_pic}` | `{ok, results: [...]}`                                  |
| `/api/history`       | GET    | —                                                                                    | `{ok, history: [{timestamp, mode, count, results}]}`    |
| `/api/history/clear` | POST   | —                                                                                    | `{ok}`                                                  |

### Result Card Display

Each scanned profile is displayed as a card showing:

- **Username** with `@` prefix
- **Label badge** (GENUINE in green, FAKE in red)
- **Probability** with colour-coded progress bar
- **Risk band** (LOW/MEDIUM/HIGH/CRITICAL) with colour coding
- **Confidence** percentage
- **Graph risk** score
- **Explanation** in natural language
- **Top SHAP features** as coloured chips (red for fake-pushing, green for genuine-pushing)

### Tech Stack

| Component | Technology                                |
| --------- | ----------------------------------------- |
| Backend   | Flask (Python)                            |
| Frontend  | Vanilla HTML5 + CSS3 + JavaScript         |
| Styling   | Custom CSS with CSS variables, dark theme |
| Fonts     | Inter (Google Fonts)                      |
| Layout    | CSS Grid + Flexbox, responsive sidebar    |

---

## 19. Complete Data Flow Diagram

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                  │
│                                                                             │
│  fake_social_media.csv ─┐                                                  │
│  global_2.0_missing.xlsx┤                                                  │
│  fake_users.csv ────────┤─── M1: Data Engineering                          │
│  real_users.csv ────────┘    │ dedup → leet-normalise → entropy            │
│                              │ → impute (MICE) → SMOTEENN balance          │
│                              ▼                                             │
│                         M2: Feature Extraction                             │
│                              │ 33 metadata features                        │
│                              │ + BERT embeddings (384-d)                   │
│                              │ + Graph features (5 centrality)             │
│                              │ → StandardScaler → PCA (95% var)           │
│                              │ → ~77 PCA components                        │
│                              ▼                                             │
│                    ┌─── 80/20 Stratified Split ───┐                        │
│                    │                               │                        │
│                    ▼                               ▼                        │
│              Training Set                     Test Set                      │
│                    │                               │                        │
│                    ▼                               │                        │
│              M4: Train 10 Models                   │                        │
│              (isotonic calibrated)                  │                        │
│                    │                               │                        │
│                    ▼                               │                        │
│              M5: Compute F1 Weights                │                        │
│                    │                               │                        │
│                    ▼                               │                        │
│              M7: 5-Fold OOF Stacking               │                        │
│                    │                               │                        │
│                    ▼                               │                        │
│              M7: Train Meta-Learner                │                        │
│              (LogisticRegression)                   │                        │
│                    │                               │                        │
│                    ▼                               │                        │
│              M8: Init SHAP Explainer               │                        │
│                    │                               │                        │
│                    ▼                               ▼                        │
│              Save pipeline.pkl          Evaluate on Test Set               │
│                                         → 97.06% acc, 0.9969 AUC              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                  │
│                                                                             │
│  INPUT SOURCES:                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Instagram   │  │ Manual      │  │ Demo        │  │ CSV         │       │
│  │ API fetch   │  │ Web form    │  │ Synthetic   │  │ File input  │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         └────────────────┼────────────────┘                │              │
│                          ▼                                 ▼              │
│                    Raw DataFrame                     Raw DataFrame        │
│                          │                                                │
│                          ▼                                                │
│                 M2: extract_metadata_features() (NO M1 balancing)         │
│                          │                                                │
│                          ▼                                                │
│                 M2: build_unified_features()                               │
│                     saved_scaler.transform()                              │
│                     saved_pca.transform()                                 │
│                          │                                                │
│                          ▼                                                │
│                 M4: Each saved model → P(fake)                            │
│                          │                                                │
│                          ▼                                                │
│                 M5: Soft vote (saved weights)                             │
│                          │                                                │
│                          ▼                                                │
│                 M7: Meta-learner → final P(fake)                          │
│                          │                                                │
│                          ▼                                                │
│                 M8: SHAP explanations (optional)                          │
│                     PCA inverse → original features                       │
│                          │                                                │
│                          ▼                                                │
│                 M11: Output formatting                                    │
│                     label + probability + risk_band                       │
│                     + confidence + explanation                            │
│                          │                                                │
│                          ▼                                                │
│                 DELIVERED VIA:                                             │
│                 ┌─────┐  ┌──────┐  ┌──────────┐                           │
│                 │ CLI │  │ JSON │  │ Web Card │                           │
│                 └─────┘  └──────┘  └──────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 20. Training vs Inference Comparison

| Aspect             | Training                    | Inference                      |
| ------------------ | --------------------------- | ------------------------------ |
| **Input**          | 4 CSV/Excel datasets        | Single profile or batch        |
| **SMOTEENN**       | Yes (balance classes)       | No                             |
| **Imputation**     | Fit + transform             | Uses saved imputation patterns |
| **Scaler**         | Fit + transform             | `scaler.transform()` only      |
| **PCA**            | Fit + transform             | `pca.transform()` only         |
| **Base Models**    | Train + calibrate           | `predict_proba()` only         |
| **Voting Weights** | Computed from F1 scores     | Saved weights applied          |
| **OOF Stacking**   | 5-fold cross-val            | Direct prediction              |
| **Meta-Learner**   | Train LogisticRegression    | `predict_proba()` only         |
| **SHAP**           | Initialise explainer        | Explain individual predictions |
| **Graph Risk**     | Computed from real graph    | Default constant (0.15)        |
| **Output**         | Metrics report + saved .pkl | Per-profile prediction card    |

---

## 21. Performance Metrics & Results

### Primary Test Set

| Metric           | Value  |
| ---------------- | ------ |
| Accuracy         | 97.06% |
| ROC-AUC          | 0.9969 |
| F1 (fake)        | 0.9693 |
| F1 (genuine)     | 0.9718 |
| Precision (fake) | 0.97   |
| Recall (fake)    | 0.97   |

### What the Numbers Mean

- **97.06% accuracy** means roughly 97 out of 100 accounts are classified correctly.
- **0.9969 ROC-AUC** means the model's ranking ability is near-perfect — it can reliably distinguish between fake and genuine accounts across all threshold settings.
- **0.9693 F1 for fake accounts** shows strong balance between precision and recall for the harder class (fakes are more varied than genuine accounts).

### Model Contribution

The meta-learner learns how much to weight each base model. Gradient boosting models (XGBoost, LightGBM, CatBoost) typically receive the highest meta-learner coefficients, while simpler models (Naive Bayes, KNN) serve as diversity regularisers.

---

## 22. File Reference

| File                            | Lines  | Purpose                               |
| ------------------------------- | ------ | ------------------------------------- |
| `main.py`                       | ~334   | CLI entry point with argparse         |
| `module1_data_engineering.py`   | ~411   | Data cleaning, imputation, SMOTEENN   |
| `module2_feature_extraction.py` | ~363   | Metadata + BERT + PCA                 |
| `module3_graph_construction.py` | varies | NetworkX graph + optional GNN         |
| `module4_base_models.py`        | ~236   | 10 base classifiers + Optuna tuning   |
| `module5_soft_voting.py`        | varies | F1-weighted soft voting               |
| `modules678.py`                 | ~416   | Graph risk, OOF stacking, SHAP        |
| `modules910.py`                 | ~542   | Training and prediction orchestration |
| `module11_output.py`            | ~250   | Output formatting and risk bands      |
| `instagram_api.py`              | ~719   | Instagram Graph API v19.0 client      |
| `pca_interpretability.py`       | varies | PCA → original feature mapping        |
| `realtime_monitor.py`           | varies | Continuous monitoring daemon          |
| `app.py`                        | ~250   | Flask web application                 |
| `templates/index.html`          | ~200   | Frontend HTML                         |
| `static/style.css`              | ~350   | Frontend styling                      |
| `static/app.js`                 | ~250   | Frontend JavaScript logic             |

---

## 23. How to Run Everything

### Initial Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python main.py                   # Full training with BERT embeddings
python main.py --no-bert         # Fast training with TF-IDF proxy
python main.py --optuna          # With Bayesian hyperparameter tuning
```

### Run Predictions via CLI

```bash
python main.py --predict         # Predict on test split
python main.py --demo            # Demo with synthetic profiles
python main.py --scan-self --token YOUR_IGAA_TOKEN   # Scan own account
python main.py --instagram user1,user2 --token TOKEN  # Scan other accounts
```

### Launch the Web Frontend

```bash
python app.py                    # Opens at http://127.0.0.1:5000
python app.py --port 8080        # Custom port
```

Then open your browser and navigate to `http://127.0.0.1:5000`.

### Continuous Monitoring

```bash
python main.py --realtime user1,user2 --token TOKEN --interval 30
# Scans every 30 minutes in a background thread
```

### PCA Report

```bash
python main.py --explain-pca     # Print which original features each PCA component represents
```

---

_Generated for the Fake Account Detection System v3.0 — February 2026_
