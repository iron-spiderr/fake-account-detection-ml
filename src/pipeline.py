"""
Modules 9 & 10 — End-to-End Training Pipeline + Real-Time Prediction Pipeline
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"


# ===========================================================================
# MODULE 9 — Training Pipeline
# ===========================================================================

def train_pipeline(
        primary_path: str = "data/fake_social_media.csv",
        fake_users_path: str = "data/fake_users.csv",
        limfadd_path: str = "data/LIMFADD.csv",
        excel_path: str = "data/fake_social_media_global_2.0_with_missing.xlsx",
        use_bert: bool = False,
        use_gnn: bool = False,
        use_optuna: bool = False,
        optuna_trials: int = 30,
        balance: bool = True,
        test_size: float = 0.2,
        save_path: str | None = None) -> dict:
    """
    Full 14-step training pipeline.
    Returns the pipeline dict (also saved to disk at `save_path`).
    """
    from .data_engineering import run_data_engineering
    from .feature_extraction import build_unified_features, get_embedder
    from .graph_construction import build_graph_features
    from .base_models import (train_base_models, get_ensemble_probabilities,
                               optuna_tune_xgboost, _build_model_zoo)
    from .soft_voting import SoftWeightedVoter
    from .stacking_shap import (OOFStackedEnsemble, compute_graph_risk,
                                 get_shap_explainer)
    from .pca_interpretability import PCAInterpreter

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║     FAKE ACCOUNT DETECTION — TRAINING    ║")
    logger.info("╚══════════════════════════════════════════╝")

    # Step 1-2: Data Engineering
    logger.info("Step 1-2: Data engineering …")
    df = run_data_engineering(primary_path, fake_users_path, limfadd_path,
                               excel_path=excel_path, balance=balance)

    label_col = "label"
    y_all = df[label_col].values.astype(int)

    # Step 3: Graph features
    logger.info("Step 3: Graph construction …")
    labels_series = pd.Series(y_all, index=df.index)
    graph_feat = build_graph_features(df, labels=labels_series, use_gnn=use_gnn)

    # Step 4-5: Feature extraction
    logger.info("Step 4-5: Feature extraction …")
    embedder = get_embedder(use_bert)
    X_all, scaler, pca, feature_names = build_unified_features(
        df, embedder=embedder, graph_features=graph_feat, fit=True)

    # fixed_meta_cols for inference alignment
    from .feature_extraction import extract_metadata_features
    fixed_meta_cols = extract_metadata_features(df.head(1)).columns.tolist()

    # Step 6: Train/test split
    logger.info("Step 6: Splitting 80/20 …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=42)

    # Graph risk for training set
    graph_risk_all = compute_graph_risk(df, graph_feat)
    mask_tr = np.zeros(len(y_all), dtype=bool)
    tr_idx, te_idx = train_test_split(np.arange(len(y_all)),
                                       test_size=test_size,
                                       stratify=y_all, random_state=42)
    graph_risk_tr = graph_risk_all[tr_idx]
    graph_risk_te = graph_risk_all[te_idx]

    # Step 7: Base models
    logger.info("Step 7: Training base models …")
    if use_optuna:
        logger.info("Running Optuna tuning …")
        best_params = optuna_tune_xgboost(X_tr, y_tr, n_trials=optuna_trials)
        from xgboost import XGBClassifier
        zoo = _build_model_zoo()
        zoo["xgboost"] = XGBClassifier(**best_params)
    else:
        zoo = None  # train_base_models will call _build_model_zoo internally

    base_models = train_base_models(X_tr, y_tr, X_val=X_te, y_val=y_te,
                                     model_zoo=zoo)

    # Step 8: Voting weights
    logger.info("Step 8: Computing voting weights …")
    voter = SoftWeightedVoter()
    voter.compute_weights(base_models, X_te, y_te)

    prob_te = get_ensemble_probabilities(base_models, X_te)
    vote_proba_te = voter.soft_vote(prob_te)
    voter.find_optimal_threshold(vote_proba_te, y_te)

    # Step 9-10: OOF stacking
    logger.info("Step 9-10: OOF stacking + meta-learner …")
    stacker = OOFStackedEnsemble(n_folds=5)
    oof_meta = stacker.build_meta_features_oof(base_models, X_tr, y_tr,
                                                graph_risk=graph_risk_tr)
    stacker.train_meta_learner(oof_meta, y_tr)

    # Step 11-12: SHAP + PCA interpreter
    logger.info("Step 11-12: SHAP + PCA interpreter …")
    pca_names = [f"PC_{i}" for i in range(X_all.shape[1])]
    shap_explainer = get_shap_explainer(base_models, pca_names)
    pca_interpreter = PCAInterpreter(pca, feature_names, pca_names)

    # Step 13: Save pipeline
    logger.info("Step 13: Saving pipeline …")
    MODEL_DIR.mkdir(exist_ok=True)
    save_path = save_path or str(PIPELINE_PATH)
    pipeline = {
        "scaler": scaler,
        "pca": pca,
        "bert_embedder": embedder,
        "base_models": base_models,
        "soft_voter": voter,
        "meta_learner": stacker,
        "shap_explainer": shap_explainer,
        "pca_interpreter": pca_interpreter,
        "feature_names": feature_names,
        "pca_feature_names": pca_names,
        "optimal_threshold": voter.optimal_threshold,
        "fixed_meta_cols": fixed_meta_cols,
    }
    joblib.dump(pipeline, save_path)
    logger.info("Pipeline saved to: %s", save_path)

    # Step 14: Evaluate
    logger.info("Step 14: Evaluating on test set …")
    metrics = _evaluate(pipeline, X_te, y_te, graph_risk_te)
    pipeline["metrics"] = metrics
    joblib.dump(pipeline, save_path)  # resave with metrics
    return pipeline


def _evaluate(pipeline: dict, X_te: np.ndarray, y_te: np.ndarray,
              graph_risk_te: np.ndarray | None = None) -> dict:
    from .base_models import get_ensemble_probabilities
    from .stacking_shap import DEFAULT_GRAPH_RISK

    base_models = pipeline["base_models"]
    voter = pipeline["soft_voter"]
    stacker = pipeline["meta_learner"]
    threshold = pipeline["optimal_threshold"]

    prob_te = get_ensemble_probabilities(base_models, X_te)
    vote_proba = voter.soft_vote(prob_te)

    gr = graph_risk_te if graph_risk_te is not None else np.full(len(y_te), DEFAULT_GRAPH_RISK)
    meta_te = np.column_stack([prob_te, gr])
    final_proba = stacker.predict_proba(meta_te)
    y_pred = (final_proba >= threshold).astype(int)

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, final_proba)
    f1_fake = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
    f1_real = f1_score(y_te, y_pred, pos_label=0, zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "f1_fake": round(f1_fake, 4),
        "f1_genuine": round(f1_real, 4),
    }

    logger.info("─" * 50)
    logger.info("Test Accuracy:  %.2f%%", acc * 100)
    logger.info("ROC-AUC:        %.4f", auc)
    logger.info("F1 (fake):      %.4f", f1_fake)
    logger.info("F1 (genuine):   %.4f", f1_real)
    logger.info("─" * 50)
    print("\nClassification Report:\n",
          classification_report(y_te, y_pred, target_names=["GENUINE", "FAKE"],
                                zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    return metrics


# ===========================================================================
# MODULE 10 — Real-Time Prediction Pipeline
# ===========================================================================

def load_pipeline(path: str | None = None) -> dict:
    path = path or str(PIPELINE_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline not found at {path}. Run training first.")
    pipeline = joblib.load(path)
    _fix_sklearn_compat(pipeline)
    return pipeline


def _fix_sklearn_compat(pipeline: dict):
    """Patch sklearn objects for cross-version compatibility.

    Models saved with sklearn ≥1.8 may lack attributes that older
    versions expect (e.g. LogisticRegression.multi_class was removed
    in 1.8 but is required by 1.5).  Add them back if missing.
    """
    from sklearn.linear_model import LogisticRegression

    stacker = pipeline.get("meta_learner")
    if stacker is not None:
        lr = getattr(stacker, "meta_learner", None)
        if isinstance(lr, LogisticRegression) and not hasattr(lr, "multi_class"):
            lr.multi_class = "auto"

    # Also patch any LogisticRegression inside calibrated base models
    for name, model in pipeline.get("base_models", {}).items():
        for attr in ("estimator", "estimators_", "calibrated_classifiers_"):
            obj = getattr(model, attr, None)
            if obj is None:
                continue
            items = obj if isinstance(obj, (list, tuple)) else [obj]
            for item in items:
                inner = getattr(item, "estimator", item)
                if isinstance(inner, LogisticRegression) and not hasattr(inner, "multi_class"):
                    inner.multi_class = "auto"


def predict(df: pd.DataFrame,
            pipeline: dict | None = None,
            pipeline_path: str | None = None,
            explain: bool = True) -> pd.DataFrame:
    """
    Run inference on a raw profile DataFrame.
    Returns a results DataFrame with label, probability, risk_band, etc.
    """
    from .feature_extraction import build_unified_features
    from .base_models import get_ensemble_probabilities
    from .stacking_shap import DEFAULT_GRAPH_RISK
    from .output import format_output

    if pipeline is None:
        pipeline = load_pipeline(pipeline_path)

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    embedder = pipeline["bert_embedder"]
    base_models = pipeline["base_models"]
    voter = pipeline["soft_voter"]
    stacker = pipeline["meta_learner"]
    threshold = pipeline["optimal_threshold"]
    shap_explainer = pipeline.get("shap_explainer")
    pca_interpreter = pipeline.get("pca_interpreter")

    # Step 1-2: Feature extraction (no fit)
    X, _, _, _ = build_unified_features(
        df, embedder=embedder, graph_features=None,
        scaler=scaler, pca=pca, fit=False)

    # Steps 3-4: Ensemble probabilities + soft vote
    prob_matrix = get_ensemble_probabilities(base_models, X)
    vote_proba = voter.soft_vote(prob_matrix)

    # Step 5-6: Add graph risk (default for unseen)
    graph_risk = np.full(len(df), DEFAULT_GRAPH_RISK)
    meta = np.column_stack([prob_matrix, graph_risk])

    # Step 7: Meta-learner final probability
    final_proba = stacker.predict_proba(meta)
    y_pred = (final_proba >= threshold).astype(int)

    # Step 8: SHAP (optional)
    shap_results = None
    if explain and shap_explainer is not None:
        try:
            shap_results = shap_explainer.explain(X)
            if pca_interpreter is not None:
                for res in shap_results:
                    top_orig = pca_interpreter.map_shap(res["shap_values"])
                    res["top_real_features"] = top_orig[:5]
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)

    # Step 9: Format output
    return format_output(df, y_pred, final_proba, graph_risk, shap_results)
