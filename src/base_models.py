"""
Module 4 — Base Model Training
Trains 10 diverse classifiers with isotonic calibration.
Optionally tunes XGBoost with Optuna.
"""

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                               RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def _build_model_zoo() -> dict:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    models = {
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0),

        "lightgbm": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),

        "catboost": CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0),

        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1),

        "svm": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),

        "knn": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),

        "naive_bayes": GaussianNB(),

        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=300, random_state=42, early_stopping=True,
            validation_fraction=0.1),

        "adaboost": AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42),
    }
    return models


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_base_models(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray | None = None,
                      y_val: np.ndarray | None = None,
                      model_zoo: dict | None = None) -> dict:
    """
    Train each model and wrap with isotonic calibration.
    Returns a dict {name: calibrated_estimator}.
    """
    if model_zoo is None:
        model_zoo = _build_model_zoo()

    calibrated_models = {}
    for name, model in model_zoo.items():
        logger.info("Training %s …", name)
        try:
            cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
            cal.fit(X_train, y_train)
            calibrated_models[name] = cal
            if X_val is not None and y_val is not None:
                y_pred = cal.predict(X_val)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                logger.info("  %s val F1: %.4f", name, f1)
        except Exception as exc:
            logger.warning("  %s failed: %s — skipping.", name, exc)

    logger.info("Trained %d base models.", len(calibrated_models))
    return calibrated_models


def get_ensemble_probabilities(models: dict, X: np.ndarray) -> np.ndarray:
    """Return an (N, M) matrix of P(fake) from each of M calibrated models.

    For single-row inference (real-time), uses ThreadPoolExecutor for
    parallel model prediction to reduce latency.
    """
    if X.shape[0] <= 5:
        # Real-time path: parallelize across models for low-latency inference
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _predict_one(name_model):
            name, model = name_model
            try:
                return model.predict_proba(X)[:, 1]
            except Exception:
                return np.full(X.shape[0], 0.5)

        model_items = list(models.items())
        probs = [None] * len(model_items)
        with ThreadPoolExecutor(max_workers=min(len(model_items), 4)) as pool:
            futures = {pool.submit(_predict_one, item): i
                       for i, item in enumerate(model_items)}
            for fut in as_completed(futures):
                probs[futures[fut]] = fut.result()
        return np.column_stack(probs)

    # Batch path: sequential is fine for larger datasets
    probs = []
    for name, model in models.items():
        try:
            p = model.predict_proba(X)[:, 1]
        except Exception:
            p = np.full(X.shape[0], 0.5)
        probs.append(p)
    return np.column_stack(probs)


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning (XGBoost)
# ---------------------------------------------------------------------------

def optuna_tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> dict:
    """Bayesian hyperparameter search for XGBoost. Returns best params."""
    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "eval_metric": "logloss", "random_state": 42,
            "n_jobs": -1, "verbosity": 0,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("Best XGBoost params: %s (F1=%.4f)",
                study.best_params, study.best_value)
    return study.best_params
