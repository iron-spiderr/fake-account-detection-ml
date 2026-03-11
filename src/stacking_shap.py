"""
Modules 6, 7, 8 — Graph Intelligence | OOF Meta-Learning | SHAP Explainability
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ===========================================================================
# MODULE 6 — Graph Risk Score
# ===========================================================================

DEFAULT_GRAPH_RISK = 0.15


def compute_graph_risk(df: pd.DataFrame,
                        graph_features: pd.DataFrame | None = None) -> np.ndarray:
    """
    Produce a per-row graph risk score ∈ [0, 1].
    If graph features are available, combines degree centrality and community
    size ratio into a heuristic risk; otherwise returns DEFAULT_GRAPH_RISK.
    """
    n = len(df)
    if graph_features is not None and len(graph_features) == n:
        # High degree + small community → higher risk
        deg = graph_features.get("degree_centrality", pd.Series(np.zeros(n))).values
        comm = graph_features.get("community_size_ratio",
                                   pd.Series(np.ones(n) * 0.5)).values
        risk = 0.5 * deg + 0.5 * (1 - comm)
        return np.clip(risk, 0, 1).astype(float)
    return np.full(n, DEFAULT_GRAPH_RISK, dtype=float)


# ===========================================================================
# MODULE 7 — OOF Stacking + Meta-Learner
# ===========================================================================

class OOFStackedEnsemble:
    """
    Out-of-Fold stacking:
      1. Generate OOF predictions from K-fold cross-validation.
      2. Fit a Logistic Regression meta-learner on the OOF predictions.
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._fitted = False

    # -----------------------------------------------------------------------
    # 7.1  Build OOF Meta-Features
    # -----------------------------------------------------------------------

    def build_meta_features_oof(self, models: dict,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  graph_risk: np.ndarray | None = None
                                  ) -> np.ndarray:
        """
        Returns an (N, M+1) matrix of OOF predictions.
        M = number of base models + 1 (graph risk).
        Data leakage is prevented — each row predicted by models that
        never saw it during training.
        """
        n = len(y)
        m = len(models)
        meta = np.zeros((n, m + 1), dtype=float)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        model_names = list(models.keys())

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info("OOF fold %d/%d …", fold + 1, self.n_folds)
            X_tr, X_v = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for j, name in enumerate(model_names):
                from sklearn.base import clone
                try:
                    m_clone = clone(models[name])
                    m_clone.fit(X_tr, y_tr)
                    meta[val_idx, j] = m_clone.predict_proba(X_v)[:, 1]
                except Exception as exc:
                    logger.warning("OOF fold %d model %s: %s", fold + 1, name, exc)
                    meta[val_idx, j] = 0.5

        # Column M+1: graph risk
        if graph_risk is not None and len(graph_risk) == n:
            meta[:, -1] = graph_risk
        else:
            meta[:, -1] = DEFAULT_GRAPH_RISK

        return meta

    # -----------------------------------------------------------------------
    # 7.2  Train Meta-Learner
    # -----------------------------------------------------------------------

    def train_meta_learner(self, meta_features: np.ndarray,
                            y: np.ndarray) -> "OOFStackedEnsemble":
        logger.info("Training meta-learner on OOF features %s …",
                    meta_features.shape)
        self.meta_learner.fit(meta_features, y)
        self._fitted = True
        return self

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Meta-learner not fitted.")
        return self.meta_learner.predict_proba(meta_features)[:, 1]

    def predict(self, meta_features: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(meta_features) >= threshold).astype(int)


# ===========================================================================
# MODULE 8 — SHAP Explainability
# ===========================================================================

class SHAPExplainer:
    """Wraps shap.TreeExplainer for XGBoost (or best available tree model).

    Caches the expected value and uses optimised paths for single-row
    inference (the common real-time case).
    """

    def __init__(self, model, feature_names: list[str] | None = None):
        import shap
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names or []
        self._expected_value = self.explainer.expected_value
        logger.info("SHAPExplainer initialised.")

    @property
    def expected_value(self):
        """Fallback for pipelines pickled before _expected_value was cached."""
        if not hasattr(self, "_expected_value"):
            self._expected_value = self.explainer.expected_value
        return self._expected_value

    def explain(self, X: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Returns a list of explanation dicts for each row in X.
        Uses check_additivity=False for faster computation on single rows.
        """
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        # For binary classification some libs return a list [neg, pos]
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # Handle single-row case (returns 1-D array)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        results = []
        for i in range(sv.shape[0]):
            row_sv = sv[i]
            named = {}
            for j, val in enumerate(row_sv):
                name = (self.feature_names[j]
                        if j < len(self.feature_names) else f"PC_{j}")
                named[name] = float(val)
            top = sorted(named.items(), key=lambda x: abs(x[1]),
                         reverse=True)[:top_k]
            results.append({
                "shap_values": named,
                "top_features": top,
                "base_value": float(self.expected_value[1])
                    if hasattr(self.expected_value, '__len__') and len(self.expected_value) > 1
                    else float(self.expected_value[0])
                    if hasattr(self.expected_value, '__len__')
                    else float(self.expected_value),
            })
        return results


class DummySHAPExplainer:
    """Fallback SHAP explainer based on feature magnitudes."""

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or []
        logger.info("DummySHAPExplainer (no shap library) initialised.")

    def explain(self, X: np.ndarray, top_k: int = 5) -> list[dict]:
        results = []
        for i in range(X.shape[0]):
            row = X[i]
            named = {}
            for j, val in enumerate(row):
                name = (self.feature_names[j]
                        if j < len(self.feature_names) else f"PC_{j}")
                named[name] = float(val) * 0.01  # dummy magnitude
            top = sorted(named.items(), key=lambda x: abs(x[1]),
                         reverse=True)[:top_k]
            results.append({"shap_values": named, "top_features": top})
        return results


def get_shap_explainer(models: dict, feature_names: list[str]) -> SHAPExplainer | DummySHAPExplainer:
    """Create the best available SHAP explainer."""
    try:
        import shap  # noqa: F401
        # Prefer XGBoost (tree explainer is fastest + most accurate)
        for name in ("xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"):
            if name in models:
                # Unwrap CalibratedClassifierCV to get base estimator
                cal = models[name]
                try:
                    base = cal.calibrated_classifiers_[0].estimator
                except Exception:
                    base = cal
                return SHAPExplainer(base, feature_names)
    except ImportError:
        pass
    return DummySHAPExplainer(feature_names)
