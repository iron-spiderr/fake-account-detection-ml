"""
Module 5 — Soft Weighted Voting
Fuses calibrated probabilities from base models using F1-weighted averaging.
"""

import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class SoftWeightedVoter:
    """F1-weighted soft voter over an ensemble of calibrated classifiers."""

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.model_names: list[str] = []
        self.optimal_threshold: float = 0.5

    # -----------------------------------------------------------------------
    # 5.1  Compute Weights
    # -----------------------------------------------------------------------

    def compute_weights(self, models: dict, X_val: np.ndarray,
                         y_val: np.ndarray) -> np.ndarray:
        """
        Compute normalised F1 weights for each model.
        w_m = F1_m / sum(F1_j)
        """
        f1_scores = []
        self.model_names = list(models.keys())

        for name, model in models.items():
            try:
                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, zero_division=0)
            except Exception:
                f1 = 0.0
            f1_scores.append(f1)
            logger.debug("  %s F1=%.4f", name, f1)

        f1_arr = np.array(f1_scores, dtype=float)
        total = f1_arr.sum()
        self.weights = (f1_arr / total) if total > 0 else np.ones(len(f1_arr)) / len(f1_arr)
        logger.info("Voting weights: %s",
                    dict(zip(self.model_names, self.weights.round(4))))
        return self.weights

    # -----------------------------------------------------------------------
    # 5.2  Soft Vote
    # -----------------------------------------------------------------------

    def soft_vote(self, prob_matrix: np.ndarray) -> np.ndarray:
        """
        prob_matrix: (N, M) — each column is P(fake) from one model.
        Returns: (N,) weighted ensemble probability.
        """
        if self.weights is None:
            # Equal weighting fallback
            return prob_matrix.mean(axis=1)
        w = self.weights[:prob_matrix.shape[1]]
        w = w / w.sum()
        return prob_matrix @ w

    # -----------------------------------------------------------------------
    # 5.3  Find Optimal Threshold
    # -----------------------------------------------------------------------

    def find_optimal_threshold(self, y_proba: np.ndarray,
                                y_true: np.ndarray) -> float:
        """Grid search threshold [0.20, 0.80] maximising F1."""
        best_thr = 0.5
        best_f1 = -1.0
        for thr in np.arange(0.20, 0.81, 0.01):
            y_pred = (y_proba >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        self.optimal_threshold = best_thr
        logger.info("Optimal threshold: %.2f (F1=%.4f)", best_thr, best_f1)
        return best_thr

    def predict(self, prob_matrix: np.ndarray) -> np.ndarray:
        proba = self.soft_vote(prob_matrix)
        return (proba >= self.optimal_threshold).astype(int)
