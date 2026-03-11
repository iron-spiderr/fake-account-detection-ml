"""
Module 2 — Feature Extraction
Converts cleaned DataFrames into the unified numeric feature matrix used by ML models.

Feature groups:
  • 33 hand-crafted metadata features
  • 384-d BERT embeddings (or 64-d TF-IDF/SVD proxy with --no-bert)
  • 5 graph centrality features (from Module 3)
  • StandardScaler + PCA (95% variance)
"""

import logging
import math
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SPAM_KEYWORDS = [
    "follow back", "followback", "f4f", "l4l", "free", "click", "buy now",
    "discount", "promo", "earn money", "work from home", "dm me", "link in bio"
]

# ---------------------------------------------------------------------------
# Text Embedders
# ---------------------------------------------------------------------------

class BERTEmbedder:
    """Full BERT embedding via sentence-transformers."""

    DIM = 384

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info("BERTEmbedder loaded: %s", model_name)

    def embed(self, texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIM))
        return self.model.encode(texts, show_progress_bar=False, batch_size=64)


class DummyBERTEmbedder:
    """TF-IDF + TruncatedSVD proxy when sentence-transformers is unavailable."""

    DIM = 64

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True,
                                     min_df=1, max_df=1.0)
        self.svd = TruncatedSVD(n_components=self.DIM, random_state=42)
        self._fitted = False
        self._vocab_empty = False

    def fit(self, texts: list):
        # Filter to non-empty texts
        non_empty = [t for t in texts if t and t.strip()]
        if len(non_empty) < 2:
            logger.info("Bio texts are empty — TF-IDF skipped; using zero embeddings.")
            self._vocab_empty = True
            self._fitted = True
            return
        try:
            mat = self.tfidf.fit_transform(non_empty)
            n_comps = min(self.DIM, mat.shape[1], mat.shape[0] - 1)
            if n_comps < 1:
                self._vocab_empty = True
            else:
                self.svd.set_params(n_components=n_comps)
                self.svd.fit(mat)
            self._fitted = True
        except Exception as exc:
            logger.warning("TF-IDF fit failed: %s — using zero embeddings.", exc)
            self._vocab_empty = True
            self._fitted = True

    def embed(self, texts: list) -> np.ndarray:
        n = len(texts) if texts else 0
        if not texts:
            return np.zeros((0, self.DIM))
        if not self._fitted:
            self.fit(texts)
        if self._vocab_empty:
            return np.zeros((n, self.DIM), dtype=np.float32)
        try:
            return self.svd.transform(
                self.tfidf.transform(texts)).astype(np.float32)
        except Exception:
            return np.zeros((n, self.DIM), dtype=np.float32)


def get_embedder(use_bert: bool = True):
    if use_bert:
        try:
            return BERTEmbedder()
        except Exception as exc:
            logger.warning("BERT unavailable (%s). Falling back to TF-IDF proxy.", exc)
    return DummyBERTEmbedder()


# ---------------------------------------------------------------------------
# 2.1  Metadata Feature Engineering
# ---------------------------------------------------------------------------

def _safe_num(col) -> pd.Series:
    if isinstance(col, (int, float, np.integer, np.floating)):
        return pd.Series([float(col)], dtype=float)
    return pd.to_numeric(col, errors="coerce").fillna(0)


def _get_col(df: pd.DataFrame, col_name: str, default: float = 0.0) -> pd.Series:
    """Safely retrieve a numeric column, returning a zero Series if absent."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), default, dtype=float), index=df.index)


def extract_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 33 numeric metadata features from a raw profile DataFrame.
    Columns with missing values are gracefully coerced to 0.
    """
    feat = pd.DataFrame(index=df.index)

    # Profile presence
    feat["has_profile_pic"] = _get_col(df, "has_profile_pic").clip(0, 1)
    feat["verified"] = _get_col(df, "verified").clip(0, 1)

    # Bio
    bio_raw = df["bio"] if "bio" in df.columns else pd.Series([""] * len(df), index=df.index)
    bio = bio_raw.astype(str).fillna("")
    feat["bio_length"] = _get_col(df, "bio_length") if "bio_length" in df.columns else bio.str.len()
    feat["bio_has_url"] = bio.str.contains(r"https?://|www\.", regex=True, na=False).astype(int)
    feat["bio_spam_score"] = bio.apply(
        lambda x: sum(1 for kw in SPAM_KEYWORDS if kw in x.lower()))

    # Network
    followers = _get_col(df, "followers")
    following = _get_col(df, "following")
    feat["followers"] = followers
    feat["following"] = following
    feat["follow_diff"] = followers - following
    feat["follower_following_ratio"] = np.where(
        following > 0, followers / following.clip(lower=1), 0)
    feat["followers_ratio"] = followers / (followers + following + 1)

    # Activity
    posts = _get_col(df, "posts")
    age = _get_col(df, "account_age_days", 365.0)
    age = age.replace(0, np.nan).fillna(365.0)
    feat["posts"] = posts
    feat["account_age_days"] = age
    feat["posts_per_day"] = posts / age.clip(lower=1)
    feat["activity_rate"] = (posts / (followers + 1)).clip(upper=100)

    # Log transforms (reduce skew)
    feat["log_followers"] = np.log1p(followers)
    feat["log_posts"] = np.log1p(posts)

    # Temporal creation signals
    feat["account_hour"] = _get_col(df, "account_age_days") % 24
    feat["account_weekend"] = (_get_col(df, "account_age_days") % 7 >= 5).astype(int)

    # Listed / likes ratios
    feat["listed_ratio"] = _get_col(df, "listed_count") / (followers + 1)
    feat["likes_per_post"] = _get_col(df, "favourites_count") / (posts + 1)

    # Content similarity
    feat["caption_similarity_score"] = _get_col(df, "caption_similarity_score")
    feat["content_similarity_score"] = _get_col(df, "content_similarity_score")

    # Spam indicators
    feat["follow_unfollow_rate"] = _get_col(df, "follow_unfollow_rate")
    feat["spam_comments_rate"] = _get_col(df, "spam_comments_rate")
    feat["generic_comment_rate"] = _get_col(df, "generic_comment_rate")

    # Suspicious signals
    feat["suspicious_links_in_bio"] = _get_col(df, "suspicious_links_in_bio").clip(0, 1)
    feat["verified_low_follow"] = ((feat["verified"] == 0) & (followers < 100)).astype(int)

    # Username string analysis
    uname_raw = df["username"] if "username" in df.columns else pd.Series([""] * len(df), index=df.index)
    uname = uname_raw.astype(str).fillna("")
    feat["username_length"] = uname.str.len()
    feat["username_randomness"] = _get_col(df, "username_randomness")
    feat["digits_count"] = uname.str.count(r"\d")
    feat["digit_ratio"] = feat["digits_count"] / (feat["username_length"].clip(lower=1))
    feat["special_char_count"] = uname.str.count(r"[^a-zA-Z0-9]")
    feat["repeat_char_count"] = uname.apply(
        lambda x: sum(1 for i in range(1, len(x)) if x[i] == x[i - 1])
        if len(x) > 1 else 0)

    # Shannon entropy (if already computed, else recompute)
    if "username_entropy" in df.columns:
        feat["username_entropy"] = _safe_num(df["username_entropy"])
    else:
        def _entropy(s: str) -> float:
            if not s:
                return 0.0
            from collections import Counter
            counts = Counter(s)
            n = len(s)
            return -sum((c / n) * math.log2(c / n) for c in counts.values())
        feat["username_entropy"] = uname.apply(_entropy)

    return feat.fillna(0).astype(float)


# ---------------------------------------------------------------------------
# 2.4  Unified Feature Vector
# ---------------------------------------------------------------------------

def build_unified_features(
        df: pd.DataFrame,
        embedder=None,
        graph_features: pd.DataFrame | None = None,
        scaler: StandardScaler | None = None,
        pca: PCA | None = None,
        fit: bool = True) -> tuple[np.ndarray, StandardScaler, PCA, list[str]]:
    """
    Build the full feature matrix:
      [ metadata (33) ] + [ BERT/TF-IDF ] + [ graph centrality (5) ]
      → StandardScaler → PCA

    Returns (X_reduced, scaler, pca, feature_names).
    If fit=False, uses provided scaler/pca for inference.
    """
    meta = extract_metadata_features(df)
    feature_names: list[str] = meta.columns.tolist()
    X = meta.values

    # Text embeddings
    if embedder is not None:
        bio_texts = df.get("bio", pd.Series([""] * len(df),
                                             index=df.index)).fillna("").astype(str).tolist()
        if isinstance(embedder, DummyBERTEmbedder) and fit:
            embedder.fit(bio_texts)
        emb = embedder.embed(bio_texts)
        if emb.shape[0] == X.shape[0]:
            emb_names = [f"bert_{i}" for i in range(emb.shape[1])]
            feature_names += emb_names
            X = np.hstack([X, emb])
        else:
            logger.warning("Embedding row count mismatch; skipping embeddings.")

    # Graph features
    if graph_features is not None:
        gf = graph_features.reindex(df.index).fillna(0).values
        g_names = graph_features.columns.tolist()
        feature_names += g_names
        X = np.hstack([X, gf])

    logger.info("Raw feature matrix: %s features, %s rows", X.shape[1], X.shape[0])

    # Scale
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Pad with the training mean so missing features become z-score 0 after
        # scaling (neutral), rather than zeros which produce extreme negative
        # z-scores and bias the model toward FAKE for every inference profile.
        expected_n = scaler.n_features_in_
        if X.shape[1] < expected_n:
            n_pad = expected_n - X.shape[1]
            mean_pad = scaler.mean_[-n_pad:]          # training-distribution means
            padding = np.tile(mean_pad, (X.shape[0], 1))
            X = np.hstack([X, padding])
            logger.debug("Padded %d missing features with training means for inference.", n_pad)
        X_scaled = scaler.transform(X)

    # PCA
    if fit:
        pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        logger.info("PCA: %d → %d components (95%% variance)", X.shape[1],
                    pca.n_components_)
    else:
        X_reduced = pca.transform(X_scaled)

    return X_reduced, scaler, pca, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from module1_data_engineering import run_data_engineering
    df = run_data_engineering(balance=False)
    X, scaler, pca, names = build_unified_features(df, embedder=None, fit=True)
    print("Feature matrix shape:", X.shape)
    print("First 5 feature names:", names[:5])
