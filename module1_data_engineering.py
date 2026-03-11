"""
Module 1 — Data Engineering
Loads the three available datasets, harmonises columns, cleans and balances data.

Datasets:
  1. fake_social_media.csv  — Primary dataset (is_fake column)
  2. fake_users.csv         — Twitter raw data (all rows are fake; INT dataset)
  3. LIMFADD.csv            — Instagram-style: Labels in {Bot, Scam, Spam, Real}
"""

import logging
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
            "@": "a", "$": "s"}

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]

# ---------------------------------------------------------------------------
# 1.1  Data Loading
# ---------------------------------------------------------------------------

def _load_fake_social_media(path: str) -> pd.DataFrame:
    """Load the primary fake_social_media.csv dataset."""
    df = pd.read_csv(path, low_memory=False)
    # rename label column to 'label'
    df = df.rename(columns={"is_fake": "label"})
    df["label"] = df["label"].astype(int)
    # numeric columns
    numeric_cols = ["has_profile_pic", "bio_length", "username_randomness",
                    "followers", "following", "follower_following_ratio",
                    "account_age_days", "posts", "posts_per_day",
                    "caption_similarity_score", "content_similarity_score",
                    "follow_unfollow_rate", "spam_comments_rate",
                    "generic_comment_rate", "suspicious_links_in_bio", "verified"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["bio"] = ""  # bio text not available in this dataset
    df["username"] = "unknown_" + df.index.astype(str)
    df["_source"] = "primary"
    return df


def _load_fake_users(path: str) -> pd.DataFrame:
    """Load fake_users.csv (Twitter; all records are fake accounts)."""
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["username"] = df["screen_name"].fillna("").astype(str)
    out["bio"] = df["description"].fillna("").astype(str)
    out["followers"] = pd.to_numeric(df.get("followers_count", 0), errors="coerce").fillna(0)
    out["following"] = pd.to_numeric(df.get("friends_count", 0), errors="coerce").fillna(0)
    out["posts"] = pd.to_numeric(df.get("statuses_count", 0), errors="coerce").fillna(0)
    out["verified"] = pd.to_numeric(df.get("verified", 0), errors="coerce").fillna(0)
    out["has_profile_pic"] = 1 - pd.to_numeric(
        df.get("default_profile_image", 1), errors="coerce").fillna(1).clip(0, 1)
    out["follower_following_ratio"] = np.where(
        out["following"] > 0, out["followers"] / out["following"], 0)
    out["account_age_days"] = np.nan
    out["posts_per_day"] = np.nan
    out["bio_length"] = out["bio"].str.len()
    out["username_randomness"] = np.nan
    out["caption_similarity_score"] = np.nan
    out["content_similarity_score"] = np.nan
    out["follow_unfollow_rate"] = np.nan
    out["spam_comments_rate"] = np.nan
    out["generic_comment_rate"] = np.nan
    out["suspicious_links_in_bio"] = out["bio"].str.contains(
        r"http[s]?://|www\.", regex=True, na=False).astype(int)
    out["label"] = 1
    out["_source"] = "twitter_fake"
    return out


def _load_limfadd(path: str) -> pd.DataFrame:
    """Load LIMFADD.csv with Labels: Bot/Scam/Spam → 1 (fake), Real → 0."""
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["followers"] = pd.to_numeric(df.get("Followers", 0), errors="coerce").fillna(0)
    out["following"] = pd.to_numeric(df.get("Following", 0), errors="coerce").fillna(0)
    out["follower_following_ratio"] = pd.to_numeric(
        df.get("Following/Followers", 0), errors="coerce").fillna(0)
    out["posts"] = pd.to_numeric(df.get("Posts", 0), errors="coerce").fillna(0)
    out["posts_per_follower"] = pd.to_numeric(
        df.get("Posts/Followers", 0), errors="coerce").fillna(0)

    # Map Bio column: N = no bio (0 length), else has bio
    bio_col = df.get("Bio", pd.Series(["N"] * len(df)))
    out["bio_length"] = (bio_col.astype(str).str.strip().str.upper() != "N").astype(int) * 80
    out["bio"] = ""

    out["has_profile_pic"] = (
        df.get("Profile Picture", pd.Series(["N"] * len(df)))
        .astype(str).str.strip().str.lower()
        .map({"yes": 1, "y": 1, "n": 0}).fillna(0).astype(int))

    out["suspicious_links_in_bio"] = (
        df.get("External Link", pd.Series(["N"] * len(df)))
        .astype(str).str.strip().str.lower()
        .map({"yes": 1, "y": 1, "n": 0}).fillna(0).astype(int))

    out["username"] = "limfadd_" + df.index.astype(str)
    out["verified"] = 0
    out["account_age_days"] = np.nan
    out["username_randomness"] = np.nan
    out["caption_similarity_score"] = np.nan
    out["content_similarity_score"] = np.nan
    out["follow_unfollow_rate"] = np.nan
    out["spam_comments_rate"] = np.nan
    out["generic_comment_rate"] = np.nan
    out["posts_per_day"] = np.nan

    # Labels: Real → 0, everything else → 1
    label_col = df.get("Labels", pd.Series(["Real"] * len(df)))
    out["label"] = label_col.astype(str).str.strip().str.lower().map(
        {"real": 0}).fillna(1).astype(int)
    out["_source"] = "limfadd"
    return out


def _load_excel_global(path: str) -> pd.DataFrame:
    """Load fake_social_media_global_2.0_with_missing.xlsx (same schema as primary CSV)."""
    df = pd.read_excel(path, engine="openpyxl")
    if "is_fake" in df.columns:
        df = df.rename(columns={"is_fake": "label"})
    elif "label" not in df.columns:
        # Assume all rows in this dataset are labelled fake=1
        df["label"] = 1
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    numeric_cols = ["has_profile_pic", "bio_length", "username_randomness",
                    "followers", "following", "follower_following_ratio",
                    "account_age_days", "posts", "posts_per_day",
                    "caption_similarity_score", "content_similarity_score",
                    "follow_unfollow_rate", "spam_comments_rate",
                    "generic_comment_rate", "suspicious_links_in_bio", "verified"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bio" not in df.columns:
        df["bio"] = ""
    if "username" not in df.columns:
        df["username"] = "excel_" + df.index.astype(str)
    df["_source"] = "excel_global"
    return df


def load_datasets(primary_path: str = "fake_social_media.csv",
                  fake_users_path: str = "fake_users.csv",
                  limfadd_path: str = "LIMFADD.csv",
                  excel_path: str = "fake_social_media_global_2.0_with_missing.xlsx"
                  ) -> pd.DataFrame:
    """
    Load and vertically merge all four datasets into one DataFrame.
    Returns a DataFrame with a unified column set and a `label` column.
    """
    dfs = []
    for loader, path in [
        (_load_fake_social_media, primary_path),
        (_load_excel_global, excel_path),
        (_load_fake_users, fake_users_path),
        (_load_limfadd, limfadd_path),
    ]:
        try:
            df = loader(path)
            dfs.append(df)
            logger.info("Loaded %s — %d rows from %s", loader.__name__, len(df), path)
        except FileNotFoundError:
            logger.warning("File not found, skipping: %s", path)
        except Exception as exc:
            logger.warning("Error loading %s: %s", path, exc)

    if not dfs:
        raise RuntimeError("No datasets could be loaded.")

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    logger.info("Combined dataset: %d rows", len(combined))
    return combined


# ---------------------------------------------------------------------------
# 1.2  Noise Removal
# ---------------------------------------------------------------------------

def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, remove constant columns, strip strings, cap outliers."""
    df = df.copy()
    original = len(df)
    df = df.drop_duplicates()
    logger.info("Dedup: %d → %d rows", original, len(df))

    # Remove constant columns (skip label and internal marker)
    skip_cols = {"label", "_source", "username", "bio"}
    nunique = df.drop(columns=[c for c in skip_cols if c in df.columns]).nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info("Dropped constant columns: %s", constant_cols)

    # Strip string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # IQR-based outlier capping on numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["label"])
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 5 * iqr
        hi = q3 + 5 * iqr
        df[col] = df[col].clip(lower=lo, upper=hi)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1.3  Adversarial Text Normalisation
# ---------------------------------------------------------------------------

def normalise_leet(text: str) -> str:
    for k, v in LEET_MAP.items():
        text = text.replace(k, v)
    return text


def adversarial_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise leet-speak, zero-width chars, and Unicode confusables."""
    text_cols = [c for c in ["username", "bio"] if c in df.columns]
    for col in text_cols:
        # Zero-width character removal
        for zw in ZERO_WIDTH:
            df[col] = df[col].astype(str).str.replace(zw, "", regex=False)
        # Unicode normalisation (NFKC collapses confusables)
        df[col] = df[col].apply(
            lambda x: unicodedata.normalize("NFKC", str(x)) if pd.notna(x) else x)
        # Leet-speak on username only
        if col == "username":
            df[col] = df[col].apply(
                lambda x: normalise_leet(str(x)) if pd.notna(x) else x)
    return df


# ---------------------------------------------------------------------------
# 1.4  Username Entropy
# ---------------------------------------------------------------------------

def username_entropy(username: str) -> float:
    """Shannon entropy of a username string."""
    if not username or username == "nan":
        return 0.0
    counts: dict = {}
    for ch in username:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(username)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def add_username_entropy(df: pd.DataFrame) -> pd.DataFrame:
    if "username" in df.columns:
        df["username_entropy"] = df["username"].apply(
            lambda x: username_entropy(str(x)))
    return df


# ---------------------------------------------------------------------------
# 1.5  Missing Value Imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric columns: IterativeImputer (MICE with ExtraTreesRegressor).
    Categorical columns: mode imputation.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["label"])
    cat_cols = df.select_dtypes(include="object").columns.difference(
        ["_source", "username", "bio"])

    # Categorical: mode
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    # Numeric: MICE
    if num_cols.size > 0 and df[num_cols].isnull().any().any():
        logger.info("Running IterativeImputer on %d numeric columns …", len(num_cols))
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
            max_iter=10, random_state=42, skip_complete=True)
        df[num_cols] = imputer.fit_transform(df[num_cols])

    return df


# ---------------------------------------------------------------------------
# 1.6  Class Balancing (SMOTEENN)
# ---------------------------------------------------------------------------

def balance_classes(df: pd.DataFrame,
                    label_col: str = "label") -> pd.DataFrame:
    """Apply SMOTEENN to balance the training data."""
    from imblearn.combine import SMOTEENN
    import warnings

    # Identify numeric and non-numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.difference([label_col])
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    X = df[num_cols].values.astype(float)
    y = df[label_col].values.astype(int)

    label_counts = pd.Series(y).value_counts()
    logger.info("Class distribution before balancing: %s", label_counts.to_dict())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smoteenn = SMOTEENN(random_state=42)
            X_res, y_res = smoteenn.fit_resample(X, y)
        logger.info("After SMOTEENN: %d rows", len(y_res))
    except Exception as exc:
        logger.warning("SMOTEENN failed (%s), returning original data.", exc)
        return df

    df_res = pd.DataFrame(X_res, columns=num_cols.tolist())
    df_res[label_col] = y_res.astype(int)
    df_res = df_res.reset_index(drop=True)

    # Re-attach categorical columns as constant (mode) values — scalar assignment
    for col in cat_cols:
        if col in df.columns:
            val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
            df_res[col] = str(val)   # scalar → fills entire column uniformly

    return df_res.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1.7  Orchestration
# ---------------------------------------------------------------------------

def run_data_engineering(primary_path: str = "fake_social_media.csv",
                         fake_users_path: str = "fake_users.csv",
                         limfadd_path: str = "LIMFADD.csv",
                         excel_path: str = "fake_social_media_global_2.0_with_missing.xlsx",
                         balance: bool = True) -> pd.DataFrame:
    logger.info("=== Module 1: Data Engineering ===")
    df = load_datasets(primary_path, fake_users_path, limfadd_path, excel_path)
    df = remove_noise(df)
    df = adversarial_normalise(df)
    df = add_username_entropy(df)
    df = impute_missing(df)
    if balance:
        df = balance_classes(df)
    logger.info("Data engineering complete. Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = run_data_engineering()
    print(df.head())
    print(df["label"].value_counts())
