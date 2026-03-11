"""
Module 11 — Output & Reporting
Converts raw predictions into structured, human-readable reports.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_BANDS = [
    (0.85, "CRITICAL"),
    (0.60, "HIGH"),
    (0.30, "MEDIUM"),
    (0.00, "LOW"),
]


def _risk_band(prob: float) -> str:
    for threshold, band in RISK_BANDS:
        if prob >= threshold:
            return band
    return "LOW"


def _get_display_desc(feature_tuple, profile_row=None):
    """Pick the right description based on actual feature value."""
    from .pca_interpretability import FEATURE_DESCRIPTIONS_ZERO
    if len(feature_tuple) == 3:
        raw_name, desc, val = feature_tuple
    else:
        # Legacy 2-tuple: (desc, val)
        return feature_tuple[0], feature_tuple[-1]

    # If we have raw profile data, check if the feature is zero/absent
    if profile_row is not None and raw_name in FEATURE_DESCRIPTIONS_ZERO:
        raw_val = profile_row.get(raw_name, None)
        if raw_val is not None and (raw_val == 0 or raw_val is False or raw_val == ""):
            return FEATURE_DESCRIPTIONS_ZERO[raw_name], val
    return desc, val


def _corrected_direction(raw_name, shap_val, profile_row):
    """Determine the correct semantic direction for a feature.

    Returns ``"increases_fake"`` or ``"decreases_fake"``.
    Applies semantic overrides for zero-value features where the raw SHAP
    sign would create a logical contradiction in user-facing text (e.g.
    "No suspicious links → increases fake likelihood").
    """
    from .pca_interpretability import (FEATURE_DESCRIPTIONS_ZERO,
                                       FEATURE_ZERO_IS_FAKE_SIGNAL)

    shap_direction = "increases_fake" if shap_val > 0 else "decreases_fake"

    # Check if we're displaying a zero-value description
    if profile_row and raw_name and raw_name in FEATURE_DESCRIPTIONS_ZERO:
        raw_val = profile_row.get(raw_name)
        if raw_val is not None and (raw_val == 0 or raw_val is False or raw_val == ""):
            # Feature is zero → apply semantic override if available
            if raw_name in FEATURE_ZERO_IS_FAKE_SIGNAL:
                is_fake_signal = FEATURE_ZERO_IS_FAKE_SIGNAL[raw_name]
                return "increases_fake" if is_fake_signal else "decreases_fake"

    return shap_direction


def generate_explanation(label: str, prob: float, risk: str,
                          top_features: list | None = None,
                          top_real: list | None = None,
                          profile_row: dict | None = None,
                          data_completeness: float | None = None) -> str:
    """Generate a natural-language risk explanation string."""
    use = top_real if top_real else top_features
    parts = [f"Prediction: {label} (probability: {prob * 100:.2f}%, risk: {risk})."]
    if use:
        # Compute total absolute impact for percentage conversion
        all_vals = [t[-1] for t in use]
        total_abs = sum(abs(v) for v in all_vals) or 1.0
        supporting_parts = []
        opposing_parts = []
        for t in use[:5]:
            raw_name = t[0] if len(t) >= 3 else None
            desc, val = _get_display_desc(t, profile_row)
            direction = _corrected_direction(raw_name, val, profile_row)
            pct = abs(val) / total_abs * 100

            if direction == "increases_fake":
                entry = f"{desc} increases fake likelihood (impact: {pct:.1f}%)"
            else:
                entry = f"{desc} reduces fake likelihood (impact: {pct:.1f}%)"

            # Does this feature support or oppose the prediction?
            supports_prediction = (
                (label == "FAKE" and direction == "increases_fake") or
                (label == "GENUINE" and direction == "decreases_fake")
            )
            if supports_prediction:
                supporting_parts.append(entry)
            else:
                opposing_parts.append(entry)

        if supporting_parts:
            parts.append("Key factors: " + "; ".join(supporting_parts[:3]) + ".")
        if opposing_parts:
            parts.append("Counterevidence: " + "; ".join(opposing_parts[:2]) + ".")

    if data_completeness is not None and data_completeness < 0.6:
        parts.append(
            f"\u26a0 Limited data available ({data_completeness * 100:.0f}%) "
            "\u2014 prediction reliability is reduced.")
    return " ".join(parts)


# Features that are directly observable from a scraped profile
OBSERVED_FEATURES = {
    "has_profile_pic", "verified", "bio_length", "bio_has_url",
    "bio_spam_score", "followers", "following", "follow_diff",
    "follower_following_ratio", "followers_ratio", "posts",
    "log_followers", "log_posts", "username_length",
    "username_entropy", "digits_count", "digit_ratio",
    "special_char_count", "repeat_char_count", "suspicious_links_in_bio",
    "verified_low_follow",
    # Descriptions for these features:
    "Whether the account has a profile picture",
    "Whether the account is verified by the platform",
    "Length of the biography text in characters",
    "Whether the biography contains a URL",
    "Count of spam keywords found in the biography",
    "Total number of followers",
    "Total number of accounts followed",
    "Difference between followers and following counts",
    "Ratio of followers to following count",
    "Followers as a fraction of total connections",
    "Total number of posts",
    "Log-transformed follower count (reduces skew)",
    "Log-transformed post count (reduces skew)",
    "Length of the username string",
    "Shannon entropy of the username (randomness measure)",
    "Number of digit characters in the username",
    "Fraction of digits in the username",
    "Number of non-alphanumeric characters in username",
    "Number of consecutively repeated characters in username",
    "Presence of suspicious external links in bio",
    "Unverified account with fewer than 100 followers flag",
}

# Features that are estimated/defaulted when scraping
ESTIMATED_FEATURES = {
    "caption_similarity_score", "content_similarity_score",
    "follow_unfollow_rate", "spam_comments_rate", "generic_comment_rate",
    "account_age_days", "posts_per_day", "activity_rate",
    "username_randomness", "account_hour", "account_weekend",
    "listed_ratio", "likes_per_post",
    # Descriptions for these:
    "Similarity between post captions (high → repetitive content)",
    "Visual/content similarity score across posts",
    "Rate of follow/unfollow cycling behaviour",
    "Fraction of comments classified as spam",
    "Fraction of generic/templated comments",
    "Age of the account in days",
    "Average posts published per day",
    "Post activity relative to follower count",
    "Whether the username appears random/bot-generated",
    "Estimated creation hour derived from account age",
    "Whether account appears to have been created on a weekend",
    "Listed count relative to followers",
    "Average likes per post",
}


def format_output(df: pd.DataFrame,
                  y_pred: np.ndarray,
                  final_proba: np.ndarray,
                  graph_risk: np.ndarray,
                  shap_results: list | None = None) -> pd.DataFrame:
    """
    Build a results DataFrame with one row per input profile.
    """
    records = []
    usernames = df.get("username",
                        pd.Series(["unknown"] * len(df), index=df.index)).tolist()

    for i in range(len(y_pred)):
        prob = float(final_proba[i])
        gr = float(graph_risk[i])
        label = "FAKE" if y_pred[i] == 1 else "GENUINE"
        band = _risk_band(prob)
        confidence = f"{max(prob, 1 - prob) * 100:.1f}%"

        # Extract raw profile row for context-aware descriptions
        profile_row = df.iloc[i].to_dict() if df is not None else None

        top_feats, top_real = None, None
        if shap_results and i < len(shap_results):
            sr = shap_results[i]
            top_feats = sr.get("top_features", [])
            top_real = sr.get("top_real_features")

        explanation = generate_explanation(
            label, prob, band, top_feats, top_real, profile_row=profile_row)

        # Annotate each SHAP feature with observed/estimated status and direction
        # top_real now returns 3-tuples (raw_name, desc, val)
        use = top_real or top_feats or []
        total_abs = sum(abs(t[-1]) for t in use) or 1.0
        annotated_features = []
        for t in use:
            desc, val = _get_display_desc(t, profile_row)
            raw_name = t[0] if len(t) == 3 else desc
            is_estimated = raw_name in ESTIMATED_FEATURES or desc in ESTIMATED_FEATURES
            pct = abs(val) / total_abs * 100
            direction = _corrected_direction(raw_name, val, profile_row)
            # Signed impact: positive = fake signal, negative = genuine signal
            signed_pct = round(pct, 1) if direction == "increases_fake" else round(-pct, 1)
            annotated_features.append({
                "name": desc,
                "impact": signed_pct,
                "direction": direction,
                "estimated": is_estimated,
            })

        # For top_features / top_shap_values, use display descriptions
        # Store signed impacts so downstream consumers preserve direction
        display_features = []
        display_shap = []
        for t in use:
            desc, val = _get_display_desc(t, profile_row)
            raw_name = t[0] if len(t) == 3 else None
            pct = abs(val) / total_abs * 100
            direction = _corrected_direction(raw_name, val, profile_row)
            signed_pct = round(pct, 1) if direction == "increases_fake" else round(-pct, 1)
            display_features.append(desc)
            display_shap.append((raw_name, desc, signed_pct))

        records.append({
            "username": str(usernames[i]).replace("unknown_", "@user_")
                if str(usernames[i]).startswith("unknown_") else str(usernames[i]),
            "label": label,
            "probability": round(prob, 4),
            "graph_risk": round(gr, 4),
            "risk_band": band,
            "confidence": confidence,
            "top_features": display_features,
            "top_shap_values": display_shap,
            "annotated_features": annotated_features,
            "explanation": explanation,
        })

    return pd.DataFrame(records)
