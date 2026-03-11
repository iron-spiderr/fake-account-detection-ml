"""
PCA Interpretability
Translates SHAP values from PCA component space back to original feature space.

SHAP_original = SHAP_PCA · V
where V is the PCA components matrix (K components × D original features).
"""

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

FEATURE_DESCRIPTIONS = {
    "has_profile_pic": "Whether the account has a profile picture",
    "bio_length": "Length of the biography text in characters",
    "username_randomness": "Whether the username appears random/bot-generated",
    "followers": "Total number of followers",
    "following": "Total number of accounts followed",
    "follower_following_ratio": "Ratio of followers to following count",
    "account_age_days": "Age of the account in days",
    "posts": "Total number of posts",
    "posts_per_day": "Average posts published per day",
    "caption_similarity_score": "Similarity between post captions (high \u2192 repetitive content)",
    "content_similarity_score": "Visual/content similarity score across posts",
    "follow_unfollow_rate": "Rate of follow/unfollow cycling behaviour",
    "spam_comments_rate": "Fraction of comments classified as spam",
    "generic_comment_rate": "Fraction of generic/templated comments",
    "suspicious_links_in_bio": "Presence of suspicious external links in bio",
    "verified": "Whether the account is verified by the platform",
    "username_entropy": "Shannon entropy of the username (randomness measure)",
    "username_length": "Length of the username string",
    "digits_count": "Number of digit characters in the username",
    "digit_ratio": "Fraction of digits in the username",
    "special_char_count": "Number of non-alphanumeric characters in username",
    "repeat_char_count": "Number of consecutively repeated characters in username",
    "bio_has_url": "Whether the biography contains a URL",
    "bio_spam_score": "Count of spam keywords found in the biography",
    "follow_diff": "Difference between followers and following counts",
    "followers_ratio": "Followers as a fraction of total connections",
    "activity_rate": "Post activity relative to follower count",
    "log_followers": "Log-transformed follower count (reduces skew)",
    "log_posts": "Log-transformed post count (reduces skew)",
    "account_hour": "Estimated creation hour derived from account age",
    "account_weekend": "Whether account appears to have been created on a weekend",
    "listed_ratio": "Listed count relative to followers",
    "likes_per_post": "Average likes per post",
    "verified_low_follow": "Unverified account with fewer than 100 followers flag",
    "community_size_ratio": "Community size ratio in social graph",
    "degree_centrality": "Degree centrality in social graph",
}

# Descriptions used when the raw feature value is zero/absent
FEATURE_DESCRIPTIONS_ZERO = {
    "bio_length": "Empty biography (no bio text)",
    "suspicious_links_in_bio": "No suspicious external links in bio",
    "bio_has_url": "No URL in biography",
    "bio_spam_score": "No spam keywords in biography",
    "has_profile_pic": "No profile picture set",
    "verified": "Account is not verified",
    "posts": "No posts published",
    "followers": "Zero followers",
    "following": "Not following anyone",
    "log_posts": "No posts (log-transformed)",
    "log_followers": "No followers (log-transformed)",
    "verified_low_follow": "Account is verified or has 100+ followers",
}

# Semantic direction for zero-value features.
# True  = zero/absent value is a FAKE signal  (increases fake likelihood)
# False = zero/absent value is a GENUINE signal (decreases fake likelihood)
# Features not listed here fall back to the raw SHAP sign.
FEATURE_ZERO_IS_FAKE_SIGNAL = {
    "bio_length": True,                # Empty biography        → suspicious
    "has_profile_pic": True,           # No profile pic         → suspicious
    "verified": True,                  # Not verified           → slightly suspicious
    "posts": True,                     # No posts               → suspicious
    "log_posts": True,                 # No posts (log)         → suspicious
    "followers": True,                 # Zero followers         → suspicious
    "log_followers": True,             # Zero followers (log)   → suspicious
    "following": True,                 # Not following anyone   → suspicious
    "suspicious_links_in_bio": False,  # No suspicious links    → genuine signal
    "bio_spam_score": False,           # No spam keywords       → genuine signal
    "bio_has_url": False,              # No URL in bio          → genuine signal
}


class PCAInterpreter:
    """Map SHAP values from PCA space back to original feature names."""

    def __init__(self, pca: PCA, feature_names: list[str],
                 pca_feature_names: list[str]):
        self.pca = pca
        self.feature_names = feature_names
        self.pca_feature_names = pca_feature_names
        # V: (n_components, n_original_features)
        self.components = pca.components_  # shape (K, D)

    def map_shap(self, shap_dict: dict, top_k: int = 10,
                 deprioritize_estimated: bool = True) -> list[tuple[str, float]]:
        """
        Convert a dict {PCA_component: shap_val} → sorted list of
        (original_feature_name, aggregated_contribution).

        When deprioritize_estimated=True, features whose values were
        estimated (not directly observed) during scrape/manual inference
        are down-weighted so that observed features rank higher in the
        explanation.  This gives more trustworthy explanations.
        """
        n_orig = len(self.feature_names)
        n_comps = self.components.shape[0]

        # Build SHAP vector in PCA space
        sv = np.zeros(n_comps)
        for name, val in shap_dict.items():
            try:
                idx = self.pca_feature_names.index(name)
                if idx < n_comps:
                    sv[idx] = val
            except ValueError:
                pass

        # Project back: (D,) = sum across K components of (shap_k * V_k)
        orig_contributions = sv @ self.components  # (D,)

        # Names of features that are estimated (not observed) at inference time
        _ESTIMATED_FEATURE_NAMES = {
            "caption_similarity_score", "content_similarity_score",
            "follow_unfollow_rate", "spam_comments_rate", "generic_comment_rate",
            "account_age_days", "posts_per_day", "activity_rate",
            "username_randomness", "account_hour", "account_weekend",
            "listed_ratio", "likes_per_post",
        }

        result = []
        for j in range(min(n_orig, len(orig_contributions))):
            name = self.feature_names[j]
            desc = FEATURE_DESCRIPTIONS.get(name, name)
            contribution = float(orig_contributions[j])

            # Down-weight estimated features for ranking (but keep real value)
            sort_weight = abs(contribution)
            if deprioritize_estimated and name in _ESTIMATED_FEATURE_NAMES:
                sort_weight *= 0.3  # reduce ranking importance

            result.append((name, desc, contribution, sort_weight))

        # Sort by adjusted weight, return (raw_name, desc, contribution) tuples
        result.sort(key=lambda x: x[3], reverse=True)
        return [(name, desc, val) for name, desc, val, _ in result[:top_k]]

    def component_report(self, n_top: int = 5) -> str:
        """Return a human-readable report of what each PCA component represents."""
        lines = ["PCA Component Interpretation", "=" * 50]
        for i, comp in enumerate(self.components):
            top_idx = np.argsort(np.abs(comp))[::-1][:n_top]
            contributors = [
                f"  {self.feature_names[j]:40s} {comp[j]:+.3f}"
                for j in top_idx if j < len(self.feature_names)
            ]
            lines.append(f"\nPC_{i} (explains {self.pca.explained_variance_ratio_[i] * 100:.1f}%):")
            lines.extend(contributors)
        return "\n".join(lines)
