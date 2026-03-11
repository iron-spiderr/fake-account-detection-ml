"""
Instagram Profile Scraper
Fetches publicly visible metadata for both public and private profiles.
No login required for basic profile data.

Strategy (two-step with automatic fallback):
  1. requests + browser headers  →  Instagram web API (fast, rarely limited)
  2. instaloader                 →  iPhone API fallback (if web API fails)
"""

import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field

import instaloader
import requests

logger = logging.getLogger(__name__)

# Browser headers that mimic Chrome visiting instagram.com
_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "X-IG-App-ID": "936619743392459",   # Instagram's own web app ID
    "Referer": "https://www.instagram.com/",
    "Origin": "https://www.instagram.com",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}
_WEB_API = "https://www.instagram.com/api/v1/users/web_profile_info/"
_WEB_TIMEOUT = 10   # seconds


class RateLimitError(Exception):
    """Raised when Instagram returns HTTP 429 Too Many Requests."""

# ---------------------------------------------------------------------------
# Profile Data Structure
# ---------------------------------------------------------------------------

@dataclass
class InstagramProfileData:
    username: str
    full_name: str
    biography: str
    follower_count: int
    following_count: int
    post_count: int
    is_private: bool
    is_verified: bool
    has_profile_pic: bool
    profile_pic_url: str
    external_url: str
    business_category: str
    is_partial: bool = False           # True when profile is private
    available_fields: list = field(default_factory=list)

    # Derived features — computed in __post_init__
    username_length: int = 0
    fullname_length: int = 0
    bio_length: int = 0
    has_external_url: bool = False
    follower_following_ratio: float = 0.0
    username_has_numbers: bool = False
    username_has_special_chars: bool = False
    fullname_word_count: int = 0
    bio_has_url: bool = False
    bio_has_phone: bool = False
    bio_has_emoji: bool = False
    digits_in_username: int = 0
    username_digit_ratio: float = 0.0

    def __post_init__(self):
        self.username_length = len(self.username)
        self.fullname_length = len(self.full_name)
        self.bio_length = len(self.biography)
        self.has_external_url = bool(self.external_url)
        self.follower_following_ratio = (
            self.follower_count / max(self.following_count, 1)
        )
        self.username_has_numbers = any(c.isdigit() for c in self.username)
        self.username_has_special_chars = any(
            not c.isalnum() and c not in ('.', '_')
            for c in self.username
        )
        self.fullname_word_count = len(self.full_name.split())
        self.bio_has_url = bool(
            re.search(r'http[s]?://|www\.', self.biography, re.IGNORECASE)
        )
        self.bio_has_phone = bool(
            re.search(r'\+?\d[\d\s\-]{7,}\d', self.biography)
        )
        self.bio_has_emoji = any(ord(c) > 127 for c in self.biography)
        self.digits_in_username = sum(c.isdigit() for c in self.username)
        self.username_digit_ratio = (
            self.digits_in_username / max(len(self.username), 1)
        )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class InstagramScraper:
    def __init__(self, delay_min: float = 1.5, delay_max: float = 3.5):
        # --- requests session (primary: Instagram web API) ---
        self._session = requests.Session()
        self._session.headers.update(_WEB_HEADERS)

        # --- instaloader (fallback: iPhone API) ---
        self._loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            quiet=True,
        )
        # Fail fast — do NOT hang for 30 min on 429
        self._loader.context.max_connection_attempts = 1

        self.delay_min = delay_min
        self.delay_max = delay_max

    def _random_delay(self):
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    # ------------------------------------------------------------------
    # Primary: Instagram web API via requests + browser headers
    # ------------------------------------------------------------------
    def _scrape_via_web(self, username: str) -> InstagramProfileData | None:
        """
        Hit Instagram's web profile API with browser-like headers.
        Returns InstagramProfileData on success, None on non-429 error,
        raises RateLimitError on HTTP 429.
        """
        try:
            resp = self._session.get(
                _WEB_API,
                params={"username": username},
                timeout=_WEB_TIMEOUT,
                allow_redirects=True,
            )

            if resp.status_code == 429:
                raise RateLimitError(
                    "Instagram rate-limited this IP (HTTP 429 on web API). "
                    "Wait ~30 minutes before scraping again."
                )
            if resp.status_code == 404:
                logger.error("Profile @%s not found (web API 404).", username)
                return None
            if resp.status_code != 200:
                logger.warning(
                    "Web API returned HTTP %d for @%s", resp.status_code, username
                )
                return None

            payload = resp.json()
            user = payload.get("data", {}).get("user")
            if not user:
                logger.warning("Web API: no user data in response for @%s", username)
                return None

            is_private = bool(user.get("is_private", False))
            available = [
                "username", "full_name", "biography",
                "follower_count", "following_count", "post_count",
                "is_verified", "has_profile_pic", "external_url",
            ]
            if not is_private:
                available.append("business_category")

            data = InstagramProfileData(
                username=user.get("username", username),
                full_name=user.get("full_name") or "",
                biography=user.get("biography") or "",
                follower_count=user.get("edge_followed_by", {}).get("count", 0),
                following_count=user.get("edge_follow", {}).get("count", 0),
                post_count=user.get("edge_owner_to_timeline_media", {}).get("count", 0),
                is_private=is_private,
                is_verified=bool(user.get("is_verified", False)),
                has_profile_pic=bool(user.get("profile_pic_url")),
                profile_pic_url=user.get("profile_pic_url") or "",
                external_url=user.get("external_url") or "",
                business_category=user.get("category_name") or "",
                is_partial=is_private,
                available_fields=available,
            )
            logger.info(
                "Web API scraped @%s — private=%s, followers=%d, posts=%d",
                username, data.is_private,
                data.follower_count, data.post_count,
            )
            return data

        except RateLimitError:
            raise
        except requests.exceptions.Timeout:
            logger.warning("Web API timed out for @%s", username)
            return None
        except Exception as exc:
            logger.warning("Web API failed for @%s: %s", username, exc)
            return None

    # ------------------------------------------------------------------
    # Fallback: instaloader (iPhone API)
    # ------------------------------------------------------------------
    def _build_from_loader(
        self, profile: instaloader.Profile
    ) -> InstagramProfileData:
        is_private = profile.is_private
        available = [
            "username", "full_name", "biography",
            "follower_count", "following_count", "post_count",
            "is_verified", "has_profile_pic", "external_url",
        ]
        if not is_private:
            available.append("business_category")

        return InstagramProfileData(
            username=profile.username,
            full_name=profile.full_name or "",
            biography=profile.biography or "",
            follower_count=profile.followers,
            following_count=profile.followees,
            post_count=profile.mediacount,
            is_private=is_private,
            is_verified=profile.is_verified,
            has_profile_pic=bool(profile.profile_pic_url),
            profile_pic_url=profile.profile_pic_url or "",
            external_url=profile.external_url or "",
            business_category="" if is_private else (
                profile.business_category_name or ""
            ),
            is_partial=is_private,
            available_fields=available,
        )

    def _scrape_via_loader(self, username: str) -> InstagramProfileData | None:
        try:
            profile = instaloader.Profile.from_username(
                self._loader.context, username
            )
            data = self._build_from_loader(profile)
            logger.info(
                "Instaloader scraped @%s — private=%s, followers=%d",
                username, data.is_private, data.follower_count,
            )
            return data
        except instaloader.exceptions.ProfileNotExistsException:
            logger.error("Profile @%s does not exist.", username)
        except instaloader.exceptions.ConnectionException as exc:
            msg = str(exc)
            if "429" in msg or "Too Many Requests" in msg:
                raise RateLimitError(
                    "Instagram rate-limited this IP on both the web API and "
                    "the iPhone API (HTTP 429). Wait ~30 minutes."
                ) from exc
            logger.error("Instaloader connection error for @%s: %s", username, exc)
        except Exception as exc:
            logger.error("Instaloader error for @%s: %s", username, exc)
        return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def scrape_profile(self, username: str) -> InstagramProfileData | None:
        """
        Try the web API first (fast, lenient rate limits).
        Falls back to instaloader if the web API returns no data.
        Raises RateLimitError only when BOTH methods are rate-limited.
        """
        self._random_delay()

        # 1. Web API (primary)
        data = self._scrape_via_web(username)
        if data is not None:
            return data

        # 2. Instaloader (fallback)
        logger.info("Web API gave no data for @%s — trying instaloader.", username)
        return self._scrape_via_loader(username)

    def profile_to_prediction_dict(self, data: InstagramProfileData) -> dict:
        """
        Convert scraped profile to a flat dict that matches the feature
        columns expected by the trained pipeline.
        Meta-only fields (_is_partial, _available_fields) are prefixed
        with '_' so the caller can strip them before building a DataFrame.
        """
        d = asdict(data)
        return {
            # --- Always available ---
            "username":                   d["username"],
            "fullname":                   d["full_name"],
            "bio":                        d["biography"],           # pipeline col
            "biography":                  d["biography"],
            "followers":                  d["follower_count"],
            "following":                  d["following_count"],
            "posts":                      d["post_count"],
            "is_private":                 int(d["is_private"]),
            "is_verified":                int(d["is_verified"]),
            "verified":                   int(d["is_verified"]),
            "has_profile_pic":            int(d["has_profile_pic"]),
            "has_external_url":           int(d["has_external_url"]),
            "suspicious_links_in_bio":    int(d["bio_has_url"]),

            # --- Derived from always-available fields ---
            "username_length":            d["username_length"],
            "fullname_length":            d["fullname_length"],
            "bio_length":                 d["bio_length"],
            "follower_following_ratio":   d["follower_following_ratio"],
            "username_has_numbers":       int(d["username_has_numbers"]),
            "username_has_special_chars": int(d["username_has_special_chars"]),
            "fullname_word_count":        d["fullname_word_count"],
            "bio_has_url":                int(d["bio_has_url"]),
            "bio_has_phone":              int(d["bio_has_phone"]),
            "bio_has_emoji":              int(d["bio_has_emoji"]),
            "digits_in_username":         d["digits_in_username"],
            "username_digit_ratio":       d["username_digit_ratio"],

            # --- Defaults for features unavailable from scraping ---
            # Use training-set genuine-class medians so that missing
            # behavioural features produce neutral z-scores rather than
            # extreme negative values that bias predictions toward FAKE.
            #
            # Account age heuristic: use follower/following count as a
            # proxy for how long the account has existed (lurker accounts
            # may have very few posts but hundreds of followers accumulated
            # over years).  Clamp to [365, 2693].
            "account_age_days":           max(365, min(max(d["post_count"] * 7, d["follower_count"] * 2, d["following_count"] * 2), 2693)),
            "posts_per_day":              d["post_count"] / max(max(365, min(max(d["post_count"] * 7, d["follower_count"] * 2, d["following_count"] * 2), 2693)), 1),
            "username_randomness":        0,
            "caption_similarity_score":   0.29,
            "content_similarity_score":   0.23,
            "follow_unfollow_rate":       68,
            "spam_comments_rate":         15,
            "generic_comment_rate":       58,

            # --- Private only — default empty/unknown ---
            "business_category":          d["business_category"] or "unknown",

            # --- Meta (strip before passing to model) ---
            "_is_partial":                d["is_partial"],
            "_available_fields":          d["available_fields"],
        }
