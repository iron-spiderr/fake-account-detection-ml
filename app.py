"""
Process 15 — Flask Web Application
Provides a browser-based interface for the Fake Account Detection System.
"""

import argparse
import logging

import pandas as pd
from flask import Flask, jsonify, render_template, request, session

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "fakedetect_secret_2026"

# Global pipeline (loaded once at startup)
_pipeline: dict | None = None

# Global scraper (lazy-init so missing instaloader doesn't break startup)
_scraper = None


def _get_scraper():
    global _scraper
    if _scraper is None:
        try:
            from scraper import InstagramScraper
            _scraper = InstagramScraper(delay_min=2.0, delay_max=4.0)
            logger.info("Instagram scraper ready.")
        except ImportError:
            logger.warning("instaloader not installed — scraper unavailable.")
    return _scraper


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from modules910 import load_pipeline
            _pipeline = load_pipeline()
            logger.info("Pipeline loaded successfully.")
        except FileNotFoundError:
            logger.warning("Pipeline not found — running in demo-only mode.")
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    pipeline = _get_pipeline()
    history = session.get("history", [])
    return jsonify({
        "ok": True,
        "pipeline_ready": pipeline is not None,
        "history_count": len(history),
    })


@app.route("/api/demo", methods=["POST"])
def api_demo():
    try:
        from instagram_api import create_demo_profiles
        from modules910 import predict
        pipeline = _get_pipeline()
        if pipeline is None:
            return jsonify({"ok": False, "error": "Pipeline not trained yet."}), 503
        df = create_demo_profiles()
        ground_truth = df["is_fake"].tolist() if "is_fake" in df.columns else None
        results_df = predict(df, pipeline=pipeline)
        results = results_df.to_dict(orient="records")

        if ground_truth is not None:
            for i, r in enumerate(results):
                r["ground_truth"] = "FAKE" if ground_truth[i] == 1 else "GENUINE"
                r["correct"] = r["label"] == r["ground_truth"]

        total   = len(results)
        correct = sum(1 for r in results if r.get("correct", False))
        tp = sum(1 for r in results if r.get("ground_truth") == "FAKE"    and r["label"] == "FAKE")
        fp = sum(1 for r in results if r.get("ground_truth") == "GENUINE" and r["label"] == "FAKE")
        fn = sum(1 for r in results if r.get("ground_truth") == "FAKE"    and r["label"] == "GENUINE")
        precision = round(tp / max(tp + fp, 1) * 100, 1)
        recall    = round(tp / max(tp + fn, 1) * 100, 1)
        f1        = round(2 * precision * recall / max(precision + recall, 0.001), 1)
        accuracy  = round(correct / total * 100, 1) if total else 0

        _add_to_history("demo", results)
        return jsonify({
            "ok": True,
            "results": results,
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        })
    except Exception as exc:
        import traceback
        logger.error("Demo failed:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/demo-test", methods=["POST"])
def api_demo_test():
    """Run predictions on the synthetic test_profiles.csv and return with ground truth."""
    import os
    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({"ok": False, "error": "Pipeline not trained yet."}), 503

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_profiles.csv")
    if not os.path.exists(csv_path):
        return jsonify({
            "ok": False,
            "error": "test_profiles.csv not found. Run: python generate_test_data.py"
        }), 404

    try:
        df = pd.read_csv(csv_path)
        ground_truth = df["is_fake"].tolist() if "is_fake" in df.columns else None

        from modules910 import predict
        results_df = predict(df, pipeline=pipeline)
        results = results_df.to_dict(orient="records")

        # Attach ground truth to each result row
        if ground_truth is not None:
            for i, r in enumerate(results):
                r["ground_truth"] = "FAKE" if ground_truth[i] == 1 else "GENUINE"
                r["correct"] = r["label"] == r["ground_truth"]

        # Compute summary metrics
        total     = len(results)
        correct   = sum(1 for r in results if r.get("correct", False))
        tp = sum(1 for r in results if r.get("ground_truth") == "FAKE"    and r["label"] == "FAKE")
        fp = sum(1 for r in results if r.get("ground_truth") == "GENUINE" and r["label"] == "FAKE")
        fn = sum(1 for r in results if r.get("ground_truth") == "FAKE"    and r["label"] == "GENUINE")
        precision = round(tp / max(tp + fp, 1) * 100, 1)
        recall    = round(tp / max(tp + fn, 1) * 100, 1)
        f1        = round(2 * precision * recall / max(precision + recall, 0.001), 1)
        accuracy  = round(correct / total * 100, 1) if total else 0

        _add_to_history("demo-test", results)
        return jsonify({
            "ok": True,
            "results": results,
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        })
    except Exception as exc:
        logger.exception("demo-test failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/scan-self", methods=["POST"])
def api_scan_self():
    body = request.get_json(force=True)
    token = body.get("token", "").strip()
    if not token:
        # Fall back to the token stored in .env / environment variable
        import os
        token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "").strip()
    if not token or token == "PASTE_YOUR_TOKEN_HERE":
        return jsonify({"ok": False, "error": "Token required. Set INSTAGRAM_ACCESS_TOKEN in .env or provide it in the request."}), 400
    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({"ok": False, "error": "Pipeline not ready."}), 503
    try:
        from instagram_api import InstagramAPIClient
        client = InstagramAPIClient(token)
        profile = client.get_own_profile()
        media = client.get_user_media(profile.get("id", ""))
        df = client.profile_to_dataframe(profile, media)
        from modules910 import predict
        results_df = predict(df, pipeline=pipeline)
        results = results_df.to_dict(orient="records")
        _add_to_history("self", results)
        return jsonify({"ok": True, "results": results})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/scan-users", methods=["POST"])
def api_scan_users():
    body = request.get_json(force=True)
    token = body.get("token", "").strip()
    usernames_raw = body.get("usernames", "")
    usernames = [u.strip().lstrip("@") for u in usernames_raw.split(",") if u.strip()]
    if not token:
        import os
        token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "").strip()
    if not token or token == "PASTE_YOUR_TOKEN_HERE" or not usernames:
        return jsonify({"ok": False, "error": "Token and usernames required. Set INSTAGRAM_ACCESS_TOKEN in .env or provide it in the request."}), 400
    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({"ok": False, "error": "Pipeline not ready."}), 503
    try:
        from instagram_api import InstagramAPIClient
        client = InstagramAPIClient(token)
        results_df = client.fetch_and_analyse(usernames, pipeline)
        results = results_df.to_dict(orient="records")
        _add_to_history("username", results)
        return jsonify({"ok": True, "results": results})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    """Scrape a real Instagram profile and run fake-account prediction."""
    body = request.get_json(force=True)
    username = body.get("username", "").strip().lstrip("@")
    if not username:
        return jsonify({"ok": False, "error": "username is required"}), 400

    scraper = _get_scraper()
    if scraper is None:
        return jsonify({"ok": False, "error": "Scraper unavailable. Run: pip install instaloader"}), 503

    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({"ok": False, "error": "Pipeline not trained yet."}), 503

    # 1. Scrape
    try:
        profile_data = scraper.scrape_profile(username)
    except Exception as exc:          # catches RateLimitError + anything else
        from scraper import RateLimitError
        if isinstance(exc, RateLimitError):
            return jsonify({
                "ok": False,
                "error": str(exc),
                "rate_limited": True,
            }), 429
        logger.exception("Scrape failed for @%s", username)
        return jsonify({"ok": False, "error": str(exc)}), 500

    if profile_data is None:
        return jsonify({
            "ok": False,
            "error": f"Could not scrape @{username}. Profile may not exist or is rate-limited."
        }), 404

    # 2. Build prediction dict and strip meta fields
    raw = scraper.profile_to_prediction_dict(profile_data)
    is_partial      = raw.pop("_is_partial", False)
    available_fields = raw.pop("_available_fields", [])

    # 3. Predict
    try:
        from modules910 import predict
        df = pd.DataFrame([raw])
        results_df = predict(df, pipeline=pipeline)
        results = results_df.to_dict(orient="records")
    except Exception as exc:
        logger.exception("Prediction failed for @%s", username)
        return jsonify({"ok": False, "error": str(exc)}), 500

    # 4. Recalibrate: scraping cannot observe behavioural features
    #    (engagement history, follow/unfollow patterns, comment quality,
    #    caption similarity, etc.).  Use Bayesian shrinkage with
    #    profile-aware genuineness heuristics instead of flat bias.
    profile_info = {
        "followers": profile_data.follower_count,
        "following": profile_data.following_count,
        "posts": profile_data.post_count,
        "has_profile_pic": profile_data.has_profile_pic,
        "is_verified": profile_data.is_verified,
        "biography": profile_data.biography,
    }
    results = _recalibrate_for_incomplete_data(results, profile_info=profile_info)

    _add_to_history("scrape", results)

    response = {
        "ok": True,
        "username": username,
        "is_private": profile_data.is_private,
        "results": results,
        "profile": {
            "followers":  profile_data.follower_count,
            "following":  profile_data.following_count,
            "posts":      profile_data.post_count,
            "biography":  profile_data.biography,
            "is_verified": profile_data.is_verified,
            "has_profile_pic": profile_data.has_profile_pic,
            "external_url": profile_data.external_url,
        },
    }
    if is_partial:
        response["warning"] = (
            "Private profile — prediction uses partial data: "
            + ", ".join(available_fields)
        )
    return jsonify(response)


@app.route("/api/manual", methods=["POST"])
def api_manual():
    body = request.get_json(force=True)
    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({"ok": False, "error": "Pipeline not ready."}), 503
    try:
        # For unobserved behavioral features we use the training genuine-class
        # medians so that unknown values produce a neutral z-score rather than
        # an extreme negative one that biases every manual entry toward FAKE.
        # Medians derived from fake_social_media.csv + LIMFADD genuine subset:
        #   follow_unfollow_rate genuine median = 68  (min=27, so 0 is out-of-dist)
        #   generic_comment_rate genuine median = 58  (min=4,  so 0 is out-of-dist)
        #   spam_comments_rate   genuine median = 15
        #   caption_similarity_score genuine mean = 0.29
        bio = body.get("biography", "") or ""
        followers = int(body.get("followers", 0))
        following = int(body.get("following", 0))
        posts     = int(body.get("posts", 0))
        has_link  = int(bool(body.get("website", ""))) or int(
            "http" in bio.lower() or "bit.ly" in bio.lower())

        # Estimate a plausible account age using follower/following counts
        # as a proxy (lurker accounts may have very few posts but hundreds
        # of followers gathered over years).  Clamp to [365, 2693].
        estimated_age = max(365, min(max(posts * 7, followers * 2, following * 2), 2693))

        df = pd.DataFrame([{
            "username":                 body.get("username", "manual_input"),
            "bio":                      bio,
            "has_profile_pic":          int(bool(body.get("has_profile_pic", False))),
            "bio_length":               len(bio),
            "username_randomness":      0,
            "followers":                followers,
            "following":                following,
            "follower_following_ratio": followers / max(following, 1),
            "account_age_days":         estimated_age,
            "posts":                    posts,
            "posts_per_day":            posts / max(estimated_age, 1),
            # Use genuine medians for latent engagement features not in the form
            "caption_similarity_score": 0.29,
            "content_similarity_score": 0.23,
            "follow_unfollow_rate":     68,
            "spam_comments_rate":       15,
            "generic_comment_rate":     58,
            "suspicious_links_in_bio":  has_link,
            "verified":                 int(bool(body.get("verified", False))),
        }])

        from modules910 import predict
        results_df = predict(df, pipeline=pipeline)
        results = results_df.to_dict(orient="records")

        # Recalibrate with profile-aware shrinkage.
        profile_info = {
            "followers": followers,
            "following": following,
            "posts": posts,
            "has_profile_pic": bool(body.get("has_profile_pic", False)),
            "is_verified": bool(body.get("verified", False)),
            "biography": bio,
        }
        results = _recalibrate_for_incomplete_data(results, profile_info=profile_info)

        _add_to_history("manual", results)
        return jsonify({"ok": True, "results": results})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/history")
def api_history():
    return jsonify({"ok": True, "history": session.get("history", [])})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    session["history"] = []
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recalibrate_for_incomplete_data(results: list,
                                      profile_info: dict | None = None) -> list:
    """
    When behavioural features (engagement history, follow/unfollow rate,
    comment quality, caption similarity, etc.) are unavailable — as in
    scrape and manual-input modes — the model systematically over-
    estimates P(fake) because it was trained with those features present.

    Two-stage recalibration:
      1. **Signal shrinkage**: metadata features alone carry only ~20% of the
         model's discriminative power.  We shrink the raw probability toward
         the prior (0.5) accordingly:
           adj_p = 0.5 + signal_weight × (raw_p − 0.5)
      2. **Profile genuineness correction**: observable profile characteristics
         that are consistent with genuine behaviour (healthy follower ratio,
         has profile pic, lurker pattern, etc.) apply a direct downward
         adjustment on P(fake).

    This replaces the old flat bias=0.38 subtraction, which over-corrected
    some accounts and under-corrected others.
    """
    from module11_output import generate_explanation, _risk_band

    # Stage 1: signal weight — how much discriminative power metadata provides
    signal_weight = 0.20

    # Stage 2: profile-based genuineness adjustment
    genuineness_bonus = 0.0
    if profile_info:
        followers = profile_info.get("followers", 0)
        following = profile_info.get("following", 0)
        posts = profile_info.get("posts", 0)
        has_pic = profile_info.get("has_profile_pic", False)
        is_verified = profile_info.get("is_verified", False)
        bio = profile_info.get("biography", "")

        ratio = followers / max(following, 1)
        if 0.3 <= ratio <= 5.0:
            genuineness_bonus += 0.04  # healthy follower/following ratio
        if followers > 100:
            genuineness_bonus += 0.03  # established audience
        if followers > 500:
            genuineness_bonus += 0.02  # larger established audience
        if has_pic:
            genuineness_bonus += 0.02
        if is_verified:
            genuineness_bonus += 0.08  # strong genuine signal
        if posts == 0 and followers > 50:
            # Lurker pattern — very common for genuine users who consume
            # but don't create content.
            genuineness_bonus += 0.05
        if 0 < len(bio) < 200:
            genuineness_bonus += 0.01  # has bio, not excessively long

    # Compute effective data completeness for reporting
    completeness = round(min(0.55, 0.35 + genuineness_bonus), 2)

    for r in results:
        raw_p = r["probability"]

        # Stage 1: Shrink toward prior (0.5) based on available signal
        adj_p = 0.5 + signal_weight * (raw_p - 0.5)

        # Stage 2: Apply genuineness correction
        adj_p -= genuineness_bonus

        adj_p = round(max(0.01, min(0.99, adj_p)), 4)

        label = "FAKE" if adj_p >= 0.5 else "GENUINE"
        band  = _risk_band(adj_p)
        conf  = f"{max(adj_p, 1 - adj_p) * 100:.1f}%"

        # Regenerate explanation with corrected values
        shap_vals = r.get("top_shap_values")
        explanation = generate_explanation(label, adj_p, band, shap_vals)

        r["raw_model_probability"] = raw_p
        r["probability"] = adj_p
        r["label"]       = label
        r["risk_band"]   = band
        r["confidence"]  = conf
        r["explanation"] = explanation
        r["data_completeness"] = completeness

    return results


def _add_to_history(mode: str, results: list):
    from datetime import datetime
    history = session.get("history", [])
    history.append({
        "timestamp": datetime.now().isoformat()[:19],
        "mode": mode,
        "count": len(results),
        "results": results,
    })
    session["history"] = history[-50:]  # keep last 50 entries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fake Account Detection Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Pre-load pipeline
    _get_pipeline()
    app.run(host=args.host, port=args.port, debug=args.debug)
