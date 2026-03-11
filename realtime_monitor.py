"""
Process 14 — Real-Time Monitoring Daemon
Continuously monitors Instagram accounts on a schedule and emits alerts.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class AlertHandler:
    """Creates and logs alerts for suspicious accounts."""

    ALERT_DIR = "alerts"

    def __init__(self):
        os.makedirs(self.ALERT_DIR, exist_ok=True)

    def handle(self, result: dict, timestamp: str):
        level = result.get("risk_band", "LOW")
        username = result.get("username", "?")
        prob = result.get("probability", 0)

        if level in ("CRITICAL", "HIGH"):
            logger.warning("⚠️  %s ALERT: %s (prob=%.2f)", level, username, prob)
            self._save(result, timestamp, level)
        elif level == "MEDIUM":
            logger.info("ℹ️  MEDIUM ALERT: %s (prob=%.2f)", username, prob)
            self._save(result, timestamp, level)

    def _save(self, result: dict, timestamp: str, level: str):
        fname = os.path.join(
            self.ALERT_DIR,
            f"alert_{level}_{result.get('username', 'unknown')}_{timestamp[:10]}.json")
        payload = {**result, "timestamp": timestamp, "alert_level": level}
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)


class RealtimeMonitor:
    """Scan accounts on demand or on a repeating schedule."""

    def __init__(self, pipeline: dict, api_token: str | None = None):
        self.pipeline = pipeline
        self.token = api_token
        self.alert_handler = AlertHandler()
        self._history: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -----------------------------------------------------------------------
    # One-shot scan
    # -----------------------------------------------------------------------

    def scan(self, usernames: list[str]) -> pd.DataFrame:
        from instagram_api import InstagramAPIClient
        if not self.token:
            raise ValueError("API token required for live scanning.")
        client = InstagramAPIClient(self.token)
        results = client.fetch_and_analyse(usernames, self.pipeline)
        self._record(results, "live")
        return results

    def scan_demo(self) -> pd.DataFrame:
        from instagram_api import create_demo_profiles
        from modules910 import predict
        df = create_demo_profiles()
        results = predict(df, pipeline=self.pipeline)
        self._record(results, "demo")
        return results

    # -----------------------------------------------------------------------
    # Continuous monitoring
    # -----------------------------------------------------------------------

    def start_continuous(self, usernames: list[str],
                          interval_minutes: int = 60):
        """Start a background thread that scans every N minutes."""
        if self._thread and self._thread.is_alive():
            logger.warning("Monitor already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(usernames, interval_minutes),
            daemon=True)
        self._thread.start()
        logger.info("Continuous monitoring started (interval=%d min).", interval_minutes)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Monitor stopped.")

    def _loop(self, usernames: list[str], interval_minutes: int):
        while not self._stop_event.is_set():
            try:
                self.scan(usernames)
            except Exception as exc:
                logger.error("Scan error: %s", exc)
            self._stop_event.wait(interval_minutes * 60)

    # -----------------------------------------------------------------------
    # History & reports
    # -----------------------------------------------------------------------

    def _record(self, results: pd.DataFrame, mode: str):
        ts = datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "mode": mode,
            "count": len(results),
            "results": results.to_dict(orient="records"),
        }
        self._history.append(entry)
        for row in results.to_dict(orient="records"):
            self.alert_handler.handle(row, ts)

    def get_history(self) -> list[dict]:
        return self._history

    def generate_report(self) -> str:
        lines = ["Real-Time Monitor Report", "=" * 50]
        total_scanned = sum(e["count"] for e in self._history)
        total_fake = sum(
            sum(1 for r in e["results"] if r.get("label") == "FAKE")
            for e in self._history)
        lines.append(f"Total scans: {len(self._history)}")
        lines.append(f"Total accounts scanned: {total_scanned}")
        lines.append(f"Fake detected: {total_fake}")
        for entry in self._history[-10:]:
            lines.append(f"\n[{entry['timestamp'][:19]}] Mode={entry['mode']} n={entry['count']}")
            for r in entry["results"]:
                lines.append(
                    f"  @{r.get('username','?')} → {r.get('label','?')} "
                    f"({r.get('probability', 0)*100:.1f}%) [{r.get('risk_band','?')}]")
        return "\n".join(lines)
