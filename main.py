"""
main.py — CLI Entry Point for the Fake Account Detection System
Usage:
  python main.py                    # Train
  python main.py --no-bert          # Train without BERT (TF-IDF proxy)
  python main.py --optuna           # Train with Optuna tuning
  python main.py --predict          # Predict on test demo profiles
  python main.py --demo             # Demo with synthetic profiles
  python main.py --scan-self --token TOKEN
  python main.py --instagram u1,u2 --token TOKEN
  python main.py --realtime u1,u2 --token TOKEN --interval 30
  python main.py --explain-pca
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fake Account Detection System v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data paths
    p.add_argument("--primary",     default="data/fake_social_media.csv")
    p.add_argument("--fake-users",  default="data/fake_users.csv")
    p.add_argument("--limfadd",     default="data/LIMFADD.csv")
    p.add_argument("--excel",       default="data/fake_social_media_global_2.0_with_missing.xlsx")

    # Training flags
    p.add_argument("--no-bert",  action="store_true",
                   help="Use TF-IDF proxy instead of BERT embeddings")
    p.add_argument("--no-balance", action="store_true",
                   help="Skip SMOTEENN class balancing")
    p.add_argument("--gnn",     action="store_true",
                   help="Enable GNN graph embeddings (requires torch_geometric)")
    p.add_argument("--optuna",  action="store_true",
                   help="Run Optuna hyperparameter tuning for XGBoost")
    p.add_argument("--optuna-trials", type=int, default=30)
    p.add_argument("--save",    default=None,
                   help="Custom pipeline save path")

    # Inference modes
    p.add_argument("--predict",      action="store_true",
                   help="Run predictions on synthetic demo profiles")
    p.add_argument("--demo",         action="store_true",
                   help="Demo scan with synthetic profiles")
    p.add_argument("--scan-self",    action="store_true")
    p.add_argument("--instagram",    type=str, default=None,
                   help="Comma-separated Instagram usernames to scan")
    p.add_argument("--token",        type=str, default=None,
                   help="Instagram/Facebook API token")
    p.add_argument("--realtime",     type=str, default=None,
                   help="Comma-separated usernames for continuous monitoring")
    p.add_argument("--interval",     type=int, default=60,
                   help="Monitoring interval in minutes (default: 60)")
    p.add_argument("--explain-pca",  action="store_true",
                   help="Print PCA component interpretability report")

    return p


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Determine if we need to train or load
    inference_modes = any([
        args.predict, args.demo, args.scan_self,
        args.instagram, args.realtime, args.explain_pca,
    ])

    if not inference_modes:
        # ── TRAINING ─────────────────────────────────────────────────────────
        logger.info("Starting training pipeline …")
        from src.pipeline import train_pipeline
        pipeline = train_pipeline(
            primary_path=args.primary,
            fake_users_path=args.fake_users,
            limfadd_path=args.limfadd,
            excel_path=args.excel,
            use_bert=not args.no_bert,
            use_gnn=args.gnn,
            use_optuna=args.optuna,
            optuna_trials=args.optuna_trials,
            balance=not args.no_balance,
            save_path=args.save,
        )
        logger.info("Training complete. Metrics: %s", pipeline.get("metrics", {}))
        return

    # ── Load saved pipeline for inference ────────────────────────────────────
    from src.pipeline import load_pipeline
    try:
        pipeline = load_pipeline(args.save)
    except FileNotFoundError:
        logger.error("No saved pipeline found. Run training first (without inference flags).")
        sys.exit(1)

    # ── Demo / Predict ────────────────────────────────────────────────────────
    if args.demo or args.predict:
        from src.instagram_api import create_demo_profiles
        from src.pipeline import predict
        df = create_demo_profiles()
        results = predict(df, pipeline=pipeline)
        print("\n" + "=" * 60)
        print("DEMO RESULTS")
        print("=" * 60)
        for _, row in results.iterrows():
            print(f"  @{row['username']:25s}  {row['label']:8s}  "
                  f"P={row['probability']:.3f}  [{row['risk_band']}]")
        print()

    # ── Explain PCA ───────────────────────────────────────────────────────────
    if args.explain_pca:
        interp = pipeline.get("pca_interpreter")
        if interp:
            print(interp.component_report(n_top=5))
        else:
            logger.warning("PCA interpreter not found in pipeline.")

    # ── Self scan ─────────────────────────────────────────────────────────────
    if args.scan_self:
        if not args.token:
            logger.error("--token required for --scan-self")
            sys.exit(1)
        from src.instagram_api import InstagramAPIClient
        client = InstagramAPIClient(args.token)
        results = client.demo_analyse(pipeline)
        _print_results(results)

    # ── Username scan ─────────────────────────────────────────────────────────
    if args.instagram:
        if not args.token:
            logger.error("--token required for --instagram")
            sys.exit(1)
        from src.instagram_api import InstagramAPIClient
        client = InstagramAPIClient(args.token)
        usernames = [u.strip() for u in args.instagram.split(",")]
        results = client.fetch_and_analyse(usernames, pipeline)
        _print_results(results)

    # ── Real-time monitoring ──────────────────────────────────────────────────
    if args.realtime:
        if not args.token:
            logger.error("--token required for --realtime")
            sys.exit(1)
        from src.realtime_monitor import RealtimeMonitor
        monitor = RealtimeMonitor(pipeline, api_token=args.token)
        usernames = [u.strip() for u in args.realtime.split(",")]
        logger.info("Starting continuous monitoring (interval=%d min) …", args.interval)
        logger.info("Press Ctrl+C to stop.")
        monitor.start_continuous(usernames, interval_minutes=args.interval)
        try:
            import time
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            monitor.stop()
            print("\n" + monitor.generate_report())


def _print_results(results):
    print("\n" + "=" * 60)
    for _, row in results.iterrows():
        print(f"  @{row['username']:25s}  {row['label']:8s}  "
              f"P={row['probability']:.3f}  [{row['risk_band']}]")
        print(f"  {row['explanation'][:90]}…")
        print()


if __name__ == "__main__":
    main()
