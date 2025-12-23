"""
Edge Factory training pipeline.
Discovers, validates, and registers trading edges.

This script is the main entry point for the offline ML system.
It should be run periodically (e.g., daily) to discover new edges.

CRITICAL: This system NEVER executes trades.
All output goes to registry/strategies.json for the trader to consume.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List

from trader.app.edge_factory.data.fetch_trades import TradeFetcher
from trader.app.edge_factory.data.datasets import TradeDataset
from trader.app.edge_factory.features.feature_engineering import FeatureEngineer
from trader.app.edge_factory.models.anomaly_iforest import (
    AnomalyDetector,
    AnomalyDetectorConfig,
)
from trader.app.edge_factory.models.meta_labeler import (
    MetaLabeler,
    MetaLabelConfig,
)
from trader.app.edge_factory.backtests.backtest_runner import (
    BacktestRunner,
    BacktestConfig,
)
from trader.app.edge_factory.selection.evaluator import (
    StrategyEvaluator,
    EvaluationConfig,
)
from shared.schemas import StrategyParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_symbols() -> List[str]:
    """Get symbols from environment or defaults."""
    symbols_env = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
    return [s.strip() for s in symbols_env.split(",")]


def get_strategy_params() -> StrategyParameters:
    """Get strategy parameters from environment or defaults."""
    return StrategyParameters(
        entry_z_score=float(os.getenv("ENTRY_Z", "2.0")),
        take_profit_bps=float(os.getenv("TP_BPS", "8")),
        stop_loss_bps=float(os.getenv("SL_BPS", "5")),
        time_stop_seconds=int(os.getenv("TIME_STOP_SEC", "60")),
        min_anomaly_score=float(os.getenv("MIN_ANOMALY_SCORE", "0.7")),
    )


def run_training_pipeline(
    symbols: List[str],
    days: int = 7,
    registry_path: str = "registry/strategies.json",
) -> None:
    """
    Run the full edge discovery pipeline.

    Steps:
    1. Fetch historical data
    2. Calculate features
    3. Train anomaly detector
    4. Meta-label anomalies
    5. Backtest candidate strategies
    6. Evaluate and register passing strategies
    """
    logger.info("=" * 60)
    logger.info("EDGE FACTORY TRAINING PIPELINE")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Days: {days}")
    logger.info(f"Registry: {registry_path}")
    logger.info("=" * 60)

    # Initialize components
    fetcher = TradeFetcher()
    feature_engineer = FeatureEngineer()
    backtest_config = BacktestConfig()
    backtester = BacktestRunner(config=backtest_config)
    evaluator = StrategyEvaluator(registry_path=registry_path)
    params = get_strategy_params()

    # Check for expired strategies
    expired = evaluator.check_expiration()
    if expired:
        logger.info(f"Expired strategies: {expired}")

    # Process each symbol
    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*40}")

        try:
            # Step 1: Fetch data
            logger.info("Step 1: Fetching historical data...")
            df = fetcher.fetch_historical_klines(
                symbol=symbol,
                interval="1",
                days=days,
            )

            if df.empty or len(df) < 1000:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                continue

            logger.info(f"Fetched {len(df)} rows")

            # Step 2: Calculate features
            logger.info("Step 2: Calculating features...")
            featured = feature_engineer.calculate_features(df)
            featured = feature_engineer.normalize_features(featured)

            # Step 3: Create dataset
            dataset = TradeDataset(featured, symbol)

            # Step 4: Run walk-forward backtest
            logger.info("Step 4: Running walk-forward backtest...")
            strategy_name = f"mean_revert_{symbol}_{datetime.utcnow().strftime('%Y%m%d')}"

            result = backtester.run_walk_forward(
                dataset=dataset,
                strategy_name=strategy_name,
                parameters=params,
                anomaly_threshold=params.min_anomaly_score,
            )

            logger.info(f"Backtest result: {result.total_trades} trades, "
                       f"Sharpe: {result.sharpe_ratio:.2f}, "
                       f"Win rate: {result.win_rate:.2%}")

            # Step 5: Evaluate and promote
            logger.info("Step 5: Evaluating strategy...")
            passed, reason, entry = evaluator.evaluate_candidate(
                result=result,
                symbols=[symbol],
            )

            if passed and entry:
                logger.info(f"Strategy PASSED: {reason}")
                evaluator.promote_strategy(entry)
            else:
                logger.warning(f"Strategy REJECTED: {reason}")

            # Step 6: Run regime validation for robustness
            logger.info("Step 6: Regime validation...")
            regime_results = backtester.run_regime_validation(
                dataset=dataset,
                strategy_name=strategy_name,
                parameters=params,
            )

            for regime, regime_result in regime_results.items():
                logger.info(
                    f"  {regime}: trades={regime_result.total_trades}, "
                    f"sharpe={regime_result.sharpe_ratio:.2f}, "
                    f"valid={regime_result.is_valid}"
                )

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    # Print registry summary
    summary = evaluator.get_registry_summary()
    logger.info("\n" + "=" * 60)
    logger.info("REGISTRY SUMMARY")
    logger.info(f"Total strategies: {summary['total_strategies']}")
    logger.info(f"Enabled strategies: {summary['enabled_strategies']}")
    logger.info(f"Average Sharpe: {summary['avg_sharpe']:.2f}")
    logger.info("=" * 60)


def run_performance_review(registry_path: str = "registry/strategies.json") -> None:
    """
    Review performance of existing strategies.
    Check for degradation and disable underperforming strategies.
    """
    logger.info("Running performance review...")

    evaluator = StrategyEvaluator(registry_path=registry_path)

    # Check strategies due for review
    due_for_review = evaluator.check_review_due()
    logger.info(f"Strategies due for review: {due_for_review}")

    # Check expiration
    expired = evaluator.check_expiration()
    logger.info(f"Expired strategies: {expired}")

    # Print summary
    summary = evaluator.get_registry_summary()
    for strategy in summary["strategies"]:
        status = "ENABLED" if strategy["enabled"] else "DISABLED"
        logger.info(
            f"  {strategy['name']}: {status}, "
            f"Sharpe={strategy['sharpe']:.2f}, "
            f"Symbols={strategy['symbols']}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Edge Factory - ML Edge Discovery Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "review"],
        default="train",
        help="Pipeline mode: train (discover edges) or review (check performance)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (overrides env)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of historical data to fetch",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="registry/strategies.json",
        help="Path to strategy registry",
    )

    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_symbols()

    if args.mode == "train":
        run_training_pipeline(
            symbols=symbols,
            days=args.days,
            registry_path=args.registry,
        )
    elif args.mode == "review":
        run_performance_review(registry_path=args.registry)


if __name__ == "__main__":
    main()
