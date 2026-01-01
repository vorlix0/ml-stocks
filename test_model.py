# test_model.py (backtest)
"""Script for ML strategy backtesting."""

import sys
import pandas as pd

from config import DATA_CONFIG, MODEL_CONFIG, BACKTEST_CONFIG
from src.model import ModelTrainer
from src.backtest import TradingStrategy, TradingSimulator, RiskMetrics
from src.utils import Plotter, load_csv_safe, validate_features_data
from src.exceptions import ForexMLError, DataNotFoundError, ModelNotFoundError


def load_test_data() -> tuple:
    """
    Loads test data and model.
    
    Returns:
        Tuple (model, df_test, feature_cols)
    
    Raises:
        ModelNotFoundError: When model doesn't exist
        DataNotFoundError: When data doesn't exist
    """
    model = ModelTrainer.load_model()
    df = load_csv_safe(DATA_CONFIG.features_file)
    validate_features_data(df)
    
    split_date = MODEL_CONFIG.SPLIT_DATE
    feature_cols = [
        col for col in df.columns 
        if col not in MODEL_CONFIG.EXCLUDED_COLUMNS
    ]
    
    df_test = df[df.index >= split_date].copy()
    X_test = df_test[feature_cols]
    
    return model, df_test, X_test


def run_backtest():
    """Main function for backtesting."""
    # Load data
    model, df_test, X_test = load_test_data()
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Strategy
    strategy = TradingStrategy(df_test, y_pred_proba)
    
    # Generate signals - variant without ADX filter
    signals_no_filter = strategy.generate_signals(use_adx_filter=False)
    strategy.print_signal_stats(signals_no_filter, "without ADX filter")
    
    # Generate signals - variant with ADX filter
    signals_adx = strategy.generate_signals(use_adx_filter=True)
    strategy.print_signal_stats(signals_adx, f"with ADX filter > {BACKTEST_CONFIG.ADX_THRESHOLD}")
    
    # ADX stats
    adx_threshold = BACKTEST_CONFIG.ADX_THRESHOLD
    adx_stats = df_test[df_test['ADX'] > adx_threshold]
    print(f"\nDays with strong trend (ADX > {adx_threshold}): "
          f"{len(adx_stats)} ({len(adx_stats)/len(df_test)*100:.1f}%)")
    print(f"ADX: min={df_test['ADX'].min():.1f}, "
          f"max={df_test['ADX'].max():.1f}, "
          f"mean={df_test['ADX'].mean():.1f}")
    
    # Calculate returns
    returns_no_filter = strategy.calculate_strategy_returns(signals_no_filter)
    returns_adx = strategy.calculate_strategy_returns(signals_adx)
    
    cum_market = strategy.get_market_returns()
    cum_no_filter = strategy.calculate_cumulative_returns(returns_no_filter)
    cum_adx = strategy.calculate_cumulative_returns(returns_adx)
    
    print(f"\nNaN in Strategy_Returns: {returns_no_filter.isna().sum()}")
    print(f"Mean Strategy_Returns (without filter): {returns_no_filter.mean():.5f}")
    print(f"Mean Strategy_Returns (with filter): {returns_adx.mean():.5f}")
    
    # Returns summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Market Return: {cum_market.iloc[-1] - 1:.2%}")
    print(f"Strategy Return (without filter): {cum_no_filter.iloc[-1] - 1:.2%}")
    print(f"Strategy Return (with ADX filter): {cum_adx.iloc[-1] - 1:.2%}")
    print(f"\nOutperformance (without filter): {(cum_no_filter.iloc[-1] - cum_market.iloc[-1]):.2%}")
    print(f"Outperformance (with ADX filter): {(cum_adx.iloc[-1] - cum_market.iloc[-1]):.2%}")
    
    # Simulation with capital
    print(f"\n{'='*60}")
    print(f"TRADING SIMULATION WITH {BACKTEST_CONFIG.INITIAL_CAPITAL:,.0f} USD")
    print(f"{'='*60}")
    
    simulator = TradingSimulator(df_test)
    
    results_market = simulator.simulate_buy_and_hold()
    results_no_filter = simulator.simulate(signals_no_filter)
    results_adx = simulator.simulate(signals_adx)
    
    simulator.print_simulation_results(results_market, "📊 Buy & Hold")
    simulator.print_simulation_results(results_no_filter, "🤖 ML Strategy WITHOUT ADX filter")
    simulator.print_simulation_results(results_adx, f"🤖 ML Strategy WITH ADX filter (>{BACKTEST_CONFIG.ADX_THRESHOLD})")
    
    simulator.compare_strategies(
        [results_market, results_no_filter, results_adx],
        ["Buy & Hold", "No filter", "ADX filter"]
    )
    
    # Risk metrics
    metrics_market = RiskMetrics(df_test['Returns'], cum_market)
    metrics_no_filter = RiskMetrics(returns_no_filter, cum_no_filter)
    metrics_adx = RiskMetrics(returns_adx, cum_adx)
    
    RiskMetrics.print_comparison(
        [metrics_market, metrics_no_filter, metrics_adx],
        ["Market", "without filter", "with ADX filter"]
    )
    
    # Charts
    Plotter.plot_backtest(
        df_test.index,
        {
            'Buy & Hold': cum_market,
            'ML Strategy (without filter)': cum_no_filter,
            'ML Strategy (with ADX filter)': cum_adx
        },
        save_path=BACKTEST_CONFIG.backtest_plot_file
    )
    
    Plotter.plot_portfolio_with_adx(
        df_test.index,
        {
            'Buy & Hold': cum_market,
            'ML Strategy (without filter)': cum_no_filter,
            'ML Strategy (with ADX filter)': cum_adx
        },
        df_test['ADX'],
        save_path=BACKTEST_CONFIG.portfolio_plot_file
    )
    
    print(f"\n{'='*60}")
    print(f"Charts saved: {BACKTEST_CONFIG.backtest_plot_file}, "
          f"{BACKTEST_CONFIG.portfolio_plot_file}")
    print(f"{'='*60}")


def main():
    """Main function with error handling."""
    try:
        run_backtest()
    except ModelNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run first: python train_model.py")
        sys.exit(1)
    except DataNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run first: python process_data.py")
        sys.exit(1)
    except ForexMLError as e:
        print(f"\n❌ BŁĄD: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Nieoczekiwany błąd: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
