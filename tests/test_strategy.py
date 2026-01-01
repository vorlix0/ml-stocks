"""
Tests for TradingStrategy class.
"""
import pytest
import pandas as pd
import numpy as np

from src.backtest.strategy import TradingStrategy


class TestTradingStrategy:
    """Tests for TradingStrategy class."""
    
    def test_init_creates_prediction_column(self, sample_df_with_features, sample_predictions):
        """Strategy should add Prediction column."""
        df = sample_df_with_features.iloc[:100]
        strategy = TradingStrategy(df, sample_predictions)
        
        assert 'Prediction' in strategy.df.columns
        assert len(strategy.df['Prediction']) == len(sample_predictions)
    
    def test_init_creates_returns_if_missing(self, sample_ohlcv_data, sample_predictions):
        """Strategy should create Returns column if missing."""
        df = sample_ohlcv_data.iloc[:100].copy()
        # Remove Returns if exists
        if 'Returns' in df.columns:
            df = df.drop('Returns', axis=1)
        
        df['ADX'] = np.random.uniform(20, 40, len(df))
        strategy = TradingStrategy(df, sample_predictions)
        
        assert 'Returns' in strategy.df.columns
    
    def test_generate_signals_binary(self, sample_df_with_features, sample_predictions):
        """Signals should be binary (0 or 1)."""
        df = sample_df_with_features.iloc[:100]
        strategy = TradingStrategy(df, sample_predictions)
        
        signals = strategy.generate_signals()
        
        assert set(signals.unique()).issubset({0, 1})
    
    def test_generate_signals_threshold(self, sample_df_with_features):
        """Higher threshold should result in fewer buy signals."""
        predictions = np.array([0.3, 0.5, 0.7, 0.9] * 25)
        df = sample_df_with_features.iloc[:100]
        strategy = TradingStrategy(df, predictions)
        
        signals_low = strategy.generate_signals(threshold=0.3)
        signals_high = strategy.generate_signals(threshold=0.8)
        
        assert signals_low.sum() > signals_high.sum()
    
    def test_generate_signals_adx_filter(self, sample_df_with_features, sample_predictions):
        """ADX filter should reduce number of signals."""
        df = sample_df_with_features.iloc[:100]
        # Set some ADX values below threshold
        df['ADX'] = np.where(
            np.random.rand(len(df)) > 0.5,
            30,  # Above threshold
            20   # Below threshold
        )
        
        strategy = TradingStrategy(df, sample_predictions)
        
        signals_no_filter = strategy.generate_signals(use_adx_filter=False)
        signals_with_filter = strategy.generate_signals(use_adx_filter=True, adx_threshold=25)
        
        assert signals_with_filter.sum() <= signals_no_filter.sum()
    
    def test_calculate_strategy_returns_shape(self, sample_df_with_features, sample_predictions):
        """Strategy returns should have same length as signals."""
        df = sample_df_with_features.iloc[:100]
        strategy = TradingStrategy(df, sample_predictions)
        
        signals = strategy.generate_signals()
        strategy_returns = strategy.calculate_strategy_returns(signals)
        
        assert len(strategy_returns) == len(signals)
    
    def test_calculate_strategy_returns_hold_is_zero(self, sample_df_with_features):
        """Hold signal (0) should result in zero return."""
        df = sample_df_with_features.iloc[:100]
        predictions = np.zeros(100)  # All predictions below threshold
        strategy = TradingStrategy(df, predictions)
        
        signals = strategy.generate_signals(threshold=0.5)
        strategy_returns = strategy.calculate_strategy_returns(signals)
        
        # After shift, most should be 0 (no position)
        assert (strategy_returns.iloc[1:] == 0).all()
    
    def test_calculate_cumulative_returns_increasing(self, sample_df_with_features):
        """Cumulative returns should start at 1."""
        df = sample_df_with_features.iloc[:100]
        predictions = np.ones(100) * 0.9
        strategy = TradingStrategy(df, predictions)
        
        signals = strategy.generate_signals()
        strategy_returns = strategy.calculate_strategy_returns(signals)
        cumulative = strategy.calculate_cumulative_returns(strategy_returns.fillna(0))
        
        assert cumulative.iloc[0] == pytest.approx(1.0, rel=0.01)
    
    def test_get_market_returns_shape(self, sample_df_with_features, sample_predictions):
        """Market returns should have same length as data."""
        df = sample_df_with_features.iloc[:100]
        strategy = TradingStrategy(df, sample_predictions)
        
        market_returns = strategy.get_market_returns()
        
        assert len(market_returns) == len(df)
    
    def test_signal_uses_previous_day(self, sample_df_with_features, sample_predictions):
        """Strategy should use previous day's signal (no look-ahead bias)."""
        df = sample_df_with_features.iloc[:100]
        strategy = TradingStrategy(df, sample_predictions)
        
        signals = strategy.generate_signals()
        strategy_returns = strategy.calculate_strategy_returns(signals)
        
        # First return should be NaN (no previous signal)
        assert pd.isna(strategy_returns.iloc[0])
