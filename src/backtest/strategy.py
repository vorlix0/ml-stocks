"""
Module for defining trading strategies.
"""
import logging

import pandas as pd
import numpy as np

from config import BACKTEST_CONFIG

logger = logging.getLogger("forex_ml.backtest.strategy")


class TradingStrategy:
    """Class for generating trading signals."""
    
    def __init__(self, df: pd.DataFrame, predictions: np.ndarray):
        """
        Initializes strategy.
        
        Args:
            df: DataFrame with data (must contain 'Close', 'ADX', 'Returns')
            predictions: Model prediction probabilities
        """
        self.df = df.copy()
        self.df['Prediction'] = predictions
        
        # Ensure we have Returns
        if 'Returns' not in self.df.columns:
            self.df['Returns'] = self.df['Close'].pct_change()
    
    def generate_signals(
        self, 
        threshold: float = None,
        adx_threshold: int = None,
        use_adx_filter: bool = False
    ) -> pd.Series:
        """
        Generates trading signals.
        
        Args:
            threshold: Probability threshold (default from config)
            adx_threshold: ADX threshold for trend filter (default from config)
            use_adx_filter: Whether to use ADX filter
        
        Returns:
            Series with signals (1 = buy, 0 = hold)
        """
        threshold = threshold or BACKTEST_CONFIG.PREDICTION_THRESHOLD
        adx_threshold = adx_threshold or BACKTEST_CONFIG.ADX_THRESHOLD
        
        if use_adx_filter:
            signals = (
                (self.df['Prediction'] > threshold) & 
                (self.df['ADX'] > adx_threshold)
            ).astype(int)
        else:
            signals = (self.df['Prediction'] > threshold).astype(int)
        
        return signals
    
    def calculate_strategy_returns(self, signals: pd.Series) -> pd.Series:
        """
        Calculates strategy returns based on signals.
        
        Args:
            signals: Series with signals (1 = buy, 0 = hold)
        
        Returns:
            Series with strategy returns
        """
        # Previous day's signal determines position
        return signals.shift(1) * self.df['Returns']
    
    def calculate_cumulative_returns(self, strategy_returns: pd.Series) -> pd.Series:
        """
        Calculates cumulative returns.
        
        Args:
            strategy_returns: Series with strategy returns
        
        Returns:
            Series with cumulative returns
        """
        return (1 + strategy_returns).cumprod()
    
    def get_market_returns(self) -> pd.Series:
        """Returns cumulative market returns (Buy & Hold)."""
        return (1 + self.df['Returns']).cumprod()
    
    def print_signal_stats(self, signals: pd.Series, name: str = "Strategy") -> None:
        """
        Prints signal statistics.
        
        Args:
            signals: Series with signals
            name: Strategy name
        """
        buy_count = (signals == 1).sum()
        hold_count = (signals == 0).sum()
        
        logger.info(f"{'='*60}")
        logger.info(f"SIGNAL DISTRIBUTION ({name}):")
        logger.info(f"Buy (1): {buy_count}, Hold (0): {hold_count}")
        logger.info(f"Min prediction: {self.df['Prediction'].min():.3f}, "
                    f"Max: {self.df['Prediction'].max():.3f}")
        logger.info(f"Mean prediction: {self.df['Prediction'].mean():.3f}")
