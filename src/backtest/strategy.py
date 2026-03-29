"""
Module for defining trading strategies.
"""
import logging

import numpy as np
import pandas as pd

from config import BACKTEST_CONFIG
from src.backtest.signal_filter import ADXFilter, NoFilter, SignalFilter

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
        use_adx_filter: bool = False,
        signal_filter: SignalFilter | None = None,
    ) -> pd.Series:
        """
        Generates trading signals.

        Accepts either a ``SignalFilter`` object (preferred) **or** the legacy
        ``use_adx_filter`` boolean flag for backward compatibility.

        Args:
            threshold: Probability threshold (default from config)
            adx_threshold: ADX threshold used when ``use_adx_filter=True``
                           and no explicit ``signal_filter`` is given.
            use_adx_filter: Legacy flag — creates an ``ADXFilter`` when
                            ``signal_filter`` is not provided.
            signal_filter: A :class:`SignalFilter` instance that decides
                           which signals survive.  When provided,
                           ``use_adx_filter`` and ``adx_threshold`` are
                           ignored.

        Returns:
            Series with signals (1 = buy, 0 = hold)
        """
        threshold = threshold or BACKTEST_CONFIG.PREDICTION_THRESHOLD

        raw_signals = (self.df['Prediction'] > threshold).astype(int)

        # Resolve the filter to use
        if signal_filter is None:
            if use_adx_filter:
                adx_threshold = adx_threshold or BACKTEST_CONFIG.ADX_THRESHOLD
                signal_filter = ADXFilter(threshold=adx_threshold)
            else:
                signal_filter = NoFilter()

        return signal_filter.apply(raw_signals, self.df)

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
