"""
Signal filter classes implementing the Strategy pattern.

Each concrete filter encapsulates one filtering algorithm, making it easy
to add new filters without modifying TradingStrategy.
"""
from abc import ABC, abstractmethod

import pandas as pd


class SignalFilter(ABC):
    """Abstract base class for trading signal filters."""

    @abstractmethod
    def apply(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Applies the filter to a raw signal series.

        Args:
            signals: Boolean/integer Series of raw buy signals (1 = buy).
            df: DataFrame with market data (must contain any columns the
                filter requires, e.g. 'ADX').

        Returns:
            Filtered signal Series (1 = buy, 0 = hold).
        """


class NoFilter(SignalFilter):
    """Pass-through filter that leaves signals unchanged."""

    def apply(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        return signals


class ADXFilter(SignalFilter):
    """Filter that requires ADX to be above a minimum threshold.

    Only signals where the trend strength indicator (ADX) exceeds
    ``threshold`` are kept; all others are set to 0.

    Args:
        threshold: Minimum ADX value required to keep a signal.
    """

    def __init__(self, threshold: int = 25):
        self.threshold = threshold

    def apply(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Keep signals only when ADX indicates a strong trend."""
        return (signals.astype(bool) & (df['ADX'] > self.threshold)).astype(int)
