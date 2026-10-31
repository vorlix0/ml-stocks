"""Equal-weight portfolio helpers for multi-ticker (panel) data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def panel_strategy_returns(
    df: pd.DataFrame, proba: np.ndarray, threshold: float
) -> tuple[pd.Series, pd.Series]:
    """
    Daily equal-weight returns of the ML strategy and of Buy & Hold.

    A ticker is held on a day when its previous day's predicted probability
    exceeded ``threshold``; otherwise its slot is held in cash (0% return).

    Args:
        df: Panel with ``Ticker`` and ``Returns`` columns (date index).
        proba: Predicted probabilities aligned with ``df`` rows.
        threshold: Probability above which the stock is held.

    Returns:
        Tuple (strategy daily returns, Buy & Hold daily returns) per date.
    """
    work = pd.DataFrame({
        'Ticker': df['Ticker'].to_numpy(),
        'Returns': df['Returns'].to_numpy(),
        'Signal': (np.asarray(proba) > threshold).astype(int),
    }, index=df.index)
    work['Position'] = work.groupby('Ticker')['Signal'].shift(1).fillna(0)
    work['Strategy'] = work['Position'] * work['Returns']

    grouped = work.groupby(level=0)
    return grouped['Strategy'].mean(), grouped['Returns'].mean()
