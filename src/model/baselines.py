"""
Simple baselines used to check whether the main model adds any value.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_CONFIG


def create_logistic_baseline() -> ClassifierMixin:
    """Strongly regularized logistic regression on standardized features."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, random_state=MODEL_CONFIG.RANDOM_STATE),
    )


def sma_trend_signals(df: pd.DataFrame, column: str = "Close_to_SMA_200") -> pd.Series:
    """Classic trend rule: long while Close is above its 200-day SMA."""
    return (df[column] > 0).astype(int)
