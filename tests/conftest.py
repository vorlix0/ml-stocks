"""
Shared fixtures for tests.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Creates sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(200) * 0.02
    close = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': close * (1 + np.random.randn(200) * 0.005),
        'High': close * (1 + np.abs(np.random.randn(200) * 0.01)),
        'Low': close * (1 - np.abs(np.random.randn(200) * 0.01)),
        'Close': close,
        'Volume': np.random.randint(1_000_000, 10_000_000, size=200)
    }, index=dates)
    
    return df


@pytest.fixture
def sample_returns() -> pd.Series:
    """Creates sample returns series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    returns = pd.Series(
        np.random.randn(100) * 0.02,
        index=dates
    )
    return returns


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """Creates sample prediction probabilities."""
    np.random.seed(42)
    return np.random.rand(100)


@pytest.fixture
def sample_df_with_features(sample_ohlcv_data) -> pd.DataFrame:
    """Creates DataFrame with basic features for strategy testing."""
    df = sample_ohlcv_data.copy()
    df['Returns'] = df['Close'].pct_change()
    df['ADX'] = np.random.uniform(15, 40, len(df))
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    return df.dropna()
