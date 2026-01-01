"""
Configuration for forex-ml-hf project.
All constants and parameters in one place.
"""
from dataclasses import dataclass
from pathlib import Path


# Directory structure
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
CHARTS_DIR = OUTPUTS_DIR / "charts"


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data downloading and processing."""
    TICKER: str = "AAPL"
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2024-12-21"
    
    @property
    def data_file(self) -> Path:
        return RAW_DATA_DIR / f"{self.TICKER}_data.csv"
    
    @property
    def features_file(self) -> Path:
        return PROCESSED_DATA_DIR / f"{self.TICKER}_features.csv"


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering."""
    # Moving Averages windows
    SMA_WINDOWS: tuple = (10, 20, 50, 200)
    EMA_WINDOWS: tuple = (12, 26, 50, 100)
    
    # Rolling windows
    ROLLING_WINDOWS: tuple = (20, 50)
    
    # Momentum/ROC windows
    ROC_WINDOWS: tuple = (5, 10, 20)
    
    # Volatility windows
    VOLATILITY_WINDOWS: tuple = (10, 20, 50)
    
    # Z-score windows
    ZSCORE_WINDOWS: tuple = (10, 20, 50)
    
    # Lag features
    LAG_PERIODS: tuple = (1, 3, 5, 10)
    
    # Target configuration
    TARGET_HORIZON_DAYS: int = 20  # Predict price movement X days ahead
    TARGET_THRESHOLD: float = 0.02  # Return threshold (2%)


@dataclass(frozen=True)
class ModelConfig:
    """ML model configuration."""
    # Train/Test split
    SPLIT_DATE: str = "2023-01-01"
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # GradientBoosting hyperparameters
    N_ESTIMATORS: int = 300
    MAX_DEPTH: int = 5
    LEARNING_RATE: float = 0.05
    SUBSAMPLE: float = 0.8
    
    # Model file
    @property
    def model_file(self) -> Path:
        return MODELS_DIR / "model.joblib"
    
    # Excluded columns from features
    EXCLUDED_COLUMNS: tuple = ('Target', 'Close', 'Open', 'High', 'Low', 'Volume')


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting configuration."""
    # Strategy thresholds
    PREDICTION_THRESHOLD: float = 0.38
    ADX_THRESHOLD: int = 25  # Strong trend indicator
    
    # Initial capital
    INITIAL_CAPITAL: float = 10000.0
    
    # Risk metrics
    TRADING_DAYS_PER_YEAR: int = 252
    RISK_FREE_RATE: float = 0.0
    
    # Output files
    @property
    def backtest_plot_file(self) -> Path:
        return CHARTS_DIR / "backtest.png"
    
    @property
    def portfolio_plot_file(self) -> Path:
        return CHARTS_DIR / "portfolio_value.png"
    
    @property
    def feature_importance_plot_file(self) -> Path:
        return CHARTS_DIR / "feature_importance.png"


# Singleton instances
DATA_CONFIG = DataConfig()
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
BACKTEST_CONFIG = BacktestConfig()
