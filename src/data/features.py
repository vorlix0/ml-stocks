"""
Module for creating features from OHLCV data.
Each feature category has a separate method for better readability and testability.
"""
import pandas as pd
import numpy as np
import ta

from config import FEATURE_CONFIG
from src.exceptions import InvalidDataError, EmptyDataError


class FeatureEngineer:
    """Class for creating features from OHLCV data."""
    
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def __init__(self, df: pd.DataFrame):
        """
        Initializes FeatureEngineer.
        
        Args:
            df: DataFrame with OHLCV data (Close, Open, High, Low, Volume)
        
        Raises:
            InvalidDataError: When required columns are missing
            EmptyDataError: When DataFrame is empty
        """
        self._validate_input(df)
        self.df = df.copy()
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validates input data.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            EmptyDataError: When DataFrame is empty
            InvalidDataError: When required OHLCV columns are missing
        """
        if df.empty:
            raise EmptyDataError("Input DataFrame is empty")
        
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise InvalidDataError(
                f"Missing OHLCV columns: {missing_cols}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Creates all features and returns DataFrame.
        
        Returns:
            DataFrame with all features
        """
        self._add_basic_features()
        self._add_moving_averages()
        self._add_rolling_high_low()
        self._add_ema_features()
        self._add_bollinger_bands()
        self._add_momentum_indicators()
        self._add_trend_indicators()
        self._add_volatility_features()
        self._add_volume_features()
        self._add_rate_of_change()
        self._add_zscore_features()
        self._add_gap_and_range()
        self._add_ema_spreads()
        self._add_cyclical_features()
        self._add_lag_features()
        self._add_trend_signals()
        self._add_feature_interactions()
        self._add_target()
        
        # Drop NaN
        self.df = self.df.dropna()
        
        return self.df
    
    def _add_basic_features(self) -> None:
        """
        Adds basic features: returns, spread, price change.
        
        Creates columns: Returns, High_Low, Price_Change
        """
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['High_Low'] = self.df['High'] - self.df['Low']
        self.df['Price_Change'] = self.df['Close'] - self.df['Open']
    
    def _add_moving_averages(self) -> None:
        """
        Adds Simple Moving Averages.
        
        Creates columns: SMA_{window} for each window in FEATURE_CONFIG.SMA_WINDOWS
        """
        for window in FEATURE_CONFIG.SMA_WINDOWS:
            self.df[f'SMA_{window}'] = self.df['Close'].rolling(window=window).mean()
    
    def _add_rolling_high_low(self) -> None:
        """
        Adds rolling high/low for trend context.
        
        Creates columns: RollingHigh_{window}, RollingLow_{window}
        """
        for window in FEATURE_CONFIG.ROLLING_WINDOWS:
            self.df[f'RollingHigh_{window}'] = self.df['High'].rolling(window=window).max()
            self.df[f'RollingLow_{window}'] = self.df['Low'].rolling(window=window).min()
    
    def _add_ema_features(self) -> None:
        """
        Adds Exponential Moving Averages.
        
        Creates columns: EMA_{window} for each window in FEATURE_CONFIG.EMA_WINDOWS
        """
        for window in FEATURE_CONFIG.EMA_WINDOWS:
            self.df[f'EMA_{window}'] = ta.trend.EMAIndicator(
                self.df['Close'], window=window
            ).ema_indicator()
    
    def _add_bollinger_bands(self) -> None:
        """
        Adds Bollinger Bands.
        
        Creates columns: BB_high, BB_low, BB_width
        """
        bb = ta.volatility.BollingerBands(self.df['Close'])
        self.df['BB_high'] = bb.bollinger_hband()
        self.df['BB_low'] = bb.bollinger_lband()
        self.df['BB_width'] = self.df['BB_high'] - self.df['BB_low']
    
    def _add_momentum_indicators(self) -> None:
        """
        Adds RSI, Stochastic Oscillator, MACD.
        
        Creates columns: RSI, Stoch_K, Stoch_D, MACD, MACD_signal
        """
        # RSI
        self.df['RSI'] = ta.momentum.RSIIndicator(self.df['Close']).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close']
        )
        self.df['Stoch_K'] = stoch.stoch()
        self.df['Stoch_D'] = stoch.stoch_signal()
        
        # MACD
        macd = ta.trend.MACD(self.df['Close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_signal'] = macd.macd_signal()
    
    def _add_trend_indicators(self) -> None:
        """
        Adds ADX (Average Directional Index).
        
        Creates columns: ADX
        """
        adx = ta.trend.ADXIndicator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close']
        )
        self.df['ADX'] = adx.adx()
    
    def _add_volatility_features(self) -> None:
        """
        Adds ATR and rolling standard deviation.
        
        Creates columns: ATR, ATR_norm, Returns_std_{window}
        """
        # ATR
        atr = ta.volatility.AverageTrueRange(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close']
        )
        self.df['ATR'] = atr.average_true_range()
        self.df['ATR_norm'] = (self.df['ATR'] / self.df['Close']).clip(lower=0, upper=10)
        
        # Rolling std of returns
        for window in FEATURE_CONFIG.VOLATILITY_WINDOWS:
            self.df[f'Returns_std_{window}'] = self.df['Returns'].rolling(window=window).std()
    
    def _add_volume_features(self) -> None:
        """
        Adds volume-related features.
        
        Creates columns: Volume_SMA, Volume_ratio, MFI, OBV
        """
        self.df['Volume_SMA'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_ratio'] = self.df['Volume'] / self.df['Volume_SMA']
        
        # MFI
        mfi = ta.volume.MFIIndicator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            volume=self.df['Volume']
        )
        self.df['MFI'] = mfi.money_flow_index()
        
        # OBV
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=self.df['Close'],
            volume=self.df['Volume']
        )
        self.df['OBV'] = obv.on_balance_volume()
    
    def _add_rate_of_change(self) -> None:
        """
        Adds Rate of Change for price and volume.
        
        Creates columns: ROC_{window}, Vol_ROC_{window}
        """
        for window in FEATURE_CONFIG.ROC_WINDOWS:
            self.df[f'ROC_{window}'] = self.df['Close'].pct_change(periods=window)
            self.df[f'Vol_ROC_{window}'] = self.df['Volume'].pct_change(periods=window)
    
    def _add_zscore_features(self) -> None:
        """
        Adds Z-score relative to rolling mean.
        
        Creates columns: ZScore_{window}
        """
        for window in FEATURE_CONFIG.ZSCORE_WINDOWS:
            roll_mean = self.df['Close'].rolling(window=window).mean()
            roll_std = self.df['Close'].rolling(window=window).std()
            self.df[f'ZScore_{window}'] = (self.df['Close'] - roll_mean) / roll_std
    
    def _add_gap_and_range(self) -> None:
        """
        Adds Gap Open and Intraday Range.
        
        Creates columns: Gap_Open, Intraday_Range
        """
        self.df['Gap_Open'] = (
            (self.df['Open'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)
        )
        self.df['Intraday_Range'] = (self.df['High'] - self.df['Low']) / self.df['Open']
    
    def _add_ema_spreads(self) -> None:
        """
        Adds EMA spreads and Bollinger %B.
        
        Creates columns: EMA_12_26_spread, EMA_26_50_spread, BB_percentB
        """
        self.df['EMA_12_26_spread'] = self.df['EMA_12'] - self.df['EMA_26']
        self.df['EMA_26_50_spread'] = self.df['EMA_26'] - self.df['EMA_50']
        self.df['BB_percentB'] = (
            (self.df['Close'] - self.df['BB_low']) / 
            (self.df['BB_high'] - self.df['BB_low'])
        )
    
    def _add_cyclical_features(self) -> None:
        """
        Adds cyclical features (day of week as sin/cos).
        
        Creates columns: DOW_sin, DOW_cos
        """
        dow = self.df.index.dayofweek
        self.df['DOW_sin'] = np.sin(2 * np.pi * dow / 7)
        self.df['DOW_cos'] = np.cos(2 * np.pi * dow / 7)
    
    def _add_lag_features(self) -> None:
        """
        Adds lag features for Close and Returns.
        
        Creates columns: Close_lag_{lag}, Returns_lag_{lag}
        """
        for lag in FEATURE_CONFIG.LAG_PERIODS:
            self.df[f'Close_lag_{lag}'] = self.df['Close'].shift(lag)
            self.df[f'Returns_lag_{lag}'] = self.df['Returns'].shift(lag)
    
    def _add_trend_signals(self) -> None:
        """
        Adds trend and momentum signals.
        
        Creates columns: Trend_5d, Trend_20d, Momentum_10d, Momentum_20d
        """
        self.df['Trend_5d'] = (self.df['Close'] > self.df['SMA_10']).astype(int)
        self.df['Trend_20d'] = (self.df['Close'] > self.df['SMA_50']).astype(int)
        self.df['Momentum_10d'] = self.df['Close'] - self.df['Close'].shift(10)
        self.df['Momentum_20d'] = self.df['Close'] - self.df['Close'].shift(20)
    
    def _add_feature_interactions(self) -> None:
        """
        Adds feature interactions.
        
        Creates columns: RSI_x_ADX, BB_width_x_ATR, Volume_ratio_x_MFI,
                        EMA_spread_x_MACD, Trend_5d_x_Momentum,
                        Range20_vs_ATR, Range50_vs_ATR
        """
        atr_base = self.df['ATR'].clip(lower=1e-6)
        
        self.df['RSI_x_ADX'] = self.df['RSI'] * self.df['ADX']
        self.df['BB_width_x_ATR'] = self.df['BB_width'] * self.df['ATR_norm']
        self.df['Volume_ratio_x_MFI'] = self.df['Volume_ratio'] * self.df['MFI']
        self.df['EMA_spread_x_MACD'] = self.df['EMA_12_26_spread'] * self.df['MACD']
        self.df['Trend_5d_x_Momentum'] = self.df['Trend_5d'] * self.df['Momentum_10d']
        
        self.df['Range20_vs_ATR'] = (
            (self.df['RollingHigh_20'] - self.df['RollingLow_20']) / atr_base
        ).clip(-10, 10)
        self.df['Range50_vs_ATR'] = (
            (self.df['RollingHigh_50'] - self.df['RollingLow_50']) / atr_base
        ).clip(-10, 10)
    
    def _add_target(self) -> None:
        """
        Adds target variable.
        
        Creates columns: Target (1 if future return > threshold, else 0)
        """
        horizon = FEATURE_CONFIG.TARGET_HORIZON_DAYS
        threshold = FEATURE_CONFIG.TARGET_THRESHOLD
        
        future_return = (
            (self.df['Close'].shift(-horizon) - self.df['Close']) / self.df['Close']
        )
        self.df['Target'] = (future_return > threshold).astype(int)
