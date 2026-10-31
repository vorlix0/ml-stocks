"""
Module for creating features from OHLCV data.
Each feature category has a separate method for better readability and testability.
"""
import numpy as np
import pandas as pd
import ta

from config import FEATURE_CONFIG
from src.exceptions import EmptyDataError, InvalidDataError

MARKET_COLUMNS = ['Benchmark_Close', 'VIX']


def build_market_frame(benchmark: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Combines benchmark (e.g. SPY) and VIX closes into one market-context frame.

    Args:
        benchmark: OHLCV DataFrame of the benchmark
        vix: OHLCV DataFrame of the volatility index

    Returns:
        DataFrame indexed by date with ``Benchmark_Close`` and ``VIX`` columns

    Raises:
        InvalidDataError: When either input has no 'Close' column
    """
    for name, frame in (('benchmark', benchmark), ('VIX', vix)):
        if 'Close' not in frame.columns:
            raise InvalidDataError(f"Missing 'Close' column in {name} data")

    return pd.DataFrame({
        'Benchmark_Close': benchmark['Close'],
        'VIX': vix['Close'],
    }).sort_index().ffill()


class FeatureEngineer:
    """Class for creating features from OHLCV data."""

    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(self, df: pd.DataFrame, market: pd.DataFrame | None = None):
        """
        Initializes FeatureEngineer.

        Args:
            df: DataFrame with OHLCV data (Close, Open, High, Low, Volume)
            market: Optional market context from :func:`build_market_frame`.
                When given, market features are added and the target becomes
                "outperforms the benchmark" instead of an absolute return.

        Raises:
            InvalidDataError: When required columns are missing
            EmptyDataError: When DataFrame is empty
        """
        self._validate_input(df)
        self.df = df.copy()
        self.market = self._validate_market(market)

    @staticmethod
    def _validate_market(market: pd.DataFrame | None) -> pd.DataFrame | None:
        if market is None:
            return None
        missing_cols = set(MARKET_COLUMNS) - set(market.columns)
        if missing_cols:
            raise InvalidDataError(f"Missing market columns: {missing_cols}")
        return market

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
        self._add_normalized_features()
        if self.market is not None:
            self._add_market_features()
        self._add_target()

        # Keep rows whose future target is unknown for inference/backtesting.
        self.df = self.df.dropna(subset=self.df.columns.drop('Target'))

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

    def _add_normalized_features(self) -> None:
        """
        Adds scale-free versions of price-level indicators.

        Raw levels (SMA, EMA, ATR, OBV, ...) drift with the price and leave
        the training range on later data, so the model uses these ratios
        instead (raw columns are listed in MODEL_CONFIG.EXCLUDED_COLUMNS).
        """
        close = self.df['Close']
        for window in FEATURE_CONFIG.SMA_WINDOWS:
            self.df[f'Close_to_SMA_{window}'] = close / self.df[f'SMA_{window}'] - 1
        for window in FEATURE_CONFIG.EMA_WINDOWS:
            self.df[f'Close_to_EMA_{window}'] = close / self.df[f'EMA_{window}'] - 1
        for window in FEATURE_CONFIG.ROLLING_WINDOWS:
            self.df[f'Close_to_RollingHigh_{window}'] = (
                close / self.df[f'RollingHigh_{window}'] - 1
            )
            self.df[f'Close_to_RollingLow_{window}'] = (
                close / self.df[f'RollingLow_{window}'] - 1
            )

        self.df['BB_width_norm'] = self.df['BB_width'] / close
        self.df['MACD_norm'] = self.df['MACD'] / close
        self.df['MACD_signal_norm'] = self.df['MACD_signal'] / close
        self.df['EMA_12_26_spread_norm'] = self.df['EMA_12_26_spread'] / close
        self.df['EMA_26_50_spread_norm'] = self.df['EMA_26_50_spread'] / close
        self.df['High_Low_norm'] = self.df['High_Low'] / close
        self.df['Price_Change_norm'] = self.df['Price_Change'] / close
        self.df['OBV_ROC_20'] = (
            self.df['OBV'].diff(20) / self.df['Volume_SMA'].clip(lower=1) / 20
        )

    def _aligned_market(self) -> pd.DataFrame:
        """Market series aligned to this stock's trading days (forward-filled)."""
        assert self.market is not None
        return self.market.reindex(self.market.index.union(self.df.index)).ffill().reindex(
            self.df.index
        )

    def _add_market_features(self) -> None:
        """
        Adds benchmark and VIX context plus the stock's relative behaviour.

        Creates columns: Mkt_ret_{window}, Mkt_Close_to_SMA_200, Mkt_vol_20,
        VIX, VIX_chg_5, VIX_to_SMA_20, Rel_ROC_{window}, Beta_{window}
        """
        market = self._aligned_market()
        bench = market['Benchmark_Close']
        vix = market['VIX']
        bench_ret = bench.pct_change()

        for window in FEATURE_CONFIG.ROC_WINDOWS:
            self.df[f'Mkt_ret_{window}'] = bench.pct_change(periods=window)
            self.df[f'Rel_ROC_{window}'] = (
                self.df[f'ROC_{window}'] - self.df[f'Mkt_ret_{window}']
            )
        self.df['Mkt_ret_60'] = bench.pct_change(periods=60)
        self.df['Mkt_Close_to_SMA_200'] = bench / bench.rolling(200).mean() - 1
        self.df['Mkt_vol_20'] = bench_ret.rolling(20).std()

        self.df['VIX'] = vix
        self.df['VIX_chg_5'] = vix.pct_change(periods=5)
        self.df['VIX_to_SMA_20'] = vix / vix.rolling(20).mean() - 1

        beta_window = FEATURE_CONFIG.MARKET_BETA_WINDOW
        self.df[f'Beta_{beta_window}'] = (
            self.df['Returns'].rolling(beta_window).cov(bench_ret)
            / bench_ret.rolling(beta_window).var()
        ).clip(-5, 5)

    def _add_target(self) -> None:
        """
        Adds target variable.

        Creates columns: Target. Without market data: 1 if the future return
        exceeds TARGET_THRESHOLD. With market data: 1 if the stock beats the
        benchmark's future return by more than TARGET_EXCESS_THRESHOLD.
        Rows whose future is not yet known keep NaN.
        """
        horizon = FEATURE_CONFIG.TARGET_HORIZON_DAYS

        future_return = (
            (self.df['Close'].shift(-horizon) - self.df['Close']) / self.df['Close']
        )
        if self.market is None:
            signal = future_return > FEATURE_CONFIG.TARGET_THRESHOLD
        else:
            bench = self._aligned_market()['Benchmark_Close']
            future_return = future_return - (bench.shift(-horizon) / bench - 1)
            signal = future_return > FEATURE_CONFIG.TARGET_EXCESS_THRESHOLD

        self.df['Target'] = signal.where(future_return.notna()).astype(float)
