"""
Tests for FeatureEngineer class.
"""
import pandas as pd
import pytest

from src.data.features import FeatureEngineer
from src.exceptions import EmptyDataError, InvalidDataError


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    @pytest.fixture
    def longer_ohlcv_data(self, sample_ohlcv_data):
        """Extends the sample beyond the longest indicator and target horizon."""
        data = pd.concat([sample_ohlcv_data, sample_ohlcv_data], ignore_index=True)
        data.index = pd.date_range(
            start=sample_ohlcv_data.index[0], periods=len(data), freq="D"
        )
        return data

    def test_init_with_valid_data(self, sample_ohlcv_data):
        """Should initialize with valid OHLCV data."""
        engineer = FeatureEngineer(sample_ohlcv_data)

        assert engineer.df is not None
        assert len(engineer.df) == len(sample_ohlcv_data)

    def test_init_raises_on_empty_data(self):
        """Should raise EmptyDataError for empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(EmptyDataError, match="empty"):
            FeatureEngineer(empty_df)

    def test_init_raises_on_missing_columns(self):
        """Should raise InvalidDataError when OHLCV columns missing."""
        incomplete_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })

        with pytest.raises(InvalidDataError, match="Missing OHLCV columns"):
            FeatureEngineer(incomplete_df)

    def test_create_all_features_adds_columns(self, sample_ohlcv_data):
        """create_all_features should add many new columns."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        initial_cols = len(engineer.df.columns)

        result = engineer.create_all_features()

        assert len(result.columns) > initial_cols
        assert len(result.columns) > 50  # Should have many features

    def test_create_all_features_has_target(self, longer_ohlcv_data):
        """Result should include Target column."""
        engineer = FeatureEngineer(longer_ohlcv_data)
        result = engineer.create_all_features()

        assert 'Target' in result.columns
        assert result['Target'].notna().any()

    def test_create_all_features_keeps_unknown_targets(self, longer_ohlcv_data):
        """Features are complete while the final target horizon remains unknown."""
        from config import FEATURE_CONFIG

        engineer = FeatureEngineer(longer_ohlcv_data)
        result = engineer.create_all_features()

        assert result.drop(columns=['Target']).isna().sum().sum() == 0
        assert result['Target'].isna().sum() == FEATURE_CONFIG.TARGET_HORIZON_DAYS

    def test_target_is_binary_and_matches_future_return(self, longer_ohlcv_data):
        """Known targets reflect the configured forward return threshold."""
        from config import FEATURE_CONFIG

        engineer = FeatureEngineer(longer_ohlcv_data)
        result = engineer.create_all_features()

        future_return = (
            longer_ohlcv_data['Close'].shift(-FEATURE_CONFIG.TARGET_HORIZON_DAYS)
            / longer_ohlcv_data['Close']
            - 1
        )
        expected = (
            (future_return > FEATURE_CONFIG.TARGET_THRESHOLD)
            .where(future_return.notna())
            .astype(float)
        )
        pd.testing.assert_series_equal(
            result['Target'], expected.loc[result.index], check_names=False
        )
        assert set(result['Target'].dropna().unique()).issubset({0, 1})

    def test_returns_column_exists(self, sample_ohlcv_data):
        """Returns column should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert 'Returns' in result.columns

    def test_sma_columns_exist(self, sample_ohlcv_data):
        """SMA columns should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        sma_cols = [col for col in result.columns if col.startswith('SMA_')]
        assert len(sma_cols) > 0

    def test_ema_columns_exist(self, sample_ohlcv_data):
        """EMA columns should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        ema_cols = [col for col in result.columns if col.startswith('EMA_')]
        assert len(ema_cols) > 0

    def test_bollinger_bands_exist(self, sample_ohlcv_data):
        """Bollinger Band columns should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert 'BB_high' in result.columns
        assert 'BB_low' in result.columns
        assert 'BB_width' in result.columns

    def test_momentum_indicators_exist(self, sample_ohlcv_data):
        """RSI, MACD, Stochastic should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'Stoch_K' in result.columns

    def test_adx_exists(self, sample_ohlcv_data):
        """ADX column should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert 'ADX' in result.columns

    def test_volume_features_exist(self, sample_ohlcv_data):
        """Volume features should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert 'Volume_SMA' in result.columns
        assert 'Volume_ratio' in result.columns
        assert 'OBV' in result.columns
        assert 'MFI' in result.columns

    def test_feature_interactions_exist(self, sample_ohlcv_data):
        """Feature interaction columns should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        interaction_cols = [col for col in result.columns if '_x_' in col]
        assert len(interaction_cols) > 0

    def test_cyclical_features_exist(self, sample_ohlcv_data):
        """Day of week cyclical features should be created."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert 'DOW_sin' in result.columns
        assert 'DOW_cos' in result.columns

    def test_cyclical_features_range(self, sample_ohlcv_data):
        """Cyclical features should be in [-1, 1] range."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.create_all_features()

        assert result['DOW_sin'].min() >= -1
        assert result['DOW_sin'].max() <= 1
        assert result['DOW_cos'].min() >= -1
        assert result['DOW_cos'].max() <= 1

    def test_data_not_modified_in_place(self, sample_ohlcv_data):
        """Original DataFrame should not be modified."""
        original_cols = list(sample_ohlcv_data.columns)
        original_len = len(sample_ohlcv_data)

        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.create_all_features()

        assert list(sample_ohlcv_data.columns) == original_cols
        assert len(sample_ohlcv_data) == original_len
