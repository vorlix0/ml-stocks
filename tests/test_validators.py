"""
Tests for Pydantic validators (src/validators.py).
"""
import pandas as pd
import pytest

from src.exceptions import EmptyDataError, InvalidDataError
from src.validators import (
    DataConfigSchema,
    FeaturesRow,
    ModelConfigSchema,
    OHLCVRow,
    validate_features_dataframe,
    validate_model_config,
    validate_ohlcv_dataframe,
)


class TestOHLCVRow:
    """Unit tests for OHLCVRow Pydantic model."""

    def test_valid_row(self):
        row = OHLCVRow(Open=100.0, High=105.0, Low=98.0, Close=102.0, Volume=1_000_000)
        assert row.Close == 102.0

    def test_high_less_than_low_raises(self):
        with pytest.raises(Exception, match="High"):
            OHLCVRow(Open=100.0, High=95.0, Low=98.0, Close=100.0, Volume=500_000)

    def test_negative_price_raises(self):
        with pytest.raises(Exception):
            OHLCVRow(Open=-1.0, High=105.0, Low=98.0, Close=102.0, Volume=1_000_000)

    def test_zero_price_raises(self):
        with pytest.raises(Exception):
            OHLCVRow(Open=0.0, High=105.0, Low=98.0, Close=102.0, Volume=1_000_000)

    def test_negative_volume_raises(self):
        with pytest.raises(Exception):
            OHLCVRow(Open=100.0, High=105.0, Low=98.0, Close=102.0, Volume=-1)

    def test_zero_volume_allowed(self):
        row = OHLCVRow(Open=100.0, High=105.0, Low=98.0, Close=102.0, Volume=0)
        assert row.Volume == 0

    def test_extra_fields_allowed(self):
        """Extra fields (e.g. features) should be silently accepted."""
        row = OHLCVRow(Open=100.0, High=105.0, Low=98.0, Close=102.0, Volume=1_000_000,
                       RSI=55.0)
        assert row.RSI == 55.0  # type: ignore[attr-defined]


class TestFeaturesRow:
    """Unit tests for FeaturesRow Pydantic model."""

    def test_valid_row_target_1(self):
        row = FeaturesRow(Open=100.0, High=105.0, Low=98.0, Close=102.0,
                          Volume=1_000_000, Target=1)
        assert row.Target == 1

    def test_valid_row_target_0(self):
        row = FeaturesRow(Open=100.0, High=105.0, Low=98.0, Close=102.0,
                          Volume=1_000_000, Target=0)
        assert row.Target == 0

    def test_invalid_target_raises(self):
        with pytest.raises(Exception, match="Target"):
            FeaturesRow(Open=100.0, High=105.0, Low=98.0, Close=102.0,
                        Volume=1_000_000, Target=2)


class TestModelConfigSchema:
    """Unit tests for ModelConfigSchema Pydantic model."""

    def test_valid_config(self):
        cfg = ModelConfigSchema(
            N_ESTIMATORS=300,
            MAX_DEPTH=5,
            LEARNING_RATE=0.05,
            SUBSAMPLE=0.8,
            VALIDATION_SIZE=0.2,
            RANDOM_STATE=42,
        )
        assert cfg.N_ESTIMATORS == 300

    def test_zero_estimators_raises(self):
        with pytest.raises(Exception):
            ModelConfigSchema(
                N_ESTIMATORS=0,
                MAX_DEPTH=5,
                LEARNING_RATE=0.05,
                SUBSAMPLE=0.8,
                VALIDATION_SIZE=0.2,
                RANDOM_STATE=42,
            )

    def test_learning_rate_ge_1_raises(self):
        with pytest.raises(Exception):
            ModelConfigSchema(
                N_ESTIMATORS=100,
                MAX_DEPTH=5,
                LEARNING_RATE=1.0,
                SUBSAMPLE=0.8,
                VALIDATION_SIZE=0.2,
                RANDOM_STATE=42,
            )

    def test_subsample_gt_1_raises(self):
        with pytest.raises(Exception):
            ModelConfigSchema(
                N_ESTIMATORS=100,
                MAX_DEPTH=5,
                LEARNING_RATE=0.1,
                SUBSAMPLE=1.5,
                VALIDATION_SIZE=0.2,
                RANDOM_STATE=42,
            )


class TestDataConfigSchema:
    """Unit tests for DataConfigSchema Pydantic model."""

    def test_valid_config(self):
        cfg = DataConfigSchema(TICKER="AAPL", START_DATE="2020-01-01", END_DATE="2024-01-01")
        assert cfg.TICKER == "AAPL"

    def test_end_before_start_raises(self):
        with pytest.raises(Exception, match="END_DATE"):
            DataConfigSchema(
                TICKER="AAPL",
                START_DATE="2024-01-01",
                END_DATE="2020-01-01",
            )

    def test_same_dates_raises(self):
        with pytest.raises(Exception):
            DataConfigSchema(
                TICKER="AAPL",
                START_DATE="2024-01-01",
                END_DATE="2024-01-01",
            )


class TestValidateOHLCVDataframe:
    """Tests for validate_ohlcv_dataframe helper."""

    def test_valid_dataframe(self, sample_ohlcv_data):
        validate_ohlcv_dataframe(sample_ohlcv_data)  # should not raise

    def test_empty_dataframe_raises(self):
        with pytest.raises(EmptyDataError):
            validate_ohlcv_dataframe(pd.DataFrame())

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Open": [100.0], "Close": [102.0]})
        with pytest.raises(InvalidDataError, match="Missing"):
            validate_ohlcv_dataframe(df)

    def test_invalid_high_low_raises(self, sample_ohlcv_data):
        bad = sample_ohlcv_data.copy()
        bad["High"] = bad["Low"] - 1  # High < Low for every row
        with pytest.raises(InvalidDataError):
            validate_ohlcv_dataframe(bad)


class TestValidateFeaturesDataframe:
    """Tests for validate_features_dataframe helper."""

    def test_valid_dataframe(self, sample_df_with_features):
        validate_features_dataframe(sample_df_with_features)  # should not raise

    def test_empty_dataframe_raises(self):
        with pytest.raises(EmptyDataError):
            validate_features_dataframe(pd.DataFrame())

    def test_missing_target_column_raises(self, sample_ohlcv_data):
        with pytest.raises(InvalidDataError, match="Target"):
            validate_features_dataframe(sample_ohlcv_data)

    def test_non_binary_target_raises(self, sample_df_with_features):
        bad = sample_df_with_features.copy()
        bad["Target"] = 5
        with pytest.raises(InvalidDataError):
            validate_features_dataframe(bad)

    def test_single_class_raises(self, sample_df_with_features):
        bad = sample_df_with_features.copy()
        bad["Target"] = 0
        with pytest.raises(InvalidDataError, match="one unique value"):
            validate_features_dataframe(bad)


class TestValidateModelConfig:
    """Tests for validate_model_config function."""

    def test_valid_config(self):
        validate_model_config(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            validation_size=0.2,
            random_state=42,
        )  # should not raise

    def test_invalid_config_raises(self):
        with pytest.raises(InvalidDataError):
            validate_model_config(
                n_estimators=0,  # invalid
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                validation_size=0.2,
                random_state=42,
            )
