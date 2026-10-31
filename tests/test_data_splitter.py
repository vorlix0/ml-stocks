"""
Tests for DataSplitter implementations (Strategy pattern).
"""
import numpy as np
import pandas as pd
import pytest

from src.model.data_splitter import ChronologicalSplitter, DataSplit, DataSplitter


class TestDataSplit:
    """Tests for the DataSplit dataclass."""

    def test_data_split_holds_all_partitions(self):
        """DataSplit should store all partition references."""
        idx = pd.date_range("2023-01-01", periods=10)
        df = pd.DataFrame({"a": range(10)}, index=idx)
        s = pd.Series(range(10), index=idx)

        split = DataSplit(
            X_train=df[:6], X_test=df[6:],
            y_train=s[:6], y_test=s[6:],
            X_tr=df[:4], X_val=df[4:6],
            y_tr=s[:4], y_val=s[4:6],
        )

        assert len(split.X_train) == 6
        assert len(split.X_test) == 4
        assert len(split.X_tr) == 4
        assert len(split.X_val) == 2


class TestChronologicalSplitter:
    """Tests for ChronologicalSplitter."""

    @pytest.fixture
    def time_series_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Creates time-series feature data spanning 2022-2024."""
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=500, freq="D")
        x = pd.DataFrame(
            np.random.randn(500, 5),
            columns=[f"f{i}" for i in range(5)],
            index=dates,
        )
        y = pd.Series(np.random.randint(0, 2, 500), index=dates)
        return x, y

    def test_split_returns_data_split(self, time_series_data):
        """split() must return a DataSplit instance."""
        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-01-01")

        result = splitter.split(x, y)

        assert isinstance(result, DataSplit)

    def test_train_before_split_date(self, time_series_data):
        """All training data indices must be before the split date."""
        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-01-01")

        result = splitter.split(x, y)

        assert (result.X_train.index < "2023-01-01").all()
        assert (result.y_train.index < "2023-01-01").all()

    def test_test_after_split_date(self, time_series_data):
        """All test data indices must be on or after the split date."""
        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-01-01")

        result = splitter.split(x, y)

        assert (result.X_test.index >= "2023-01-01").all()
        assert (result.y_test.index >= "2023-01-01").all()

    def test_no_data_leakage_between_train_and_test(self, time_series_data):
        """Train and test indices must not overlap."""
        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-01-01")

        result = splitter.split(x, y)

        train_idx = set(result.X_train.index)
        test_idx = set(result.X_test.index)
        assert train_idx.isdisjoint(test_idx)

    def test_validation_is_subset_of_train(self, time_series_data):
        """Validation data must come from the training period."""
        from config import FEATURE_CONFIG

        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-01-01")

        result = splitter.split(x, y)

        all_train_idx = set(result.X_train.index)
        tr_idx = set(result.X_tr.index)
        val_idx = set(result.X_val.index)

        assert tr_idx | val_idx < all_train_idx
        assert tr_idx.isdisjoint(val_idx)
        assert len(all_train_idx - (tr_idx | val_idx)) == FEATURE_CONFIG.TARGET_HORIZON_DAYS

    def test_validation_size_respected(self, time_series_data):
        """Validation set size should approximate the requested fraction."""
        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-01-01", validation_size=0.2)

        result = splitter.split(x, y)

        total_train = len(result.X_tr) + len(result.X_val)
        actual_ratio = len(result.X_val) / total_train
        assert abs(actual_ratio - 0.2) < 0.05

    def test_purges_horizon_before_test_and_validation(self, time_series_data):
        """Training labels cannot extend into validation or test periods."""
        from config import FEATURE_CONFIG

        x, y = time_series_data
        split_date = pd.Timestamp("2023-01-01")
        splitter = ChronologicalSplitter(split_date=str(split_date.date()))
        result = splitter.split(x, y)
        horizon = FEATURE_CONFIG.TARGET_HORIZON_DAYS

        unpurged_train = x[x.index < split_date]
        assert result.X_train.index[-1] == unpurged_train.index[-horizon - 1]
        train_position = x.index.get_loc(result.X_tr.index[-1])
        validation_position = x.index.get_loc(result.X_val.index[0])
        assert validation_position - train_position == horizon + 1

    def test_custom_split_date(self, time_series_data):
        """Splitter should respect a custom split_date."""
        x, y = time_series_data
        splitter = ChronologicalSplitter(split_date="2023-06-01")

        result = splitter.split(x, y)

        assert (result.X_train.index < "2023-06-01").all()
        assert (result.X_test.index >= "2023-06-01").all()

    def test_uses_config_defaults(self, time_series_data):
        """When no params given, should use MODEL_CONFIG defaults."""
        from config import MODEL_CONFIG

        x, y = time_series_data
        splitter = ChronologicalSplitter()

        assert splitter.split_date == MODEL_CONFIG.SPLIT_DATE
        assert splitter.validation_size == MODEL_CONFIG.VALIDATION_SIZE
        assert splitter.random_state == MODEL_CONFIG.RANDOM_STATE

    def test_splitter_is_abstract_base_class(self):
        """DataSplitter must be abstract and not instantiable."""
        with pytest.raises(TypeError):
            DataSplitter()
