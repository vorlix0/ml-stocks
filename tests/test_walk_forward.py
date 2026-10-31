"""
Tests for walk-forward validation and baselines.
"""
import numpy as np
import pandas as pd
import pytest

from config import FEATURE_CONFIG
from src.model.baselines import create_logistic_baseline, sma_trend_signals
from src.model.walk_forward import pre_split_data, walk_forward_evaluate


@pytest.fixture
def series_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=600, freq="D")
    x = pd.DataFrame(rng.normal(size=(600, 4)), index=idx, columns=list("abcd"))
    y = pd.Series((x["a"] + rng.normal(scale=0.5, size=600) > 0).astype(float), index=idx)
    return x, y


class TestPreSplitData:
    def test_excludes_test_period_and_horizon(self, series_data):
        x, y = series_data
        x_pre, y_pre = pre_split_data(x, y, split_date="2022-01-01")

        before = x[x.index < "2022-01-01"]
        assert len(x_pre) == len(before) - FEATURE_CONFIG.TARGET_HORIZON_DAYS
        assert x_pre.index[-1] < before.index[-1]
        assert len(y_pre) == len(x_pre)

    def test_drops_unlabelled_rows(self, series_data):
        x, y = series_data
        y = y.copy()
        y.iloc[:10] = np.nan
        x_pre, _ = pre_split_data(x, y, split_date="2022-01-01")

        assert x_pre.index[0] == x.index[10]


class TestWalkForward:
    def test_returns_fold_aucs_for_predictable_signal(self, series_data):
        from sklearn.linear_model import LogisticRegression

        x, y = series_data
        result = walk_forward_evaluate(lambda: LogisticRegression(), x, y, n_splits=4)

        assert len(result.fold_aucs) == 4
        assert result.mean_auc > 0.8

    def test_train_never_overlaps_validation_horizon(self, series_data):
        from sklearn.dummy import DummyClassifier

        x, y = series_data
        seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        class Recorder(DummyClassifier):
            def fit(self, x_, y_, sample_weight=None):
                self._last_train = x_.index[-1]
                return super().fit(x_, y_, sample_weight)

            def predict_proba(self, x_):
                seen.append((self._last_train, x_.index[0]))
                return super().predict_proba(x_)

        walk_forward_evaluate(lambda: Recorder(strategy="prior"), x, y, n_splits=3)

        gap = pd.Timedelta(days=FEATURE_CONFIG.TARGET_HORIZON_DAYS)
        assert seen
        assert all(val_start - train_end > gap for train_end, val_start in seen)


class TestBaselines:
    def test_logistic_baseline_fits_and_predicts(self, series_data):
        x, y = series_data
        model = create_logistic_baseline().fit(x, y)

        proba = model.predict_proba(x)[:, 1]
        assert proba.min() >= 0 and proba.max() <= 1

    def test_sma_trend_signals(self):
        df = pd.DataFrame({"Close_to_SMA_200": [-0.1, 0.0, 0.2]})

        assert sma_trend_signals(df).tolist() == [0, 0, 1]
