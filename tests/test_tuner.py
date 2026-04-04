"""
Tests for HyperparameterTuner (src/model/tuner.py).

These tests use small trial counts to run fast in CI.
"""
import numpy as np
import pandas as pd
import pytest

from src.model.tuner import HyperparameterTuner


@pytest.fixture
def large_training_data() -> pd.DataFrame:
    """Creates larger sample data for tuner tests (needs to span split date)."""
    from src.data.features import FeatureEngineer

    np.random.seed(0)
    dates = pd.date_range(start="2021-01-01", periods=1000, freq="D")

    base_price = 100.0
    returns = np.random.randn(1000) * 0.02
    close = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": close * (1 + np.random.randn(1000) * 0.005),
            "High": close * (1 + np.abs(np.random.randn(1000) * 0.01)),
            "Low": close * (1 - np.abs(np.random.randn(1000) * 0.01)),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=1000),
        },
        index=dates,
    )

    engineer = FeatureEngineer(df)
    return engineer.create_all_features()


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner."""

    def test_init_valid(self, large_training_data):
        """Should initialise without errors."""
        tuner = HyperparameterTuner(large_training_data)
        assert tuner.model_type == "gradient_boosting"

    def test_init_custom_model_type(self, large_training_data):
        """Should accept a custom model type."""
        tuner = HyperparameterTuner(large_training_data, model_type="random_forest")
        assert tuner.model_type == "random_forest"

    def test_run_returns_dict(self, large_training_data):
        """run() should return a non-empty dict of hyperparameters."""
        tuner = HyperparameterTuner(large_training_data)
        best = tuner.run(n_trials=2, show_progress_bar=False)

        assert isinstance(best, dict)
        assert len(best) > 0

    def test_run_returns_expected_keys(self, large_training_data):
        """Best params dict should contain the expected hyperparameter keys."""
        tuner = HyperparameterTuner(large_training_data)
        best = tuner.run(n_trials=2, show_progress_bar=False)

        expected_keys = {"n_estimators", "max_depth", "learning_rate", "subsample"}
        assert expected_keys.issubset(set(best.keys()))

    def test_run_values_in_valid_range(self, large_training_data):
        """Returned hyperparameter values should be within the search space."""
        tuner = HyperparameterTuner(large_training_data)
        best = tuner.run(n_trials=2, show_progress_bar=False)

        assert 50 <= best["n_estimators"] <= 500
        assert 2 <= best["max_depth"] <= 8
        assert 0.0 < best["learning_rate"] < 1.0
        assert 0.5 <= best["subsample"] <= 1.0

    def test_run_random_forest(self, large_training_data):
        """Tuner should also work with random_forest model type."""
        tuner = HyperparameterTuner(large_training_data, model_type="random_forest")
        best = tuner.run(n_trials=2, show_progress_bar=False)
        assert isinstance(best, dict)
        # random_forest uses max_features, not learning_rate
        assert "n_estimators" in best
        assert "learning_rate" not in best
