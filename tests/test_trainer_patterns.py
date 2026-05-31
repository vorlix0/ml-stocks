"""
Integration tests for refactored ModelTrainer with dependency injection.

These tests prove that the new patterns (DataSplitter, ModelRepository)
do not break any existing functionality when injected into ModelTrainer.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.exceptions import ModelNotTrainedError
from src.model.data_splitter import ChronologicalSplitter, DataSplitter
from src.model.model_repository import JoblibModelRepository, ModelRepository
from src.model.trainer import ModelTrainer


class TestTrainerWithCustomSplitter:
    """Tests that ModelTrainer works with injected DataSplitter."""

    @pytest.fixture
    def large_training_data(self) -> pd.DataFrame:
        """Creates data spanning the split date."""
        from src.data.features import FeatureEngineer

        np.random.seed(42)
        dates = pd.date_range(start="2021-01-01", periods=1000, freq="D")
        base_price = 100
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

    def test_default_splitter_is_chronological(self, large_training_data):
        """Without explicit splitter, trainer should use ChronologicalSplitter."""
        trainer = ModelTrainer(large_training_data)

        assert isinstance(trainer._splitter, ChronologicalSplitter)

    def test_custom_splitter_is_used(self, large_training_data):
        """Injected splitter should be used for data preparation."""

        class MockSplitter(DataSplitter):
            def __init__(self):
                self.called = False

            def split(self, x, y):
                self.called = True
                # Delegate to real splitter for valid results
                real = ChronologicalSplitter()
                return real.split(x, y)

        splitter = MockSplitter()
        trainer = ModelTrainer(large_training_data, splitter=splitter)
        trainer.prepare_data()

        assert splitter.called

    def test_trainer_train_with_injected_splitter(self, large_training_data):
        """Training should succeed with a custom splitter."""
        splitter = ChronologicalSplitter(split_date="2023-06-01")
        trainer = ModelTrainer(large_training_data, splitter=splitter)
        model = trainer.train()

        assert model is not None
        assert trainer.X_test is not None
        assert (trainer.X_test.index >= "2023-06-01").all()


class TestTrainerWithCustomRepository:
    """Tests that ModelTrainer works with injected ModelRepository."""

    @pytest.fixture
    def large_training_data(self) -> pd.DataFrame:
        """Creates data spanning the split date."""
        from src.data.features import FeatureEngineer

        np.random.seed(42)
        dates = pd.date_range(start="2021-01-01", periods=1000, freq="D")
        base_price = 100
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

    def test_default_repository_is_joblib(self, large_training_data):
        """Without explicit repo, trainer should use JoblibModelRepository."""
        trainer = ModelTrainer(large_training_data)

        assert isinstance(trainer._repository, JoblibModelRepository)

    def test_save_model_uses_repository(self, large_training_data):
        """save_model should delegate to the injected repository."""

        class MockRepository(ModelRepository):
            def __init__(self):
                self.saved_model = None
                self.default_path = None

            def save(self, model, path=None):
                self.saved_model = model

            def load(self, path=None):
                return self.saved_model

        repo = MockRepository()
        trainer = ModelTrainer(large_training_data, repository=repo)
        trainer.train()
        trainer.save_model()

        assert repo.saved_model is not None

    def test_save_model_before_train_raises(self, large_training_data):
        """save_model without training should raise ModelNotTrainedError."""
        trainer = ModelTrainer(large_training_data)

        with pytest.raises(ModelNotTrainedError):
            trainer.save_model()

    def test_save_and_load_round_trip(self, large_training_data):
        """Model should survive save/load cycle via repository."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        try:
            trainer = ModelTrainer(large_training_data)
            trainer.train()
            trainer.save_model(path)

            loaded = ModelTrainer.load_model(path)
            preds_original = trainer.model.predict_proba(trainer.X_test)[:, 1]
            preds_loaded = loaded.predict_proba(trainer.X_test)[:, 1]

            np.testing.assert_array_almost_equal(preds_original, preds_loaded)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestBackwardCompatibility:
    """Ensure the refactored trainer is fully backward-compatible."""

    @pytest.fixture
    def large_training_data(self) -> pd.DataFrame:
        """Creates data spanning the split date."""
        from src.data.features import FeatureEngineer

        np.random.seed(42)
        dates = pd.date_range(start="2021-01-01", periods=1000, freq="D")
        base_price = 100
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

    def test_trainer_without_args_works_as_before(self, large_training_data):
        """ModelTrainer(df) without extra args should work identically."""
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        model = trainer.train()

        assert model is not None
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.X_tr is not None
        assert trainer.X_val is not None

    def test_feature_importances_still_work(self, large_training_data):
        """get_feature_importances should work after refactoring."""
        trainer = ModelTrainer(large_training_data)
        trainer.train()

        importances = trainer.get_feature_importances()

        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
        assert len(importances) == len(trainer.feature_cols)

    def test_predictions_shape_unchanged(self, large_training_data):
        """Model predictions should have expected shape."""
        trainer = ModelTrainer(large_training_data)
        trainer.train()

        preds = trainer.model.predict_proba(trainer.X_test)[:, 1]

        assert len(preds) == len(trainer.X_test)
        assert preds.min() >= 0
        assert preds.max() <= 1
