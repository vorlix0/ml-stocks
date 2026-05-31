"""
Tests for ModelRepository implementations (Repository pattern).
"""
import os
import tempfile

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.exceptions import ModelNotFoundError
from src.model.model_repository import JoblibModelRepository, ModelRepository


class TestModelRepositoryAbstract:
    """Tests for the ModelRepository abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """ModelRepository must be abstract and not instantiable."""
        with pytest.raises(TypeError):
            ModelRepository()


class TestJoblibModelRepository:
    """Tests for JoblibModelRepository."""

    @pytest.fixture
    def trained_model(self) -> GradientBoostingClassifier:
        """Creates a small trained model for testing."""
        np.random.seed(42)
        features = np.random.randn(50, 3)
        y = (features[:, 0] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(features, y)
        return model

    @pytest.fixture
    def temp_path(self) -> str:
        """Creates a temporary file path for model persistence."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        # Remove so save can create it fresh
        os.remove(path)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_save_creates_file(self, trained_model, temp_path):
        """save() must create a file at the given path."""
        repo = JoblibModelRepository()

        repo.save(trained_model, temp_path)

        assert os.path.exists(temp_path)

    def test_load_returns_equivalent_model(self, trained_model, temp_path):
        """load() must return a model producing identical predictions."""
        repo = JoblibModelRepository()
        repo.save(trained_model, temp_path)

        loaded = repo.load(temp_path)

        np.random.seed(0)
        x_test = np.random.randn(10, 3)
        np.testing.assert_array_equal(
            trained_model.predict(x_test),
            loaded.predict(x_test),
        )

    def test_load_nonexistent_raises_model_not_found(self):
        """load() on a missing path must raise ModelNotFoundError."""
        repo = JoblibModelRepository()

        with pytest.raises(ModelNotFoundError):
            repo.load("/tmp/nonexistent_model_12345.joblib")

    def test_save_creates_parent_directories(self, trained_model):
        """save() should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "model.joblib")
            repo = JoblibModelRepository()

            repo.save(trained_model, path)

            assert os.path.exists(path)

    def test_default_path_from_config(self):
        """Default path should come from MODEL_CONFIG."""
        from config import MODEL_CONFIG

        repo = JoblibModelRepository()

        assert repo.default_path == MODEL_CONFIG.model_file

    def test_custom_compress_level(self, trained_model, temp_path):
        """Repository should use the configured compression level."""
        repo = JoblibModelRepository(compress=0)

        repo.save(trained_model, temp_path)
        loaded = repo.load(temp_path)

        # Model should still work after save/load
        np.random.seed(0)
        x_test = np.random.randn(10, 3)
        np.testing.assert_array_equal(
            trained_model.predict(x_test),
            loaded.predict(x_test),
        )

    def test_round_trip_preserves_probabilities(self, trained_model, temp_path):
        """Probabilities should be identical after save/load."""
        repo = JoblibModelRepository()
        repo.save(trained_model, temp_path)
        loaded = repo.load(temp_path)

        np.random.seed(0)
        x_test = np.random.randn(10, 3)
        np.testing.assert_array_almost_equal(
            trained_model.predict_proba(x_test),
            loaded.predict_proba(x_test),
        )
