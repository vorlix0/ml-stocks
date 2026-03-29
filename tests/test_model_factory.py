"""
Tests for model_factory and BaseTrainer interface.
"""
import pytest
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src.model.base_trainer import BaseTrainer
from src.model.model_factory import MODEL_REGISTRY, create_model
from src.model.trainer import ModelTrainer


class TestCreateModel:
    """Tests for create_model() factory function."""

    def test_creates_gradient_boosting(self):
        """Should return a GradientBoostingClassifier for 'gradient_boosting'."""
        model = create_model("gradient_boosting")
        assert isinstance(model, GradientBoostingClassifier)

    def test_creates_random_forest(self):
        """Should return a RandomForestClassifier for 'random_forest'."""
        model = create_model("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_raises_for_unknown_model(self):
        """Should raise ValueError for an unknown model name."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("unknown_model")

    def test_kwargs_forwarded_to_estimator(self):
        """Keyword arguments should be forwarded to the estimator."""
        model = create_model("gradient_boosting", n_estimators=50, max_depth=3)
        assert model.n_estimators == 50
        assert model.max_depth == 3

    def test_all_registry_entries_are_classifiers(self):
        """Every entry in MODEL_REGISTRY should be a ClassifierMixin subclass."""
        for name, cls in MODEL_REGISTRY.items():
            instance = cls()
            assert isinstance(instance, ClassifierMixin), (
                f"MODEL_REGISTRY['{name}'] is not a ClassifierMixin"
            )


class TestBaseTrainerInterface:
    """Verify BaseTrainer is a proper ABC and ModelTrainer satisfies it."""

    def test_cannot_instantiate_base_trainer(self):
        """Instantiating BaseTrainer directly should raise TypeError."""
        with pytest.raises(TypeError):
            BaseTrainer()  # type: ignore[abstract]

    def test_model_trainer_is_base_trainer(self, sample_ohlcv_data):
        """ModelTrainer must be a subclass of BaseTrainer."""
        from src.data.features import FeatureEngineer

        engineer = FeatureEngineer(sample_ohlcv_data)
        df = engineer.create_all_features()
        trainer = ModelTrainer(df)

        assert isinstance(trainer, BaseTrainer)


class TestModelTrainerSplitAttributesAlwaysExist:
    """Ensure split attributes exist even when _validate_input raises."""

    def test_attributes_exist_after_valid_init(self, sample_ohlcv_data):
        """All split attributes must be set after successful __init__."""
        from src.data.features import FeatureEngineer

        engineer = FeatureEngineer(sample_ohlcv_data)
        df = engineer.create_all_features()
        trainer = ModelTrainer(df)

        for attr in ('X_train', 'X_test', 'y_train', 'y_test',
                     'X_tr', 'X_val', 'y_tr', 'y_val'):
            assert hasattr(trainer, attr), f"Missing attribute: {attr}"
            assert getattr(trainer, attr) is None
