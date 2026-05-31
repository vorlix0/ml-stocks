"""
Tests for ModelEvaluator refactored with lazy initialization pattern.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.model.evaluator import ModelEvaluator


class TestModelEvaluatorLazyInit:
    """Tests that ModelEvaluator correctly uses lazy prediction initialization."""

    @pytest.fixture
    def evaluator(self) -> ModelEvaluator:
        """Creates an evaluator with a small trained model."""
        np.random.seed(42)
        x_train = np.random.randn(100, 5)
        y_train = (x_train[:, 0] > 0).astype(int)
        x_test = np.random.randn(30, 5)
        y_test = (x_test[:, 0] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=20, random_state=42)
        model.fit(x_train, y_train)

        return ModelEvaluator(
            model=model,
            x_test=pd.DataFrame(x_test),
            y_test=pd.Series(y_test),
        )

    def test_predictions_none_before_first_call(self, evaluator):
        """Predictions should be None initially."""
        assert evaluator.y_pred is None
        assert evaluator.y_pred_proba is None

    def test_get_metrics_triggers_predict(self, evaluator):
        """get_metrics() should auto-compute predictions if None."""
        metrics = evaluator.get_metrics()

        assert evaluator.y_pred is not None
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'confusion_matrix' in metrics

    def test_get_classification_report_triggers_predict(self, evaluator):
        """get_classification_report() should auto-compute predictions."""
        report = evaluator.get_classification_report()

        assert evaluator.y_pred is not None
        assert isinstance(report, str)
        assert "precision" in report

    def test_explicit_predict_sets_attributes(self, evaluator):
        """predict() should set y_pred and y_pred_proba."""
        evaluator.predict()

        assert evaluator.y_pred is not None
        assert evaluator.y_pred_proba is not None
        assert len(evaluator.y_pred) == 30
        assert len(evaluator.y_pred_proba) == 30

    def test_predictions_not_recomputed(self, evaluator):
        """Once predicted, calling get_metrics again shouldn't change them."""
        evaluator.predict()
        first_pred = evaluator.y_pred.copy()

        evaluator.get_metrics()

        np.testing.assert_array_equal(evaluator.y_pred, first_pred)

    def test_metrics_accuracy_in_valid_range(self, evaluator):
        """Accuracy should be between 0 and 1."""
        metrics = evaluator.get_metrics()

        assert 0 <= metrics['accuracy'] <= 1

    def test_metrics_roc_auc_in_valid_range(self, evaluator):
        """ROC AUC should be between 0 and 1."""
        metrics = evaluator.get_metrics()

        assert 0 <= metrics['roc_auc'] <= 1

    def test_evaluate_validation_returns_auc(self, evaluator):
        """evaluate_validation should return a float AUC score."""
        np.random.seed(99)
        x_val = pd.DataFrame(np.random.randn(20, 5))
        y_val = pd.Series((np.random.randn(20) > 0).astype(int))

        auc = evaluator.evaluate_validation(x_val, y_val)

        assert isinstance(auc, float)
        assert 0 <= auc <= 1

    def test_print_evaluation_does_not_raise(self, evaluator):
        """print_evaluation should complete without errors."""
        # Should not raise any exceptions
        evaluator.print_evaluation()
