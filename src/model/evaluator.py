"""
Module for ML model evaluation.
"""
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger("forex_ml.model.evaluator")


class ModelEvaluator:
    """Class for ML model evaluation."""

    def __init__(self, model, x_test: pd.DataFrame, y_test: pd.Series):
        """
        Initializes evaluator.

        Args:
            model: Trained sklearn model
            x_test: Test features
            y_test: Test target
        """
        self.model = model
        self.X_test = x_test
        self.y_test = y_test

        # Predictions
        self.y_pred = None
        self.y_pred_proba = None

    def predict(self) -> None:
        """Makes predictions on test set."""
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

    def get_metrics(self) -> dict[str, Any]:
        """
        Calculates evaluation metrics.

        Returns:
            Dictionary with metrics
        """
        if self.y_pred is None:
            self.predict()

        return {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, self.y_pred)
        }

    def get_classification_report(self) -> str:
        """
        Returns classification report.

        Returns:
            String with classification report
        """
        if self.y_pred is None:
            self.predict()

        return classification_report(self.y_test, self.y_pred)

    def print_evaluation(self) -> None:
        """Prints full model evaluation."""
        if self.y_pred is None:
            self.predict()

        metrics = self.get_metrics()

        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"Classification:\n{self.get_classification_report()}")
        logger.info(f"Test set balance: % UP: {(self.y_test == 1).sum() / len(self.y_test) * 100:.2f}%")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    def evaluate_validation(self, x_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Evaluates model on validation set.

        Args:
            x_val: Validation features
            y_val: Validation target

        Returns:
            AUC on validation set
        """
        y_val_pred_proba = self.model.predict_proba(x_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        logger.info(f"Validation AUC: {val_auc:.3f}")
        return val_auc
