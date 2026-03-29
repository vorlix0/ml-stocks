"""
Abstract base class for ML model trainers.

Defines the interface every trainer implementation must satisfy, making it
possible to swap ``GradientBoostingClassifier`` for any other estimator
(e.g. XGBoost, RandomForest) without touching the backtest or evaluation
code.
"""
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import ClassifierMixin


class BaseTrainer(ABC):
    """Abstract interface for model trainers."""

    @abstractmethod
    def train(self) -> ClassifierMixin:
        """Train the model and return the fitted estimator."""

    @abstractmethod
    def get_feature_importances(self) -> pd.DataFrame:
        """Return a DataFrame with 'feature' and 'importance' columns."""

    @abstractmethod
    def save_model(self, path: str | None = None) -> None:
        """Persist the trained model to disk."""

    @staticmethod
    @abstractmethod
    def load_model(path: str | None = None) -> ClassifierMixin:
        """Load and return a previously saved model."""
