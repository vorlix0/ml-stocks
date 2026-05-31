"""
Repository pattern for model persistence.

Separates model storage concerns from training logic, following the
Single Responsibility Principle. New storage backends (S3, MLflow
registry, etc.) can be added by implementing ModelRepository.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin

from config import MODEL_CONFIG
from src.exceptions import InvalidDataError, ModelNotFoundError


class ModelRepository(ABC):
    """Abstract interface for model persistence."""

    @abstractmethod
    def save(self, model: ClassifierMixin, path: str | None = None) -> None:
        """Persist a trained model.

        Args:
            model: Fitted sklearn estimator.
            path: Optional destination path.

        Raises:
            OSError: When the model cannot be saved.
        """

    @abstractmethod
    def load(self, path: str | None = None) -> ClassifierMixin:
        """Load a previously persisted model.

        Args:
            path: Optional source path.

        Returns:
            The loaded estimator.

        Raises:
            ModelNotFoundError: When no model is found at the path.
            InvalidDataError: When the file cannot be deserialized.
        """


class JoblibModelRepository(ModelRepository):
    """Persists models using joblib serialization (default).

    Args:
        default_path: Fallback path when none is explicitly given.
        compress: Joblib compression level (0-9).
    """

    def __init__(
        self,
        default_path: Path | None = None,
        compress: int = 3,
    ) -> None:
        self.default_path = default_path or MODEL_CONFIG.model_file
        self.compress = compress

    def save(self, model: ClassifierMixin, path: str | None = None) -> None:
        """Save model to a joblib file."""
        dest = Path(path) if path else self.default_path
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, dest, compress=self.compress)
        except Exception as e:
            raise OSError(f"Cannot save model to {dest}: {e}") from e

    def load(self, path: str | None = None) -> ClassifierMixin:
        """Load model from a joblib file."""
        source = Path(path) if path else self.default_path
        if not source.exists():
            raise ModelNotFoundError(
                f"Model file doesn't exist: {source}. Run train_model.py first"
            )
        try:
            return joblib.load(source)
        except Exception as e:
            raise InvalidDataError(f"Error loading model {source}: {e}") from e
