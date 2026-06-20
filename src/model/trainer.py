"""
Module for training ML models.
"""
import logging

import pandas as pd
from sklearn.base import ClassifierMixin

from config import MODEL_CONFIG
from src.exceptions import (
    EmptyDataError,
    InvalidDataError,
    ModelNotTrainedError,
)
from src.model.base_trainer import BaseTrainer
from src.model.data_splitter import ChronologicalSplitter, DataSplitter
from src.model.model_factory import create_model
from src.model.model_repository import JoblibModelRepository, ModelRepository
from src.validators import validate_model_config

logger = logging.getLogger("forex_ml.model.trainer")


class ModelTrainer(BaseTrainer):
    """Class for training ML models.

    Supports dependency injection of splitting strategy and persistence
    backend, following the Open/Closed Principle.

    Args:
        df: DataFrame with features and target.
        splitter: Data splitting strategy (defaults to ChronologicalSplitter).
        repository: Model persistence backend (defaults to JoblibModelRepository).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        splitter: DataSplitter | None = None,
        repository: ModelRepository | None = None,
    ):
        """
        Initializes trainer.

        Args:
            df: DataFrame with features and target
            splitter: Data splitting strategy (defaults to ChronologicalSplitter)
            repository: Model persistence backend (defaults to JoblibModelRepository)

        Raises:
            EmptyDataError: When DataFrame is empty
            InvalidDataError: When Target column is missing
        """
        self._validate_input(df)
        self.df = df
        self.model: ClassifierMixin | None = None
        self.feature_cols = self._get_feature_columns()
        self._splitter = splitter or ChronologicalSplitter()
        self._repository = repository or JoblibModelRepository()

        # Data splits – initialized here so attributes always exist
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.X_tr: pd.DataFrame | None = None
        self.X_val: pd.DataFrame | None = None
        self.y_tr: pd.Series | None = None
        self.y_val: pd.Series | None = None

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validates input data."""
        if df.empty:
            raise EmptyDataError("Input DataFrame is empty")

        if 'Target' not in df.columns:
            raise InvalidDataError(
                "Missing 'Target' column in data. "
                "Run process_data.py first"
            )

    def _get_feature_columns(self) -> list[str]:
        """Returns list of feature columns."""
        return [
            col for col in self.df.columns
            if col not in MODEL_CONFIG.EXCLUDED_COLUMNS
        ]

    def prepare_data(self) -> None:
        """Prepares data for training using the configured splitter."""
        x = self.df[self.feature_cols]
        y = self.df['Target']

        # Remove rows with NaN
        mask = x.notna().all(axis=1) & y.notna()
        x = x[mask]
        y = y[mask]

        logger.info(f"After removing NaN: {x.shape}")

        split = self._splitter.split(x, y)

        self.X_train = split.X_train
        self.X_test = split.X_test
        self.y_train = split.y_train
        self.y_test = split.y_test
        self.X_tr = split.X_tr
        self.X_val = split.X_val
        self.y_tr = split.y_tr
        self.y_val = split.y_val

        logger.info(f"Train: {len(split.X_train)}, Test: {len(split.X_test)}")

    def train(self, use_mlflow: bool = False) -> ClassifierMixin:
        """
        Trains model configured by MODEL_CONFIG.MODEL_TYPE.

        Optionally logs parameters, metrics and the trained model to an
        MLflow experiment (requires ``mlflow`` to be installed and an active
        tracking server or local store).

        Args:
            use_mlflow: When *True*, wrap training in an MLflow run and log
                hyperparameters, validation AUC and the fitted estimator.

        Returns:
            Trained model
        """
        if self.X_tr is None:
            self.prepare_data()

        # Validate hyperparameters with Pydantic before training
        validate_model_config(
            n_estimators=MODEL_CONFIG.N_ESTIMATORS,
            max_depth=MODEL_CONFIG.MAX_DEPTH,
            learning_rate=MODEL_CONFIG.LEARNING_RATE,
            subsample=MODEL_CONFIG.SUBSAMPLE,
            validation_size=MODEL_CONFIG.VALIDATION_SIZE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
        )

        model = create_model(
            MODEL_CONFIG.MODEL_TYPE,
            n_estimators=MODEL_CONFIG.N_ESTIMATORS,
            max_depth=MODEL_CONFIG.MAX_DEPTH,
            learning_rate=MODEL_CONFIG.LEARNING_RATE,
            subsample=MODEL_CONFIG.SUBSAMPLE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            verbose=0,
        )
        self.model = model

        assert self.X_tr is not None and self.y_tr is not None

        if use_mlflow:
            self._train_with_mlflow()
        else:
            model.fit(self.X_tr, self.y_tr)

        return model

    def _train_with_mlflow(self) -> None:
        """Fits the model and logs run artefacts to MLflow."""
        try:
            import mlflow
            import mlflow.sklearn
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:
            raise ImportError(
                "MLflow is required for experiment tracking. "
                "Install it with: pip install mlflow"
            ) from exc

        params = {
            "model_type": MODEL_CONFIG.MODEL_TYPE,
            "n_estimators": MODEL_CONFIG.N_ESTIMATORS,
            "max_depth": MODEL_CONFIG.MAX_DEPTH,
            "learning_rate": MODEL_CONFIG.LEARNING_RATE,
            "subsample": MODEL_CONFIG.SUBSAMPLE,
            "random_state": MODEL_CONFIG.RANDOM_STATE,
        }

        with mlflow.start_run():
            mlflow.log_params(params)

            assert self.model is not None
            assert self.X_tr is not None and self.y_tr is not None
            assert self.X_val is not None and self.y_val is not None
            assert self.X_test is not None and self.y_test is not None
            model = self.model

            model.fit(self.X_tr, self.y_tr)

            # Validation metrics
            val_proba = model.predict_proba(self.X_val)[:, 1]
            val_auc = roc_auc_score(self.y_val, val_proba)
            mlflow.log_metric("val_auc", val_auc)

            # Test metrics
            test_proba = model.predict_proba(self.X_test)[:, 1]
            test_auc = roc_auc_score(self.y_test, test_proba)
            mlflow.log_metric("test_auc", test_auc)

            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info(
                f"MLflow run finished – val_auc={val_auc:.4f}, test_auc={test_auc:.4f}"
            )

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns DataFrame with feature importances.

        Returns:
            DataFrame with 'feature' and 'importance' columns

        Raises:
            ModelNotTrainedError: When model has not been trained
        """
        if self.model is None:
            raise ModelNotTrainedError(
                "Model has not been trained yet! "
                "Call train() method first."
            )

        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def save_model(self, path: str | None = None) -> None:
        """
        Saves model to file using the configured repository.

        Args:
            path: File path (default from config)

        Raises:
            ModelNotTrainedError: When model has not been trained
            OSError: When file cannot be saved
        """
        if self.model is None:
            raise ModelNotTrainedError(
                "Model has not been trained yet! "
                "Call train() method first."
            )

        self._repository.save(self.model, path)
        saved_to = path if path is not None else "repository default path"
        logger.info(f"Model saved to: {saved_to}")

    @staticmethod
    def load_model(path: str | None = None) -> ClassifierMixin:
        """
        Loads model from file.

        Args:
            path: File path (default from config)

        Returns:
            Loaded model

        Raises:
            ModelNotFoundError: When model file doesn't exist
            InvalidDataError: When model cannot be loaded
        """
        repository = JoblibModelRepository()
        return repository.load(path)

