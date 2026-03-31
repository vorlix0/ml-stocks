"""
Module for training ML models.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split

from config import MODEL_CONFIG
from src.exceptions import (
    EmptyDataError,
    InvalidDataError,
    ModelNotFoundError,
    ModelNotTrainedError,
)
from src.model.base_trainer import BaseTrainer
from src.model.model_factory import create_model
from src.validators import validate_model_config

logger = logging.getLogger("forex_ml.model.trainer")


class ModelTrainer(BaseTrainer):
    """Class for training ML models."""

    def __init__(self, df: pd.DataFrame):
        """
        Initializes trainer.

        Args:
            df: DataFrame with features and target

        Raises:
            EmptyDataError: When DataFrame is empty
            InvalidDataError: When Target column is missing
        """
        self._validate_input(df)
        self.df = df
        self.model = None
        self.feature_cols = self._get_feature_columns()

        # Data splits – initialized here so attributes always exist
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_tr = None
        self.X_val = None
        self.y_tr = None
        self.y_val = None

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
        """Prepares data for training (chronological split)."""
        x = self.df[self.feature_cols]
        y = self.df['Target']

        # Remove rows with NaN
        mask = x.notna().all(axis=1) & y.notna()
        x = x[mask]
        y = y[mask]

        logger.info(f"After removing NaN: {x.shape}")

        # Chronological train/test split
        split_date = MODEL_CONFIG.SPLIT_DATE
        self.X_train = x[x.index < split_date]
        self.X_test = x[x.index >= split_date]
        self.y_train = y[y.index < split_date]
        self.y_test = y[y.index >= split_date]

        logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")

        # Split train into train/validation
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=MODEL_CONFIG.VALIDATION_SIZE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            shuffle=False
        )

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

        self.model = create_model(
            MODEL_CONFIG.MODEL_TYPE,
            n_estimators=MODEL_CONFIG.N_ESTIMATORS,
            max_depth=MODEL_CONFIG.MAX_DEPTH,
            learning_rate=MODEL_CONFIG.LEARNING_RATE,
            subsample=MODEL_CONFIG.SUBSAMPLE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            verbose=0,
        )

        if use_mlflow:
            self._train_with_mlflow()
        else:
            self.model.fit(self.X_tr, self.y_tr)

        return self.model

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

            self.model.fit(self.X_tr, self.y_tr)

            # Validation metrics
            val_proba = self.model.predict_proba(self.X_val)[:, 1]
            val_auc = roc_auc_score(self.y_val, val_proba)
            mlflow.log_metric("val_auc", val_auc)

            # Test metrics
            test_proba = self.model.predict_proba(self.X_test)[:, 1]
            test_auc = roc_auc_score(self.y_test, test_proba)
            mlflow.log_metric("test_auc", test_auc)

            mlflow.sklearn.log_model(self.model, artifact_path="model")
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

    def save_model(self, path: str = None) -> None:
        """
        Saves model to file.

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

        path = Path(path) if path else MODEL_CONFIG.model_file

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path, compress=3)
        except Exception as e:
            raise OSError(f"Cannot save model to {path}: {e}") from e

        logger.info(f"Model saved to: {path}")

    @staticmethod
    def load_model(path: str = None) -> ClassifierMixin:
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
        path = Path(path) if path else MODEL_CONFIG.model_file

        if not path.exists():
            raise ModelNotFoundError(
                f"Model file doesn't exist: {path}. "
                "Run train_model.py first"
            )

        try:
            return joblib.load(path)
        except Exception as e:
            raise InvalidDataError(f"Error loading model {path}: {e}") from e
