"""
Module for training ML models.
"""
import logging
import joblib
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from config import MODEL_CONFIG
from src.exceptions import (
    ModelNotFoundError,
    ModelNotTrainedError,
    InvalidDataError,
    EmptyDataError
)

logger = logging.getLogger("forex_ml.model.trainer")


class ModelTrainer:
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
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validates input data."""
        if df.empty:
            raise EmptyDataError("Input DataFrame is empty")
        
        if 'Target' not in df.columns:
            raise InvalidDataError(
                "Missing 'Target' column in data. "
                "Run process_data.py first"
            )
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_tr = None
        self.X_val = None
        self.y_tr = None
        self.y_val = None
    
    def _get_feature_columns(self) -> List[str]:
        """Returns list of feature columns."""
        return [
            col for col in self.df.columns 
            if col not in MODEL_CONFIG.EXCLUDED_COLUMNS
        ]
    
    def prepare_data(self) -> None:
        """Prepares data for training (chronological split)."""
        X = self.df[self.feature_cols]
        y = self.df['Target']
        
        # Remove rows with NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"After removing NaN: {X.shape}")
        
        # Chronological train/test split
        split_date = MODEL_CONFIG.SPLIT_DATE
        self.X_train = X[X.index < split_date]
        self.X_test = X[X.index >= split_date]
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
    
    def train(self) -> GradientBoostingClassifier:
        """
        Trains GradientBoosting model.
        
        Returns:
            Trained model
        """
        if self.X_tr is None:
            self.prepare_data()
        
        self.model = GradientBoostingClassifier(
            n_estimators=MODEL_CONFIG.N_ESTIMATORS,
            max_depth=MODEL_CONFIG.MAX_DEPTH,
            learning_rate=MODEL_CONFIG.LEARNING_RATE,
            subsample=MODEL_CONFIG.SUBSAMPLE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            verbose=0
        )
        
        self.model.fit(self.X_tr, self.y_tr)
        return self.model
    
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
            IOError: When file cannot be saved
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
            raise IOError(f"Cannot save model to {path}: {e}")
        
        logger.info(f"Model saved to: {path}")
    
    @staticmethod
    def load_model(path: str = None) -> GradientBoostingClassifier:
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
            raise InvalidDataError(f"Error loading model {path}: {e}")
