"""
Data splitting strategies for ML model training.

Implements the Strategy pattern to decouple data preparation logic from
the trainer, making it easy to swap splitting strategies (chronological,
random, walk-forward) without modifying ModelTrainer.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODEL_CONFIG


@dataclass
class DataSplit:
    """Holds the result of a data splitting operation."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_tr: pd.DataFrame
    X_val: pd.DataFrame
    y_tr: pd.Series
    y_val: pd.Series


class DataSplitter(ABC):
    """Abstract base class for data splitting strategies."""

    @abstractmethod
    def split(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> DataSplit:
        """Split features and target into train/test/validation sets.

        Args:
            x: Feature DataFrame.
            y: Target Series.

        Returns:
            A DataSplit containing all partitions.
        """


class ChronologicalSplitter(DataSplitter):
    """Splits data chronologically based on a date boundary.

    This is the default strategy for time-series financial data where
    temporal ordering must be preserved to avoid look-ahead bias.

    Args:
        split_date: Date string used to separate train and test sets.
        validation_size: Fraction of training data used for validation.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        split_date: str | None = None,
        validation_size: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.split_date = split_date or MODEL_CONFIG.SPLIT_DATE
        self.validation_size = validation_size or MODEL_CONFIG.VALIDATION_SIZE
        # Use explicit None check: 0 is a valid random_state
        self.random_state = random_state if random_state is not None else MODEL_CONFIG.RANDOM_STATE

    def split(self, x: pd.DataFrame, y: pd.Series) -> DataSplit:
        """Split chronologically by split_date, then sub-split train into train/val."""
        x_train = x[x.index < self.split_date]
        x_test = x[x.index >= self.split_date]
        y_train = y[y.index < self.split_date]
        y_test = y[y.index >= self.split_date]

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.validation_size,
            random_state=self.random_state,
            shuffle=False,
        )

        return DataSplit(
            X_train=x_train,
            X_test=x_test,
            y_train=y_train,
            y_test=y_test,
            X_tr=x_tr,
            X_val=x_val,
            y_tr=y_tr,
            y_val=y_val,
        )
