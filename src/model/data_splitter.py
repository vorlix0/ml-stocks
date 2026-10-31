"""
Data splitting strategies for ML model training.

Implements the Strategy pattern to decouple data preparation logic from
the trainer, making it easy to swap splitting strategies (chronological,
random, walk-forward) without modifying ModelTrainer.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import FEATURE_CONFIG, MODEL_CONFIG


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
        """Split chronologically, purging overlapping target horizons at boundaries."""
        # Gaps are counted in unique dates so panels (several rows per date)
        # are purged the same way as a single ticker.
        pre_mask = np.asarray(x.index < self.split_date)
        x_train, y_train = x[pre_mask], y[pre_mask]
        x_test, y_test = x[~pre_mask], y[~pre_mask]

        gap = FEATURE_CONFIG.TARGET_HORIZON_DAYS
        dates = x_train.index.unique().sort_values()
        if len(dates) <= 2 * gap + 1:
            raise ValueError(
                f"Not enough pre-split samples ({len(dates)}) for "
                f"a {gap}-sample gap and train/validation partitions."
            )

        # Exclude pre-test dates whose forward labels would cross split_date.
        dates = dates[:-gap]
        validation_start = int(len(dates) * (1 - self.validation_size))
        train_end = validation_start - gap
        if train_end <= 0 or validation_start >= len(dates):
            raise ValueError(
                "Not enough samples for train/validation partitions after "
                f"applying the {gap}-sample gap."
            )

        keep = np.asarray(x_train.index.isin(dates))
        x_train, y_train = x_train[keep], y_train[keep]

        tr_mask = np.asarray(x_train.index.isin(dates[:train_end]))
        val_mask = np.asarray(x_train.index.isin(dates[validation_start:]))
        x_tr, y_tr = x_train[tr_mask], y_train[tr_mask]
        x_val, y_val = x_train[val_mask], y_train[val_mask]

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
