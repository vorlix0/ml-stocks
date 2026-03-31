"""
Pydantic models for input data validation.

Provides type-safe schema validation for:
- Raw OHLCV market data rows
- Feature-engineered rows (must include Target)
- Model configuration parameters
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.exceptions import EmptyDataError, InvalidDataError

# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------


class OHLCVRow(BaseModel):
    """Validates a single OHLCV record."""

    model_config = ConfigDict(extra="allow")

    Open: float = Field(gt=0, description="Opening price (must be positive)")
    High: float = Field(gt=0, description="High price (must be positive)")
    Low: float = Field(gt=0, description="Low price (must be positive)")
    Close: float = Field(gt=0, description="Closing price (must be positive)")
    Volume: float = Field(ge=0, description="Trading volume (must be non-negative)")

    @model_validator(mode="after")
    def high_gte_low(self) -> OHLCVRow:
        if self.High < self.Low:
            raise ValueError(f"High ({self.High}) must be >= Low ({self.Low})")
        return self


class FeaturesRow(OHLCVRow):
    """Validates a single row of feature-engineered data (must have Target)."""

    Target: int = Field(description="Binary target variable (0 or 1)")

    @field_validator("Target")
    @classmethod
    def target_is_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"Target must be 0 or 1, got {v}")
        return v


class DataConfigSchema(BaseModel):
    """Validates DataConfig parameters."""

    TICKER: str = Field(min_length=1, max_length=10)
    START_DATE: date
    END_DATE: date

    @model_validator(mode="after")
    def end_after_start(self) -> DataConfigSchema:
        if self.END_DATE <= self.START_DATE:
            raise ValueError(
                f"END_DATE ({self.END_DATE}) must be after START_DATE ({self.START_DATE})"
            )
        return self


class ModelConfigSchema(BaseModel):
    """Validates ModelConfig hyperparameters."""

    N_ESTIMATORS: int = Field(gt=0)
    MAX_DEPTH: int = Field(gt=0)
    LEARNING_RATE: float = Field(gt=0, lt=1)
    SUBSAMPLE: float = Field(gt=0, le=1)
    VALIDATION_SIZE: float = Field(gt=0, lt=1)
    RANDOM_STATE: int = Field(ge=0)


# ---------------------------------------------------------------------------
# DataFrame-level validators
# ---------------------------------------------------------------------------


def validate_ohlcv_dataframe(df: pd.DataFrame) -> None:
    """
    Validates an entire OHLCV DataFrame using Pydantic.

    Checks column presence, basic value constraints (High ≥ Low, prices > 0,
    volume ≥ 0), and acceptable NaN rate.

    Args:
        df: DataFrame to validate.

    Raises:
        EmptyDataError: When *df* is empty.
        InvalidDataError: When schema validation fails for any row.
    """
    if df.empty:
        raise EmptyDataError("OHLCV DataFrame is empty")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise InvalidDataError(f"Missing OHLCV columns: {missing}")

    # Check NaN rate on required columns
    nan_ratio = df[required].isna().sum().sum() / (len(df) * len(required))
    if nan_ratio > 0.5:
        raise InvalidDataError(
            f"Too many missing values in OHLCV data: {nan_ratio:.1%}"
        )

    # Sample-based Pydantic validation (first, last, and up to 50 random rows)
    sample_idx = _sample_indices(df, n=50)
    errors: list[str] = []
    for idx in sample_idx:
        row = df.iloc[idx][required].to_dict()
        try:
            OHLCVRow(**row)
        except Exception as exc:
            errors.append(f"Row {df.index[idx]}: {exc}")

    if errors:
        raise InvalidDataError(
            "OHLCV validation failed:\n" + "\n".join(errors[:5])
        )


def validate_features_dataframe(df: pd.DataFrame) -> None:
    """
    Validates a feature-engineered DataFrame using Pydantic.

    Checks that *Target* column is present, contains only 0/1 values, and
    that the DataFrame is not empty.

    Args:
        df: DataFrame to validate.

    Raises:
        EmptyDataError: When *df* is empty.
        InvalidDataError: When schema validation fails.
    """
    if df.empty:
        raise EmptyDataError("Features DataFrame is empty")

    if "Target" not in df.columns:
        raise InvalidDataError("Missing 'Target' column in features data")

    target_values = set(df["Target"].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise InvalidDataError(
            f"Target column contains values other than 0/1: {target_values}"
        )

    if len(target_values) < 2:
        raise InvalidDataError(
            f"Target has only one unique value: {target_values}. "
            "At least 2 classes required."
        )


def validate_model_config(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    validation_size: float,
    random_state: int,
) -> None:
    """
    Validates model hyperparameters using Pydantic.

    Args:
        n_estimators: Number of boosting stages.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate.
        subsample: Fraction of samples used per tree.
        validation_size: Fraction of training data used for validation.
        random_state: Random seed.

    Raises:
        InvalidDataError: When any parameter is out of range.
    """
    try:
        ModelConfigSchema(
            N_ESTIMATORS=n_estimators,
            MAX_DEPTH=max_depth,
            LEARNING_RATE=learning_rate,
            SUBSAMPLE=subsample,
            VALIDATION_SIZE=validation_size,
            RANDOM_STATE=random_state,
        )
    except Exception as exc:
        raise InvalidDataError(f"Invalid model configuration: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_indices(df: pd.DataFrame, n: int) -> list[int]:
    """Returns up to *n* representative row indices (first, last, random middle)."""
    length = len(df)
    if length <= n:
        return list(range(length))
    import random

    random.seed(0)
    middle = random.sample(range(1, length - 1), min(n - 2, length - 2))
    return sorted({0, length - 1} | set(middle))
