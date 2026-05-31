"""
Module with helper functions for file handling and data validation.
"""
from pathlib import Path

import pandas as pd

from src.exceptions import DataNotFoundError, EmptyDataError, InvalidDataError, ModelNotFoundError


def validate_file_exists(path: str, file_type: str = "file") -> Path:
    """
    Checks if file exists.

    Args:
        path: File path
        file_type: File type (for error message)

    Returns:
        Path to file

    Raises:
        DataNotFoundError: When file doesn't exist
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataNotFoundError(f"{file_type.capitalize()} doesn't exist: {path}")
    return file_path


def load_csv_safe(
    path: str,
    required_columns: list[str] | None = None,
    index_col: int = 0,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Safely loads CSV file with validation.

    Args:
        path: CSV file path
        required_columns: List of required columns (optional)
        index_col: Index column
        parse_dates: Whether to parse dates

    Returns:
        DataFrame with data

    Raises:
        DataNotFoundError: When file doesn't exist
        EmptyDataError: When file is empty
        InvalidDataError: When required columns are missing
    """
    validate_file_exists(path, "CSV file")

    try:
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    except pd.errors.EmptyDataError as e:
        raise EmptyDataError(f"CSV file is empty: {path}") from e
    except Exception as e:
        raise InvalidDataError(f"Error loading CSV {path}: {e}") from e

    if df.empty:
        raise EmptyDataError(f"DataFrame is empty after loading: {path}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise InvalidDataError(
                f"Missing columns in {path}: {missing_cols}"
            )

    return df


def load_model_safe(path: str) -> object:
    """
    Safely loads model file (joblib format).

    Args:
        path: Model file path

    Returns:
        Loaded object

    Raises:
        ModelNotFoundError: When file doesn't exist
        InvalidDataError: When file cannot be loaded
    """
    import joblib

    file_path = Path(path)
    if not file_path.exists():
        raise ModelNotFoundError(f"Model file doesn't exist: {path}")

    try:
        return joblib.load(path)
    except Exception as e:
        raise InvalidDataError(f"Error loading model {path}: {e}") from e


def validate_ohlcv_data(df: pd.DataFrame) -> None:
    """
    Validates OHLCV data.

    Delegates to the canonical Pydantic-based validator in ``src.validators``
    to avoid duplicating validation logic.

    Args:
        df: DataFrame to validate

    Raises:
        EmptyDataError: When data is empty
        InvalidDataError: When data is invalid
    """
    from src.validators import validate_ohlcv_dataframe

    validate_ohlcv_dataframe(df)


def validate_features_data(df: pd.DataFrame) -> None:
    """
    Validates features data.

    Delegates to the canonical Pydantic-based validator in ``src.validators``
    to avoid duplicating validation logic.

    Args:
        df: DataFrame to validate

    Raises:
        EmptyDataError: When data is empty
        InvalidDataError: When data is invalid
    """
    from src.validators import validate_features_dataframe

    validate_features_dataframe(df)
