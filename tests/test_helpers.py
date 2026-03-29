"""
Tests for helper functions (src/utils/helpers.py).
"""
from pathlib import Path

import pandas as pd
import pytest

from src.exceptions import (
    DataNotFoundError,
    EmptyDataError,
    InvalidDataError,
    ModelNotFoundError,
)
from src.utils.helpers import (
    load_csv_safe,
    load_model_safe,
    validate_features_data,
    validate_file_exists,
    validate_ohlcv_data,
)


class TestValidateFileExists:
    """Tests for validate_file_exists()."""

    def test_existing_file_returns_path(self, tmp_path):
        """Should return a Path object for an existing file."""
        f = tmp_path / "test.csv"
        f.write_text("data")

        result = validate_file_exists(str(f))

        assert isinstance(result, Path)
        assert result == f

    def test_missing_file_raises(self, tmp_path):
        """Should raise DataNotFoundError for a non-existent file."""
        missing = tmp_path / "missing.csv"

        with pytest.raises(DataNotFoundError):
            validate_file_exists(str(missing))


class TestLoadCsvSafe:
    """Tests for load_csv_safe()."""

    def _write_csv(self, tmp_path: Path, content: str, name: str = "data.csv") -> Path:
        p = tmp_path / name
        p.write_text(content)
        return p

    def test_loads_valid_csv(self, tmp_path):
        """Should return a non-empty DataFrame for a valid CSV."""
        p = self._write_csv(tmp_path, "idx,A,B\n0,1,2\n1,3,4\n")

        df = load_csv_safe(str(p))

        assert not df.empty
        assert list(df.columns) == ['A', 'B']

    def test_raises_for_missing_file(self, tmp_path):
        """Should raise DataNotFoundError when file does not exist."""
        with pytest.raises(DataNotFoundError):
            load_csv_safe(str(tmp_path / "missing.csv"))

    def test_raises_for_empty_file(self, tmp_path):
        """Should raise EmptyDataError when file is empty."""
        p = tmp_path / "empty.csv"
        p.write_text("")

        with pytest.raises(EmptyDataError):
            load_csv_safe(str(p))

    def test_raises_for_missing_required_columns(self, tmp_path):
        """Should raise InvalidDataError when required columns are absent."""
        p = self._write_csv(tmp_path, "idx,A\n0,1\n")

        with pytest.raises(InvalidDataError, match="Missing columns"):
            load_csv_safe(str(p), required_columns=['A', 'B'])

    def test_validates_required_columns_present(self, tmp_path):
        """Should not raise when all required columns are present."""
        p = self._write_csv(tmp_path, "idx,A,B\n0,1,2\n")

        df = load_csv_safe(str(p), required_columns=['A', 'B'])

        assert 'A' in df.columns and 'B' in df.columns


class TestLoadModelSafe:
    """Tests for load_model_safe()."""

    def test_raises_for_missing_model_file(self, tmp_path):
        """Should raise ModelNotFoundError when the model file is absent."""
        with pytest.raises(ModelNotFoundError):
            load_model_safe(str(tmp_path / "model.joblib"))

    def test_loads_valid_joblib_file(self, tmp_path):
        """Should load an object that was saved with joblib."""
        import joblib

        obj = {"key": "value"}
        path = tmp_path / "model.joblib"
        joblib.dump(obj, path)

        loaded = load_model_safe(str(path))

        assert loaded == obj

    def test_raises_for_corrupt_file(self, tmp_path):
        """Should raise InvalidDataError when the file is corrupted."""
        bad = tmp_path / "bad.joblib"
        bad.write_bytes(b"not a valid joblib file")

        with pytest.raises(InvalidDataError):
            load_model_safe(str(bad))


class TestValidateOhlcvData:
    """Tests for validate_ohlcv_data()."""

    def _make_ohlcv(self, rows: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Open': [1.0] * rows,
            'High': [1.1] * rows,
            'Low': [0.9] * rows,
            'Close': [1.0] * rows,
            'Volume': [1_000] * rows,
        })

    def test_valid_data_does_not_raise(self):
        """Should not raise for a well-formed OHLCV DataFrame."""
        validate_ohlcv_data(self._make_ohlcv())

    def test_raises_for_missing_column(self):
        """Should raise InvalidDataError when an OHLCV column is missing."""
        df = self._make_ohlcv().drop(columns=['Volume'])

        with pytest.raises(InvalidDataError, match="Missing OHLCV columns"):
            validate_ohlcv_data(df)

    def test_raises_for_empty_dataframe(self):
        """Should raise EmptyDataError for a zero-row DataFrame."""
        df = self._make_ohlcv(rows=0)

        with pytest.raises(EmptyDataError):
            validate_ohlcv_data(df)


class TestValidateFeaturesData:
    """Tests for validate_features_data()."""

    def _make_features(self, rows: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Feature1': [1.0] * rows,
            'Target': [0, 1] * (rows // 2),
        })

    def test_valid_data_does_not_raise(self):
        """Should not raise for a valid features DataFrame."""
        validate_features_data(self._make_features())

    def test_raises_for_missing_target(self):
        """Should raise InvalidDataError when Target column is absent."""
        df = self._make_features().drop(columns=['Target'])

        with pytest.raises(InvalidDataError, match="Target"):
            validate_features_data(df)

    def test_raises_for_empty_dataframe(self):
        """Should raise EmptyDataError for a zero-row DataFrame."""
        df = self._make_features(rows=0)

        with pytest.raises(EmptyDataError):
            validate_features_data(df)

    def test_raises_for_single_class_target(self):
        """Should raise InvalidDataError when Target has only one unique value."""
        df = pd.DataFrame({
            'Feature1': [1.0] * 10,
            'Target': [0] * 10,
        })

        with pytest.raises(InvalidDataError, match="one value"):
            validate_features_data(df)
