"""
Tests for SignalFilter ABC and concrete implementations.
"""
import pandas as pd
import pytest

from src.backtest.signal_filter import ADXFilter, NoFilter, SignalFilter


class TestNoFilter:
    """Tests for NoFilter — the pass-through implementation."""

    def test_returns_signals_unchanged(self):
        """NoFilter should return the same signals it receives."""
        signals = pd.Series([1, 0, 1, 1, 0])
        df = pd.DataFrame({'ADX': [30, 20, 35, 10, 28]})

        result = NoFilter().apply(signals, df)

        pd.testing.assert_series_equal(result, signals)

    def test_result_has_same_index(self):
        """Output index must match input index."""
        idx = pd.date_range('2023-01-01', periods=5)
        signals = pd.Series([1, 0, 1, 0, 1], index=idx)
        df = pd.DataFrame({'ADX': [25] * 5}, index=idx)

        result = NoFilter().apply(signals, df)

        assert list(result.index) == list(idx)


class TestADXFilter:
    """Tests for ADXFilter."""

    def _make_inputs(
        self, signals: list[int], adx_values: list[float]
    ) -> tuple[pd.Series, pd.DataFrame]:
        signals_s = pd.Series(signals)
        df = pd.DataFrame({'ADX': adx_values})
        return signals_s, df

    def test_keeps_signals_above_threshold(self):
        """Signals where ADX > threshold should survive."""
        signals, df = self._make_inputs([1, 1], [30.0, 26.0])

        result = ADXFilter(threshold=25).apply(signals, df)

        assert list(result) == [1, 1]

    def test_removes_signals_below_threshold(self):
        """Signals where ADX <= threshold should be zeroed out."""
        signals, df = self._make_inputs([1, 1], [20.0, 25.0])

        result = ADXFilter(threshold=25).apply(signals, df)

        # ADX 20 < 25 → suppressed; ADX 25 is NOT > 25 → also suppressed
        assert list(result) == [0, 0]

    def test_hold_signals_stay_zero(self):
        """Hold signals (0) should stay 0 regardless of ADX."""
        signals, df = self._make_inputs([0, 0, 0], [30.0, 40.0, 50.0])

        result = ADXFilter(threshold=25).apply(signals, df)

        assert list(result) == [0, 0, 0]

    def test_mixed_signals_and_adx(self):
        """Only rows with signal=1 AND ADX > threshold should yield 1."""
        signals, df = self._make_inputs(
            [1, 1, 0, 1],
            [30.0, 20.0, 35.0, 26.0]
        )

        result = ADXFilter(threshold=25).apply(signals, df)

        assert list(result) == [1, 0, 0, 1]

    def test_output_is_binary(self):
        """All values in result must be 0 or 1."""
        signals, df = self._make_inputs(
            [1, 0, 1, 0, 1],
            [28.0, 30.0, 22.0, 15.0, 40.0]
        )

        result = ADXFilter(threshold=25).apply(signals, df)

        assert set(result.unique()).issubset({0, 1})

    def test_custom_threshold_respected(self):
        """A high threshold should filter out more signals."""
        signals, df = self._make_inputs([1, 1, 1], [30.0, 40.0, 50.0])

        result_low = ADXFilter(threshold=25).apply(signals, df)
        result_high = ADXFilter(threshold=45).apply(signals, df)

        assert result_low.sum() >= result_high.sum()


class TestSignalFilterInterface:
    """Verify SignalFilter is a proper ABC."""

    def test_cannot_instantiate_abstract_class(self):
        """Instantiating SignalFilter directly should raise TypeError."""
        with pytest.raises(TypeError):
            SignalFilter()  # type: ignore[abstract]

    def test_concrete_subclass_satisfies_interface(self):
        """Both NoFilter and ADXFilter are valid SignalFilter instances."""
        assert isinstance(NoFilter(), SignalFilter)
        assert isinstance(ADXFilter(), SignalFilter)
