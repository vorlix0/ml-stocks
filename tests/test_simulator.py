"""
Tests for TradingSimulator class.
"""
import pandas as pd
import pytest

from src.backtest.simulator import TradingSimulator


class TestTradingSimulator:
    """Tests for TradingSimulator class."""

    @pytest.fixture
    def simple_df(self) -> pd.DataFrame:
        """Creates a simple DataFrame with a monotonically increasing price."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        return pd.DataFrame({'Close': [100.0] * 10}, index=dates)

    @pytest.fixture
    def growing_df(self) -> pd.DataFrame:
        """Creates a DataFrame with a steadily rising price."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        prices = [100.0 + i * 5 for i in range(10)]  # 100, 105, 110, …
        return pd.DataFrame({'Close': prices}, index=dates)

    # ── simulate() ──────────────────────────────────────────────────────────

    def test_simulate_returns_dict_with_required_keys(self, simple_df):
        """simulate() must return a dict with all expected keys."""
        sim = TradingSimulator(simple_df, initial_capital=10_000)
        signals = pd.Series([1] * 10, index=simple_df.index)

        result = sim.simulate(signals)

        for key in ('initial_capital', 'final_value', 'profit', 'roi', 'portfolio_values'):
            assert key in result

    def test_simulate_initial_capital_preserved(self, simple_df):
        """The 'initial_capital' entry must equal the constructor argument."""
        sim = TradingSimulator(simple_df, initial_capital=5_000)
        signals = pd.Series([0] * 10, index=simple_df.index)

        result = sim.simulate(signals)

        assert result['initial_capital'] == 5_000

    def test_simulate_all_hold_keeps_capital(self, simple_df):
        """All-hold signals (0) should produce zero profit (no exposure)."""
        sim = TradingSimulator(simple_df, initial_capital=10_000)
        signals = pd.Series([0] * 10, index=simple_df.index)

        result = sim.simulate(signals)

        # No position → portfolio should stay near initial capital
        assert abs(result['profit']) < 1e-6

    def test_simulate_portfolio_values_length(self, simple_df):
        """portfolio_values must have the same length as the input data."""
        sim = TradingSimulator(simple_df, initial_capital=10_000)
        signals = pd.Series([1] * 10, index=simple_df.index)

        result = sim.simulate(signals)

        assert len(result['portfolio_values']) == len(simple_df)

    def test_simulate_roi_positive_for_rising_price(self, growing_df):
        """ROI should be positive when price rises and we are fully invested."""
        sim = TradingSimulator(growing_df, initial_capital=10_000)
        # Buy on day 1 and stay in (signal 1 every day)
        signals = pd.Series([1] * 10, index=growing_df.index)

        result = sim.simulate(signals)

        assert result['roi'] > 0

    def test_simulate_roi_formula_consistent(self, simple_df):
        """roi must be consistent with final_value and initial_capital."""
        sim = TradingSimulator(simple_df, initial_capital=10_000)
        signals = pd.Series([1] * 10, index=simple_df.index)

        result = sim.simulate(signals)

        expected_roi = (result['profit'] / result['initial_capital']) * 100
        assert result['roi'] == pytest.approx(expected_roi, rel=1e-6)

    def test_simulate_profit_equals_final_minus_initial(self, growing_df):
        """profit must equal final_value - initial_capital."""
        sim = TradingSimulator(growing_df, initial_capital=10_000)
        signals = pd.Series([1] * 10, index=growing_df.index)

        result = sim.simulate(signals)

        assert result['profit'] == pytest.approx(
            result['final_value'] - result['initial_capital'], rel=1e-6
        )

    # ── simulate_buy_and_hold() ──────────────────────────────────────────────

    def test_buy_and_hold_returns_dict_with_required_keys(self, simple_df):
        """simulate_buy_and_hold() must return a dict with all expected keys."""
        sim = TradingSimulator(simple_df, initial_capital=10_000)

        result = sim.simulate_buy_and_hold()

        for key in ('initial_capital', 'final_value', 'profit', 'roi'):
            assert key in result

    def test_buy_and_hold_flat_price_zero_profit(self, simple_df):
        """Buy & Hold on a flat price should produce ~zero profit."""
        sim = TradingSimulator(simple_df, initial_capital=10_000)

        result = sim.simulate_buy_and_hold()

        assert abs(result['profit']) < 1e-6

    def test_buy_and_hold_rising_price_positive_roi(self, growing_df):
        """Buy & Hold on a rising price should produce positive ROI."""
        sim = TradingSimulator(growing_df, initial_capital=10_000)

        result = sim.simulate_buy_and_hold()

        assert result['roi'] > 0

    def test_buy_and_hold_roi_formula_consistent(self, growing_df):
        """roi must equal (profit / initial_capital) * 100."""
        sim = TradingSimulator(growing_df, initial_capital=10_000)

        result = sim.simulate_buy_and_hold()

        expected_roi = (result['profit'] / result['initial_capital']) * 100
        assert result['roi'] == pytest.approx(expected_roi, rel=1e-6)

    # ── default capital from config ──────────────────────────────────────────

    def test_default_capital_from_config(self, simple_df):
        """When initial_capital is not given, config default should be used."""
        from config import BACKTEST_CONFIG

        sim = TradingSimulator(simple_df)

        assert sim.initial_capital == BACKTEST_CONFIG.INITIAL_CAPITAL
