"""
Tests for RiskMetrics class.
"""
import pytest
import pandas as pd
import numpy as np

from src.backtest.metrics import RiskMetrics


class TestRiskMetrics:
    """Tests for RiskMetrics class."""
    
    def test_sharpe_ratio_positive_returns(self, sample_returns):
        """Sharpe ratio should be positive for positive mean returns."""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02] * 20)
        metrics = RiskMetrics(positive_returns)
        
        assert metrics.sharpe_ratio() > 0
    
    def test_sharpe_ratio_negative_returns(self):
        """Sharpe ratio should be negative for negative mean returns."""
        negative_returns = pd.Series([-0.01, -0.02, -0.015, -0.01, -0.02] * 20)
        metrics = RiskMetrics(negative_returns)
        
        assert metrics.sharpe_ratio() < 0
    
    def test_sharpe_ratio_zero_std(self):
        """Sharpe ratio should be 0 when std is 0."""
        constant_returns = pd.Series([0.01] * 50)
        metrics = RiskMetrics(constant_returns)
        
        assert metrics.sharpe_ratio() == 0.0
    
    def test_max_drawdown_negative(self, sample_returns):
        """Max drawdown should always be negative or zero."""
        metrics = RiskMetrics(sample_returns)
        
        assert metrics.max_drawdown() <= 0
    
    def test_max_drawdown_severe_loss(self):
        """Max drawdown should reflect severe losses."""
        # 50% loss followed by recovery
        returns = pd.Series([0.1, -0.5, 0.2, 0.1])
        metrics = RiskMetrics(returns)
        
        drawdown = metrics.max_drawdown()
        assert drawdown < -0.3  # Significant drawdown
    
    def test_max_drawdown_no_loss(self):
        """Max drawdown should be 0 for only positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        metrics = RiskMetrics(returns)
        
        assert metrics.max_drawdown() == 0.0
    
    def test_total_return_positive(self):
        """Total return should be positive for positive returns."""
        returns = pd.Series([0.1, 0.1, 0.1])  # ~33% total
        metrics = RiskMetrics(returns)
        
        total = metrics.total_return()
        assert total > 0.3
        assert total < 0.4
    
    def test_total_return_negative(self):
        """Total return should be negative for negative returns."""
        returns = pd.Series([-0.1, -0.1, -0.1])
        metrics = RiskMetrics(returns)
        
        assert metrics.total_return() < 0
    
    def test_volatility_positive(self, sample_returns):
        """Volatility should always be positive."""
        metrics = RiskMetrics(sample_returns)
        
        assert metrics.volatility() > 0
    
    def test_volatility_zero_for_constant(self):
        """Volatility should be 0 for constant returns."""
        constant_returns = pd.Series([0.01] * 50)
        metrics = RiskMetrics(constant_returns)
        
        assert metrics.volatility() == 0.0
    
    def test_get_all_metrics_keys(self, sample_returns):
        """get_all_metrics should return all expected keys."""
        metrics = RiskMetrics(sample_returns)
        all_metrics = metrics.get_all_metrics()
        
        expected_keys = {'sharpe_ratio', 'max_drawdown', 'total_return', 'volatility'}
        assert set(all_metrics.keys()) == expected_keys
    
    def test_custom_rf_rate(self, sample_returns):
        """Custom risk-free rate should be used in calculations."""
        metrics_low_rf = RiskMetrics(sample_returns, rf_rate=0.0)
        metrics_high_rf = RiskMetrics(sample_returns, rf_rate=0.1)
        
        # Higher rf should result in lower Sharpe
        assert metrics_low_rf.sharpe_ratio() > metrics_high_rf.sharpe_ratio()
    
    def test_handles_nan_in_returns(self):
        """Should handle NaN values in returns."""
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, np.nan, 0.01])
        metrics = RiskMetrics(returns_with_nan)
        
        # Should not raise, NaN filled with 0
        sharpe = metrics.sharpe_ratio()
        assert np.isfinite(sharpe)
