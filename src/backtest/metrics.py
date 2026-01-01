"""
Module for calculating risk metrics.
"""
import logging
from typing import Dict

import pandas as pd
import numpy as np

from config import BACKTEST_CONFIG

logger = logging.getLogger("forex_ml.backtest.metrics")


class RiskMetrics:
    """Class for calculating risk metrics."""
    
    def __init__(
        self, 
        returns: pd.Series,
        cumulative_returns: pd.Series = None,
        rf_rate: float = None
    ):
        """
        Initializes risk metrics.
        
        Args:
            returns: Series with daily returns
            cumulative_returns: Series with cumulative returns (optional)
            rf_rate: Risk-free rate (default from config)
        """
        self.returns = returns.fillna(0)
        self.cumulative_returns = cumulative_returns
        self.rf_rate = rf_rate if rf_rate is not None else BACKTEST_CONFIG.RISK_FREE_RATE
    
    def sharpe_ratio(self) -> float:
        """
        Calculates annualized Sharpe ratio.
        
        Returns:
            Sharpe ratio
        """
        excess_returns = self.returns - self.rf_rate
        if excess_returns.std() == 0:
            return 0.0
        
        return (
            np.sqrt(BACKTEST_CONFIG.TRADING_DAYS_PER_YEAR) * 
            (excess_returns.mean() / excess_returns.std())
        )
    
    def max_drawdown(self) -> float:
        """
        Calculates maximum drawdown.
        
        Returns:
            Maximum drawdown (negative value)
        """
        if self.cumulative_returns is None:
            self.cumulative_returns = (1 + self.returns).cumprod()
        
        running_max = self.cumulative_returns.cummax()
        drawdown = (self.cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def total_return(self) -> float:
        """
        Calculates total return.
        
        Returns:
            Total return (e.g., 0.5 = 50%)
        """
        if self.cumulative_returns is None:
            self.cumulative_returns = (1 + self.returns).cumprod()
        
        return self.cumulative_returns.iloc[-1] - 1
    
    def volatility(self) -> float:
        """
        Calculates annualized volatility.
        
        Returns:
            Annualized volatility
        """
        return self.returns.std() * np.sqrt(BACKTEST_CONFIG.TRADING_DAYS_PER_YEAR)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Returns all risk metrics.
        
        Returns:
            Dictionary with metrics
        """
        return {
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'total_return': self.total_return(),
            'volatility': self.volatility()
        }
    
    @staticmethod
    def print_comparison(
        metrics_list: list,
        names: list
    ) -> None:
        """
        Prints risk metrics comparison.
        
        Args:
            metrics_list: List of RiskMetrics objects
            names: List of strategy names
        """
        logger.info(f"📈 Risk metrics:")
        
        for name, metrics in zip(names, metrics_list):
            all_metrics = metrics.get_all_metrics()
            logger.info(f"   Sharpe Ratio ({name}): {all_metrics['sharpe_ratio']:.2f}")
        
        for name, metrics in zip(names, metrics_list):
            all_metrics = metrics.get_all_metrics()
            logger.info(f"   Max Drawdown ({name}): {all_metrics['max_drawdown']:.2%}")
