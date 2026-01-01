"""
Module for trading simulation with capital.
"""
import logging
from typing import List, Dict

import pandas as pd
import numpy as np

from config import BACKTEST_CONFIG

logger = logging.getLogger("forex_ml.backtest.simulator")


class TradingSimulator:
    """Class for trading simulation with capital."""
    
    def __init__(self, df: pd.DataFrame, initial_capital: float = None):
        """
        Initializes simulator.
        
        Args:
            df: DataFrame with data (must contain 'Close')
            initial_capital: Initial capital (default from config)
        """
        self.df = df
        self.initial_capital = initial_capital or BACKTEST_CONFIG.INITIAL_CAPITAL
    
    def simulate(self, signals: pd.Series) -> Dict[str, float]:
        """
        Simulates trading based on signals.
        
        Args:
            signals: Series with signals (1 = buy, 0 = hold/sell)
        
        Returns:
            Dictionary with simulation results
        """
        cash = self.initial_capital
        shares = 0.0
        portfolio_values: List[float] = []
        
        df_sim = self.df.copy()
        df_sim['Signal'] = signals
        
        for idx, row in df_sim.iterrows():
            signal = row['Signal']
            close_price = row['Close']
            
            if signal == 1 and shares == 0:  # Buy if we don't have shares
                shares = cash / close_price
                cash = 0
            elif signal == 0 and shares > 0:  # Sell if we have shares
                cash = shares * close_price
                shares = 0
            
            # Portfolio value
            current_value = cash + (shares * close_price)
            portfolio_values.append(current_value)
        
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        profit = final_value - self.initial_capital
        roi = (profit / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'profit': profit,
            'roi': roi,
            'portfolio_values': portfolio_values
        }
    
    def simulate_buy_and_hold(self) -> Dict[str, float]:
        """
        Simulates Buy & Hold strategy.
        
        Returns:
            Dictionary with simulation results
        """
        shares_bought = self.initial_capital / self.df['Close'].iloc[0]
        final_value = shares_bought * self.df['Close'].iloc[-1]
        profit = final_value - self.initial_capital
        roi = (profit / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'profit': profit,
            'roi': roi
        }
    
    def print_simulation_results(
        self, 
        results: Dict[str, float], 
        name: str = "Strategy"
    ) -> None:
        """
        Prints simulation results.
        
        Args:
            results: Dictionary with results
            name: Strategy name
        """
        logger.info(f"{name}:")
        logger.info(f"   Initial capital:  {results['initial_capital']:,.0f} USD")
        logger.info(f"   Final capital:    {results['final_value']:,.0f} USD")
        logger.info(f"   Profit:           {results['profit']:,.0f} USD")
        logger.info(f"   ROI:              {results['roi']:.2f}%")
    
    def compare_strategies(
        self, 
        results_list: List[Dict[str, float]], 
        names: List[str]
    ) -> None:
        """
        Compares results of different strategies.
        
        Args:
            results_list: List of dictionaries with results
            names: List of strategy names
        """
        logger.info(f"💰 Strategy comparison:")
        
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                diff_profit = results_list[j]['profit'] - results_list[i]['profit']
                diff_roi = results_list[j]['roi'] - results_list[i]['roi']
                logger.info(f"   {names[j]} vs {names[i]}: {diff_profit:,.0f} USD ({diff_roi:.2f}%)")
