"""
Module for results visualization.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import BACKTEST_CONFIG

logger = logging.getLogger("forex_ml.utils.visualization")


class Plotter:
    """Class for creating charts."""
    
    @staticmethod
    def _save_figure(save_path: Union[str, Path]) -> None:
        """Saves figure ensuring directory exists."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Chart saved: {save_path}")
    
    @staticmethod
    def plot_feature_importance(
        importances: pd.DataFrame,
        top_n: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """
        Creates feature importance chart.
        
        Args:
            importances: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            save_path: Path to save chart
            show: Whether to display chart
        """
        plt.figure(figsize=(10, 6))
        importances.head(top_n).plot(
            x='feature', 
            y='importance', 
            kind='barh',
            ax=plt.gca()
        )
        plt.title("Feature Importance")
        plt.tight_layout()
        
        if save_path:
            Plotter._save_figure(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_backtest(
        index: pd.DatetimeIndex,
        cumulative_returns: Dict[str, pd.Series],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """
        Creates backtest chart with cumulative returns.
        
        Args:
            index: DatetimeIndex for X axis
            cumulative_returns: Dictionary {name: Series with returns}
            save_path: Path to save chart
            show: Whether to display chart
        """
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        plt.figure(figsize=(12, 6))
        
        for i, (name, returns) in enumerate(cumulative_returns.items()):
            plt.plot(
                index, 
                returns, 
                label=name, 
                linewidth=2, 
                color=colors[i % len(colors)]
            )
        
        plt.legend()
        plt.title('Backtest: ML Strategy vs Buy & Hold')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            Plotter._save_figure(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_portfolio_with_adx(
        index: pd.DatetimeIndex,
        cumulative_returns: Dict[str, pd.Series],
        adx: pd.Series,
        adx_threshold: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """
        Creates portfolio chart with ADX.
        
        Args:
            index: DatetimeIndex for X axis
            cumulative_returns: Dictionary {name: Series with returns %}
            adx: Series with ADX
            adx_threshold: ADX threshold to mark
            save_path: Path to save chart
            show: Whether to display chart
        """
        adx_threshold = adx_threshold or BACKTEST_CONFIG.ADX_THRESHOLD
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        plt.figure(figsize=(14, 7))
        
        # Upper chart - percentage returns
        plt.subplot(2, 1, 1)
        for i, (name, returns) in enumerate(cumulative_returns.items()):
            plt.plot(
                index,
                (returns - 1) * 100,
                label=name,
                linewidth=2,
                color=colors[i % len(colors)]
            )
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.ylabel('Return (%)')
        plt.title('Returns Comparison - Backtest ML Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Lower chart - ADX
        plt.subplot(2, 1, 2)
        plt.plot(index, adx, label='ADX', linewidth=2, color='purple')
        plt.axhline(
            y=adx_threshold, 
            color='red', 
            linestyle='--', 
            alpha=0.5, 
            label=f'Strong trend threshold ({adx_threshold})'
        )
        plt.ylabel('ADX')
        plt.xlabel('Date')
        plt.title('ADX (trend strength indicator)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Plotter._save_figure(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def print_feature_importance_analysis(
        importances: pd.DataFrame,
        top_n: int = 10
    ) -> None:
        """
        Prints feature importance analysis.
        
        Args:
            importances: DataFrame with importances
            top_n: Number of top features
        """
        # Interaction features
        interaction_features = [
            f for f in importances['feature'] if '_x_' in f
        ]
        
        if interaction_features:
            logger.info(f"Interaction features ({len(interaction_features)}):")
            logger.info(f"\n{importances[importances['feature'].isin(interaction_features)]}")
        
        logger.info(f"Top {top_n} features:")
        logger.info(f"\n{importances.head(top_n)}")
        
        # Warning check
        top_importance = importances.iloc[0]['importance']
        bottom_importance = importances.iloc[-1]['importance']
        
        logger.info(f"Top feature importance: {top_importance:.4f}")
        logger.info(f"Bottom feature importance: {bottom_importance:.4f}")
        
        if top_importance < 0.15:
            logger.warning("⚠️ WARNING: Features have too uniform importance - model is learning poorly!")
