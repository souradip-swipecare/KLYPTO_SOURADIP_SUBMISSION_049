"""
Results Analysis Script
Analyze and visualize trading results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Create visualizations for trading results."""
    
    def __init__(self, output_dir: str = 'results'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")
        logger.info(f"Visualizer initialized: {self.output_dir}")
    
    def plot_equity_curve(self, equity_curve: pd.Series, filename: str = 'equity_curve.png'):
        """
        Plot equity curve over time.
        
        Args:
            equity_curve: Series with cumulative equity
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#2E86AB')
        ax.fill_between(equity_curve.index, equity_curve.values, alpha=0.3, color='#2E86AB')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Equity curve saved: {filepath}")
    
    def plot_drawdown(self, equity_curve: pd.Series, filename: str = 'drawdown.png'):
        """
        Plot drawdown over time.
        
        Args:
            equity_curve: Series with cumulative equity
            filename: Output filename
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(drawdown.index, drawdown.values, color='#A23B72', alpha=0.6)
        ax.plot(drawdown.index, drawdown.values, color='#A23B72', linewidth=1)
        
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Drawdown plot saved: {filepath}")
    
    def plot_trade_distribution(self, trades: list, filename: str = 'trade_distribution.png'):
        """
        Plot distribution of trade P&L.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
        """
        df = pd.DataFrame(trades)
        pnls = df['pnl'].values
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(pnls, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0].axvline(pnls.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${pnls.mean():.2f}')
        axes[0].set_title('P&L Distribution', fontweight='bold')
        axes[0].set_xlabel('P&L ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Win vs Loss
        winning = len(pnls[pnls > 0])
        losing = len(pnls[pnls < 0])
        colors = ['#06A77D', '#D62839']
        axes[1].bar(['Winning', 'Losing'], [winning, losing], color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_title('Win vs Loss Trades', fontweight='bold')
        axes[1].set_ylabel('Count')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        total = winning + losing
        axes[1].text(0, winning + 2, f'{winning/total*100:.1f}%', ha='center', fontweight='bold')
        axes[1].text(1, losing + 2, f'{losing/total*100:.1f}%', ha='center', fontweight='bold')
        
        fig.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Trade distribution saved: {filepath}")
    
    def plot_monthly_returns(self, equity_curve: pd.Series, filename: str = 'monthly_returns.png'):
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_curve: Series with cumulative equity
            filename: Output filename
        """
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().fillna(0)
        
        # Group by month and calculate returns
        if hasattr(equity_curve.index, 'to_period'):
            monthly = equity_curve.resample('M').last()
            monthly_returns = monthly.pct_change() * 100
        else:
            monthly_returns = daily_returns.groupby(daily_returns.index.month).sum() * 100
        
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#06A77D' if x > 0 else '#D62839' for x in monthly_returns.values]
        ax.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_title('Monthly Returns', fontweight='bold', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Monthly returns saved: {filepath}")
    
    def plot_trade_duration_vs_pnl(self, trades: list, filename: str = 'duration_vs_pnl.png'):
        """
        Plot trade duration vs P&L scatter.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
        """
        df = pd.DataFrame(trades)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color points by win/loss
        colors = ['#06A77D' if x > 0 else '#D62839' for x in df['pnl'].values]
        ax.scatter(df['bars_held'] if 'bars_held' in df.columns else range(len(df)), 
                  df['pnl'], c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Trade Duration vs P&L', fontweight='bold', fontsize=14)
        ax.set_xlabel('Duration (bars)')
        ax.set_ylabel('P&L ($)')
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Duration vs P&L scatter saved: {filepath}")


class ResultsComparison:
    """Compare multiple backtest results."""
    
    @staticmethod
    def compare_strategies(results_dict: dict, filename: str = 'strategy_comparison.csv'):
        """
        Compare multiple strategy results.
        
        Args:
            results_dict: Dictionary with strategy_name: metrics
            filename: Output filename
        """
        comparison_df = pd.DataFrame(results_dict).T
        comparison_df.to_csv(filename)
        logger.info(f"Comparison saved: {filename}")
        return comparison_df
    
    @staticmethod
    def create_comparison_chart(results_dict: dict, filename: str = 'comparison_chart.png'):
        """
        Create visual comparison of strategies.
        
        Args:
            results_dict: Dictionary with strategy_name: metrics
            filename: Output filename
        """
        df = pd.DataFrame(results_dict).T
        
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 4))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            axes[idx].barh(df.index, df[metric].values, color='#2E86AB', alpha=0.7, edgecolor='black')
            axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison chart saved: {filename}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Results visualization module loaded")
