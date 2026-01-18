"""
NIFTY 50 Algorithmic Trading System - Backtest Engine
Professional backtesting with comprehensive metrics and analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    num_trades: int
    payoff_ratio: float
    recovery_factor: float


class BacktestEngine:
    """
    Professional backtesting engine with comprehensive metrics.
    """
    
    def __init__(self, config: Dict):
        """Initialize backtest engine."""
        self.config = config
        self.risk_free_rate = config.get('backtest', {}).get('risk_free_rate', 0.065)
        self.trading_days_per_year = config.get('backtest', {}).get('trading_days_per_year', 252)
        logger.info("BacktestEngine initialized")
    
    def run_backtest(self, df: pd.DataFrame, trades: List[Dict], 
                     initial_capital: float = 100000) -> Tuple[BacktestMetrics, pd.DataFrame]:
        """
        Run complete backtest with metrics calculation.
        
        Args:
            df: DataFrame with market data
            trades: List of executed trades
            initial_capital: Starting capital
            
        Returns:
            Tuple of (BacktestMetrics, DataFrame with equity curve)
        """
        logger.info("="*60)
        logger.info("STARTING BACKTEST")
        logger.info("="*60)
        
        # Initialize equity curve
        df_backtest = df.copy()
        df_backtest['equity'] = initial_capital
        df_backtest['cash'] = initial_capital
        df_backtest['pnl'] = 0.0
        df_backtest['cumulative_pnl'] = 0.0
        df_backtest['equity_curve'] = initial_capital
        
        # Process trades
        cumulative_pnl = 0.0
        for trade in trades:
            exit_idx = trade['exit_idx']
            pnl = trade['pnl']
            cumulative_pnl += pnl
            
            # Update equity from exit point onward
            if exit_idx < len(df_backtest):
                df_backtest.loc[df_backtest.index[exit_idx:], 'cumulative_pnl'] = cumulative_pnl
                df_backtest.loc[df_backtest.index[exit_idx:], 'equity_curve'] = initial_capital + cumulative_pnl
        
        # Calculate daily returns
        df_backtest['daily_return'] = df_backtest['equity_curve'].pct_change().fillna(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(df_backtest, trades, initial_capital)
        
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Total Return: {metrics.total_return:.2f}%")
        logger.info(f"Annual Return: {metrics.annual_return:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
        logger.info(f"Max Drawdown Duration: {metrics.max_drawdown_duration} bars")
        logger.info(f"Win Rate: {metrics.win_rate:.2f}%")
        logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"Average Trade PnL: ${metrics.avg_trade_pnl:.2f}")
        logger.info(f"Total Trades: {metrics.num_trades}")
        logger.info("="*60)
        
        return metrics, df_backtest
    
    def _calculate_metrics(self, df: pd.DataFrame, trades: List[Dict], 
                          initial_capital: float) -> BacktestMetrics:
        """
        Calculate comprehensive backtest metrics.
        
        Args:
            df: DataFrame with backtest data
            trades: List of trades
            initial_capital: Starting capital
            
        Returns:
            BacktestMetrics object
        """
        # Basic returns
        final_equity = df['equity_curve'].iloc[-1]
        total_return_pct = (final_equity - initial_capital) / initial_capital * 100
        
        # Annualized return
        trading_days = len(df)
        years = trading_days / 252
        annual_return = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        annual_return_pct = annual_return * 100
        
        # Sharpe Ratio
        daily_returns = df['daily_return']
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-6)
        
        # Sortino Ratio (only downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-6)
        
        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Max Drawdown Duration
        drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # Trade statistics
        trade_stats = self._calculate_trade_stats(trades)
        
        return BacktestMetrics(
            total_return=total_return_pct,
            annual_return=annual_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            avg_trade_pnl=trade_stats['avg_pnl'],
            num_trades=len(trades),
            payoff_ratio=trade_stats['payoff_ratio'],
            recovery_factor=total_return_pct / (abs(max_drawdown) + 1e-6)
        )
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """
        Calculate the longest drawdown duration in bars.
        
        Args:
            drawdown: Series of drawdown values
            
        Returns:
            Maximum drawdown duration in bars
        """
        in_drawdown = drawdown < -0.001
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        
        durations = []
        for group_id in drawdown_groups[in_drawdown].unique():
            duration = sum(drawdown_groups == group_id)
            durations.append(duration)
        
        return max(durations) if durations else 0
    
    def _calculate_trade_stats(self, trades: List[Dict]) -> Dict:
        """
        Calculate trade-based statistics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_pnl': 0,
                'payoff_ratio': 0
            }
        
        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        
        stats = {
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'profit_factor': total_wins / (total_losses + 1e-6),
            'avg_pnl': np.mean(pnls),
            'payoff_ratio': (np.mean(winning_trades) / abs(np.mean(losing_trades))) if losing_trades else 0
        }
        
        return stats
    
    def generate_backtest_report(self, metrics: BacktestMetrics, trades: List[Dict]) -> str:
        """
        Generate professional backtest report.
        
        Args:
            metrics: BacktestMetrics object
            trades: List of trades
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("NIFTY 50 ALGORITHMIC TRADING SYSTEM - BACKTEST REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 70)
        report.append(f"Total Return:                   {metrics.total_return:>10.2f}%")
        report.append(f"Annualized Return:             {metrics.annual_return:>10.2f}%")
        report.append(f"Sharpe Ratio:                   {metrics.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:                  {metrics.sortino_ratio:>10.2f}")
        report.append(f"Max Drawdown:                   {metrics.max_drawdown:>10.2f}%")
        report.append(f"Max Drawdown Duration:          {metrics.max_drawdown_duration:>10} bars")
        report.append(f"Recovery Factor:                {metrics.recovery_factor:>10.2f}")
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 70)
        report.append(f"Total Trades:                   {metrics.num_trades:>10}")
        report.append(f"Win Rate:                       {metrics.win_rate:>10.2f}%")
        report.append(f"Profit Factor:                  {metrics.profit_factor:>10.2f}")
        report.append(f"Average Trade PnL:             ${metrics.avg_trade_pnl:>10.2f}")
        report.append(f"Payoff Ratio:                   {metrics.payoff_ratio:>10.2f}")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


class MonteCarloAnalysis:
    """Monte Carlo analysis for strategy robustness."""
    
    @staticmethod
    def run_monte_carlo(returns: pd.Series, num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation on returns.
        
        Args:
            returns: Series of returns
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary of Monte Carlo statistics
        """
        if len(returns) == 0:
            return {}
        
        final_returns = []
        final_drawdowns = []
        
        for _ in range(num_simulations):
            simulated_returns = np.random.choice(returns.values, size=len(returns), replace=True)
            cumulative = (1 + simulated_returns).cumprod()
            final_return = (cumulative[-1] - 1) * 100
            
            drawdown = (cumulative - cumulative.max()) / cumulative.max()
            max_dd = drawdown.min() * 100
            
            final_returns.append(final_return)
            final_drawdowns.append(max_dd)
        
        return {
            'mean_return': np.mean(final_returns),
            'std_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'mean_max_dd': np.mean(final_drawdowns),
            'worst_case_dd': np.percentile(final_drawdowns, 95)
        }


if __name__ == "__main__":
    config = {
        'backtest': {
            'risk_free_rate': 0.065,
            'trading_days_per_year': 252
        }
    }
    
    engine = BacktestEngine(config)
    print("Backtest Engine Module Ready")
