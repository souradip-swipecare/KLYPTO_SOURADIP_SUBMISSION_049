"""
Professional Reports Generator
===============================
Comprehensive reporting system for trading results including PDF, Excel, JSON exports.

Author: AlgoTrading System
Date: 2026-01-18
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import logging


class TradingReportsGenerator:
    """
    Professional trading reports generator
    """
    
    def __init__(self, output_dir: str = "results", logger: Optional[logging.Logger] = None):
        """
        Initialize Reports Generator
        
        Args:
            output_dir: Output directory for reports
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_executive_summary(self,
                                  trades: pd.DataFrame,
                                  metrics: Dict[str, Any],
                                  strategy_name: str = "Trading Strategy") -> str:
        """
        Generate executive summary report
        
        Args:
            trades: DataFrame of trades
            metrics: Performance metrics
            strategy_name: Strategy name
            
        Returns:
            Formatted text report
        """
        report = f"""
{'='*80}
EXECUTIVE SUMMARY REPORT
{'='*80}
Strategy:        {strategy_name}
Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

1. PERFORMANCE OVERVIEW
{'-'*80}
Total Return:              {metrics.get('total_return_pct', 0):>10.2f}%
Annual Return:             {metrics.get('annual_return_pct', 0):>10.2f}%
Initial Capital:           ${metrics.get('initial_capital', 0):>10,.2f}
Final Capital:             ${metrics.get('final_capital', 0):>10,.2f}
Total P&L:                 ${metrics.get('total_pnl', 0):>10,.2f}

2. RISK METRICS
{'-'*80}
Max Drawdown:              {metrics.get('max_drawdown_pct', 0):>10.2f}%
Volatility (Annual):       {metrics.get('volatility_pct', 0):>10.2f}%
Sharpe Ratio:              {metrics.get('sharpe_ratio', 0):>10.2f}
Sortino Ratio:             {metrics.get('sortino_ratio', 0):>10.2f}
Calmar Ratio:              {metrics.get('calmar_ratio', 0):>10.2f}
Recovery Factor:           {metrics.get('recovery_factor', 0):>10.2f}

3. TRADE STATISTICS
{'-'*80}
Total Trades:              {metrics.get('trade_count', 0):>10.0f}
Winning Trades:            {metrics.get('winning_trades', 0):>10.0f}
Losing Trades:             {metrics.get('losing_trades', 0):>10.0f}
Win Rate:                  {metrics.get('win_rate_pct', 0):>10.2f}%
Profit Factor:             {metrics.get('profit_factor', 0):>10.2f}
Average Win:               ${metrics.get('avg_win', 0):>10,.2f}
Average Loss:              ${metrics.get('avg_loss', 0):>10,.2f}
Largest Win:               ${metrics.get('largest_win', 0):>10,.2f}
Largest Loss:              ${metrics.get('largest_loss', 0):>10,.2f}

4. EFFICIENCY METRICS
{'-'*80}
Payoff Ratio:              {metrics.get('payoff_ratio', 0):>10.2f}
Return on Risk:            {metrics.get('return_on_risk', 0):>10.2f}
Average Trade Duration:    {metrics.get('avg_bars_held', 0):>10.1f} bars
Max Consecutive Wins:      {metrics.get('max_consecutive_wins', 0):>10.0f}
Max Consecutive Losses:    {metrics.get('max_consecutive_losses', 0):>10.0f}

{'='*80}
"""
        return report
    
    def generate_detailed_trades_report(self,
                                       trades: pd.DataFrame,
                                       output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate detailed trades report
        
        Args:
            trades: DataFrame of trades
            output_file: Optional output CSV file
            
        Returns:
            Enhanced trades DataFrame
        """
        report_df = trades.copy()
        
        # Add cumulative metrics
        report_df['cumulative_pnl'] = report_df['realized_pnl'].cumsum()
        report_df['trade_number'] = range(1, len(report_df) + 1)
        
        # Calculate trade efficiency
        report_df['bars_to_profit'] = report_df['bars_held']
        report_df['pnl_per_bar'] = report_df['realized_pnl'] / (report_df['bars_held'] + 1)
        
        # Win/Loss streak tracking
        report_df['is_win'] = report_df['realized_pnl'] > 0
        report_df['consecutive_streak'] = (report_df['is_win'] != 
                                          report_df['is_win'].shift()).cumsum()
        
        if output_file:
            filepath = f"{self.output_dir}/{output_file}"
            report_df.to_csv(filepath, index=False)
            self.logger.info(f"Detailed trades report saved: {filepath}")
        
        return report_df
    
    def generate_monthly_summary(self,
                                trades: pd.DataFrame,
                                output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate monthly summary
        
        Args:
            trades: DataFrame of trades
            output_file: Optional output CSV file
            
        Returns:
            Monthly summary DataFrame
        """
        trades_copy = trades.copy()
        trades_copy['entry_time'] = pd.to_datetime(trades_copy['entry_time'])
        trades_copy['year_month'] = trades_copy['entry_time'].dt.to_period('M')
        
        monthly = trades_copy.groupby('year_month').agg({
            'realized_pnl': ['sum', 'mean', 'count', 'min', 'max'],
            'realized_pct': ['mean', 'min', 'max']
        }).round(2)
        
        monthly.columns = ['Total_PnL', 'Avg_PnL', 'Trades', 'Min_Trade', 'Max_Trade',
                          'Avg_Return_%', 'Min_Return_%', 'Max_Return_%']
        
        if output_file:
            filepath = f"{self.output_dir}/{output_file}"
            monthly.to_csv(filepath)
            self.logger.info(f"Monthly summary saved: {filepath}")
        
        return monthly
    
    def generate_strategy_comparison_report(self,
                                          strategies: Dict[str, Dict[str, Any]],
                                          output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comparison report for multiple strategies
        
        Args:
            strategies: Dict of strategy_name -> metrics
            output_file: Optional output file
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for strategy_name, metrics in strategies.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total_Return_%': metrics.get('total_return_pct', 0),
                'Annual_Return_%': metrics.get('annual_return_pct', 0),
                'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                'Max_Drawdown_%': metrics.get('max_drawdown_pct', 0),
                'Win_Rate_%': metrics.get('win_rate_pct', 0),
                'Profit_Factor': metrics.get('profit_factor', 0),
                'Total_Trades': metrics.get('trade_count', 0),
                'Avg_Win_$': metrics.get('avg_win', 0),
                'Avg_Loss_$': metrics.get('avg_loss', 0),
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if output_file:
            filepath = f"{self.output_dir}/{output_file}"
            comparison_df.to_csv(filepath, index=False)
            self.logger.info(f"Strategy comparison saved: {filepath}")
        
        return comparison_df
    
    def generate_risk_report(self,
                            metrics: Dict[str, Any],
                            trades: pd.DataFrame,
                            returns: Optional[pd.Series] = None) -> str:
        """
        Generate comprehensive risk report
        
        Args:
            metrics: Performance metrics
            trades: DataFrame of trades
            returns: Series of returns
            
        Returns:
            Formatted risk report
        """
        report = f"""
{'='*80}
RISK ANALYSIS REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DRAWDOWN ANALYSIS
{'-'*80}
Maximum Drawdown:          {metrics.get('max_drawdown_pct', 0):>10.2f}%
Average Drawdown:          {metrics.get('avg_drawdown_pct', 0):>10.2f}%
Max Drawdown Duration:     {metrics.get('max_drawdown_duration', 0):>10} bars
Current Drawdown:          {metrics.get('current_drawdown_pct', 0):>10.2f}%

2. VOLATILITY & STANDARD DEVIATION
{'-'*80}
Daily Volatility:          {metrics.get('daily_volatility_pct', 0):>10.2f}%
Annual Volatility:         {metrics.get('volatility_pct', 0):>10.2f}%
Return/Volatility:         {metrics.get('return_risk_ratio', 0):>10.2f}

3. VALUE AT RISK
{'-'*80}
VaR (95%):                 {metrics.get('var_95', 0):>10.4f}
CVaR (95%):                {metrics.get('cvar_95', 0):>10.4f}
Expected Shortfall:        {metrics.get('expected_shortfall', 0):>10.4f}

4. TRADE LOSS ANALYSIS
{'-'*80}
"""
        if len(trades) > 0:
            losing_trades = trades[trades['realized_pnl'] < 0]
            if len(losing_trades) > 0:
                avg_loss = losing_trades['realized_pnl'].mean()
                max_loss = losing_trades['realized_pnl'].min()
                report += f"""
Total Losing Trades:       {len(losing_trades):>10}
Average Loss Amount:       ${avg_loss:>10,.2f}
Largest Loss:              ${max_loss:>10,.2f}
% of Total Trades:         {(len(losing_trades)/len(trades)*100):>10.2f}%
"""
        
        report += f"""
5. RISK-ADJUSTED RETURNS
{'-'*80}
Sharpe Ratio:              {metrics.get('sharpe_ratio', 0):>10.2f}
Sortino Ratio:             {metrics.get('sortino_ratio', 0):>10.2f}
Calmar Ratio:              {metrics.get('calmar_ratio', 0):>10.2f}
Information Ratio:         {metrics.get('information_ratio', 0):>10.2f}

{'='*80}
"""
        return report
    
    def generate_json_report(self,
                            trades: pd.DataFrame,
                            metrics: Dict[str, Any],
                            output_file: str = "report.json") -> str:
        """
        Generate JSON format report
        
        Args:
            trades: DataFrame of trades
            metrics: Performance metrics
            output_file: Output JSON filename
            
        Returns:
            Path to saved JSON file
        """
        # Convert DataFrames to JSON-serializable format
        trades_json = trades.to_dict('records')
        
        report_json = {
            'generated_at': datetime.now().isoformat(),
            'summary': metrics,
            'trades': trades_json,
            'statistics': {
                'total_trades': len(trades),
                'winning_trades': len(trades[trades['realized_pnl'] > 0]),
                'losing_trades': len(trades[trades['realized_pnl'] < 0]),
            }
        }
        
        filepath = f"{self.output_dir}/{output_file}"
        with open(filepath, 'w') as f:
            json.dump(report_json, f, indent=2, default=str)
        
        self.logger.info(f"JSON report saved: {filepath}")
        return filepath
    
    def generate_performance_metrics_table(self,
                                          metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate metrics as DataFrame
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Metrics DataFrame
        """
        metrics_df = pd.DataFrame([
            {'Metric': 'Total Return (%)', 'Value': metrics.get('total_return_pct', 0)},
            {'Metric': 'Annual Return (%)', 'Value': metrics.get('annual_return_pct', 0)},
            {'Metric': 'Sharpe Ratio', 'Value': metrics.get('sharpe_ratio', 0)},
            {'Metric': 'Sortino Ratio', 'Value': metrics.get('sortino_ratio', 0)},
            {'Metric': 'Max Drawdown (%)', 'Value': metrics.get('max_drawdown_pct', 0)},
            {'Metric': 'Win Rate (%)', 'Value': metrics.get('win_rate_pct', 0)},
            {'Metric': 'Profit Factor', 'Value': metrics.get('profit_factor', 0)},
            {'Metric': 'Payoff Ratio', 'Value': metrics.get('payoff_ratio', 0)},
            {'Metric': 'Recovery Factor', 'Value': metrics.get('recovery_factor', 0)},
            {'Metric': 'Total Trades', 'Value': metrics.get('trade_count', 0)},
            {'Metric': 'Avg Trade Duration (bars)', 'Value': metrics.get('avg_bars_held', 0)},
        ])
        return metrics_df
    
    def export_to_excel(self,
                       trades: pd.DataFrame,
                       metrics: Dict[str, Any],
                       monthly_summary: Optional[pd.DataFrame] = None,
                       output_file: str = "trading_report.xlsx") -> str:
        """
        Export comprehensive report to Excel
        
        Args:
            trades: DataFrame of trades
            metrics: Performance metrics
            monthly_summary: Optional monthly summary DataFrame
            output_file: Output Excel filename
            
        Returns:
            Path to saved Excel file
        """
        filepath = f"{self.output_dir}/{output_file}"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_df = self.generate_performance_metrics_table(metrics)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed Trades
            trades_report = self.generate_detailed_trades_report(trades)
            trades_report.to_excel(writer, sheet_name='Trades', index=False)
            
            # Sheet 3: Monthly Summary
            if monthly_summary is not None:
                monthly_summary.to_excel(writer, sheet_name='Monthly')
        
        self.logger.info(f"Excel report saved: {filepath}")
        return filepath


if __name__ == "__main__":
    print("Trading Reports Generator Module Loaded Successfully")
