"""
Trading Dashboard Module
========================
Real-time and historical trading dashboard with comprehensive metrics visualization.

Author: AlgoTrading System
Date: 2026-01-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging


class TradingDashboard:
    """
    Professional trading dashboard for monitoring and analysis
    """
    
    def __init__(self, output_dir: str = "results", style: str = "seaborn"):
        """
        Initialize Trading Dashboard
        
        Args:
            output_dir: Output directory for dashboard files
            style: Matplotlib style (seaborn, dark_background, etc.)
        """
        self.output_dir = output_dir
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (16, 10)
        self.logger = logging.getLogger(__name__)
    
    def create_performance_dashboard(self, 
                                    trades: pd.DataFrame,
                                    metrics: Dict[str, Any],
                                    equity_curve: pd.Series,
                                    filename: str = "performance_dashboard.png") -> str:
        """
        Create comprehensive performance dashboard
        
        Args:
            trades: DataFrame of trades
            metrics: Dictionary of performance metrics
            equity_curve: Equity curve series
            filename: Output filename
            
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#1f77b4')
        ax1.fill_between(equity_curve.index, equity_curve.values, alpha=0.3)
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Key Metrics Box
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        metrics_text = self._format_metrics_box(metrics)
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        if len(trades) > 0:
            monthly_returns = self._calculate_monthly_returns(trades)
            sns.heatmap(monthly_returns, annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=ax3, cbar_kws={'label': 'Return %'})
            ax3.set_title('Monthly Returns (%)', fontsize=11, fontweight='bold')
        
        # 4. Drawdown Chart
        ax4 = fig.add_subplot(gs[1, 1])
        drawdown = self._calculate_drawdown(equity_curve)
        ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
        ax4.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        ax4.set_title('Drawdown', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Drawdown %')
        ax4.grid(True, alpha=0.3)
        
        # 5. Win/Loss Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        if len(trades) > 0:
            pnl = trades['realized_pnl'].values
            colors = ['green' if x > 0 else 'red' for x in pnl]
            ax5.bar(range(len(pnl)), pnl, color=colors, alpha=0.7)
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_title('Trade P&L Distribution', fontsize=11, fontweight='bold')
            ax5.set_ylabel('P&L ($)')
            ax5.set_xlabel('Trade #')
        
        # 6. Cumulative P&L
        ax6 = fig.add_subplot(gs[2, 0])
        if len(trades) > 0:
            cumulative_pnl = trades['realized_pnl'].cumsum()
            colors = ['green' if x > 0 else 'red' for x in cumulative_pnl]
            ax6.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.5)
            ax6.plot(cumulative_pnl.values, linewidth=2)
            ax6.set_title('Cumulative P&L', fontsize=11, fontweight='bold')
            ax6.set_ylabel('Cumulative P&L ($)')
            ax6.grid(True, alpha=0.3)
        
        # 7. Win Rate & Profit Factor
        ax7 = fig.add_subplot(gs[2, 1])
        if len(trades) > 0:
            labels = ['Wins', 'Losses', 'Breakeven']
            win_count = len(trades[trades['realized_pnl'] > 0])
            loss_count = len(trades[trades['realized_pnl'] < 0])
            be_count = len(trades[trades['realized_pnl'] == 0])
            sizes = [win_count, loss_count, be_count]
            colors_pie = ['green', 'red', 'gray']
            ax7.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90)
            ax7.set_title('Trade Outcomes', fontsize=11, fontweight='bold')
        
        # 8. Daily Returns Distribution
        ax8 = fig.add_subplot(gs[2, 2])
        if len(equity_curve) > 1:
            daily_returns = equity_curve.pct_change().dropna() * 100
            ax8.hist(daily_returns, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax8.axvline(daily_returns.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
            ax8.set_title('Daily Returns Distribution', fontsize=11, fontweight='bold')
            ax8.set_xlabel('Daily Return (%)')
            ax8.set_ylabel('Frequency')
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        fig.suptitle('Trading Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Dashboard saved: {filepath}")
        return filepath
    
    def create_risk_dashboard(self,
                             trades: pd.DataFrame,
                             metrics: Dict[str, Any],
                             returns: pd.Series,
                             filename: str = "risk_dashboard.png") -> str:
        """
        Create risk analysis dashboard
        
        Args:
            trades: DataFrame of trades
            metrics: Performance metrics
            returns: Series of returns
            filename: Output filename
            
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Return Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(returns, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2, label='Median')
        ax1.set_title('Return Distribution', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Return')
        ax1.legend()
        
        # 2. Risk Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        risk_text = self._format_risk_metrics(metrics)
        ax2.text(0.05, 0.95, risk_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 3. VaR Visualization
        ax3 = fig.add_subplot(gs[0, 2])
        var_95 = returns.quantile(0.05)
        ax3.hist(returns, bins=40, color='steelblue', alpha=0.7)
        ax3.axvline(var_95, color='red', linestyle='--', linewidth=2, 
                   label=f'VaR 95%: {var_95:.4f}')
        ax3.set_title('Value at Risk', fontsize=11, fontweight='bold')
        ax3.legend()
        
        # 4. Rolling Volatility
        ax4 = fig.add_subplot(gs[1, :2])
        rolling_vol = returns.rolling(window=30).std()
        ax4.plot(rolling_vol.index, rolling_vol.values, color='darkred', linewidth=2)
        ax4.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.3, color='red')
        ax4.set_title('Rolling 30-Day Volatility', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Volatility')
        ax4.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio Over Time
        ax5 = fig.add_subplot(gs[1, 2])
        sharpe_rolling = (returns.rolling(window=60).mean() / 
                         returns.rolling(window=60).std()) * np.sqrt(252)
        ax5.plot(sharpe_rolling.index, sharpe_rolling.values, color='blue', linewidth=2)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_title('Rolling Sharpe Ratio (60d)', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Losing Trades Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        if len(trades) > 0:
            losing_trades = trades[trades['realized_pnl'] < 0]['realized_pnl'].values
            ax6.hist(losing_trades, bins=20, color='red', alpha=0.7, edgecolor='black')
            ax6.set_title('Losing Trades Distribution', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Loss Amount ($)')
        
        # 7. Trade Duration vs Loss
        ax7 = fig.add_subplot(gs[2, 1])
        if len(trades) > 0:
            losing = trades[trades['realized_pnl'] < 0]
            if len(losing) > 0:
                ax7.scatter(losing['bars_held'], losing['realized_pnl'], alpha=0.6, s=50)
                ax7.set_title('Losing Trades: Duration vs Loss', fontsize=11, fontweight='bold')
                ax7.set_xlabel('Duration (bars)')
                ax7.set_ylabel('Loss ($)')
                ax7.grid(True, alpha=0.3)
        
        # 8. Consecutive Losses
        ax8 = fig.add_subplot(gs[2, 2])
        if len(trades) > 0:
            pnl = trades['realized_pnl'].values
            consecutive_losses = []
            current = 0
            for p in pnl:
                if p < 0:
                    current += 1
                else:
                    if current > 0:
                        consecutive_losses.append(current)
                    current = 0
            if consecutive_losses:
                ax8.hist(consecutive_losses, bins=10, color='darkred', alpha=0.7, edgecolor='black')
                ax8.set_title('Consecutive Losses Distribution', fontsize=11, fontweight='bold')
                ax8.set_xlabel('Consecutive Losses')
                ax8.set_ylabel('Frequency')
        
        fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Risk dashboard saved: {filepath}")
        return filepath
    
    def create_summary_report(self,
                             trades: pd.DataFrame,
                             metrics: Dict[str, Any],
                             filename: str = "summary_report.html") -> str:
        """
        Create HTML summary report
        
        Args:
            trades: DataFrame of trades
            metrics: Performance metrics
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        html_content = self._generate_html_report(trades, metrics)
        
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Summary report saved: {filepath}")
        return filepath
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _format_metrics_box(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display"""
        return f"""PERFORMANCE METRICS
{'='*35}
Total Return:        {metrics.get('total_return_pct', 0):.2f}%
Annual Return:       {metrics.get('annual_return_pct', 0):.2f}%
Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}
Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}
Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%
Win Rate:            {metrics.get('win_rate_pct', 0):.2f}%
Profit Factor:       {metrics.get('profit_factor', 0):.2f}
Total Trades:        {metrics.get('trade_count', 0)}
Winning Trades:      {metrics.get('winning_trades', 0)}
Losing Trades:       {metrics.get('losing_trades', 0)}
"""
    
    def _format_risk_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format risk metrics for display"""
        return f"""RISK ANALYSIS
{'='*35}
Volatility:          {metrics.get('volatility_pct', 0):.2f}%
VaR (95%):           {metrics.get('var_95', 0):.4f}
CVaR (95%):          {metrics.get('cvar_95', 0):.4f}
Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%
Drawdown Duration:   {metrics.get('max_drawdown_duration', 0)} bars
Recovery Factor:     {metrics.get('recovery_factor', 0):.2f}
Return/Risk Ratio:   {metrics.get('return_risk_ratio', 0):.2f}
Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}
"""
    
    def _calculate_monthly_returns(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns matrix"""
        if len(trades) == 0:
            return pd.DataFrame()
        
        trades_copy = trades.copy()
        trades_copy['entry_time'] = pd.to_datetime(trades_copy['entry_time'])
        trades_copy['year_month'] = trades_copy['entry_time'].dt.to_period('M')
        
        monthly = trades_copy.groupby('year_month')['realized_pnl'].sum()
        return pd.DataFrame(monthly).T
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        running_max = equity_curve.expanding().max()
        drawdown = ((equity_curve - running_max) / running_max) * 100
        return drawdown
    
    def _generate_html_report(self, trades: pd.DataFrame, metrics: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Summary Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
                .metric-box { background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; border-radius: 4px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
                .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th { background-color: #007bff; color: white; padding: 12px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background-color: #f5f5f5; }
                .positive { color: green; }
                .negative { color: red; }
                .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading System Summary Report</h1>
                <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <h2>Performance Overview</h2>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics.get('total_return_pct', 0):.2f}%" + """</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics.get('sharpe_ratio', 0):.2f}" + """</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics.get('max_drawdown_pct', 0):.2f}%" + """</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics.get('win_rate_pct', 0):.2f}%" + """</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                </div>
                
                <h2>Trade Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Trades</td>
                        <td>""" + str(metrics.get('trade_count', 0)) + """</td>
                    </tr>
                    <tr>
                        <td>Winning Trades</td>
                        <td class="positive">""" + str(metrics.get('winning_trades', 0)) + """</td>
                    </tr>
                    <tr>
                        <td>Losing Trades</td>
                        <td class="negative">""" + str(metrics.get('losing_trades', 0)) + """</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>""" + f"{metrics.get('profit_factor', 0):.2f}" + """</td>
                    </tr>
                    <tr>
                        <td>Avg Win</td>
                        <td class="positive">$""" + f"{metrics.get('avg_win', 0):.2f}" + """</td>
                    </tr>
                    <tr>
                        <td>Avg Loss</td>
                        <td class="negative">$""" + f"{metrics.get('avg_loss', 0):.2f}" + """</td>
                    </tr>
                </table>
                
                <div class="footer">
                    <p>This report was automatically generated by the Trading System.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html


if __name__ == "__main__":
    print("Trading Dashboard Module Loaded Successfully")
