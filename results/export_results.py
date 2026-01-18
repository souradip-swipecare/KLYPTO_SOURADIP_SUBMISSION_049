"""
Results Export Module
Export backtest results to various formats (CSV, JSON, Excel, HTML)
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ResultsExporter:
    """Export trading results to multiple formats."""
    
    def __init__(self, output_dir: str = 'results'):
        """Initialize exporter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Results exporter initialized: {self.output_dir}")
    
    def export_trades_csv(self, trades: List[Dict], filename: str = 'trades.csv') -> str:
        """
        Export trades to CSV format.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        df = pd.DataFrame(trades)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Trades exported to CSV: {filepath}")
        return str(filepath)
    
    def export_trades_json(self, trades: List[Dict], filename: str = 'trades.json') -> str:
        """
        Export trades to JSON format.
        
        Args:
            trades: List of trade dictionaries
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        logger.info(f"Trades exported to JSON: {filepath}")
        return str(filepath)
    
    def export_metrics_summary(self, metrics: Dict[str, Any], 
                              filename: str = 'metrics_summary.json') -> str:
        """
        Export performance metrics summary.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics exported: {filepath}")
        return str(filepath)
    
    def export_equity_curve(self, equity_curve: pd.Series, 
                           filename: str = 'equity_curve.csv') -> str:
        """
        Export equity curve to CSV.
        
        Args:
            equity_curve: Series with equity values
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        df = pd.DataFrame({'equity': equity_curve})
        filepath = self.output_dir / filename
        df.to_csv(filepath)
        logger.info(f"Equity curve exported: {filepath}")
        return str(filepath)
    
    def export_to_excel(self, trades: List[Dict], metrics: Dict,
                       filename: str = 'backtest_results.xlsx') -> str:
        """
        Export trades and metrics to Excel with multiple sheets.
        
        Args:
            trades: List of trade dictionaries
            metrics: Dictionary of metrics
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Trades sheet
            df_trades = pd.DataFrame(trades)
            df_trades.to_excel(writer, sheet_name='Trades', index=False)
            
            # Metrics sheet
            df_metrics = pd.DataFrame([metrics])
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Total Trades', 'Win Rate', 'Total Return', 'Sharpe Ratio'],
                'Value': [
                    len(trades),
                    metrics.get('win_rate', 0),
                    metrics.get('total_return', 0),
                    metrics.get('sharpe_ratio', 0)
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Excel export complete: {filepath}")
        return str(filepath)
    
    def export_html_report(self, trades: List[Dict], metrics: Dict,
                          filename: str = 'backtest_report.html') -> str:
        """
        Export comprehensive HTML report.
        
        Args:
            trades: List of trade dictionaries
            metrics: Dictionary of metrics
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        
        df_trades = pd.DataFrame(trades)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            
            <h2>Performance Metrics</h2>
            <div class="metric">
                <strong>Total Trades:</strong> {len(trades)}
            </div>
            <div class="metric">
                <strong>Win Rate:</strong> {metrics.get('win_rate', 0):.2f}%
            </div>
            <div class="metric">
                <strong>Total Return:</strong> {metrics.get('total_return', 0):.2f}%
            </div>
            <div class="metric">
                <strong>Sharpe Ratio:</strong> {metrics.get('sharpe_ratio', 0):.2f}
            </div>
            <div class="metric">
                <strong>Max Drawdown:</strong> {metrics.get('max_drawdown', 0):.2f}%
            </div>
            
            <h2>Trade Details</h2>
            {df_trades.to_html(index=False)}
            
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported: {filepath}")
        return str(filepath)


class ResultsAnalyzer:
    """Analyze and generate insights from trading results."""
    
    @staticmethod
    def analyze_trades(trades: List[Dict]) -> Dict[str, Any]:
        """
        Analyze trade results and generate statistics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        pnls = df['pnl'].values
        
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        
        analysis = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
            'total_profit': pnls.sum(),
            'profit_factor': abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else 0,
            'avg_trade_duration': df['bars_held'].mean() if 'bars_held' in df.columns else 0
        }
        
        return analysis
    
    @staticmethod
    def generate_report(trades: List[Dict], metrics: Dict) -> str:
        """
        Generate comprehensive text report.
        
        Args:
            trades: List of trade dictionaries
            metrics: Dictionary of metrics
            
        Returns:
            Formatted report string
        """
        analysis = ResultsAnalyzer.analyze_trades(trades)
        
        report = [
            "=" * 70,
            "TRADING SYSTEM BACKTEST REPORT",
            "=" * 70,
            "",
            "PERFORMANCE METRICS",
            "-" * 70,
            f"Total Return:           {metrics.get('total_return', 0):>10.2f}%",
            f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):>10.2f}",
            f"Max Drawdown:           {metrics.get('max_drawdown', 0):>10.2f}%",
            f"Win Rate:               {metrics.get('win_rate', 0):>10.2f}%",
            "",
            "TRADE STATISTICS",
            "-" * 70,
            f"Total Trades:           {analysis.get('total_trades', 0):>10}",
            f"Winning Trades:         {analysis.get('winning_trades', 0):>10}",
            f"Losing Trades:          {analysis.get('losing_trades', 0):>10}",
            f"Average Win:            ${analysis.get('avg_win', 0):>10.2f}",
            f"Average Loss:           ${analysis.get('avg_loss', 0):>10.2f}",
            f"Profit Factor:          {analysis.get('profit_factor', 0):>10.2f}",
            f"Largest Win:            ${analysis.get('largest_win', 0):>10.2f}",
            f"Largest Loss:           ${analysis.get('largest_loss', 0):>10.2f}",
            "",
            "=" * 70
        ]
        
        return "\n".join(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exporter = ResultsExporter()
    print("Results export module loaded")
