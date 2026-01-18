"""
Results Summary Generator
Generate comprehensive summary statistics from backtest results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Generate summary statistics from results."""
    
    @staticmethod
    def create_summary(trades: List[Dict], metrics: Dict, 
                      output_file: str = None) -> Dict[str, Any]:
        """
        Create comprehensive summary from trades and metrics.
        
        Args:
            trades: List of trade dictionaries
            metrics: Dictionary of performance metrics
            output_file: Optional file to save summary
            
        Returns:
            Dictionary with summary statistics
        """
        df = pd.DataFrame(trades)
        pnls = df['pnl'].values if not df.empty else np.array([])
        
        # Calculate statistics
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        
        summary = {
            # Trade Statistics
            'trade_count': {
                'total': len(trades),
                'winning': len(winning_trades),
                'losing': len(losing_trades),
                'breakeven': len(pnls[pnls == 0])
            },
            
            # Return Statistics
            'returns': {
                'total_return_pct': metrics.get('total_return', 0),
                'annual_return_pct': metrics.get('annual_return', 0),
                'total_pnl': pnls.sum() if len(pnls) > 0 else 0,
                'avg_pnl': pnls.mean() if len(pnls) > 0 else 0
            },
            
            # Risk Statistics
            'risk': {
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown', 0),
                'max_drawdown_duration': metrics.get('max_drawdown_duration', 0),
                'recovery_factor': metrics.get('recovery_factor', 0)
            },
            
            # Trade Quality
            'trade_quality': {
                'win_rate_pct': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'payoff_ratio': metrics.get('payoff_ratio', 0),
                'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
                'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
                'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
                'win_loss_ratio': (winning_trades.mean() / abs(losing_trades.mean())) 
                                 if len(losing_trades) > 0 else 0
            },
            
            # Trade Duration
            'duration': {
                'avg_bars': df['bars_held'].mean() if 'bars_held' in df.columns else 0,
                'min_bars': df['bars_held'].min() if 'bars_held' in df.columns else 0,
                'max_bars': df['bars_held'].max() if 'bars_held' in df.columns else 0
            },
            
            # Winning & Losing Streaks
            'streaks': SummaryGenerator._calculate_streaks(pnls)
        }
        
        # Save if requested
        if output_file:
            SummaryGenerator._save_summary(summary, output_file)
        
        return summary
    
    @staticmethod
    def _calculate_streaks(pnls: np.ndarray) -> Dict[str, int]:
        """Calculate longest winning and losing streaks."""
        if len(pnls) == 0:
            return {'longest_win_streak': 0, 'longest_loss_streak': 0}
        
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0
            
            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)
        
        return {
            'longest_win_streak': max_win_streak,
            'longest_loss_streak': max_loss_streak
        }
    
    @staticmethod
    def _save_summary(summary: Dict, filepath: str):
        """Save summary to JSON file."""
        import json
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved: {filepath}")
    
    @staticmethod
    def format_summary_text(summary: Dict) -> str:
        """
        Format summary as readable text.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Formatted text string
        """
        lines = [
            "=" * 80,
            "BACKTEST SUMMARY REPORT",
            "=" * 80,
            "",
            "TRADE STATISTICS",
            "-" * 80,
            f"  Total Trades:           {summary['trade_count']['total']}",
            f"  Winning Trades:         {summary['trade_count']['winning']}",
            f"  Losing Trades:          {summary['trade_count']['losing']}",
            f"  Win Rate:               {summary['trade_quality']['win_rate_pct']:.2f}%",
            "",
            "RETURN STATISTICS",
            "-" * 80,
            f"  Total Return:           {summary['returns']['total_return_pct']:.2f}%",
            f"  Annual Return:          {summary['returns']['annual_return_pct']:.2f}%",
            f"  Total P&L:              ${summary['returns']['total_pnl']:.2f}",
            f"  Average Trade P&L:      ${summary['returns']['avg_pnl']:.2f}",
            "",
            "RISK METRICS",
            "-" * 80,
            f"  Sharpe Ratio:           {summary['risk']['sharpe_ratio']:.2f}",
            f"  Max Drawdown:           {summary['risk']['max_drawdown_pct']:.2f}%",
            f"  Recovery Factor:        {summary['risk']['recovery_factor']:.2f}",
            "",
            "TRADE QUALITY",
            "-" * 80,
            f"  Profit Factor:          {summary['trade_quality']['profit_factor']:.2f}",
            f"  Average Win:            ${summary['trade_quality']['avg_win']:.2f}",
            f"  Average Loss:           ${summary['trade_quality']['avg_loss']:.2f}",
            f"  Largest Win:            ${summary['trade_quality']['largest_win']:.2f}",
            f"  Largest Loss:           ${summary['trade_quality']['largest_loss']:.2f}",
            f"  Win/Loss Ratio:         {summary['trade_quality']['win_loss_ratio']:.2f}",
            "",
            "TRADE DURATION (in bars)",
            "-" * 80,
            f"  Average Duration:       {summary['duration']['avg_bars']:.1f}",
            f"  Shortest Trade:         {summary['duration']['min_bars']}",
            f"  Longest Trade:          {summary['duration']['max_bars']}",
            "",
            "STREAKS",
            "-" * 80,
            f"  Longest Winning Streak: {summary['streaks']['longest_win_streak']}",
            f"  Longest Losing Streak:  {summary['streaks']['longest_loss_streak']}",
            "",
            "=" * 80
        ]
        
        return "\n".join(lines)


class ResultsComparator:
    """Compare results from different runs or strategies."""
    
    @staticmethod
    def compare_runs(runs_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.
        
        Args:
            runs_data: Dictionary with run_name: metrics
            
        Returns:
            DataFrame with comparison results
        """
        comparison_df = pd.DataFrame(runs_data).T
        return comparison_df
    
    @staticmethod
    def analyze_parameter_sensitivity(param_results: Dict[str, float]) -> Dict:
        """
        Analyze how results change with parameter variations.
        
        Args:
            param_results: Dictionary with parameter_value: metric_value
            
        Returns:
            Dictionary with sensitivity analysis
        """
        values = list(param_results.values())
        
        analysis = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'sensitivity': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        }
        
        return analysis


class PerformanceReport:
    """Generate detailed performance reports."""
    
    @staticmethod
    def generate_pdf_report(summary: Dict, trades: List[Dict], 
                           output_file: str = 'report.pdf'):
        """
        Generate PDF report (requires reportlab).
        
        Args:
            summary: Summary statistics
            trades: List of trades
            output_file: Output PDF filename
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Add title
            story.append(Paragraph("Backtest Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Add summary table
            summary_data = [
                ['Metric', 'Value'],
                ['Total Trades', str(summary['trade_count']['total'])],
                ['Win Rate', f"{summary['trade_quality']['win_rate_pct']:.2f}%"],
                ['Total Return', f"{summary['returns']['total_return_pct']:.2f}%"],
                ['Sharpe Ratio', f"{summary['risk']['sharpe_ratio']:.2f}"],
                ['Max Drawdown', f"{summary['risk']['max_drawdown_pct']:.2f}%"],
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            story.append(summary_table)
            doc.build(story)
            logger.info(f"PDF report generated: {output_file}")
            
        except ImportError:
            logger.warning("reportlab not installed. Skipping PDF generation.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Summary generator module loaded")
