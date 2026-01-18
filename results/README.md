# Results Module

Comprehensive results handling for the trading system including export, analysis, and visualization.

## Features

### 1. **Export Results** (`export_results.py`)
Export backtest results to multiple formats:
- **CSV** - Trade-by-trade detailed results
- **JSON** - Machine-readable format for programmatic access
- **Excel** - Multiple sheets with trades, metrics, and summary
- **HTML** - Interactive web-readable report
- **Text** - Professional text reports

**Usage:**
```python
from results.export_results import ResultsExporter

exporter = ResultsExporter(output_dir='results')

# Export trades
exporter.export_trades_csv(trades, 'my_trades.csv')
exporter.export_trades_json(trades, 'my_trades.json')

# Export metrics
exporter.export_metrics_summary(metrics, 'metrics.json')

# Export equity curve
exporter.export_equity_curve(equity_series, 'equity.csv')

# Export to Excel
exporter.export_to_excel(trades, metrics, 'backtest.xlsx')

# Generate HTML report
exporter.export_html_report(trades, metrics, 'report.html')
```

### 2. **Visualize Results** (`visualize_results.py`)
Create professional visualizations:
- **Equity Curve** - Cumulative returns over time
- **Drawdown Chart** - Maximum loss tracking
- **Trade Distribution** - Win/loss histogram and counts
- **Monthly Returns** - Monthly performance heatmap
- **Duration vs P&L** - Scatter plot of trade characteristics
- **Strategy Comparison** - Compare multiple strategies

**Usage:**
```python
from results.visualize_results import ResultsVisualizer, ResultsComparison

visualizer = ResultsVisualizer(output_dir='results')

# Create visualizations
visualizer.plot_equity_curve(equity_curve, 'equity_curve.png')
visualizer.plot_drawdown(equity_curve, 'drawdown.png')
visualizer.plot_trade_distribution(trades, 'distribution.png')
visualizer.plot_monthly_returns(equity_curve, 'monthly.png')
visualizer.plot_trade_duration_vs_pnl(trades, 'duration_vs_pnl.png')

# Compare strategies
results_dict = {
    'Strategy A': metrics_a,
    'Strategy B': metrics_b
}
ResultsComparison.create_comparison_chart(results_dict, 'comparison.png')
```

### 3. **Generate Summary** (`summary_generator.py`)
Create comprehensive summary statistics:
- Trade count and win rate
- Return statistics (total, annual, average)
- Risk metrics (Sharpe, Sortino, drawdown)
- Trade quality (profit factor, win/loss ratio)
- Trade duration statistics
- Winning and losing streaks
- Performance reports (text, PDF, JSON)

**Usage:**
```python
from results.summary_generator import SummaryGenerator, PerformanceReport

# Create summary
summary = SummaryGenerator.create_summary(trades, metrics, 'summary.json')

# Format as text
text_report = SummaryGenerator.format_summary_text(summary)
print(text_report)

# Generate PDF report (requires reportlab)
PerformanceReport.generate_pdf_report(summary, trades, 'report.pdf')
```

### 4. **Compare Results** 
Compare multiple backtest runs or parameter variations:

**Usage:**
```python
from results.summary_generator import ResultsComparator

# Compare different runs
runs = {
    'Run 1': metrics1,
    'Run 2': metrics2,
    'Run 3': metrics3
}

comparison_df = ResultsComparator.compare_runs(runs)
print(comparison_df)

# Analyze parameter sensitivity
param_results = {
    'ema_5_15': 0.023,
    'ema_5_20': 0.031,
    'ema_5_25': 0.018
}

sensitivity = ResultsComparator.analyze_parameter_sensitivity(param_results)
print(f"Sensitivity: {sensitivity['sensitivity']:.3f}")
```

## Output Files

Generated results are saved in the `results/` directory:

```
results/
├── trades.csv                    # All trades with details
├── trades.json                   # Machine-readable format
├── metrics_summary.json          # Performance metrics
├── equity_curve.csv              # Equity values over time
├── backtest_results.xlsx         # Multi-sheet Excel file
├── backtest_report.html          # Web-viewable report
├── summary.json                  # Summary statistics
├── equity_curve.png              # Equity chart
├── drawdown.png                  # Drawdown chart
├── trade_distribution.png        # Trade statistics chart
├── monthly_returns.png           # Monthly performance chart
├── duration_vs_pnl.png           # Trade analysis scatter
├── strategy_comparison.png       # Strategy comparison chart
└── report.pdf                    # Professional PDF report
```

## Trade Data Format

Each trade record contains:
```python
{
    'trade_id': int,              # Unique trade identifier
    'entry_idx': int,             # Entry bar index
    'exit_idx': int,              # Exit bar index
    'entry_price': float,         # Entry price
    'exit_price': float,          # Exit price
    'position_type': str,         # 'LONG' or 'SHORT'
    'pnl': float,                 # Profit/Loss in dollars
    'pnl_pct': float,             # Profit/Loss percentage
    'bars_held': int,             # Duration in bars
    'exit_reason': str,           # 'Signal', 'Stop Loss', 'Take Profit'
    'entry_time': datetime,       # Entry timestamp
    'exit_time': datetime         # Exit timestamp
}
```

## Metrics Format

Performance metrics dictionary:
```python
{
    'total_return': float,        # Total return percentage
    'annual_return': float,       # Annualized return
    'sharpe_ratio': float,        # Risk-adjusted return
    'sortino_ratio': float,       # Downside risk ratio
    'max_drawdown': float,        # Maximum loss percentage
    'win_rate': float,            # Percentage of winning trades
    'profit_factor': float,       # Total wins / total losses
    'num_trades': int,            # Total number of trades
    'payoff_ratio': float,        # Avg win / avg loss
    'recovery_factor': float      # Return / max drawdown
}
```

## Integration with Main System

```python
from core.backtest_engine import BacktestEngine
from results import ResultsExporter, ResultsVisualizer, SummaryGenerator

# Run backtest
metrics, df_backtest = engine.run_backtest(df, trades)

# Export results
exporter = ResultsExporter()
exporter.export_trades_csv(trades)
exporter.export_to_excel(trades, metrics)

# Visualize
visualizer = ResultsVisualizer()
visualizer.plot_equity_curve(df_backtest['equity_curve'])
visualizer.plot_drawdown(df_backtest['equity_curve'])

# Generate summary
summary = SummaryGenerator.create_summary(trades, metrics)
print(SummaryGenerator.format_summary_text(summary))
```

## Best Practices

1. **Always Export Results** - Save all results immediately after backtesting
2. **Create Summaries** - Generate text summaries for quick review
3. **Visualize Key Metrics** - Create charts for stakeholder presentations
4. **Compare Strategies** - Use comparison tools to evaluate different approaches
5. **Track Parameters** - Keep records of what parameters produced what results
6. **Archive Reports** - Save reports with timestamp for audit trail

## Requirements

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- openpyxl >= 3.0.0 (for Excel export)
- reportlab >= 3.6.0 (optional, for PDF reports)

## See Also

- [DataPipeline](../core/data_pipeline.py) - Data loading and processing
- [BacktestEngine](../core/backtest_engine.py) - Performance metrics calculation
- [StrategyExecutor](../core/strategy_executor.py) - Trade generation
- [Analysis](../core/analysis.py) - Statistical analysis and outlier detection
