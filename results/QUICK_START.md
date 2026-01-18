# Results Scripts Quick Reference

## 📊 Results Folder Contents

```
results/
├── __init__.py                 # Package initialization
├── export_results.py           # Export to CSV, JSON, Excel, HTML
├── visualize_results.py        # Create charts and visualizations
├── summary_generator.py        # Generate summary statistics
└── README.md                   # Full documentation
```

## 🚀 Quick Usage Examples

### 1. Export Results

```python
from results.export_results import ResultsExporter, ResultsAnalyzer

exporter = ResultsExporter(output_dir='results')

# Export trades to CSV
exporter.export_trades_csv(trades, 'trades.csv')

# Export to JSON
exporter.export_trades_json(trades, 'trades.json')

# Export metrics
exporter.export_metrics_summary(metrics, 'metrics.json')

# Multi-sheet Excel file
exporter.export_to_excel(trades, metrics, 'backtest.xlsx')

# HTML report
exporter.export_html_report(trades, metrics, 'report.html')

# Generate text report
analyzer = ResultsAnalyzer()
report = analyzer.generate_report(trades, metrics)
print(report)
```

### 2. Visualize Results

```python
from results.visualize_results import ResultsVisualizer, ResultsComparison

visualizer = ResultsVisualizer(output_dir='results')

# Create equity curve
visualizer.plot_equity_curve(equity_curve, 'equity.png')

# Plot drawdown
visualizer.plot_drawdown(equity_curve, 'drawdown.png')

# Trade distribution
visualizer.plot_trade_distribution(trades, 'distribution.png')

# Monthly returns
visualizer.plot_monthly_returns(equity_curve, 'monthly.png')

# Duration vs P&L
visualizer.plot_trade_duration_vs_pnl(trades, 'duration_pnl.png')

# Compare strategies
strategies = {'Strategy A': metrics_a, 'Strategy B': metrics_b}
ResultsComparison.create_comparison_chart(strategies, 'comparison.png')
```

### 3. Generate Summaries

```python
from results.summary_generator import SummaryGenerator, PerformanceReport

# Create comprehensive summary
summary = SummaryGenerator.create_summary(trades, metrics, 'summary.json')

# Format as text
text_summary = SummaryGenerator.format_summary_text(summary)
print(text_summary)

# Save to file
with open('summary_report.txt', 'w') as f:
    f.write(text_summary)

# Generate PDF report
PerformanceReport.generate_pdf_report(summary, trades, 'report.pdf')
```

### 4. Compare Results

```python
from results.summary_generator import ResultsComparator

# Compare multiple runs
runs = {
    'Run 1': metrics1,
    'Run 2': metrics2,
    'Run 3': metrics3
}
comparison = ResultsComparator.compare_runs(runs)
print(comparison)

# Parameter sensitivity analysis
results = {
    'ema_5_10': 0.023,
    'ema_5_15': 0.031,
    'ema_5_20': 0.018
}
sensitivity = ResultsComparator.analyze_parameter_sensitivity(results)
print(f"Sensitivity: {sensitivity['sensitivity']:.3f}")
```

## 📋 Summary Output Format

The summary contains organized metrics:

```
summary = {
    'trade_count': {
        'total': 143,
        'winning': 32,
        'losing': 111,
        'breakeven': 0
    },
    'returns': {
        'total_return_pct': -0.31,
        'annual_return_pct': -1.24,
        'total_pnl': -310.00,
        'avg_pnl': -2.17
    },
    'risk': {
        'sharpe_ratio': -0.12,
        'sortino_ratio': 0.00,
        'max_drawdown_pct': -0.77,
        'max_drawdown_duration': 45,
        'recovery_factor': 0.40
    },
    'trade_quality': {
        'win_rate_pct': 22.38,
        'profit_factor': 0.45,
        'payoff_ratio': 4.47,
        'avg_win': 12.65,
        'avg_loss': -2.83,
        'largest_win': 45.23,
        'largest_loss': -89.54
    },
    'duration': {
        'avg_bars': 8.5,
        'min_bars': 1,
        'max_bars': 35
    },
    'streaks': {
        'longest_win_streak': 3,
        'longest_loss_streak': 15
    }
}
```

## 🔗 Integration with Core System

```python
# Example: Complete workflow
from core.backtest_engine import BacktestEngine
from core.strategy_executor import StrategyExecutor
from results.export_results import ResultsExporter
from results.visualize_results import ResultsVisualizer
from results.summary_generator import SummaryGenerator

# Run backtest
executor = StrategyExecutor(config)
df, trades = executor.execute_trades(df)

engine = BacktestEngine(config)
metrics, df_backtest = engine.run_backtest(df, trades)

# Export results
exporter = ResultsExporter()
exporter.export_trades_csv(trades)
exporter.export_to_excel(trades, metrics)

# Create visualizations
visualizer = ResultsVisualizer()
visualizer.plot_equity_curve(df_backtest['equity_curve'])
visualizer.plot_trade_distribution(trades)

# Generate summary
summary = SummaryGenerator.create_summary(trades, metrics)
print(SummaryGenerator.format_summary_text(summary))
```

## 📊 Class Reference

### ResultsExporter
- `export_trades_csv()` - Export trades to CSV
- `export_trades_json()` - Export trades to JSON
- `export_metrics_summary()` - Export performance metrics
- `export_equity_curve()` - Export equity series
- `export_to_excel()` - Multi-sheet Excel export
- `export_html_report()` - Generate HTML report

### ResultsAnalyzer
- `analyze_trades()` - Calculate trade statistics
- `generate_report()` - Create text report

### ResultsVisualizer
- `plot_equity_curve()` - Line chart of equity
- `plot_drawdown()` - Drawdown area chart
- `plot_trade_distribution()` - Histogram and win/loss
- `plot_monthly_returns()` - Bar chart by month
- `plot_trade_duration_vs_pnl()` - Scatter plot

### ResultsComparison
- `compare_strategies()` - DataFrame comparison
- `create_comparison_chart()` - Visual comparison

### SummaryGenerator
- `create_summary()` - Full summary statistics
- `format_summary_text()` - Format as readable text

### ResultsComparator
- `compare_runs()` - Compare multiple backtests
- `analyze_parameter_sensitivity()` - Parameter analysis

### PerformanceReport
- `generate_pdf_report()` - Create PDF report

## 💡 Tips & Best Practices

1. **Always export immediately** after backtesting for record-keeping
2. **Use Excel export** for sharing results with non-technical stakeholders
3. **Create HTML reports** for web-based sharing
4. **Generate summaries** for quick review of key metrics
5. **Compare strategies** to identify best approaches
6. **Track parameter sensitivity** to understand what matters most
7. **Save visualizations** for presentations and reports
8. **Archive results** with timestamps for audit trails

## 🔧 Customization

### Custom Export Format
```python
class CustomExporter(ResultsExporter):
    def export_custom_format(self, trades, filename):
        # Your custom export logic
        pass
```

### Custom Visualization
```python
class CustomVisualizer(ResultsVisualizer):
    def plot_custom_chart(self, data, filename):
        # Your custom visualization logic
        pass
```

## ⚙️ Configuration

Set output directories:
```python
exporter = ResultsExporter(output_dir='my_results')
visualizer = ResultsVisualizer(output_dir='my_results')
```

## 📞 Support

For detailed documentation, see [README.md](README.md)

For integration examples, check core modules documentation:
- [BacktestEngine](../core/backtest_engine.py)
- [StrategyExecutor](../core/strategy_executor.py)
- [Analysis](../core/analysis.py)
