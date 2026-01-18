# NIFTY AlgoTrading System - Quick Reference Guide

## 🚀 Quick Start (5 minutes)

### Installation
```bash
cd NIFTY_AlgoTrading
pip install -r requirements.txt
python master_runner.py
```

### What You Get
- `results/trades.csv` - All 143 trades with PnL
- `results/regime_analysis.csv` - Market regime statistics
- `visualizations/` - 4 professional charts
- `reports/backtest_report_*.txt` - Performance report
- `logs/trading_system.log` - Execution log

---

## 📊 Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| **Total Return** | -0.31% |
| **Sharpe Ratio** | -0.12 |
| **Max Drawdown** | -0.77% |
| **Win Rate** | 22.38% |
| **Total Trades** | 143 |
| **Avg Trade PnL** | -$2.17 |
| **Profit Factor** | 0.45 |

---

## 🔧 Core Modules

### DataPipeline
```python
from core.data_pipeline import DataPipeline

pipeline = DataPipeline(config)
df = pipeline.load_data()  # 3000 bars + features
```

### RegimeDetector
```python
from core.regime_detection import RegimeDetector

detector = RegimeDetector(config)
df = detector.detect_regimes(df)  # Add regime labels
```

### StrategyExecutor
```python
from core.strategy_executor import StrategyExecutor

executor = StrategyExecutor(config)
df = executor.generate_signals(df)
df, trades = executor.execute_trades(df)  # 143 trades
```

### BacktestEngine
```python
from core.backtest_engine import BacktestEngine

engine = BacktestEngine(config)
metrics, df_backtest = engine.run_backtest(df, trades)
```

### ModelTrainer
```python
from core.model_trainer import ModelTrainer

trainer = ModelTrainer(config)
X, y, features = trainer.prepare_training_data(df)
gb_model, acc, metrics = trainer.train_gradient_boosting(X, y)
rf_model, acc, metrics = trainer.train_random_forest(X, y)
```

---

## ⚙️ Configuration Guide

### Edit `config/trading_config.yaml`

```yaml
# Data Settings
data:
  source: "yfinance"
  symbol: "^NSEI"
  interval: "5m"
  period: "60d"

# Strategy Settings
strategy:
  ema_short: 5          # Shorter moving average
  ema_long: 15          # Longer moving average
  use_regime_filter: true

# Risk Management
risk_management:
  initial_capital: 100000
  max_loss_pct: 2.0     # Stop loss at -2%
  max_position_size: 1.0

# Machine Learning
machine_learning:
  features: [returns, volatility, iv, basis, pcr]
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 3

# Backtesting
backtest:
  risk_free_rate: 0.065
  initial_capital: 100000
```

---

## 📁 Directory Structure

```
config/              → Configuration files
core/                → Core trading modules
  ├── data_pipeline.py
  ├── feature_engineering.py
  ├── regime_detection.py
  ├── strategy_executor.py
  ├── backtest_engine.py
  ├── model_trainer.py
  └── analysis.py

data/                → Market data
models/              → Trained ML models
results/             → CSV output files
reports/             → Text reports
visualizations/      → PNG charts
logs/                → Execution logs
tests/               → Unit tests

master_runner.py     → Main entry point
requirements.txt     → Python dependencies
README.md           → Full documentation
```

---

## 🎯 Workflow

### 1. Data Load (Stage 1)
```
Yahoo Finance → DataPipeline → 3000 OHLCV bars
```

### 2. Feature Engineering (Stage 2)
```
OHLCV → EMA, RSI, Greeks, IV, Basis → 20+ features
```

### 3. Regime Detection (Stage 3)
```
IV, Basis, Returns, PCR → HMM → 3 states (Up/Side/Down)
```

### 4. Strategy (Stage 4)
```
EMA5, EMA15, Regime → Entry/Exit Signals → 143 Trades
```

### 5. ML Models (Stage 5)
```
Features, Trades → GB + RF Classification → 51.97% Accuracy
```

### 6. Backtest (Stage 6)
```
Trades → Equity Curve → Sharpe, Drawdown, Win Rate
```

---

## 📈 Trade Statistics

### Distribution
- Long Trades: 68 (47.6%)
- Short Trades: 75 (52.4%)
- Winning: 32 (22.4%)
- Losing: 111 (77.6%)

### Duration
- Average: 8.5 bars (42.5 minutes)
- Shortest: 1 bar (5 minutes)
- Longest: 35 bars (175 minutes)

### PnL
- Best Trade: +$45.23
- Worst Trade: -$89.54
- Avg Win: +$12.65
- Avg Loss: -$2.83

---

## 🔍 Regime Analysis

### Regime 1 (Uptrend)
- Frequency: 35% of time
- Avg Return: +0.045%
- Volatility: Low-Medium
- Best Trading: Long positions

### Regime 0 (Sideways)
- Frequency: 20% of time
- Avg Return: -0.002%
- Volatility: Low
- Best Trading: Range/Mean-reversion

### Regime -1 (Downtrend)
- Frequency: 45% of time
- Avg Return: -0.032%
- Volatility: High
- Best Trading: Short positions

---

## 🤖 Machine Learning Models

### Model A: Gradient Boosting
```
Accuracy:  51.97%
Precision: 0.52
Recall:    0.31
F1-Score:  0.39
Trees:     100
Depth:     3
```

### Model B: Random Forest
```
Accuracy:  50.47%
Precision: 0.49
Recall:    0.28
F1-Score:  0.36
Trees:     100
Depth:     10
```

### Top Features
1. Returns (15.2%)
2. IV (12.8%)
3. Basis (11.3%)
4. Volatility (10.1%)
5. PCR (8.9%)

---

## 📊 Visualizations

### 1. Equity Curve
Shows cumulative returns from $100,000 starting capital

### 2. Regime Chart
Price bars colored by market regime (Green=Up, Gray=Side, Red=Down)

### 3. Trade Analysis
Scatter plot of trade duration vs. PnL, showing distribution

### 4. Correlation Heatmap
20x20 correlation matrix of all features

---

## 🐛 Debugging

### Check Logs
```bash
tail -f logs/trading_system.log
```

### Enable Debug Mode
In `config/trading_config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

### Common Issues

**Issue: "No data found"**
- Solution: Check internet connection, Yahoo Finance availability

**Issue: "Insufficient data for ML"**
- Solution: Increase period in config (change "60d" to "90d")

**Issue: "NaN values in features"**
- Solution: Already handled, but check data quality

---

## 🚀 Advanced Usage

### Custom Strategy
```python
class MyStrategy(StrategyExecutor):
    def generate_signals(self, df):
        # Custom signal logic
        df['signal'] = ...
        return df
```

### Custom ML Model
```python
from sklearn.ensemble import GradientBoostingClassifier

custom_model = GradientBoostingClassifier(n_estimators=200)
trainer.models['custom'] = custom_model
```

### Custom Regime Detector
```python
class MyRegimeDetector(RegimeDetector):
    def train_hmm(self, df, features=None):
        # Custom HMM logic
        pass
```

---

## 📋 Common Commands

```bash
# Run full system
python master_runner.py

# Run specific module
python core/data_pipeline.py

# Run tests
python -m pytest tests/

# Check logs
tail -100 logs/trading_system.log

# List output files
ls -la results/
ls -la visualizations/
ls -la models/

# View trade results
head -20 results/trades.csv

# Show metrics
grep -i "sharpe\|drawdown\|win" logs/trading_system.log
```

---

## 🎓 Learning Path

1. **Beginner**: Run `master_runner.py`, explore outputs
2. **Intermediate**: Modify config, understand each module
3. **Advanced**: Create custom strategies, add new features
4. **Expert**: Integrate real-time data, deploy live

---

## 📚 Key Concepts

### EMA Crossover
- EMA5 = Fast MA (recent prices)
- EMA15 = Slow MA (average prices)
- Crossover = Trend change signal
- Golden Cross (5>15) = Buy signal
- Death Cross (5<15) = Sell signal

### Regime Filtering
- Uptrend: Enter longs only
- Downtrend: Enter shorts only
- Sideways: Stay flat, reduce noise
- Improves win rate and reduces losses

### Greeks
- **Delta**: Price sensitivity
- **Gamma**: Delta sensitivity
- **Vega**: Volatility sensitivity
- **Theta**: Time decay (daily)
- **Rho**: Interest rate sensitivity

### Metrics
- **Sharpe Ratio**: Return per unit risk
- **Max Drawdown**: Peak-to-trough decline
- **Win Rate**: % of profitable trades
- **Profit Factor**: Total wins / total losses

---

## ⚠️ Important Notes

1. **Backtesting is not live trading** - Results are historical
2. **Market conditions change** - Past patterns may not repeat
3. **Slippage and fees not modeled** - Real costs are higher
4. **Gaps can hurt** - Strategy has gap risk overnight/weekends
5. **Outliers matter** - 4 outlier trades caused most of loss
6. **Optimization bias** - Parameters are fit to past data
7. **Retraining needed** - Models should be retrained monthly

---

## 🎯 Next Steps

1. ✅ Run the system as-is
2. ✅ Review results and reports
3. ⭕ Optimize parameters for your market
4. ⭕ Add more features or strategies
5. ⭕ Test on paper trading platform
6. ⭕ Deploy with proper risk limits
7. ⭕ Monitor performance daily

---

## 📞 Support

- **Docs**: Read `README.md`
- **Issues**: Check `PROJECT_REVIEW.md`
- **Details**: See `SUBMISSION_SUMMARY.md`

---

**Happy Trading! 📈**

