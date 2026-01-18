# NIFTY 50 Algorithmic Trading System

Professional-grade algorithmic trading system for NIFTY 50 index trading with advanced machine learning, regime detection, and risk management.

## 📊 System Architecture

```
NIFTY_AlgoTrading/
├── config/
│   └── trading_config.yaml              # Central configuration
├── core/
│   ├── __init__.py
│   ├── data_pipeline.py                 # Data acquisition & engineering
│   ├── feature_engineering.py           # 20+ technical indicators
│   ├── regime_detection.py              # HMM-based market regimes
│   ├── strategy_executor.py             # EMA crossover with filters
│   ├── backtest_engine.py               # Comprehensive metrics
│   ├── model_trainer.py                 # ML model training
│   ├── report_generator.py              # Professional reporting
│   └── analysis.py                      # Outlier detection & viz
├── strategies/
│   └── ema_crossover_regime.py          # Strategy implementation
├── backtests/
│   └── backtest_results/                # Historical results
├── data/
│   ├── raw/                             # Raw market data
│   └── processed/                       # Feature engineering output
├── models/
│   ├── gradient_boosting_model.pkl      # Trained GB classifier
│   └── random_forest_model.pkl          # Trained RF classifier
├── results/
│   ├── trades.csv                       # Detailed trade log
│   ├── regime_analysis.csv              # Regime statistics
│   └── ml_results.csv                   # ML performance
├── reports/
│   └── backtest_report_*.txt            # Backtest reports
├── visualizations/
│   ├── equity_curve.png
│   ├── regime_chart.png
│   ├── trade_analysis.png
│   └── correlation_heatmap.png
├── logs/
│   └── trading_system.log               # System logs
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_strategy.py
│   └── test_backtest.py
├── master_runner.py                     # Main orchestrator
├── requirements.txt
└── README.md
```

## 🎯 Key Features

### 1. **Data Pipeline** (Stage 1)
- **Data Sources**: Yahoo Finance (yfinance) for NIFTY 50 spot data
- **Data Points**: 3000+ 5-minute OHLCV bars
- **Options Generation**: ATM and adjacent strikes with Greeks
- **Futures Generation**: Basis calculation with current contracts
- **Data Validation**: NaN handling, forward/backward fill, outlier detection

### 2. **Feature Engineering** (Stage 2)
- **Technical Indicators**: EMA, RSI, MACD, ATR, Bollinger Bands
- **Volatility Metrics**: IV (Implied Volatility), Realized Volatility
- **Greeks**: Delta, Gamma, Vega, Theta, Rho (Black-Scholes)
- **Derivatives**: Returns, Basis, PCR (Put-Call Ratio)
- **Total Features**: 20+ engineered features

```python
Key Features:
- EMA 5, EMA 15 (trend identification)
- RSI (momentum, overbought/oversold)
- MACD (trend confirmation)
- IV (option pricing, volatility proxy)
- Basis (futures premium/discount)
- PCR (put-call sentiment)
```

### 3. **Regime Detection** (Stage 3)
- **Algorithm**: Hidden Markov Model (HMM) with 3 states
- **States**: Uptrend (1), Sideways (0), Downtrend (-1)
- **Features**: IV, Basis, Returns, PCR
- **Application**: Entry/exit filtering, signal confidence
- **Characteristics**: Regime transition analysis, duration statistics

### 4. **Trading Strategy** (Stage 4)
- **Strategy Type**: EMA 5/15 Crossover with Regime Filter
- **Entry Rules**:
  - Long: EMA5 > EMA15 AND Regime = Uptrend
  - Short: EMA5 < EMA15 AND Regime = Downtrend
- **Exit Rules**: 
  - Signal reversal
  - Stop loss (2% by default)
  - Take profit targets
- **Risk Management**: Position sizing, max loss limits
- **Signals**: ~143 trades on test data (22.38% win rate)

### 5. **Machine Learning** (Stage 5)
- **Models Trained**:
  - Gradient Boosting: 51.97% accuracy
  - Random Forest: 50.47% accuracy
- **Target Variable**: Binary classification (profitable vs. loss trades)
- **Feature Set**: 10+ engineered features
- **Cross-Validation**: Time-series aware splitting
- **Output**: Model predictions, probability scores, feature importance

### 6. **Backtesting & Analysis** (Stage 6)
- **Metrics Calculated**:
  - Total Return: -0.31% (sample)
  - Sharpe Ratio: -0.12
  - Sortino Ratio: N/A (negative returns)
  - Max Drawdown: -0.77%
  - Win Rate: 22.38%
  - Profit Factor: 0.45
  - Recovery Factor: 0.40
- **Outlier Detection**: 3-sigma Z-score analysis
- **Trade Statistics**: Duration, PnL distribution, regime impact

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
hmmlearn >= 0.2.7
yfinance >= 0.1.70
matplotlib >= 3.4.0
seaborn >= 0.11.0
ta >= 0.10.0
py_vollib >= 0.5.3
PyYAML >= 5.4.0
```

### Installation
```bash
# Clone repository
git clone <repo-url>
cd NIFTY_AlgoTrading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config/trading_config.yaml`:

```yaml
data:
  source: "yfinance"
  symbol: "^NSEI"
  interval: "5m"
  period: "60d"

strategy:
  ema_short: 5
  ema_long: 15
  use_regime_filter: true

risk_management:
  initial_capital: 100000
  max_loss_pct: 2.0
  max_position_size: 1.0

machine_learning:
  features: [returns, volatility, iv, basis, pcr, rsi, macd, atr]
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 3

backtest:
  risk_free_rate: 0.065
  trading_days_per_year: 252
  initial_capital: 100000
```

### Running the System

```bash
# Run complete pipeline
python master_runner.py

# Run specific stages
from core.data_pipeline import DataPipeline
from core.feature_engineering import FeatureEngineer

config = load_config('config/trading_config.yaml')
pipeline = DataPipeline(config)
df = pipeline.load_data()
```

## 📈 Expected Outputs

### 1. CSV Files
- `results/trades.csv` - Detailed trade log with PnL, duration, reasons
- `results/regime_analysis.csv` - Regime statistics and characteristics
- `results/ml_results.csv` - Model accuracy and performance metrics
- `results/full_data_with_signals.csv` - Complete dataset with signals

### 2. Visualizations
- `visualizations/equity_curve.png` - Cumulative returns over time
- `visualizations/regime_chart.png` - Price with regime coloring
- `visualizations/trade_analysis.png` - Win/loss distribution
- `visualizations/correlation_heatmap.png` - Feature correlations

### 3. Reports
- `reports/backtest_report_*.txt` - Comprehensive text report
- `logs/trading_system.log` - Complete execution log

## 📊 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Total Return | -0.31% | Small loss over test period |
| Sharpe Ratio | -0.12 | Poor risk-adjusted returns |
| Max Drawdown | -0.77% | Maximum loss experienced |
| Win Rate | 22.38% | ~1 in 4-5 trades profitable |
| Profit Factor | 0.45 | Losses exceed wins |
| Average Trade | -$2.17 | Negative expectancy |
| Num Trades | 143 | Sufficient trading frequency |

**Note**: This is a developmental system showing realistic trading challenges. Live deployment requires optimization and validation.

## 🔧 Advanced Features

### 1. Regime Analysis
```python
from core.regime_detection import RegimeAnalyzer

analyzer = RegimeAnalyzer()
transitions = analyzer.get_transition_matrix(df)
characteristics = analyzer.analyze_regime_characteristics(df, regime_id=1)
```

### 2. Monte Carlo Analysis
```python
from core.backtest_engine import MonteCarloAnalysis

mc_results = MonteCarloAnalysis.run_monte_carlo(returns, num_simulations=1000)
print(f"95% Confidence Loss: {mc_results['percentile_5']:.2f}%")
```

### 3. Model Persistence
```python
# Save trained models
trainer.save_models('models/')

# Load models
trainer.load_models('models/')

# Make predictions
predictions = trainer.predict(X, model_name='gradient_boosting')
probabilities = trainer.predict_proba(X)
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_strategy.py -v

# Run with coverage
python -m pytest tests/ --cov=core/
```

## 📚 Module Documentation

### DataPipeline
- Fetches NIFTY 50 data from yfinance
- Generates options chains with strikes
- Calculates futures basis
- Handles missing data

### FeatureEngineer
- Computes 20+ technical indicators
- Calculates option Greeks
- Normalizes features
- Handles NaN values

### RegimeDetector
- Trains 3-state HMM
- Maps states to economic regimes
- Analyzes regime transitions
- Calculates regime characteristics

### StrategyExecutor
- Generates trading signals
- Executes trades with risk limits
- Tracks positions
- Calculates trade statistics

### BacktestEngine
- Calculates equity curve
- Computes Sharpe, Sortino, Drawdown
- Analyzes trade statistics
- Generates reports

### ModelTrainer
- Trains Gradient Boosting and Random Forest
- Performs time-series cross-validation
- Calculates feature importance
- Saves/loads models

## ⚠️ Risk Disclaimer

This is a **BACKTESTED** system for **EDUCATIONAL PURPOSES ONLY**:
- Past performance ≠ future results
- Market conditions change constantly
- Actual trading may face slippage, gaps, liquidity issues
- Always validate with real-time paper trading first
- Use proper position sizing and risk management
- Consult financial advisor before live trading

## 🔐 Best Practices

1. **Data Validation**: Always verify data integrity before trading
2. **Parameter Tuning**: Optimize parameters on recent data only
3. **Walk-Forward Testing**: Test on unseen future data
4. **Risk Management**: Never risk more than 2% per trade
5. **Monitoring**: Track system performance daily
6. **Drawdown Limits**: Stop trading if max drawdown exceeded
7. **Regime Awareness**: Always consider market regime
8. **Model Retraining**: Retrain models monthly/quarterly

## 📞 Support & Development

- **Logging**: Check `logs/trading_system.log` for detailed execution info
- **Debugging**: Enable DEBUG level logging in config
- **Customization**: Modify strategy, features, or ML models as needed
- **Backtesting**: Run `master_runner.py` to validate changes

## 📝 Future Enhancements

- [ ] Real-time data streaming with WebSocket
- [ ] Multi-strategy ensemble system
- [ ] Advanced portfolio optimization
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Options strategy implementation
- [ ] Risk parity portfolio management
- [ ] Ensemble model voting
- [ ] Alternative data integration

## 📄 License

Proprietary - For authorized use only

## 👤 Author

Quantitative Trading Research Team

---

**Last Updated**: January 2025
**Status**: Development/Educational
**Version**: 1.0.0

---

### Quick Reference Commands

```bash
# Run full pipeline
python master_runner.py

# Run specific module
python core/data_pipeline.py

# Generate reports
python core/report_generator.py

# Analyze results
python core/analysis.py

# Check logs
tail -f logs/trading_system.log
```

### Typical Execution Time
- Data Loading: ~5 seconds
- Feature Engineering: ~8 seconds
- Regime Detection: ~12 seconds
- Strategy Execution: ~3 seconds
- ML Training: ~8 seconds
- Backtesting: ~4 seconds
- Report Generation: ~3 seconds
- **Total: ~45 seconds**

