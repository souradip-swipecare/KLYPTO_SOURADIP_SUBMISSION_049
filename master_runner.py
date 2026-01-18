"""
NIFTY 50 Algorithmic Trading System - Master Orchestrator
Comprehensive end-to-end trading system execution
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import yaml

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules
from core.data_pipeline import DataPipeline
from core.feature_engineering import FeatureEngineer
from core.regime_detection import RegimeDetector, RegimeAnalyzer
from core.strategy_executor import StrategyExecutor
from core.backtest_engine import BacktestEngine
from core.model_trainer import ModelTrainer
from core.report_generator import ReportGenerator
from core.analysis import OutlierAnalyzer, Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingSystemOrchestrator:
    """
    Master orchestrator for the NIFTY 50 algorithmic trading system.
    Coordinates all pipeline stages.
    """
    
    def __init__(self, config_path: str = 'config/trading_config.yaml'):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
        self._setup_logging()
        
        # Initialize components
        self.data_pipeline = DataPipeline(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.regime_detector = RegimeDetector(self.config)
        self.strategy_executor = StrategyExecutor(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.outlier_analyzer = OutlierAnalyzer(self.config)
        self.visualizer = Visualizer(self.config)
        
        logger.info("✓ Trading System Orchestrator initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Configuration loaded from {self.config_path}")
        return config
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            'data',
            'results',
            'models',
            'reports',
            'visualizations',
            'logs',
            'backtest_results'
        ]
        
        for dir_name in dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
        
        logger.info("✓ Directories created")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
    
    def run_pipeline(self) -> Dict:
        """
        Run complete trading system pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING NIFTY 50 ALGORITHMIC TRADING SYSTEM")
        logger.info("="*80 + "\n")
        
        results = {}
        
        try:
            # Stage 1: Data Acquisition
            logger.info("\n[STAGE 1/6] DATA ACQUISITION AND ENGINEERING")
            logger.info("-" * 80)
            df = self._stage_data_acquisition()
            results['data'] = df
            
            # Stage 2: Feature Engineering
            logger.info("\n[STAGE 2/6] FEATURE ENGINEERING")
            logger.info("-" * 80)
            df = self._stage_feature_engineering(df)
            results['features'] = df
            
            # Stage 3: Regime Detection
            logger.info("\n[STAGE 3/6] REGIME DETECTION")
            logger.info("-" * 80)
            df = self._stage_regime_detection(df)
            results['regime_analysis'] = self._analyze_regimes(df)
            
            # Stage 4: Strategy Execution
            logger.info("\n[STAGE 4/6] STRATEGY EXECUTION")
            logger.info("-" * 80)
            df, trades = self._stage_strategy_execution(df)
            results['trades'] = trades
            
            # Stage 5: Machine Learning
            logger.info("\n[STAGE 5/6] MACHINE LEARNING MODEL TRAINING")
            logger.info("-" * 80)
            ml_results = self._stage_machine_learning(df)
            results['ml_results'] = ml_results
            
            # Stage 6: Backtesting and Analysis
            logger.info("\n[STAGE 6/6] BACKTESTING AND ANALYSIS")
            logger.info("-" * 80)
            backtest_results = self._stage_backtesting_analysis(df, trades)
            results['backtest'] = backtest_results
            
            # Generate Reports
            logger.info("\n[FINAL] REPORT GENERATION AND VISUALIZATION")
            logger.info("-" * 80)
            self._generate_reports(df, trades, results)
            
            logger.info("\n" + "="*80)
            logger.info("✓ TRADING SYSTEM PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80 + "\n")
            
            return results
        
        except Exception as e:
            logger.error(f"✗ Pipeline execution failed: {str(e)}", exc_info=True)
            raise
    
    def _stage_data_acquisition(self) -> pd.DataFrame:
        """Stage 1: Data Acquisition."""
        df = self.data_pipeline.load_data()
        
        logger.info(f"✓ Data loaded: {len(df)} bars")
        logger.info(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"✓ Features: {list(df.columns)}")
        
        return df
    
    def _stage_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Feature Engineering."""
        df = self.feature_engineer.engineer_features(df)
        
        logger.info(f"✓ Features engineered: {len(df.columns)} total columns")
        logger.info(f"✓ New features added: {', '.join(self.feature_engineer.new_features)}")
        
        return df
    
    def _stage_regime_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: Regime Detection."""
        df = self.regime_detector.detect_regimes(df)
        
        regime_counts = df['regime'].value_counts()
        logger.info(f"✓ Regimes detected:")
        for regime_id, count in regime_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  Regime {self.regime_detector.get_regime_name(int(regime_id))}: {count} bars ({pct:.1f}%)")
        
        return df
    
    def _analyze_regimes(self, df: pd.DataFrame) -> Dict:
        """Analyze regime characteristics."""
        analyzer = RegimeAnalyzer()
        
        analysis = {}
        for regime_id in df['regime'].unique():
            characteristics = analyzer.analyze_regime_characteristics(df, regime_id)
            analysis[f"regime_{int(regime_id)}"] = characteristics
        
        return analysis
    
    def _stage_strategy_execution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Stage 4: Strategy Execution."""
        df = self.strategy_executor.generate_signals(df)
        df, trades = self.strategy_executor.execute_trades(df)
        
        logger.info(f"✓ Trades generated: {len(trades)}")
        
        if trades:
            trade_stats = self.strategy_executor.get_trade_statistics()
            logger.info(f"✓ Win Rate: {trade_stats['win_rate']:.2f}%")
            logger.info(f"✓ Average Trade PnL: ${trade_stats['avg_pnl']:.2f}")
        
        return df, trades
    
    def _stage_machine_learning(self, df: pd.DataFrame) -> Dict:
        """Stage 5: Machine Learning."""
        X, y, feature_cols = self.model_trainer.prepare_training_data(df)
        
        if len(X) < 100:
            logger.warning("Insufficient data for ML training")
            return {}
        
        gb_model, gb_accuracy, gb_metrics = self.model_trainer.train_gradient_boosting(X, y)
        rf_model, rf_accuracy, rf_metrics = self.model_trainer.train_random_forest(X, y)
        
        # Get feature importance
        gb_importance = self.model_trainer.get_feature_importance('gradient_boosting')
        
        logger.info(f"\nTop 5 Important Features (Gradient Boosting):")
        for idx, row in gb_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        results = {
            'gradient_boosting': {
                'accuracy': gb_accuracy,
                'metrics': gb_metrics
            },
            'random_forest': {
                'accuracy': rf_accuracy,
                'metrics': rf_metrics
            },
            'feature_importance': gb_importance.to_dict('records')
        }
        
        return results
    
    def _stage_backtesting_analysis(self, df: pd.DataFrame, 
                                    trades: List) -> Dict:
        """Stage 6: Backtesting and Analysis."""
        initial_capital = self.config.get('backtest', {}).get('initial_capital', 100000)
        
        metrics, df_backtest = self.backtest_engine.run_backtest(
            df, trades, initial_capital
        )
        
        # Outlier analysis
        outlier_stats = self.outlier_analyzer.detect_outliers(df)
        
        logger.info(f"\n✓ Outlier Analysis:")
        logger.info(f"  Outlier Trades: {outlier_stats.get('num_outliers', 0)}")
        logger.info(f"  Outlier %: {outlier_stats.get('outlier_pct', 0):.2f}%")
        
        return {
            'metrics': metrics,
            'backtest_df': df_backtest,
            'outliers': outlier_stats
        }
    
    def _generate_reports(self, df: pd.DataFrame, trades: List, 
                         results: Dict):
        """Generate comprehensive reports."""
        # Generate backtest report
        if 'backtest' in results and results['backtest']:
            metrics = results['backtest']['metrics']
            report = self.backtest_engine.generate_backtest_report(metrics, trades)
            logger.info("\n" + report)
            
            # Save report
            report_path = Path('reports') / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"\n✓ Report saved: {report_path}")
        
        # Generate visualizations
        logger.info("\n✓ Generating visualizations...")
        self.visualizer.create_comprehensive_dashboard(df, trades)
        logger.info("✓ Visualizations saved to visualizations/")
        
        # Save results to CSV
        self._save_results_to_csv(df, trades, results)
    
    def _save_results_to_csv(self, df: pd.DataFrame, trades: List, 
                            results: Dict):
        """Save results to CSV files."""
        # Save trades
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv('results/trades.csv', index=False)
            logger.info(f"✓ Trades saved: results/trades.csv")
        
        # Save full data with signals
        df.to_csv('results/full_data_with_signals.csv')
        logger.info(f"✓ Full data saved: results/full_data_with_signals.csv")
        
        # Save regime analysis
        if 'regime_analysis' in results:
            regime_df = pd.DataFrame(results['regime_analysis']).T
            regime_df.to_csv('results/regime_analysis.csv')
            logger.info(f"✓ Regime analysis saved: results/regime_analysis.csv")
        
        # Save ML results summary
        if 'ml_results' in results and results['ml_results']:
            ml_summary = {
                'Model': ['Gradient Boosting', 'Random Forest'],
                'Accuracy': [
                    results['ml_results'].get('gradient_boosting', {}).get('accuracy', 0),
                    results['ml_results'].get('random_forest', {}).get('accuracy', 0)
                ]
            }
            ml_df = pd.DataFrame(ml_summary)
            ml_df.to_csv('results/ml_results.csv', index=False)
            logger.info(f"✓ ML results saved: results/ml_results.csv")


class TradeDetailedExporter:
    """Export detailed trade information."""
    
    @staticmethod
    def export_detailed_trades(df: pd.DataFrame, trades: List, 
                              output_file: str = 'results/detailed_trades.csv'):
        """
        Export trades with detailed analysis.
        
        Args:
            df: Market data DataFrame
            trades: List of trade dictionaries
            output_file: Output file path
        """
        detailed_trades = []
        
        for trade in trades:
            entry_bar = df.iloc[trade['entry_idx']]
            exit_bar = df.iloc[trade['exit_idx']]
            
            detailed_trade = {
                'trade_id': trade['trade_id'],
                'type': trade['position_type'],
                'entry_time': entry_bar.get('timestamp', trade['entry_idx']),
                'entry_price': trade['entry_price'],
                'entry_regime': entry_bar.get('regime', np.nan),
                'entry_iv': entry_bar.get('iv', np.nan),
                'exit_time': exit_bar.get('timestamp', trade['exit_idx']),
                'exit_price': trade['exit_price'],
                'exit_reason': trade['exit_reason'],
                'pnl': trade['pnl'],
                'pnl_pct': trade['pnl_pct'],
                'bars_held': trade['bars_held'],
                'risk_reward': abs(trade['pnl'] / (trade['entry_price'] * 0.02)) if trade['entry_price'] > 0 else 0
            }
            detailed_trades.append(detailed_trade)
        
        detailed_df = pd.DataFrame(detailed_trades)
        detailed_df.to_csv(output_file, index=False)
        logger.info(f"✓ Detailed trades exported: {output_file}")
        
        return detailed_df


def main():
    """Main execution function."""
    try:
        # Initialize orchestrator
        orchestrator = TradingSystemOrchestrator()
        
        # Run complete pipeline
        results = orchestrator.run_pipeline()
        
        # Export detailed trades
        if 'trades' in results and results['trades']:
            TradeDetailedExporter.export_detailed_trades(
                results['features'],
                results['trades']
            )
        
        logger.info("\n✓ System execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
