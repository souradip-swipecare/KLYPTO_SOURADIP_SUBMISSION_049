"""
NIFTY 50 Algorithmic Trading System - Feature Engineering
Advanced technical indicators and derived features for trading
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Professional feature engineering for algorithmic trading.
    Calculates technical indicators, Greeks, and derived metrics.
    """
    
    def __init__(self, config: Dict):
        """Initialize feature engineer with configuration."""
        self.config = config
        logger.info("FeatureEngineer initialized")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators (EMAs).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Calculating technical indicators...")
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # EMA periods from config
        ema_fast = self.config.get('technical_indicators', {}).get('ema_fast', 5)
        ema_slow = self.config.get('technical_indicators', {}).get('ema_slow', 15)
        
        # Calculate EMAs
        df['ema_fast'] = ta.trend.ema_indicator(close=df['close'], window=ema_fast)
        df['ema_slow'] = ta.trend.ema_indicator(close=df['close'], window=ema_slow)
        
        # Additional indicators for robustness
        df['rsi'] = ta.momentum.rsi(close=df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], 
                                                      close=df['close'], window=14)
        
        logger.info(f"✓ Technical indicators calculated")
        return df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from market data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Calculating derived features...")
        
        # IV metrics
        if 'iv' in df.columns:
            df['iv_ma_20'] = df['iv'].rolling(window=20).mean()
            df['iv_std'] = df['iv'].rolling(window=20).std()
            df['iv_zscore'] = (df['iv'] - df['iv_ma_20']) / (df['iv_std'] + 1e-6)
        
        # Returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['returns_rolling_std'] = df['log_returns'].rolling(window=20).std()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(20)
        df['momentum_pct'] = df['momentum'] / df['close'].shift(20)
        
        # Volume analysis
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-6)
        
        # Price patterns
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['hl_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
        
        logger.info(f"✓ Derived features calculated")
        return df
    
    def calculate_greeks_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare Greek-based features.
        
        Args:
            df: DataFrame with Greeks data
            
        Returns:
            DataFrame with Greek features
        """
        logger.info("Calculating Greek-based features...")
        
        if 'delta' not in df.columns:
            logger.warning("Delta not found, skipping Greek features")
            return df
        
        # Delta exposure
        df['delta_exposure'] = df['delta'].abs()
        df['delta_trend'] = df['delta'].diff()
        
        # Gamma exposure (convexity)
        if 'gamma' in df.columns:
            df['gamma_exposure'] = df['gamma'] * df['close']
            df['gamma_trend'] = df['gamma'].diff()
        
        # Vega exposure (volatility sensitivity)
        if 'vega' in df.columns:
            df['vega_exposure'] = df['vega']
            df['vega_trend'] = df['vega'].diff()
        
        # Theta exposure (time decay)
        if 'theta' in df.columns:
            df['theta_daily'] = df['theta']  # Daily decay
        
        logger.info(f"✓ Greek features calculated")
        return df
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk-related metrics.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with risk metrics
        """
        logger.info("Calculating risk metrics...")
        
        # Volatility metrics
        df['volatility'] = df['log_returns'].rolling(window=20).std() * np.sqrt(252*75)  # Annualized
        df['volatility_regime'] = pd.cut(df['volatility'], bins=3, labels=['Low', 'Medium', 'High'])
        
        # Drawdown from peak
        df['cumulative_return'] = (1 + df['log_returns']).cumprod()
        df['running_max'] = df['cumulative_return'].expanding().max()
        df['drawdown'] = (df['cumulative_return'] - df['running_max']) / df['running_max']
        
        # Value at Risk (VaR) - 95% confidence
        df['var_95'] = df['log_returns'].rolling(window=252).quantile(0.05)
        
        logger.info(f"✓ Risk metrics calculated")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Args:
            df: Raw market data
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("="*60)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*60)
        
        # Step 1: Technical Indicators
        df = self.calculate_technical_indicators(df)
        
        # Step 2: Derived Features
        df = self.calculate_derived_features(df)
        
        # Step 3: Greek Features
        df = self.calculate_greeks_features(df)
        
        # Step 4: Risk Metrics
        df = self.calculate_risk_metrics(df)
        
        # Remove NaN rows created by rolling windows
        df = df.dropna()
        
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info(f"Total features engineered: {len(df.columns)}")
        logger.info("="*60)
        
        return df


class FeatureSelector:
    """Feature selection and importance analysis."""
    
    @staticmethod
    def select_features_for_ml(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Select relevant features for ML models.
        
        Args:
            df: DataFrame with all features
            target_col: Target column name
            
        Returns:
            DataFrame with selected features
        """
        
        ml_features = [
            'ema_fast', 'ema_slow', 'rsi', 'macd', 'atr',
            'iv', 'iv_zscore', 'delta', 'gamma', 'vega',
            'basis', 'pcr', 'returns', 'momentum',
            'volatility', 'log_returns'
        ]
        
        available_features = [f for f in ml_features if f in df.columns]
        
        if target_col and target_col in df.columns:
            available_features.append(target_col)
        
        return df[available_features]


if __name__ == "__main__":
    config = {
        'technical_indicators': {
            'ema_fast': 5,
            'ema_slow': 15
        }
    }
    
    engineer = FeatureEngineer(config)
    print("Feature Engineering Module Ready")
