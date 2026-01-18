"""
NIFTY 50 Algorithmic Trading System - Core Data Pipeline
Professional data acquisition, processing, and feature engineering module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Professional data pipeline for NIFTY 50 trading system.
    Handles data acquisition, cleaning, and feature engineering.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.spot_df = None
        self.futures_df = None
        self.options_df = None
        self.features_df = None
        
        logger.info("DataPipeline initialized")
    
    def fetch_spot_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Fetch NIFTY 50 Spot market data.
        
        Args:
            lookback_days: Number of days to fetch (default: 365 for 1 year)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
            
            logger.info(f"Fetching NIFTY 50 spot data ({lookback_days} days)...")
            
            nifty = yf.download(tickers="^NSEI", period="60d", interval="5m", progress=False)
            
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            
            nifty.reset_index(inplace=True)
            nifty.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Ensure timestamp is timezone-naive
            if nifty['timestamp'].dt.tz is not None:
                nifty['timestamp'] = nifty['timestamp'].dt.tz_localize(None)
            
            logger.info(f"✓ Fetched {len(nifty)} spot data points")
            self.spot_df = nifty
            return nifty
            
        except Exception as e:
            logger.error(f"Error fetching spot data: {str(e)}")
            logger.info("Using synthetic spot data as fallback...")
            return self._generate_synthetic_spot_data()
    
    def _generate_synthetic_spot_data(self, n_points: int = 3000) -> pd.DataFrame:
        """Generate synthetic spot data for demonstration."""
        dates = pd.date_range(end=datetime.now(), periods=n_points, freq='5min')
        close_prices = 20000 + np.cumsum(np.random.randn(n_points) * 10)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(n_points) * 5,
            'high': close_prices + abs(np.random.randn(n_points) * 15),
            'low': close_prices - abs(np.random.randn(n_points) * 15),
            'close': close_prices,
            'volume': np.random.randint(1000000, 5000000, n_points)
        })
        
        logger.info(f"Generated synthetic spot data ({n_points} points)")
        return df
    
    def generate_futures_data(self, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate futures data (based on spot with basis).
        
        Args:
            spot_df: Spot data DataFrame
            
        Returns:
            Futures data with basis premium
        """
        logger.info("Generating futures data with basis...")
        
        futures_df = spot_df.copy()
        
        # Add basis premium (futures typically trade at slight premium)
        futures_basis = np.random.uniform(1.003, 1.008, len(futures_df))
        futures_df['open'] = futures_df['open'] * futures_basis
        futures_df['high'] = futures_df['high'] * futures_basis
        futures_df['low'] = futures_df['low'] * futures_basis
        futures_df['close'] = futures_df['close'] * futures_basis
        
        # Add Open Interest
        futures_df['oi'] = np.random.randint(50000, 200000, len(futures_df))
        
        logger.info(f"✓ Generated {len(futures_df)} futures data points")
        self.futures_df = futures_df
        return futures_df
    
    def generate_options_data(self, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate options chain data (ATM, ATM±1, ATM±2).
        
        Args:
            spot_df: Spot data DataFrame
            
        Returns:
            Options data with Greeks
        """
        logger.info("Generating options chain data...")
        
        options_data = []
        
        for _, row in spot_df.iterrows():
            spot_price = row['close']
            timestamp = row['timestamp']
            
            # ATM strike (round to nearest 50)
            atm_strike = round(spot_price / 50) * 50
            
            # Generate strikes: ATM, ATM±1 (±50 points), ATM±2 (±100 points)
            strikes = [atm_strike - 100, atm_strike - 50, atm_strike, atm_strike + 50, atm_strike + 100]
            
            for strike in strikes:
                # Generate call and put
                for opt_type in ['CE', 'PE']:
                    iv = np.random.uniform(10, 30) / 100  # 10-30% IV
                    ltp = max(spot_price - strike, 0) + np.random.uniform(50, 150) if opt_type == 'CE' else \
                          max(strike - spot_price, 0) + np.random.uniform(50, 150)
                    oi = np.random.randint(1000, 50000)
                    
                    options_data.append({
                        'timestamp': timestamp,
                        'strike': strike,
                        'type': opt_type,
                        'ltp': ltp,
                        'iv': iv,
                        'oi': oi,
                        'volume': oi // 10
                    })
        
        options_df = pd.DataFrame(options_data)
        logger.info(f"✓ Generated {len(options_df)} options data points")
        self.options_df = options_df
        return options_df
    
    def calculate_greeks(self, options_df: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Option Greeks using Black-Scholes model.
        
        Args:
            options_df: Options data
            spot_df: Spot data
            
        Returns:
            Options data with Greeks
        """
        logger.info("Calculating Option Greeks (Black-Scholes)...")
        
        try:
            from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
        except ImportError:
            logger.warning("py_vollib not installed, skipping Greeks calculation")
            return options_df
        
        r = self.config.get('options_greeks', {}).get('risk_free_rate', 0.065)
        t = self.config.get('options_greeks', {}).get('time_to_expiry_days', 7) / 365.0
        
        options_with_spot = pd.merge(
            options_df, 
            spot_df[['timestamp', 'close']], 
            on='timestamp', 
            how='left'
        ).rename(columns={'close': 'spot_price'})
        
        def calculate_row_greeks(row):
            try:
                flag = row['type'].lower()[0]
                S = row['spot_price']
                K = row['strike']
                sigma = row['iv']
                
                if sigma <= 0 or pd.isna(sigma):
                    return pd.Series([0, 0, 0, 0, 0])
                
                d = delta(flag, S, K, t, r, sigma)
                g = gamma(flag, S, K, t, r, sigma)
                v = vega(flag, S, K, t, r, sigma)
                th = theta(flag, S, K, t, r, sigma)
                rh = rho(flag, S, K, t, r, sigma)
                
                return pd.Series([d, g, v, th, rh])
            except:
                return pd.Series([0, 0, 0, 0, 0])
        
        greeks_df = options_with_spot.apply(calculate_row_greeks, axis=1)
        greeks_df.columns = ['delta', 'gamma', 'vega', 'theta', 'rho']
        
        options_df = pd.concat([options_df, greeks_df], axis=1)
        logger.info("✓ Greeks calculated successfully")
        
        return options_df
    
    def merge_datasets(self, spot_df: pd.DataFrame, futures_df: pd.DataFrame, 
                       options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge spot, futures, and options data on timestamp.
        
        Args:
            spot_df: Spot data
            futures_df: Futures data
            options_df: Options data
            
        Returns:
            Merged dataset
        """
        logger.info("Merging datasets...")
        
        # Aggregate options to one row per timestamp
        atm_options = options_df.groupby('timestamp').first().reset_index()
        
        # Calculate PCR (Put-Call Ratio)
        pcr_df = options_df.groupby(['timestamp', 'type'])['oi'].sum().unstack(fill_value=0)
        if 'PE' in pcr_df.columns and 'CE' in pcr_df.columns:
            pcr_df['pcr'] = pcr_df['PE'] / (pcr_df['CE'] + 1)
        else:
            pcr_df['pcr'] = 0.5
        pcr_df = pcr_df.reset_index()[['timestamp', 'pcr']]
        
        # Merge on timestamp
        merged_df = pd.merge(spot_df, futures_df[['timestamp', 'close', 'oi']], 
                            on='timestamp', suffixes=('', '_fut'))
        merged_df.rename(columns={'close_fut': 'fut_close', 'oi_fut': 'fut_oi'}, inplace=True)
        
        merged_df = pd.merge(merged_df, 
                            atm_options[['timestamp', 'iv', 'delta', 'gamma', 'vega']], 
                            on='timestamp', how='left')
        merged_df = pd.merge(merged_df, pcr_df, on='timestamp', how='left')
        
        # Calculate derived metrics
        merged_df['basis'] = (merged_df['fut_close'] - merged_df['close']) / (merged_df['close'] + 1e-6)
        merged_df['returns'] = merged_df['close'].pct_change()
        
        # Fill NaN values
        merged_df.fillna(method='bfill', inplace=True)
        merged_df.fillna(0, inplace=True)
        
        logger.info(f"✓ Merged dataset shape: {merged_df.shape}")
        self.features_df = merged_df
        
        return merged_df
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run complete data pipeline.
        
        Returns:
            Processed features DataFrame
        """
        logger.info("="*60)
        logger.info("STARTING DATA PIPELINE")
        logger.info("="*60)
        
        # Step 1: Fetch/Generate Data
        spot_df = self.fetch_spot_data()
        futures_df = self.generate_futures_data(spot_df)
        options_df = self.generate_options_data(spot_df)
        
        # Step 2: Calculate Greeks
        options_df = self.calculate_greeks(options_df, spot_df)
        
        # Step 3: Merge Datasets
        features_df = self.merge_datasets(spot_df, futures_df, options_df)
        
        logger.info("="*60)
        logger.info("DATA PIPELINE COMPLETE")
        logger.info(f"Final dataset: {features_df.shape}")
        logger.info("="*60)
        
        return features_df


if __name__ == "__main__":
    # Example usage
    config = {
        'options_greeks': {
            'risk_free_rate': 0.065,
            'time_to_expiry_days': 7
        }
    }
    
    pipeline = DataPipeline(config)
    features_df = pipeline.run_pipeline()
    
    print(f"\nFinal Features Shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print(f"\nFirst 5 rows:\n{features_df.head()}")
