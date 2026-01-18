"""
NIFTY 50 Algorithmic Trading System - Regime Detection
Hidden Markov Model for market state identification and classification
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Professional regime detection using Hidden Markov Models.
    Identifies market states: Uptrend, Downtrend, Sideways
    """
    
    def __init__(self, config: Dict):
        """Initialize regime detector with configuration."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.regime_mapping = {}
        logger.info("RegimeDetector initialized")
    
    def train_hmm(self, df: pd.DataFrame, features: list = None) -> Tuple[object, np.ndarray]:
        """
        Train HMM for regime detection.
        
        Args:
            df: DataFrame with features
            features: Features to use for HMM training
            
        Returns:
            Tuple of (trained model, predicted states)
        """
        logger.info("Training HMM for regime detection...")
        
        # Default features from config
        if features is None:
            features = self.config.get('regime_detection', {}).get('features', 
                                                                    ['iv', 'basis', 'returns', 'pcr'])
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        logger.info(f"Using features: {available_features}")
        
        # Prepare data
        X = df[available_features].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(X) < 100:
            logger.warning(f"Insufficient data ({len(X)} rows). Using synthetic regimes.")
            return self._create_fallback_model(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Get HMM parameters from config
        hmm_config = self.config.get('regime_detection', {})
        n_states = hmm_config.get('n_states', 3)
        covariance_type = hmm_config.get('covariance_type', 'diag')
        max_iter = hmm_config.get('max_iterations', 1000)
        
        # Train HMM
        try:
            self.model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=max_iter,
                random_state=hmm_config.get('random_state', 42),
                min_covar=0.01
            )
            
            self.model.fit(X_scaled)
            hidden_states = self.model.predict(X_scaled)
            
            logger.info(f"✓ HMM trained successfully with {n_states} states")
            
        except Exception as e:
            logger.warning(f"HMM training failed: {str(e)}. Using fallback model.")
            return self._create_fallback_model(df)
        
        return self.model, hidden_states
    
    def _create_fallback_model(self, df: pd.DataFrame) -> Tuple[object, np.ndarray]:
        """Create fallback regime classification based on returns."""
        returns = df['returns'].fillna(0)
        states = np.where(returns > returns.median(), 1, -1)
        return None, states
    
    def map_states_to_regimes(self, df: pd.DataFrame, states: np.ndarray) -> Dict[int, int]:
        """
        Map HMM states to economic regimes.
        
        Args:
            df: DataFrame with returns
            states: Predicted HMM states
            
        Returns:
            Mapping of state to regime label
        """
        logger.info("Mapping HMM states to economic regimes...")
        
        # Calculate mean return per state
        state_returns = {}
        for state in np.unique(states):
            mask = states == state
            mean_return = df.loc[mask, 'returns'].mean()
            state_returns[state] = mean_return
        
        # Sort by returns
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        
        # Create mapping
        regime_map = {}
        n_states = len(sorted_states)
        
        if n_states == 3:
            regime_map = {
                sorted_states[0][0]: -1,  # Low return -> Downtrend
                sorted_states[1][0]: 0,   # Medium -> Sideways
                sorted_states[2][0]: 1    # High return -> Uptrend
            }
        elif n_states == 2:
            regime_map = {
                sorted_states[0][0]: -1,  # Low -> Downtrend
                sorted_states[1][0]: 1    # High -> Uptrend
            }
        else:
            regime_map = {sorted_states[0][0]: 0}
        
        logger.info(f"Regime Mapping: {regime_map}")
        self.regime_mapping = regime_map
        
        return regime_map
    
    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete regime detection pipeline.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with regime labels
        """
        logger.info("="*60)
        logger.info("STARTING REGIME DETECTION")
        logger.info("="*60)
        
        # Train HMM
        features = self.config.get('regime_detection', {}).get('features')
        model, hidden_states = self.train_hmm(df, features)
        
        # Add to dataframe (with alignment)
        df_copy = df.copy()
        
        # Create temporary dataframe for states (aligned with training data)
        features = self.config.get('regime_detection', {}).get('features', 
                                                                ['iv', 'basis', 'returns', 'pcr'])
        available_features = [f for f in features if f in df.columns]
        X = df_copy[available_features].replace([np.inf, -np.inf], np.nan)
        
        # Initialize regime column
        df_copy['regime_hidden'] = np.nan
        df_copy['regime_hidden'] = df_copy['regime_hidden'].astype('float64')
        
        # Align indices
        valid_idx = X.dropna().index
        df_copy.loc[valid_idx, 'regime_hidden'] = hidden_states
        
        # Forward fill for NaN values
        df_copy['regime_hidden'] = df_copy['regime_hidden'].fillna(method='bfill').fillna(method='ffill')
        
        # Map to regimes
        regime_map = self.map_states_to_regimes(df_copy, df_copy['regime_hidden'].fillna(0).astype(int).values)
        
        # Apply mapping
        df_copy['regime'] = df_copy['regime_hidden'].map(regime_map).fillna(0)
        
        # Regime statistics
        regime_stats = df_copy.groupby('regime').agg({
            'returns': ['mean', 'std', 'count'],
            'iv': 'mean',
            'volatility': 'mean' if 'volatility' in df_copy.columns else None
        })
        
        logger.info("\nRegime Statistics:")
        logger.info(regime_stats)
        
        logger.info("="*60)
        logger.info("REGIME DETECTION COMPLETE")
        logger.info("="*60)
        
        return df_copy
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get human-readable regime name."""
        names = {1: "Uptrend", 0: "Sideways", -1: "Downtrend"}
        return names.get(regime_id, "Unknown")
    
    def get_regime_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate duration of each regime.
        
        Args:
            df: DataFrame with regime column
            
        Returns:
            DataFrame with regime durations
        """
        regime_changes = (df['regime'] != df['regime'].shift()).cumsum()
        duration_df = df.groupby(regime_changes).agg({
            'regime': 'first',
            'timestamp': ['min', 'max', 'count']
        })
        
        duration_df.columns = ['regime', 'start_time', 'end_time', 'duration_bars']
        duration_df['regime_name'] = duration_df['regime'].apply(self.get_regime_name)
        duration_df['duration_hours'] = duration_df['duration_bars'] * 5 / 60
        
        return duration_df


class RegimeAnalyzer:
    """Analyze regime characteristics and transitions."""
    
    @staticmethod
    def get_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix.
        
        Args:
            df: DataFrame with regime column
            
        Returns:
            Transition probability matrix
        """
        transitions = pd.crosstab(
            df['regime'],
            df['regime'].shift(-1),
            normalize='index'
        )
        return transitions
    
    @staticmethod
    def analyze_regime_characteristics(df: pd.DataFrame, regime_id: int) -> Dict:
        """
        Analyze characteristics specific to a regime.
        
        Args:
            df: DataFrame with regime and other columns
            regime_id: Regime to analyze
            
        Returns:
            Dictionary of regime characteristics
        """
        regime_data = df[df['regime'] == regime_id]
        
        characteristics = {
            'regime': regime_id,
            'data_points': len(regime_data),
            'avg_return': regime_data['returns'].mean(),
            'std_return': regime_data['returns'].std(),
            'avg_iv': regime_data.get('iv', pd.Series()).mean(),
            'avg_volatility': regime_data.get('volatility', pd.Series()).mean(),
            'sharpe_ratio': regime_data['returns'].mean() / (regime_data['returns'].std() + 1e-6) if regime_data['returns'].std() > 0 else 0
        }
        
        return characteristics


if __name__ == "__main__":
    config = {
        'regime_detection': {
            'n_states': 3,
            'features': ['iv', 'basis', 'returns', 'pcr'],
            'covariance_type': 'diag'
        }
    }
    
    detector = RegimeDetector(config)
    print("Regime Detection Module Ready")
