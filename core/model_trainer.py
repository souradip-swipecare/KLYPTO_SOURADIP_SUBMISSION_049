"""
NIFTY 50 Algorithmic Trading System - Model Trainer
Machine learning model training and evaluation
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Tuple, Any
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Professional ML model training for trade classification.
    """
    
    def __init__(self, config: Dict):
        """Initialize model trainer."""
        self.config = config
        self.ml_config = config.get('machine_learning', {})
        self.models = {}
        self.scalers = {}
        self.feature_cols = None
        logger.info("ModelTrainer initialized")
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare data for ML training.
        
        Args:
            df: DataFrame with market data and trades
            
        Returns:
            Tuple of (features, labels, feature_columns)
        """
        logger.info("Preparing training data...")
        
        # Select feature columns
        feature_cols = self.ml_config.get('features', [
            'returns', 'volatility', 'iv', 'basis', 'pcr',
            'rsi', 'macd', 'atr'
        ])
        
        # Filter to available features
        available_features = [f for f in feature_cols if f in df.columns]
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # Extract features
        X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Create labels (1 if profitable trade, 0 otherwise)
        y = np.where(df['pnl'] > 0, 1, 0)
        
        # Remove NaN rows
        valid_idx = ~X.isnull().any(axis=1)
        X = X[valid_idx].values
        y = y[valid_idx]
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        self.feature_cols = available_features
        
        return X, y, available_features
    
    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> Tuple[object, float, Dict]:
        """
        Train Gradient Boosting classifier.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Tuple of (trained model, accuracy, metrics dictionary)
        """
        logger.info("\nTraining Gradient Boosting Classifier...")
        
        # Get model parameters
        gb_params = self.ml_config.get('gradient_boosting', {})
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        gb_model = GradientBoostingClassifier(
            n_estimators=gb_params.get('n_estimators', 100),
            learning_rate=gb_params.get('learning_rate', 0.1),
            max_depth=gb_params.get('max_depth', 3),
            min_samples_split=gb_params.get('min_samples_split', 5),
            min_samples_leaf=gb_params.get('min_samples_leaf', 2),
            random_state=42
        )
        
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = gb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"✓ Gradient Boosting Accuracy: {accuracy*100:.2f}%")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        
        self.models['gradient_boosting'] = gb_model
        self.scalers['gradient_boosting'] = scaler
        
        return gb_model, accuracy, metrics
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> Tuple[object, float, Dict]:
        """
        Train Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Tuple of (trained model, accuracy, metrics dictionary)
        """
        logger.info("\nTraining Random Forest Classifier...")
        
        # Get model parameters
        rf_params = self.ml_config.get('random_forest', {})
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth', 10),
            min_samples_split=rf_params.get('min_samples_split', 5),
            min_samples_leaf=rf_params.get('min_samples_leaf', 2),
            n_jobs=-1,
            random_state=42
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"✓ Random Forest Accuracy: {accuracy*100:.2f}%")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        
        self.models['random_forest'] = rf_model
        self.scalers['random_forest'] = scaler
        
        return rf_model, accuracy, metrics
    
    def get_feature_importance(self, model_name: str = 'gradient_boosting') -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        importances = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_models(self, output_dir: str):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = output_path / f"{model_name}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            scaler_file = output_path / f"{model_name}_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[model_name], f)
            
            logger.info(f"✓ Saved {model_name} to {model_file}")
    
    def load_models(self, models_dir: str):
        """
        Load trained models from disk.
        
        Args:
            models_dir: Directory containing model files
        """
        models_path = Path(models_dir)
        
        for model_file in models_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            
            with open(model_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            
            scaler_file = models_path / f"{model_name}_scaler.pkl"
            with open(scaler_file, 'rb') as f:
                self.scalers[model_name] = pickle.load(f)
            
            logger.info(f"✓ Loaded {model_name} from {model_file}")
    
    def predict(self, X: np.ndarray, model_name: str = 'gradient_boosting') -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            Predicted labels
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not trained")
            return np.zeros(len(X))
        
        scaler = self.scalers[model_name]
        X_scaled = scaler.transform(X)
        
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray, model_name: str = 'gradient_boosting') -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            Probability predictions
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not trained")
            return np.zeros((len(X), 2))
        
        scaler = self.scalers[model_name]
        X_scaled = scaler.transform(X)
        
        model = self.models[model_name]
        probabilities = model.predict_proba(X_scaled)
        
        return probabilities


class ModelEvaluator:
    """Evaluate and compare trained models."""
    
    @staticmethod
    def cross_validate_model(model: object, X: np.ndarray, y: np.ndarray, 
                            cv_splits: int = 5) -> Dict:
        """
        Perform time-series cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Label vector
            cv_splits: Number of CV splits
            
        Returns:
            Cross-validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train_scaled, y_train)
            
            score = model_clone.score(X_test_scaled, y_test)
            scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }


if __name__ == "__main__":
    config = {
        'machine_learning': {
            'features': ['returns', 'volatility', 'iv'],
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10
            }
        }
    }
    
    trainer = ModelTrainer(config)
    print("Model Trainer Module Ready")
