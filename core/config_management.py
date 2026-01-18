"""
Configuration Management System
================================
Centralized configuration management for the trading system.

Author: AlgoTrading System
Date: 2026-01-18
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import logging


@dataclass
class DataConfig:
    """Data configuration"""
    timeframe: str = "5min"
    lookback_period: int = 1
    data_sources: Dict[str, str] = field(default_factory=lambda: {
        "spot": "yfinance",
        "futures": "synthetic",
        "options": "synthetic"
    })


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    ema_fast: int = 5
    ema_fast: int = 15
    min_rsi: int = 30
    max_rsi: int = 70
    min_atr_threshold: float = 0.0


@dataclass
class RiskConfig:
    """Risk management configuration"""
    initial_capital: float = 100000.0
    position_sizing_method: str = "fixed"
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    max_daily_loss_pct: float = 0.05
    max_drawdown_limit: float = 0.20
    stop_loss_enabled: bool = True
    stop_loss_method: str = "fixed"
    stop_loss_points: float = 50.0
    atr_multiplier: float = 2.0
    take_profit_enabled: bool = True
    take_profit_method: str = "risk_reward"
    risk_reward_ratio: float = 3.0


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    models: List[str] = field(default_factory=lambda: ["gb", "rf"])
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_depth: int = 10
    n_estimators: int = 100
    learning_rate: float = 0.1


class ConfigManager:
    """
    Professional configuration management system
    """
    
    def __init__(self, config_file: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize Configuration Manager
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config: Dict[str, Any] = {}
        self.config_file = config_file
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self._load_defaults()
    
    def load_config(self, config_file: str) -> bool:
        """
        Load configuration from file
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if successful
        """
        try:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.logger.error(f"Unsupported config format: {config_file}")
                return False
            
            self.config_file = config_file
            self.logger.info(f"Configuration loaded from {config_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return False
    
    def save_config(self, output_file: Optional[str] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            output_file: Output file path
            
        Returns:
            True if successful
        """
        try:
            output_file = output_file or self.config_file
            if not output_file:
                self.logger.error("No output file specified")
                return False
            
            # Ensure directory exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                with open(output_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation: section.key)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to parent key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            self.logger.debug(f"Set {key} = {value}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error setting config: {e}")
            return False
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of key-value pairs to update
            
        Returns:
            True if successful
        """
        try:
            for key, value in updates.items():
                self.set(key, value)
            return True
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name
            
        Returns:
            Section dictionary
        """
        return self.config.get(section, {})
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration
        
        Args:
            None
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required sections
        required_sections = ['data', 'strategy', 'risk', 'ml']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate values
        if 'risk' in self.config:
            risk = self.config['risk']
            if risk.get('initial_capital', 0) <= 0:
                errors.append("initial_capital must be positive")
            if not 0 <= risk.get('max_daily_loss_pct', 0.05) <= 1:
                errors.append("max_daily_loss_pct must be between 0 and 1")
        
        if 'strategy' in self.config:
            strategy = self.config['strategy']
            ema_fast = strategy.get('ema_fast')
            ema_slow = strategy.get('ema_slow')
            if ema_fast and ema_slow and ema_fast >= ema_slow:
                errors.append("ema_fast must be less than ema_slow")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def print_config(self) -> None:
        """Print configuration to console"""
        print("\n" + "="*80)
        print("TRADING SYSTEM CONFIGURATION")
        print("="*80)
        print(yaml.dump(self.config, default_flow_style=False))
        print("="*80 + "\n")
    
    def export_to_json(self, output_file: str) -> bool:
        """Export configuration as JSON"""
        return self.save_config(output_file)
    
    def export_to_yaml(self, output_file: str) -> bool:
        """Export configuration as YAML"""
        return self.save_config(output_file)
    
    def _load_defaults(self) -> None:
        """Load default configuration"""
        self.config = {
            'market': {
                'symbol': 'NIFTY',
                'exchange': 'NSE',
                'asset_class': 'INDEX',
                'base_currency': 'INR'
            },
            'data': {
                'timeframe': '5min',
                'lookback_period': 1,
                'data_sources': {
                    'spot': 'yfinance',
                    'futures': 'synthetic',
                    'options': 'synthetic'
                }
            },
            'strategy': {
                'ema_fast': 5,
                'ema_slow': 15,
                'min_rsi': 30,
                'max_rsi': 70
            },
            'risk': {
                'initial_capital': 100000.0,
                'position_sizing_method': 'fixed',
                'risk_per_trade': 0.02,
                'max_position_size': 0.1,
                'max_daily_loss_pct': 0.05,
                'stop_loss_enabled': True,
                'stop_loss_method': 'fixed',
                'stop_loss_points': 50.0
            },
            'ml': {
                'models': ['gb', 'rf'],
                'test_size': 0.2,
                'cv_folds': 5
            },
            'backtest': {
                'initial_capital': 100000.0,
                'commission_pct': 0.0,
                'slippage_pct': 0.0
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading_system.log',
                'max_bytes': 10485760,
                'backup_count': 5
            }
        }
        self.logger.info("Default configuration loaded")


class EnvironmentConfig:
    """
    Environment-specific configuration
    """
    
    @staticmethod
    def get_environment() -> str:
        """Get current environment"""
        import os
        return os.getenv('TRADING_ENV', 'development')
    
    @staticmethod
    def load_environment_config(env: Optional[str] = None) -> Dict[str, Any]:
        """
        Load environment-specific configuration
        
        Args:
            env: Environment name (development, testing, production)
            
        Returns:
            Environment configuration
        """
        env = env or EnvironmentConfig.get_environment()
        
        configs = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'dry_run': True,
                'max_threads': 2,
                'cache_enabled': False
            },
            'testing': {
                'debug': False,
                'log_level': 'INFO',
                'dry_run': True,
                'max_threads': 4,
                'cache_enabled': True
            },
            'production': {
                'debug': False,
                'log_level': 'WARNING',
                'dry_run': False,
                'max_threads': 8,
                'cache_enabled': True,
                'monitoring_enabled': True,
                'alerting_enabled': True
            }
        }
        
        return configs.get(env, configs['development'])


def create_default_config_file(output_path: str = "config/trading_config.yaml") -> str:
    """
    Create a default configuration file
    
    Args:
        output_path: Output file path
        
    Returns:
        Path to created configuration file
    """
    manager = ConfigManager()
    manager.save_config(output_path)
    return output_path


if __name__ == "__main__":
    print("Configuration Management System Loaded Successfully")
