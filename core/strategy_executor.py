"""
NIFTY 50 Algorithmic Trading System - Strategy Executor
Advanced trading strategy with risk management and position sizing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position types in trading."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradeSignal:
    """Structure for trade signals."""
    timestamp: datetime
    signal: int  # 1 for long, -1 for short, 0 for no signal
    price: float
    regime: int
    confidence: float = 1.0
    reason: str = ""


class StrategyExecutor:
    """
    Professional strategy execution with risk management.
    Implements EMA crossover with regime filtering.
    """
    
    def __init__(self, config: Dict):
        """Initialize strategy executor."""
        self.config = config
        self.strategy_config = config.get('strategy', {})
        self.risk_config = config.get('risk_management', {})
        self.trades: List[Dict] = []
        self.current_position = PositionType.FLAT
        self.entry_price = None
        logger.info("StrategyExecutor initialized")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with OHLCV, EMA, regime, and other indicators
            
        Returns:
            DataFrame with signal column added
        """
        logger.info("Generating trading signals...")
        
        df_signals = df.copy()
        
        # Initialize signal column
        df_signals['signal'] = 0
        df_signals['confidence'] = 1.0
        df_signals['signal_reason'] = ""
        
        # Get strategy parameters
        ema_short = self.strategy_config.get('ema_short', 5)
        ema_long = self.strategy_config.get('ema_long', 15)
        use_regime_filter = self.strategy_config.get('use_regime_filter', True)
        
        logger.info(f"EMA Parameters: Short={ema_short}, Long={ema_long}")
        logger.info(f"Regime Filter: {use_regime_filter}")
        
        # Ensure EMA columns exist
        if f'ema_{ema_short}' not in df_signals.columns:
            df_signals[f'ema_{ema_short}'] = df_signals['close'].ewm(span=ema_short).mean()
        if f'ema_{ema_long}' not in df_signals.columns:
            df_signals[f'ema_{ema_long}'] = df_signals['close'].ewm(span=ema_long).mean()
        
        ema_short_col = f'ema_{ema_short}'
        ema_long_col = f'ema_{ema_long}'
        
        # Generate signals
        for idx in range(1, len(df_signals)):
            current = df_signals.iloc[idx]
            previous = df_signals.iloc[idx-1]
            
            ema_short_curr = current[ema_short_col]
            ema_long_curr = current[ema_long_col]
            ema_short_prev = previous[ema_short_col]
            ema_long_prev = previous[ema_long_col]
            
            signal = 0
            reason = ""
            confidence = 1.0
            
            # EMA crossover logic
            # Long signal: EMA5 crosses above EMA15
            if ema_short_prev <= ema_long_prev and ema_short_curr > ema_long_curr:
                signal = 1
                reason = "Golden Cross (EMA5 > EMA15)"
            
            # Short signal: EMA5 crosses below EMA15
            elif ema_short_prev >= ema_long_prev and ema_short_curr < ema_long_curr:
                signal = -1
                reason = "Death Cross (EMA5 < EMA15)"
            
            # Apply regime filter if enabled
            if use_regime_filter and 'regime' in current.index:
                regime = current['regime']
                
                if signal == 1 and regime != 1:  # Only long in uptrend
                    signal = 0
                    reason = f"Regime filter blocked long signal (regime={regime})"
                    confidence = 0.5
                
                elif signal == -1 and regime != -1:  # Only short in downtrend
                    signal = 0
                    reason = f"Regime filter blocked short signal (regime={regime})"
                    confidence = 0.5
            
            # Check RSI filter if available
            if 'rsi' in current.index:
                rsi = current['rsi']
                if signal == 1 and rsi > 70:  # Overbought
                    confidence = 0.7
                    reason += " [RSI Overbought]"
                elif signal == -1 and rsi < 30:  # Oversold
                    confidence = 0.7
                    reason += " [RSI Oversold]"
            
            # Assign signal
            df_signals.at[df_signals.index[idx], 'signal'] = signal
            df_signals.at[df_signals.index[idx], 'confidence'] = confidence
            df_signals.at[df_signals.index[idx], 'signal_reason'] = reason
        
        # Signal statistics
        signal_counts = df_signals['signal'].value_counts()
        logger.info(f"\nSignal Distribution:")
        logger.info(f"  Long Signals:  {signal_counts.get(1, 0)}")
        logger.info(f"  Short Signals: {signal_counts.get(-1, 0)}")
        logger.info(f"  No Signal:     {signal_counts.get(0, 0)}")
        
        return df_signals
    
    def execute_trades(self, df: pd.DataFrame, initial_capital: float = 100000) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Execute trades based on signals.
        
        Args:
            df: DataFrame with signals
            initial_capital: Initial capital for trading
            
        Returns:
            Tuple of (DataFrame with trade info, list of completed trades)
        """
        logger.info("Executing trades...")
        
        df_trades = df.copy()
        df_trades['position'] = 0
        df_trades['entry_price'] = np.nan
        df_trades['pnl'] = 0.0
        df_trades['pnl_pct'] = 0.0
        df_trades['trade_id'] = 0
        
        trades = []
        current_position = PositionType.FLAT
        entry_price = None
        entry_idx = None
        trade_id = 0
        
        # Get risk management parameters
        max_loss_pct = self.risk_config.get('max_loss_pct', 2.0) / 100
        max_pos_size = self.risk_config.get('max_position_size', 1.0)
        
        logger.info(f"Risk Parameters: Max Loss={max_loss_pct*100}%, Max Position={max_pos_size*100}%")
        
        for idx in range(len(df_trades)):
            signal = df_trades.iloc[idx]['signal']
            price = df_trades.iloc[idx]['close']
            
            # Exit condition: Stop loss or take profit
            if current_position != PositionType.FLAT:
                pnl_pct = (price - entry_price) / entry_price
                
                if current_position == PositionType.LONG:
                    if pnl_pct < -max_loss_pct or signal == -1:
                        # Close long position
                        trade = {
                            'trade_id': trade_id,
                            'entry_idx': entry_idx,
                            'exit_idx': idx,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'pnl': price - entry_price,
                            'pnl_pct': pnl_pct * 100,
                            'bars_held': idx - entry_idx,
                            'position_type': 'LONG',
                            'entry_time': df_trades.iloc[entry_idx]['timestamp'] if 'timestamp' in df_trades.columns else entry_idx,
                            'exit_time': df_trades.iloc[idx]['timestamp'] if 'timestamp' in df_trades.columns else idx,
                            'exit_reason': 'Stop Loss' if pnl_pct < -max_loss_pct else 'Signal'
                        }
                        trades.append(trade)
                        current_position = PositionType.FLAT
                        df_trades.at[df_trades.index[idx], 'position'] = 0
                
                elif current_position == PositionType.SHORT:
                    if pnl_pct > max_loss_pct or signal == 1:
                        # Close short position
                        trade = {
                            'trade_id': trade_id,
                            'entry_idx': entry_idx,
                            'exit_idx': idx,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'pnl': entry_price - price,
                            'pnl_pct': -pnl_pct * 100,
                            'bars_held': idx - entry_idx,
                            'position_type': 'SHORT',
                            'entry_time': df_trades.iloc[entry_idx]['timestamp'] if 'timestamp' in df_trades.columns else entry_idx,
                            'exit_time': df_trades.iloc[idx]['timestamp'] if 'timestamp' in df_trades.columns else idx,
                            'exit_reason': 'Stop Loss' if pnl_pct > max_loss_pct else 'Signal'
                        }
                        trades.append(trade)
                        current_position = PositionType.FLAT
                        df_trades.at[df_trades.index[idx], 'position'] = 0
                
                # Track PnL
                df_trades.at[df_trades.index[idx], 'pnl'] = (price - entry_price) if current_position == PositionType.LONG else (entry_price - price if current_position == PositionType.SHORT else 0)
                df_trades.at[df_trades.index[idx], 'pnl_pct'] = pnl_pct * 100
            
            # Entry condition
            if current_position == PositionType.FLAT and signal != 0:
                if signal == 1:
                    current_position = PositionType.LONG
                    df_trades.at[df_trades.index[idx], 'position'] = 1
                elif signal == -1:
                    current_position = PositionType.SHORT
                    df_trades.at[df_trades.index[idx], 'position'] = -1
                
                entry_price = price
                entry_idx = idx
                trade_id += 1
        
        # Trade statistics
        logger.info(f"\nTrade Statistics:")
        logger.info(f"  Total Trades: {len(trades)}")
        if trades:
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            logger.info(f"  Winning Trades: {winning_trades} ({winning_trades/len(trades)*100:.2f}%)")
            logger.info(f"  Average PnL: {np.mean([t['pnl'] for t in trades]):.2f}")
            logger.info(f"  Max Drawdown Trade: {min([t['pnl'] for t in trades]):.2f}")
        
        self.trades = trades
        return df_trades, trades
    
    def get_trade_statistics(self) -> Dict:
        """
        Calculate comprehensive trade statistics.
        
        Args:
            
        Returns:
            Dictionary of trade statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        pnls = [t['pnl'] for t in self.trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'avg_pnl': np.mean(pnls),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0
        }
        
        return stats


if __name__ == "__main__":
    config = {
        'strategy': {
            'ema_short': 5,
            'ema_long': 15,
            'use_regime_filter': True
        },
        'risk_management': {
            'max_loss_pct': 2.0,
            'max_position_size': 1.0
        }
    }
    
    executor = StrategyExecutor(config)
    print("Strategy Executor Module Ready")
