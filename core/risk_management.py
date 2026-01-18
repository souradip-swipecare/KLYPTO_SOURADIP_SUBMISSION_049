"""
Risk Management Module
======================
Comprehensive risk management framework including position sizing, drawdown tracking,
stop-loss management, portfolio optimization, and risk metrics calculation.

Author: AlgoTrading System
Date: 2026-01-18
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging
from datetime import datetime


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class RiskLevel(Enum):
    """Risk classification levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional Value at Risk
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    daily_volatility: float = 0.0
    return_on_risk: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


@dataclass
class PositionSizing:
    """Position sizing parameters"""
    method: str = "fixed"  # fixed, kelly, volatility, risk_parity
    size: float = 1.0  # Number of shares/contracts
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.1  # Max 10% of portfolio
    leverage: float = 1.0  # Position leverage


@dataclass
class StopLoss:
    """Stop loss configuration"""
    enabled: bool = True
    method: str = "fixed"  # fixed, atr, percentage
    fixed_points: float = 50.0  # Fixed stop loss in points
    atr_multiplier: float = 2.0  # ATR * multiplier
    percentage: float = 2.0  # 2% stop loss
    trailing_stop: bool = False
    trailing_pct: float = 1.5


@dataclass
class TakeProfit:
    """Take profit configuration"""
    enabled: bool = True
    method: str = "fixed"  # fixed, atr, percentage, risk_reward
    fixed_points: float = 150.0  # Fixed take profit in points
    risk_reward_ratio: float = 3.0  # TP = SL * risk_reward_ratio


# ============================================================================
# RISK MANAGER CLASS
# ============================================================================

class RiskManager:
    """
    Professional risk management system for algorithmic trading.
    
    Responsibilities:
    - Calculate optimal position sizes
    - Monitor and enforce position limits
    - Track drawdown and recovery
    - Calculate comprehensive risk metrics
    - Manage stop losses and take profits
    - Generate risk reports
    """
    
    def __init__(self, initial_capital: float, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize Risk Manager
        
        Args:
            initial_capital: Starting portfolio capital
            config: Configuration dictionary with risk parameters
            logger: Logger instance
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Risk parameters
        self.position_sizing = PositionSizing(**config.get('position_sizing', {}))
        self.stop_loss = StopLoss(**config.get('stop_loss', {}))
        self.take_profit = TakeProfit(**config.get('take_profit', {}))
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.05)  # 5%
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.20)  # 20%
        
        # Tracking variables
        self.daily_pnl = 0.0
        self.open_positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        
    def calculate_position_size(self, 
                               entry_price: float, 
                               stop_price: float,
                               current_capital: Optional[float] = None,
                               volatility: Optional[float] = None) -> float:
        """
        Calculate optimal position size using specified method
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            current_capital: Current portfolio capital
            volatility: Current volatility for volatility-based sizing
            
        Returns:
            Optimal position size (shares/contracts)
        """
        capital = current_capital or self.current_capital
        risk_amount = capital * self.position_sizing.risk_per_trade
        
        if self.position_sizing.method == "fixed":
            size = self.position_sizing.size
            
        elif self.position_sizing.method == "kelly":
            # Kelly Criterion (simplified)
            win_rate = self._estimate_win_rate()
            avg_win = self._estimate_avg_win()
            avg_loss = self._estimate_avg_loss()
            
            if avg_loss == 0:
                size = self.position_sizing.size
            else:
                kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
                size = (capital * kelly_pct) / entry_price
                
        elif self.position_sizing.method == "volatility":
            # Volatility-based sizing
            if volatility and volatility > 0:
                size = risk_amount / (entry_price * volatility * 100)
            else:
                size = self.position_sizing.size
                
        elif self.position_sizing.method == "risk_parity":
            # Risk parity sizing
            risk_points = abs(entry_price - stop_price)
            if risk_points > 0:
                size = risk_amount / risk_points
            else:
                size = self.position_sizing.size
        else:
            size = self.position_sizing.size
        
        # Apply max position size limit
        max_size = (capital * self.position_sizing.max_position_size) / entry_price
        size = min(size, max_size)
        
        return max(1.0, size)  # Minimum 1 share/contract
    
    def calculate_stop_loss(self, 
                           entry_price: float, 
                           direction: str,
                           atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range for ATR-based stops
            
        Returns:
            Stop loss price
        """
        if not self.stop_loss.enabled:
            return None
        
        if self.stop_loss.method == "fixed":
            stop_points = self.stop_loss.fixed_points
            
        elif self.stop_loss.method == "atr":
            if atr is None:
                stop_points = entry_price * self.stop_loss.percentage / 100
            else:
                stop_points = atr * self.stop_loss.atr_multiplier
                
        elif self.stop_loss.method == "percentage":
            stop_points = entry_price * self.stop_loss.percentage / 100
        else:
            stop_points = self.stop_loss.fixed_points
        
        if direction.lower() == "long":
            return entry_price - stop_points
        else:  # short
            return entry_price + stop_points
    
    def calculate_take_profit(self, 
                             entry_price: float, 
                             stop_price: float,
                             direction: str) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            direction: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        if not self.take_profit.enabled:
            return None
        
        if self.take_profit.method == "fixed":
            tp_points = self.take_profit.fixed_points
            
        elif self.take_profit.method == "risk_reward":
            risk_points = abs(entry_price - stop_price)
            tp_points = risk_points * self.take_profit.risk_reward_ratio
        else:
            tp_points = self.take_profit.fixed_points
        
        if direction.lower() == "long":
            return entry_price + tp_points
        else:  # short
            return entry_price - tp_points
    
    def update_position(self, 
                       trade_id: str, 
                       current_price: float, 
                       timestamp: datetime) -> Dict[str, Any]:
        """
        Update open position with current market price
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Position update information
        """
        if trade_id not in self.open_positions:
            return {"error": f"Position {trade_id} not found"}
        
        pos = self.open_positions[trade_id]
        entry_price = pos['entry_price']
        direction = pos['direction']
        size = pos['size']
        
        # Calculate unrealized P&L
        if direction == "long":
            unrealized_pnl = (current_price - entry_price) * size
            unrealized_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            unrealized_pnl = (entry_price - current_price) * size
            unrealized_pct = ((entry_price - current_price) / entry_price) * 100
        
        pos['current_price'] = current_price
        pos['unrealized_pnl'] = unrealized_pnl
        pos['unrealized_pct'] = unrealized_pct
        pos['last_updated'] = timestamp
        
        return {
            "trade_id": trade_id,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pct": unrealized_pct,
            "status": "open"
        }
    
    def check_stop_loss(self, 
                       trade_id: str, 
                       current_price: float) -> Optional[bool]:
        """
        Check if stop loss has been hit
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            
        Returns:
            True if stop loss hit, False otherwise, None if position not found
        """
        if trade_id not in self.open_positions:
            return None
        
        pos = self.open_positions[trade_id]
        stop_price = pos['stop_price']
        direction = pos['direction']
        
        if direction == "long":
            hit = current_price <= stop_price
        else:
            hit = current_price >= stop_price
        
        return hit
    
    def check_take_profit(self, 
                         trade_id: str, 
                         current_price: float) -> Optional[bool]:
        """
        Check if take profit has been hit
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            
        Returns:
            True if take profit hit, False otherwise, None if position not found
        """
        if trade_id not in self.open_positions:
            return None
        
        pos = self.open_positions[trade_id]
        if 'take_profit' not in pos or pos['take_profit'] is None:
            return False
        
        tp_price = pos['take_profit']
        direction = pos['direction']
        
        if direction == "long":
            hit = current_price >= tp_price
        else:
            hit = current_price <= tp_price
        
        return hit
    
    def close_position(self, 
                      trade_id: str, 
                      exit_price: float, 
                      exit_reason: str,
                      timestamp: datetime) -> Dict[str, Any]:
        """
        Close an open position
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: Reason for exit (tp, sl, manual, etc.)
            timestamp: Exit timestamp
            
        Returns:
            Closed position details with P&L
        """
        if trade_id not in self.open_positions:
            return {"error": f"Position {trade_id} not found"}
        
        pos = self.open_positions.pop(trade_id)
        
        entry_price = pos['entry_price']
        size = pos['size']
        direction = pos['direction']
        
        # Calculate realized P&L
        if direction == "long":
            realized_pnl = (exit_price - entry_price) * size
        else:
            realized_pnl = (entry_price - exit_price) * size
        
        realized_pct = (realized_pnl / (entry_price * size)) * 100
        
        # Close position
        closed_pos = {
            "trade_id": trade_id,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "size": size,
            "realized_pnl": realized_pnl,
            "realized_pct": realized_pct,
            "exit_reason": exit_reason,
            "entry_time": pos['entry_time'],
            "exit_time": timestamp,
            "bars_held": pos.get('bars_held', 0)
        }
        
        self.closed_positions.append(closed_pos)
        self.current_capital += realized_pnl
        self.daily_pnl += realized_pnl
        self.equity_curve.append(self.current_capital)
        
        # Update peak equity and drawdown
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.current_capital - self.peak_equity) / self.peak_equity
        
        return closed_pos
    
    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Series of returns
            
        Returns:
            RiskMetrics object
        """
        if len(returns) < 2:
            return RiskMetrics()
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0
        
        # Return metrics
        annual_return = returns.mean() * 252
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Risk-adjusted returns
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Additional metrics from closed trades
        if self.closed_positions:
            df_trades = pd.DataFrame(self.closed_positions)
            winning_trades = df_trades[df_trades['realized_pnl'] > 0]
            losing_trades = df_trades[df_trades['realized_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0
            
            total_wins = winning_trades['realized_pnl'].sum()
            total_losses = abs(losing_trades['realized_pnl'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            total_pnl = df_trades['realized_pnl'].sum()
            recovery_factor = total_pnl / abs(max_drawdown * self.initial_capital) if max_drawdown != 0 else 0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            recovery_factor = 0.0
        
        return RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            drawdown_duration=drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
            daily_volatility=daily_volatility,
            return_on_risk=annual_return / annual_volatility if annual_volatility > 0 else 0,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_pnl": self.current_capital - self.initial_capital,
            "total_return_pct": ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            "daily_pnl": self.daily_pnl,
            "peak_equity": self.peak_equity,
            "current_drawdown_pct": self.current_drawdown * 100,
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions)
        }
    
    def _estimate_win_rate(self) -> float:
        """Estimate win rate from closed trades"""
        if not self.closed_positions:
            return 0.5
        df = pd.DataFrame(self.closed_positions)
        return len(df[df['realized_pnl'] > 0]) / len(df)
    
    def _estimate_avg_win(self) -> float:
        """Estimate average win from closed trades"""
        if not self.closed_positions:
            return 1.0
        df = pd.DataFrame(self.closed_positions)
        winning = df[df['realized_pnl'] > 0]
        return winning['realized_pnl'].mean() if len(winning) > 0 else 1.0
    
    def _estimate_avg_loss(self) -> float:
        """Estimate average loss from closed trades"""
        if not self.closed_positions:
            return 1.0
        df = pd.DataFrame(self.closed_positions)
        losing = df[df['realized_pnl'] < 0]
        return losing['realized_pnl'].mean() if len(losing) > 0 else -1.0


# ============================================================================
# PORTFOLIO OPTIMIZER CLASS
# ============================================================================

class PortfolioOptimizer:
    """Portfolio optimization and allocation"""
    
    @staticmethod
    def optimize_kelly_allocation(win_rate: float, 
                                  avg_win: float, 
                                  avg_loss: float) -> float:
        """
        Calculate Kelly Criterion allocation
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            
        Returns:
            Optimal allocation percentage (0-1)
        """
        if avg_loss == 0:
            return 0.25  # Default
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(kelly, 0.25))  # Cap at 25%
    
    @staticmethod
    def optimize_risk_parity(volatilities: List[float], 
                            correlations: np.ndarray) -> np.ndarray:
        """
        Calculate risk parity weights
        
        Args:
            volatilities: List of asset volatilities
            correlations: Correlation matrix
            
        Returns:
            Optimal weights
        """
        inv_vol = 1 / np.array(volatilities)
        weights = inv_vol / inv_vol.sum()
        return weights


if __name__ == "__main__":
    print("Risk Management Module Loaded Successfully")
