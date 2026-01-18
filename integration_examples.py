"""
Integration Examples
====================
Practical examples showing how to use all enhanced modules together.

Author: AlgoTrading System
Date: 2026-01-18
"""

# ============================================================================
# EXAMPLE 1: Complete Trading System with All Components
# ============================================================================

def example_complete_system():
    """Complete trading system with configuration, risk management, and reporting"""
    
    from core.config_management import ConfigManager
    from core.risk_management import RiskManager
    from core.trading_reports import TradingReportsGenerator
    from core.trading_dashboard import TradingDashboard
    from enhanced_master_runner import EnhancedMasterRunner
    import pandas as pd
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Complete Trading System")
    print("="*80)
    
    # Step 1: Load Configuration
    config = ConfigManager('config/trading_config.yaml')
    initial_capital = config.get('risk.initial_capital', 100000.0)
    print(f"✓ Configuration loaded (Capital: ${initial_capital:,.2f})")
    
    # Step 2: Run Trading Pipeline
    runner = EnhancedMasterRunner()
    results, metrics = runner.run_complete_pipeline()
    print(f"✓ Pipeline completed")
    
    # Step 3: Initialize Risk Manager
    risk_manager = RiskManager(initial_capital, config.get_section('risk'))
    print(f"✓ Risk Manager initialized")
    
    # Step 4: Generate Reports
    reporter = TradingReportsGenerator()
    
    # Executive Summary
    exec_summary = reporter.generate_executive_summary(
        results['trades'],
        metrics,
        strategy_name="EMA Crossover with Regime Filter"
    )
    print("\n" + exec_summary)
    
    # Monthly Summary
    monthly = reporter.generate_monthly_summary(results['trades'])
    print(f"✓ Generated monthly summary with {len(monthly)} months of data")
    
    # Export to Excel
    reporter.export_to_excel(
        results['trades'],
        metrics,
        monthly,
        output_file="trading_report.xlsx"
    )
    print("✓ Exported to trading_report.xlsx")
    
    # Step 5: Create Dashboards
    dashboard = TradingDashboard()
    dashboard.create_performance_dashboard(
        results['trades'],
        metrics,
        results.get('equity_curve', pd.Series())
    )
    print("✓ Created performance dashboard")
    
    # Step 6: Export All Results
    runner.export_all_results('results')
    print("✓ All results exported to results/ directory")
    
    return results, metrics


# ============================================================================
# EXAMPLE 2: Risk Management and Position Sizing
# ============================================================================

def example_risk_management():
    """Demonstrate position sizing and risk management"""
    
    from core.config_management import ConfigManager
    from core.risk_management import RiskManager
    from datetime import datetime
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Risk Management and Position Sizing")
    print("="*80)
    
    # Load configuration
    config = ConfigManager('config/trading_config.yaml')
    
    # Initialize risk manager
    risk_manager = RiskManager(
        initial_capital=100000.0,
        config=config.get_section('risk')
    )
    print(f"✓ Risk Manager initialized with ${100000:,.2f} capital")
    
    # Example 1: Calculate position size
    entry_price = 100.0
    stop_price = 95.0
    position_size = risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_price=stop_price
    )
    print(f"\nPosition Sizing:")
    print(f"  Entry Price: ${entry_price:.2f}")
    print(f"  Stop Price: ${stop_price:.2f}")
    print(f"  Optimal Size: {position_size:.2f} shares")
    
    # Example 2: Calculate stop loss
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=entry_price,
        direction='long',
        atr=2.5
    )
    print(f"\nStop Loss:")
    print(f"  Entry Price: ${entry_price:.2f}")
    print(f"  Stop Loss (ATR-based): ${stop_loss:.2f}")
    
    # Example 3: Calculate take profit
    take_profit = risk_manager.calculate_take_profit(
        entry_price=entry_price,
        stop_price=stop_loss,
        direction='long'
    )
    print(f"\nTake Profit:")
    print(f"  Entry Price: ${entry_price:.2f}")
    print(f"  Take Profit (3:1 RR): ${take_profit:.2f}")
    
    # Example 4: Track trades
    risk_manager.open_positions['TRADE_001'] = {
        'entry_price': entry_price,
        'size': position_size,
        'stop_price': stop_loss,
        'take_profit': take_profit,
        'direction': 'long',
        'entry_time': datetime.now(),
        'bars_held': 0
    }
    
    # Update position
    update = risk_manager.update_position('TRADE_001', 102.5, datetime.now())
    print(f"\nPosition Update:")
    print(f"  Unrealized P&L: ${update['unrealized_pnl']:,.2f}")
    print(f"  Unrealized Return: {update['unrealized_pct']:.2f}%")
    
    # Get portfolio summary
    summary = risk_manager.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 3: Configuration Management
# ============================================================================

def example_configuration_management():
    """Demonstrate configuration management features"""
    
    from core.config_management import ConfigManager, EnvironmentConfig
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Configuration Management")
    print("="*80)
    
    # Create and customize configuration
    config = ConfigManager()
    print("✓ Configuration manager initialized with defaults")
    
    # Get values
    ema_fast = config.get('strategy.ema_fast')
    ema_slow = config.get('strategy.ema_slow')
    print(f"\nCurrent Strategy:")
    print(f"  EMA Fast: {ema_fast}")
    print(f"  EMA Slow: {ema_slow}")
    
    # Update values
    config.update({
        'strategy.ema_fast': 8,
        'strategy.ema_slow': 20,
        'risk.initial_capital': 50000.0,
        'risk.position_sizing_method': 'kelly'
    })
    print(f"\n✓ Configuration updated")
    print(f"  EMA Fast: {config.get('strategy.ema_fast')}")
    print(f"  EMA Slow: {config.get('strategy.ema_slow')}")
    print(f"  Capital: ${config.get('risk.initial_capital'):,.2f}")
    
    # Validate configuration
    is_valid, errors = config.validate()
    print(f"\n✓ Configuration validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"  ERROR: {error}")
    
    # Get environment
    env = EnvironmentConfig.get_environment()
    env_config = EnvironmentConfig.load_environment_config(env)
    print(f"\nEnvironment: {env}")
    print(f"  Debug: {env_config.get('debug')}")
    print(f"  Log Level: {env_config.get('log_level')}")
    
    # Save configuration
    config.save_config('config/custom_config.yaml')
    print(f"\n✓ Configuration saved to config/custom_config.yaml")


# ============================================================================
# EXAMPLE 4: Reporting and Analysis
# ============================================================================

def example_reporting_and_analysis():
    """Demonstrate comprehensive reporting capabilities"""
    
    from core.trading_reports import TradingReportsGenerator
    from enhanced_master_runner import EnhancedMasterRunner
    import pandas as pd
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Reporting and Analysis")
    print("="*80)
    
    # Get sample data
    runner = EnhancedMasterRunner()
    results, metrics = runner.run_complete_pipeline()
    trades = results['trades']
    
    # Initialize reporter
    reporter = TradingReportsGenerator(output_dir='reports')
    print("✓ Reporter initialized")
    
    # Generate executive summary
    summary = reporter.generate_executive_summary(trades, metrics)
    print("\nExecutive Summary (first 20 lines):")
    print('\n'.join(summary.split('\n')[:20]))
    
    # Generate monthly summary
    monthly = reporter.generate_monthly_summary(trades, output_file='monthly.csv')
    print(f"\n✓ Monthly summary generated ({len(monthly)} months)")
    
    # Generate strategy comparison
    strategies = {
        'Strategy A': metrics,
        'Strategy B': {**metrics, 'total_return_pct': metrics.get('total_return_pct', 0) * 0.8},
        'Strategy C': {**metrics, 'total_return_pct': metrics.get('total_return_pct', 0) * 1.2}
    }
    comparison = reporter.generate_strategy_comparison_report(
        strategies,
        output_file='strategy_comparison.csv'
    )
    print(f"\n✓ Strategy comparison generated")
    print(comparison.to_string())
    
    # Generate risk report
    risk_report = reporter.generate_risk_report(metrics, trades)
    print("\nRisk Analysis (first 25 lines):")
    print('\n'.join(risk_report.split('\n')[:25]))
    
    # Export to JSON
    reporter.generate_json_report(trades, metrics, output_file='report.json')
    print("\n✓ JSON report exported")
    
    # Export to Excel
    reporter.export_to_excel(
        trades,
        metrics,
        monthly,
        output_file='complete_report.xlsx'
    )
    print("✓ Excel report exported")


# ============================================================================
# EXAMPLE 5: Dashboard Creation
# ============================================================================

def example_dashboard_creation():
    """Demonstrate dashboard creation"""
    
    from core.trading_dashboard import TradingDashboard
    from enhanced_master_runner import EnhancedMasterRunner
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Dashboard Creation")
    print("="*80)
    
    # Get sample data
    runner = EnhancedMasterRunner()
    results, metrics = runner.run_complete_pipeline()
    trades = results['trades']
    
    # Initialize dashboard
    dashboard = TradingDashboard(output_dir='dashboards')
    print("✓ Dashboard initialized")
    
    # Create performance dashboard
    path = dashboard.create_performance_dashboard(
        trades,
        metrics,
        results.get('equity_curve', pd.Series()),
        filename='performance_dashboard.png'
    )
    print(f"✓ Performance dashboard saved to {path}")
    
    # Create risk dashboard
    returns = results.get('returns', pd.Series())
    if len(returns) > 0:
        path = dashboard.create_risk_dashboard(
            trades,
            metrics,
            returns,
            filename='risk_dashboard.png'
        )
        print(f"✓ Risk dashboard saved to {path}")
    
    # Create HTML report
    path = dashboard.create_summary_report(
        trades,
        metrics,
        filename='summary_report.html'
    )
    print(f"✓ HTML summary report saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("NIFTY ALGORITHMIC TRADING SYSTEM - INTEGRATION EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        # Example 1: Complete System
        results, metrics = example_complete_system()
        
        # Example 2: Risk Management
        example_risk_management()
        
        # Example 3: Configuration
        example_configuration_management()
        
        # Example 4: Reporting
        example_reporting_and_analysis()
        
        # Example 5: Dashboard
        example_dashboard_creation()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
