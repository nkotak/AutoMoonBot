"""
Comprehensive usage examples for risk management module.

This file demonstrates:
1. Position sizing strategies
2. Stop loss management
3. Take profit management
4. Portfolio risk limits
5. Complete trading workflows
6. Dynamic risk adjustment
"""

import numpy as np
from typing import Dict, List

from automoonbot.moonpy.risk_management import (
    # Position Sizing
    KellyCriterionSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer,
    ATRBasedSizer,
    DynamicPositionSizer,
    create_position_sizer,
    # Stop Loss & Take Profit
    StopLossState,
    TakeProfitState,
    FixedStopLoss,
    TrailingStopLoss,
    ATRBasedStopLoss,
    FixedTakeProfit,
    RiskRewardTakeProfit,
    ScaledTakeProfit,
    # Risk Limits
    MaxDrawdownLimit,
    ConcentrationLimit,
    RiskLimitManager,
    create_default_limits,
)


def example_1_fixed_fractional_sizing():
    """
    Example 1: Basic Fixed Fractional Position Sizing.

    Risk a fixed percentage of capital per trade.
    """
    print("\n" + "="*80)
    print("Example 1: Fixed Fractional Position Sizing")
    print("="*80)

    # Create sizer: risk 2% per trade with 5% stop loss
    sizer = FixedFractionalSizer(
        risk_per_trade=0.02,
        stop_loss_pct=0.05,
        max_position_size=0.50  # Max 50% of portfolio
    )

    portfolio_value = 100000
    asset_price = 50

    position_size = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        asset_price=asset_price
    )

    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Asset Price: ${asset_price}")
    print(f"Risk Per Trade: 2%")
    print(f"Stop Loss: 5%")
    print(f"Position Size: {position_size:.2%} of portfolio")
    print(f"Dollar Amount: ${position_size * portfolio_value:,.0f}")
    print(f"Number of Shares: {int((position_size * portfolio_value) / asset_price)}")


def example_2_kelly_criterion():
    """
    Example 2: Kelly Criterion Optimal Sizing.

    Optimal bet sizing for maximum long-term growth.
    """
    print("\n" + "="*80)
    print("Example 2: Kelly Criterion Position Sizing")
    print("="*80)

    # Create Kelly sizer with historical statistics
    sizer = KellyCriterionSizer(
        fractional_kelly=0.5,  # Use half-Kelly for safety
        win_rate=0.55,  # 55% win rate
        avg_win=0.06,  # Average 6% win
        avg_loss=0.03  # Average 3% loss
    )

    portfolio_value = 100000
    asset_price = 100

    position_size = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        asset_price=asset_price
    )

    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Win Rate: 55%")
    print(f"Average Win: 6%")
    print(f"Average Loss: 3%")
    print(f"Fractional Kelly: 0.5 (half-Kelly)")
    print(f"Position Size: {position_size:.2%} of portfolio")


def example_3_volatility_based_sizing():
    """
    Example 3: Volatility-Based Position Sizing.

    Adjust position size based on asset volatility.
    """
    print("\n" + "="*80)
    print("Example 3: Volatility-Based Position Sizing")
    print("="*80)

    # Create volatility-based sizer
    sizer = VolatilityBasedSizer(
        target_volatility=0.15,  # Target 15% annual portfolio volatility
        annualization_factor=252
    )

    portfolio_value = 100000

    # Different assets with different volatilities
    assets = [
        ("Low Vol Stock", 100, 0.10),
        ("Medium Vol Stock", 50, 0.20),
        ("High Vol Stock", 25, 0.40)
    ]

    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Target Portfolio Volatility: 15%\n")

    for name, price, volatility in assets:
        position_size = sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            asset_price=price,
            asset_volatility=volatility
        )

        print(f"{name}:")
        print(f"  Price: ${price}")
        print(f"  Volatility: {volatility:.0%}")
        print(f"  Position Size: {position_size:.2%}")
        print(f"  Dollar Amount: ${position_size * portfolio_value:,.0f}\n")


def example_4_atr_based_sizing():
    """
    Example 4: ATR-Based Position Sizing.

    Size positions based on Average True Range.
    """
    print("\n" + "="*80)
    print("Example 4: ATR-Based Position Sizing")
    print("="*80)

    # Create ATR-based sizer
    sizer = ATRBasedSizer(
        risk_per_trade=0.02,
        atr_multiplier=2.0,
        atr_period=14
    )

    portfolio_value = 100000
    asset_price = 100

    # Simulate OHLC data
    np.random.seed(42)
    base_prices = np.linspace(95, 105, 20)
    highs = base_prices + np.random.uniform(1, 3, 20)
    lows = base_prices - np.random.uniform(1, 3, 20)
    closes = base_prices + np.random.uniform(-1, 1, 20)

    position_size = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        asset_price=asset_price,
        high_prices=highs.tolist(),
        low_prices=lows.tolist(),
        close_prices=closes.tolist()
    )

    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Asset Price: ${asset_price}")
    print(f"Risk Per Trade: 2%")
    print(f"ATR Multiplier: 2.0x")
    print(f"Position Size: {position_size:.2%} of portfolio")


def example_5_fixed_stop_loss():
    """
    Example 5: Fixed Stop Loss.

    Set a fixed stop loss at percentage from entry.
    """
    print("\n" + "="*80)
    print("Example 5: Fixed Stop Loss")
    print("="*80)

    # Create fixed stop loss (5% for long position)
    stop = FixedStopLoss(
        stop_pct=0.05,
        is_long=True
    )

    entry_price = 100.0
    state = StopLossState(
        entry_price=entry_price,
        current_price=entry_price,
        stop_price=stop.calculate_stop_price(
            StopLossState(entry_price, entry_price, 0.0)
        )
    )

    print(f"\nEntry Price: ${entry_price}")
    print(f"Stop Loss Price: ${state.stop_price}")
    print(f"Stop Distance: {((entry_price - state.stop_price) / entry_price):.1%}\n")

    # Simulate price movements
    prices = [100, 98, 96, 94, 92]

    print("Price movements:")
    for price in prices:
        state = stop.update(state, price)
        status = "TRIGGERED!" if state.is_triggered else "Active"
        print(f"  Price: ${price:6.2f} | Stop: ${state.stop_price:6.2f} | Status: {status}")

        if state.is_triggered:
            break


def example_6_trailing_stop_loss():
    """
    Example 6: Trailing Stop Loss.

    Stop loss that trails price as it moves favorably.
    """
    print("\n" + "="*80)
    print("Example 6: Trailing Stop Loss")
    print("="*80)

    # Create trailing stop
    stop = TrailingStopLoss(
        trail_pct=0.05,
        activation_pct=0.02,  # Start trailing after 2% profit
        is_long=True
    )

    entry_price = 100.0
    state = StopLossState(
        entry_price=entry_price,
        current_price=entry_price,
        stop_price=entry_price * 0.95
    )

    print(f"\nEntry Price: ${entry_price}")
    print(f"Trail Distance: 5%")
    print(f"Activation: 2% profit\n")

    # Simulate price movements: up then down
    prices = [100, 102, 105, 108, 110, 108, 106, 104, 103]

    print("Price movements:")
    for price in prices:
        state = stop.update(state, price)
        status = "TRIGGERED!" if state.is_triggered else "Trailing"
        print(f"  Price: ${price:6.2f} | Stop: ${state.stop_price:6.2f} | Status: {status}")

        if state.is_triggered:
            print(f"\n  Stopped out at ${price} with profit of ${price - entry_price:.2f} ({((price - entry_price) / entry_price):.1%})")
            break


def example_7_atr_based_stop():
    """
    Example 7: ATR-Based Stop Loss.

    Stop loss based on Average True Range.
    """
    print("\n" + "="*80)
    print("Example 7: ATR-Based Stop Loss")
    print("="*80)

    # Create ATR-based stop
    stop = ATRBasedStopLoss(
        atr_multiplier=2.0,
        trailing=True,
        is_long=True
    )

    entry_price = 100.0
    atr = 3.0  # ATR of $3

    state = StopLossState(
        entry_price=entry_price,
        current_price=entry_price,
        stop_price=stop.calculate_stop_price(
            StopLossState(entry_price, entry_price, 0.0),
            atr=atr
        )
    )

    print(f"\nEntry Price: ${entry_price}")
    print(f"ATR: ${atr}")
    print(f"ATR Multiplier: 2.0x")
    print(f"Initial Stop: ${state.stop_price} (${entry_price - state.stop_price:.2f} below entry)\n")

    # Simulate price movements
    prices = [(102, 3.0), (105, 3.5), (108, 4.0), (106, 3.8)]

    print("Price movements:")
    for price, current_atr in prices:
        state = stop.update(state, price, atr=current_atr)
        print(f"  Price: ${price:6.2f} | ATR: ${current_atr:4.2f} | Stop: ${state.stop_price:6.2f}")


def example_8_fixed_take_profit():
    """
    Example 8: Fixed Take Profit.

    Set a fixed profit target.
    """
    print("\n" + "="*80)
    print("Example 8: Fixed Take Profit")
    print("="*80)

    # Create fixed take profit (10% target)
    tp = FixedTakeProfit(
        target_pct=0.10,
        is_long=True
    )

    entry_price = 100.0
    state = TakeProfitState(
        entry_price=entry_price,
        current_price=entry_price,
        target_price=tp.calculate_target_price(
            TakeProfitState(entry_price, entry_price, 0.0)
        )
    )

    print(f"\nEntry Price: ${entry_price}")
    print(f"Target Price: ${state.target_price}")
    print(f"Target Profit: {((state.target_price - entry_price) / entry_price):.1%}\n")

    # Simulate price movements
    prices = [100, 102, 105, 108, 110, 112]

    print("Price movements:")
    for price in prices:
        state = tp.update(state, price)
        status = "HIT!" if state.is_triggered else "Pending"
        print(f"  Price: ${price:6.2f} | Target: ${state.target_price:6.2f} | Status: {status}")

        if state.is_triggered:
            print(f"\n  Target hit! Profit: ${price - entry_price:.2f} ({((price - entry_price) / entry_price):.1%})")
            break


def example_9_risk_reward_take_profit():
    """
    Example 9: Risk/Reward Take Profit.

    Set target based on risk/reward ratio.
    """
    print("\n" + "="*80)
    print("Example 9: Risk/Reward Take Profit")
    print("="*80)

    # Create risk/reward take profit (2:1 ratio)
    tp = RiskRewardTakeProfit(
        risk_reward_ratio=2.0,
        is_long=True
    )

    entry_price = 100.0
    stop_price = 95.0  # 5% stop

    state = TakeProfitState(
        entry_price=entry_price,
        current_price=entry_price,
        target_price=tp.calculate_target_price(
            TakeProfitState(entry_price, entry_price, 0.0),
            stop_price=stop_price
        )
    )

    print(f"\nEntry Price: ${entry_price}")
    print(f"Stop Price: ${stop_price}")
    print(f"Risk: ${entry_price - stop_price} ({((entry_price - stop_price) / entry_price):.1%})")
    print(f"Risk/Reward Ratio: 2:1")
    print(f"Target Price: ${state.target_price}")
    print(f"Potential Reward: ${state.target_price - entry_price} ({((state.target_price - entry_price) / entry_price):.1%})")


def example_10_scaled_take_profit():
    """
    Example 10: Scaled Take Profit.

    Take profits in multiple steps.
    """
    print("\n" + "="*80)
    print("Example 10: Scaled Take Profit")
    print("="*80)

    # Create scaled take profit
    targets = [
        (0.05, 0.33),  # Take 33% at 5% profit
        (0.10, 0.33),  # Take 33% at 10% profit
        (0.15, 0.34),  # Take remaining 34% at 15% profit
    ]

    tp = ScaledTakeProfit(
        targets=targets,
        is_long=True
    )

    entry_price = 100.0
    state = TakeProfitState(
        entry_price=entry_price,
        current_price=entry_price,
        target_price=105.0,
        remaining_position=1.0
    )

    print(f"\nEntry Price: ${entry_price}")
    print(f"Targets:")
    for target_pct, exit_fraction in targets:
        target_price = entry_price * (1 + target_pct)
        print(f"  {exit_fraction:.0%} at ${target_price:.2f} ({target_pct:.0%} profit)")

    print("\nPrice movements:")
    prices = [100, 103, 105.5, 108, 110.5, 115.5]

    for price in prices:
        state = tp.update(state, price)
        print(f"  Price: ${price:6.2f} | Remaining: {state.remaining_position:.0%} | Exits: {len(state.partial_exits)}")

        if state.is_triggered:
            print(f"\n  All targets hit! Position fully exited.")
            break


def example_11_risk_limits():
    """
    Example 11: Portfolio Risk Limits.

    Enforce risk limits on portfolio.
    """
    print("\n" + "="*80)
    print("Example 11: Portfolio Risk Limits")
    print("="*80)

    # Create risk limits
    drawdown_limit = MaxDrawdownLimit(
        hard_limit=0.20,  # Halt at 20% drawdown
        soft_limit=0.15   # Warn at 15% drawdown
    )

    concentration_limit = ConcentrationLimit(
        hard_limit=0.30,  # Max 30% in single position
        soft_limit=0.25   # Warn at 25%
    )

    print("\nRisk Limits:")
    print(f"  Max Drawdown: {drawdown_limit.soft_limit:.0%} (warn), {drawdown_limit.hard_limit:.0%} (halt)")
    print(f"  Max Concentration: {concentration_limit.soft_limit:.0%} (warn), {concentration_limit.hard_limit:.0%} (halt)")

    # Scenario 1: Normal conditions
    print("\nScenario 1: Normal Conditions")
    is_breached, breach = drawdown_limit.check_limit(current_drawdown=0.10)
    print(f"  Drawdown: 10% -> {breach.severity.value if breach else 'OK'}")

    # Scenario 2: Warning level
    print("\nScenario 2: Warning Level")
    is_breached, breach = drawdown_limit.check_limit(current_drawdown=0.18)
    print(f"  Drawdown: 18% -> {breach.severity.value if breach else 'OK'}")
    if breach:
        print(f"  {breach.message}")

    # Scenario 3: Critical level
    print("\nScenario 3: Critical Level")
    is_breached, breach = drawdown_limit.check_limit(current_drawdown=0.25)
    print(f"  Drawdown: 25% -> {breach.severity.value if breach else 'OK'}")
    if breach:
        print(f"  {breach.message}")
        print(f"  Trading would be HALTED")


def example_12_risk_limit_manager():
    """
    Example 12: Risk Limit Manager.

    Manage multiple risk limits together.
    """
    print("\n" + "="*80)
    print("Example 12: Risk Limit Manager")
    print("="*80)

    # Create limit manager with default limits
    manager = create_default_limits(
        portfolio_value=100000,
        max_drawdown=0.20,
        max_concentration=0.25,
        max_leverage=2.0
    )

    print("\nActive Limits:")
    status = manager.get_limit_status()
    for limit_name, info in status.items():
        print(f"  {limit_name}:")
        print(f"    Soft Limit: {info['soft_limit']:.2f}")
        print(f"    Hard Limit: {info['hard_limit']:.2f}")

    # Check all limits
    print("\nChecking Portfolio:")
    portfolio_data = {
        'current_drawdown': 0.12,
        'position_values': {
            'AAPL': 28000,
            'GOOGL': 22000,
            'MSFT': 15000
        },
        'portfolio_value': 100000,
        'total_exposure': 65000
    }

    is_halted, breaches = manager.check_all_limits(**portfolio_data)

    if breaches:
        print(f"\nLimit Breaches ({len(breaches)}):")
        for breach in breaches:
            print(f"  - {breach.message}")
    else:
        print("\n  All limits OK")

    print(f"\nTrading Status: {'HALTED' if is_halted else 'ACTIVE'}")


def example_13_complete_trade_workflow():
    """
    Example 13: Complete Trade Workflow.

    Integrate position sizing, stops, and limits.
    """
    print("\n" + "="*80)
    print("Example 13: Complete Trade Workflow")
    print("="*80)

    portfolio_value = 100000
    asset_price = 100

    # Step 1: Position Sizing
    print("\nStep 1: Calculate Position Size")
    sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.05)
    position_size = sizer.calculate_position_size(portfolio_value, asset_price)
    position_dollars = position_size * portfolio_value
    print(f"  Position Size: {position_size:.1%} (${position_dollars:,.0f})")

    # Step 2: Set Stop Loss
    print("\nStep 2: Set Stop Loss")
    stop = FixedStopLoss(stop_pct=0.05, is_long=True)
    stop_state = StopLossState(
        entry_price=asset_price,
        current_price=asset_price,
        stop_price=stop.calculate_stop_price(
            StopLossState(asset_price, asset_price, 0.0)
        )
    )
    print(f"  Entry: ${asset_price}")
    print(f"  Stop: ${stop_state.stop_price}")

    # Step 3: Set Take Profit
    print("\nStep 3: Set Take Profit")
    tp = RiskRewardTakeProfit(risk_reward_ratio=2.0, is_long=True)
    tp_state = TakeProfitState(
        entry_price=asset_price,
        current_price=asset_price,
        target_price=tp.calculate_target_price(
            TakeProfitState(asset_price, asset_price, 0.0),
            stop_price=stop_state.stop_price
        )
    )
    print(f"  Target: ${tp_state.target_price}")

    # Step 4: Check Risk Limits
    print("\nStep 4: Check Risk Limits")
    manager = create_default_limits(portfolio_value=portfolio_value)
    is_halted, breaches = manager.check_all_limits(
        current_drawdown=0.05,
        position_values={'STOCK': position_dollars},
        portfolio_value=portfolio_value,
        total_exposure=position_dollars
    )
    print(f"  Limits: {'PASSED' if not breaches else 'WARNINGS'}")

    # Step 5: Execute Trade
    if not is_halted:
        print("\nStep 5: Execute Trade")
        print(f"  BUY {int(position_dollars / asset_price)} shares at ${asset_price}")
        print(f"  Stop Loss: ${stop_state.stop_price}")
        print(f"  Take Profit: ${tp_state.target_price}")
        print(f"  Risk: ${asset_price - stop_state.stop_price:.2f} per share")
        print(f"  Reward: ${tp_state.target_price - asset_price:.2f} per share")
        print(f"  R:R Ratio: {(tp_state.target_price - asset_price)/(asset_price - stop_state.stop_price):.1f}:1")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("RISK MANAGEMENT - USAGE EXAMPLES")
    print("="*80)

    examples = [
        example_1_fixed_fractional_sizing,
        example_2_kelly_criterion,
        example_3_volatility_based_sizing,
        example_4_atr_based_sizing,
        example_5_fixed_stop_loss,
        example_6_trailing_stop_loss,
        example_7_atr_based_stop,
        example_8_fixed_take_profit,
        example_9_risk_reward_take_profit,
        example_10_scaled_take_profit,
        example_11_risk_limits,
        example_12_risk_limit_manager,
        example_13_complete_trade_workflow,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    run_all_examples()
