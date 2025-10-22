"""
Comprehensive unit tests for risk management module.

Tests cover:
- Position sizing strategies
- Stop loss management
- Take profit management
- Portfolio risk limits
- Risk monitoring
"""

import pytest
import numpy as np
from typing import Dict, List

from automoonbot.moonpy.risk_management.position_sizing import (
    KellyCriterionSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer,
    RiskParitySizer,
    ATRBasedSizer,
    OptimalFSizer,
    DynamicPositionSizer,
    create_position_sizer,
    PositionSizingMethod,
)

from automoonbot.moonpy.risk_management.stops import (
    StopLossState,
    TakeProfitState,
    FixedStopLoss,
    TrailingStopLoss,
    ATRBasedStopLoss,
    VolatilityBasedStopLoss,
    TimeBasedStopLoss,
    ChandelierStopLoss,
    FixedTakeProfit,
    RiskRewardTakeProfit,
    ScaledTakeProfit,
    create_stop_loss,
    create_take_profit,
    StopType,
    TakeProfitType,
)

from automoonbot.moonpy.risk_management.limits import (
    MaxDrawdownLimit,
    VaRLimit,
    ExposureLimit,
    ConcentrationLimit,
    LeverageLimit,
    SectorExposureLimit,
    RiskLimitManager,
    LimitSeverity,
    create_default_limits,
)


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestKellyCriterionSizer:
    """Test Kelly Criterion position sizing."""

    def test_initialization(self):
        """Test Kelly sizer initialization."""
        sizer = KellyCriterionSizer(fractional_kelly=0.5)
        assert sizer.fractional_kelly == 0.5
        assert sizer.lookback_window == 100

    def test_calculate_with_provided_statistics(self):
        """Test Kelly calculation with provided win rate and avg win/loss."""
        sizer = KellyCriterionSizer(fractional_kelly=1.0)

        size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100,
            win_rate=0.6,
            avg_win=0.04,
            avg_loss=0.02
        )

        assert isinstance(size, float)
        assert size >= 0.0
        assert size <= 1.0

    def test_update_statistics(self):
        """Test updating trade statistics."""
        sizer = KellyCriterionSizer()

        sizer.update_statistics(0.05)
        sizer.update_statistics(-0.02)
        sizer.update_statistics(0.03)

        assert len(sizer.trade_results) == 3


class TestFixedFractionalSizer:
    """Test Fixed Fractional position sizing."""

    def test_initialization(self):
        """Test initialization."""
        sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.05)
        assert sizer.risk_per_trade == 0.02
        assert sizer.stop_loss_pct == 0.05

    def test_calculate_position_size(self):
        """Test position size calculation."""
        sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.05)

        size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100
        )

        # Risk 2% on 5% stop = 40% position
        expected = 0.4
        assert size == pytest.approx(expected, abs=0.01)

    def test_zero_stop_loss(self):
        """Test with zero stop loss."""
        sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.0)

        size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100
        )

        assert size == 0.0


class TestVolatilityBasedSizer:
    """Test Volatility-Based position sizing."""

    def test_initialization(self):
        """Test initialization."""
        sizer = VolatilityBasedSizer(target_volatility=0.15)
        assert sizer.target_volatility == 0.15

    def test_calculate_with_volatility(self):
        """Test calculation with provided volatility."""
        sizer = VolatilityBasedSizer(target_volatility=0.15)

        size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100,
            asset_volatility=0.30
        )

        # target_vol / asset_vol = 0.15 / 0.30 = 0.5
        assert size == pytest.approx(0.5, abs=0.01)

    def test_calculate_with_returns(self):
        """Test calculation from returns."""
        sizer = VolatilityBasedSizer(target_volatility=0.15, lookback_window=20)

        returns = np.random.normal(0, 0.02, 50)

        size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100,
            returns=returns.tolist()
        )

        assert isinstance(size, float)
        assert size >= 0.0


class TestATRBasedSizer:
    """Test ATR-Based position sizing."""

    def test_initialization(self):
        """Test initialization."""
        sizer = ATRBasedSizer(risk_per_trade=0.02, atr_multiplier=2.0)
        assert sizer.risk_per_trade == 0.02
        assert sizer.atr_multiplier == 2.0

    def test_calculate_with_atr(self):
        """Test calculation with provided ATR."""
        sizer = ATRBasedSizer(risk_per_trade=0.02, atr_multiplier=2.0)

        size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100,
            atr=5.0
        )

        assert isinstance(size, float)
        assert size > 0.0

    def test_atr_calculation(self):
        """Test ATR calculation from OHLC data."""
        sizer = ATRBasedSizer(atr_period=14)

        highs = [105, 107, 106, 108, 110] + [109] * 10
        lows = [95, 97, 96, 98, 100] + [99] * 10
        closes = [100, 102, 101, 103, 105] + [104] * 10

        atr = sizer._calculate_atr(highs, lows, closes)

        assert isinstance(atr, float)
        assert atr > 0.0


class TestDynamicPositionSizer:
    """Test Dynamic position sizer."""

    def test_initialization(self):
        """Test initialization."""
        base_sizer = FixedFractionalSizer(risk_per_trade=0.02)
        dynamic = DynamicPositionSizer(
            primary_sizer=base_sizer,
            adjustment_factor=1.0
        )

        assert dynamic.adjustment_factor == 1.0

    def test_regime_adjustment(self):
        """Test regime-based adjustment."""
        base_sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.05)

        regime_adjustments = {
            'high_vol': 0.5,
            'low_vol': 1.5
        }

        dynamic = DynamicPositionSizer(
            primary_sizer=base_sizer,
            regime_adjustments=regime_adjustments
        )

        # Normal regime
        size_normal = dynamic.calculate_position_size(10000, 100)

        # High volatility regime
        dynamic.set_regime('high_vol')
        size_high_vol = dynamic.calculate_position_size(10000, 100)

        # Should be smaller in high vol
        assert size_high_vol < size_normal


class TestPositionSizerFactory:
    """Test position sizer factory function."""

    def test_create_kelly(self):
        """Test creating Kelly sizer."""
        sizer = create_position_sizer('kelly_criterion', fractional_kelly=0.5)
        assert isinstance(sizer, KellyCriterionSizer)

    def test_create_fixed_fractional(self):
        """Test creating fixed fractional sizer."""
        sizer = create_position_sizer(PositionSizingMethod.FIXED_FRACTIONAL, risk_per_trade=0.02)
        assert isinstance(sizer, FixedFractionalSizer)


# ============================================================================
# Stop Loss Tests
# ============================================================================

class TestFixedStopLoss:
    """Test Fixed Stop Loss."""

    def test_initialization(self):
        """Test initialization."""
        stop = FixedStopLoss(stop_pct=0.05, is_long=True)
        assert stop.stop_pct == 0.05
        assert stop.is_long is True

    def test_calculate_stop_price_long(self):
        """Test stop price for long position."""
        stop = FixedStopLoss(stop_pct=0.05, is_long=True)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=0.0
        )

        stop_price = stop.calculate_stop_price(state)

        # 5% below entry
        assert stop_price == pytest.approx(95.0)

    def test_calculate_stop_price_short(self):
        """Test stop price for short position."""
        stop = FixedStopLoss(stop_pct=0.05, is_long=False)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=0.0
        )

        stop_price = stop.calculate_stop_price(state)

        # 5% above entry
        assert stop_price == pytest.approx(105.0)

    def test_stop_triggered(self):
        """Test stop trigger detection."""
        stop = FixedStopLoss(stop_pct=0.05, is_long=True)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=95.0
        )

        # Price drops to stop
        state = stop.update(state, 94.0)
        assert state.is_triggered is True

        # Price above stop
        state.is_triggered = False
        state = stop.update(state, 96.0)
        assert state.is_triggered is False


class TestTrailingStopLoss:
    """Test Trailing Stop Loss."""

    def test_initialization(self):
        """Test initialization."""
        stop = TrailingStopLoss(trail_pct=0.05, activation_pct=0.02)
        assert stop.trail_pct == 0.05
        assert stop.activation_pct == 0.02

    def test_trailing_upward(self):
        """Test stop trails upward for long position."""
        stop = TrailingStopLoss(trail_pct=0.05, activation_pct=0.0, is_long=True)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=95.0
        )

        # Price rises
        state = stop.update(state, 110.0)

        # Stop should trail up to 110 - 5% = 104.5
        assert state.stop_price >= 104.0

    def test_activation_threshold(self):
        """Test trailing activation threshold."""
        stop = TrailingStopLoss(trail_pct=0.05, activation_pct=0.05, is_long=True)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=95.0
        )

        # Small price increase (below activation)
        state = stop.update(state, 102.0)
        # Should still be at initial stop
        assert state.stop_price == pytest.approx(95.0, abs=0.1)


class TestATRBasedStopLoss:
    """Test ATR-Based Stop Loss."""

    def test_initialization(self):
        """Test initialization."""
        stop = ATRBasedStopLoss(atr_multiplier=2.0, trailing=False)
        assert stop.atr_multiplier == 2.0
        assert stop.trailing is False

    def test_calculate_stop_price(self):
        """Test stop price calculation."""
        stop = ATRBasedStopLoss(atr_multiplier=2.0, is_long=True)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=0.0
        )

        stop_price = stop.calculate_stop_price(state, atr=5.0)

        # 100 - (5 * 2) = 90
        assert stop_price == pytest.approx(90.0)


class TestTimeBasedStopLoss:
    """Test Time-Based Stop Loss."""

    def test_initialization(self):
        """Test initialization."""
        stop = TimeBasedStopLoss(max_holding_periods=100)
        assert stop.max_holding_periods == 100

    def test_time_trigger(self):
        """Test time-based triggering."""
        stop = TimeBasedStopLoss(max_holding_periods=5)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=0.0
        )

        # Update multiple times
        for i in range(6):
            state = stop.update(state, 100.0)

        # Should trigger after 5 periods
        assert state.is_triggered is True


# ============================================================================
# Take Profit Tests
# ============================================================================

class TestFixedTakeProfit:
    """Test Fixed Take Profit."""

    def test_initialization(self):
        """Test initialization."""
        tp = FixedTakeProfit(target_pct=0.10, is_long=True)
        assert tp.target_pct == 0.10

    def test_calculate_target_price(self):
        """Test target price calculation."""
        tp = FixedTakeProfit(target_pct=0.10, is_long=True)

        state = TakeProfitState(
            entry_price=100.0,
            current_price=100.0,
            target_price=0.0
        )

        target = tp.calculate_target_price(state)

        # 10% above entry
        assert target == pytest.approx(110.0)

    def test_target_hit(self):
        """Test target detection."""
        tp = FixedTakeProfit(target_pct=0.10, is_long=True)

        state = TakeProfitState(
            entry_price=100.0,
            current_price=100.0,
            target_price=110.0
        )

        # Price reaches target
        state = tp.update(state, 111.0)
        assert state.is_triggered is True


class TestRiskRewardTakeProfit:
    """Test Risk/Reward Take Profit."""

    def test_initialization(self):
        """Test initialization."""
        tp = RiskRewardTakeProfit(risk_reward_ratio=2.0)
        assert tp.risk_reward_ratio == 2.0

    def test_calculate_target_from_stop(self):
        """Test target calculation based on stop distance."""
        tp = RiskRewardTakeProfit(risk_reward_ratio=2.0, is_long=True)

        state = TakeProfitState(
            entry_price=100.0,
            current_price=100.0,
            target_price=0.0
        )

        # Stop at 95 (5 points risk)
        target = tp.calculate_target_price(state, stop_price=95.0)

        # 2:1 reward = 10 points reward -> target at 110
        assert target == pytest.approx(110.0)


class TestScaledTakeProfit:
    """Test Scaled Take Profit."""

    def test_initialization(self):
        """Test initialization."""
        targets = [(0.05, 0.5), (0.10, 0.5)]
        tp = ScaledTakeProfit(targets=targets)
        assert len(tp.targets) == 2

    def test_partial_exits(self):
        """Test partial exit logic."""
        targets = [(0.05, 0.5), (0.10, 0.5)]
        tp = ScaledTakeProfit(targets=targets, is_long=True)

        state = TakeProfitState(
            entry_price=100.0,
            current_price=100.0,
            target_price=105.0,
            remaining_position=1.0
        )

        # First target hit
        state = tp.update(state, 105.5)
        assert len(state.partial_exits) == 1
        assert state.remaining_position == pytest.approx(0.5)

        # Second target hit
        state = tp.update(state, 110.5)
        assert len(state.partial_exits) == 2
        assert state.remaining_position == pytest.approx(0.0, abs=0.01)
        assert state.is_triggered is True


# ============================================================================
# Risk Limits Tests
# ============================================================================

class TestMaxDrawdownLimit:
    """Test Maximum Drawdown Limit."""

    def test_initialization(self):
        """Test initialization."""
        limit = MaxDrawdownLimit(hard_limit=0.20, soft_limit=0.15)
        assert limit.hard_limit == 0.20
        assert limit.soft_limit == 0.15

    def test_no_breach(self):
        """Test no breach scenario."""
        limit = MaxDrawdownLimit(hard_limit=0.20, soft_limit=0.15)

        is_breached, breach = limit.check_limit(current_drawdown=0.10)

        assert is_breached is False
        assert breach is None

    def test_soft_limit_breach(self):
        """Test soft limit breach."""
        limit = MaxDrawdownLimit(hard_limit=0.20, soft_limit=0.15)

        is_breached, breach = limit.check_limit(current_drawdown=0.18)

        assert is_breached is False
        assert breach is not None
        assert breach.severity == LimitSeverity.WARNING

    def test_hard_limit_breach(self):
        """Test hard limit breach."""
        limit = MaxDrawdownLimit(hard_limit=0.20, soft_limit=0.15)

        is_breached, breach = limit.check_limit(current_drawdown=0.25)

        assert is_breached is True
        assert breach is not None
        assert breach.severity == LimitSeverity.CRITICAL


class TestConcentrationLimit:
    """Test Concentration Limit."""

    def test_initialization(self):
        """Test initialization."""
        limit = ConcentrationLimit(hard_limit=0.25, soft_limit=0.20)
        assert limit.hard_limit == 0.25

    def test_check_limit(self):
        """Test concentration check."""
        limit = ConcentrationLimit(hard_limit=0.25, soft_limit=0.20)

        position_values = {
            'AAPL': 3000,
            'GOOGL': 2000,
            'MSFT': 1500
        }

        is_breached, breach = limit.check_limit(
            position_values=position_values,
            portfolio_value=10000
        )

        # 30% concentration on AAPL exceeds both limits
        assert is_breached is True


class TestRiskLimitManager:
    """Test Risk Limit Manager."""

    def test_initialization(self):
        """Test initialization."""
        manager = RiskLimitManager()
        assert len(manager.limits) == 0
        assert manager.is_halted is False

    def test_add_limit(self):
        """Test adding limits."""
        manager = RiskLimitManager()

        limit1 = MaxDrawdownLimit(hard_limit=0.20)
        limit2 = ConcentrationLimit(hard_limit=0.25)

        manager.add_limit(limit1)
        manager.add_limit(limit2)

        assert len(manager.limits) == 2

    def test_check_all_limits(self):
        """Test checking all limits."""
        manager = RiskLimitManager()

        manager.add_limit(MaxDrawdownLimit(hard_limit=0.20))
        manager.add_limit(ConcentrationLimit(hard_limit=0.25))

        is_halted, breaches = manager.check_all_limits(
            current_drawdown=0.25,
            position_values={'AAPL': 3000},
            portfolio_value=10000
        )

        # Both limits breached
        assert is_halted is True
        assert len(breaches) >= 1

    def test_halt_on_breach(self):
        """Test halting on breach."""
        manager = RiskLimitManager(halt_on_breach=True)
        manager.add_limit(MaxDrawdownLimit(hard_limit=0.20))

        # Breach hard limit
        is_halted, _ = manager.check_all_limits(current_drawdown=0.25)

        assert manager.is_halted is True


class TestDefaultLimits:
    """Test default limits factory."""

    def test_create_default_limits(self):
        """Test creating default limits."""
        manager = create_default_limits(
            portfolio_value=100000,
            max_drawdown=0.20,
            max_concentration=0.25,
            max_leverage=2.0
        )

        assert isinstance(manager, RiskLimitManager)
        assert len(manager.limits) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete risk management workflows."""

    def test_position_sizing_with_stops(self):
        """Test position sizing integrated with stop loss."""

        # Create position sizer
        sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.05)

        # Calculate position size
        position_size = sizer.calculate_position_size(
            portfolio_value=10000,
            asset_price=100
        )

        # Create stop loss
        stop = FixedStopLoss(stop_pct=0.05, is_long=True)

        state = StopLossState(
            entry_price=100.0,
            current_price=100.0,
            stop_price=stop.calculate_stop_price(
                StopLossState(100.0, 100.0, 0.0)
            )
        )

        # Position size and stop are consistent
        assert position_size > 0.0
        assert state.stop_price < state.entry_price

    def test_complete_trade_workflow(self):
        """Test complete trade with position sizing, stops, and limits."""

        # Setup
        portfolio_value = 100000
        asset_price = 100

        # Position sizing
        sizer = FixedFractionalSizer(risk_per_trade=0.02, stop_loss_pct=0.05)
        position_size = sizer.calculate_position_size(portfolio_value, asset_price)

        # Stop loss
        stop = FixedStopLoss(stop_pct=0.05, is_long=True)
        stop_state = StopLossState(
            entry_price=asset_price,
            current_price=asset_price,
            stop_price=stop.calculate_stop_price(
                StopLossState(asset_price, asset_price, 0.0)
            )
        )

        # Take profit
        tp = FixedTakeProfit(target_pct=0.10, is_long=True)
        tp_state = TakeProfitState(
            entry_price=asset_price,
            current_price=asset_price,
            target_price=tp.calculate_target_price(
                TakeProfitState(asset_price, asset_price, 0.0)
            )
        )

        # Risk limits
        manager = create_default_limits(portfolio_value=portfolio_value)

        # Simulate trade
        # Price drops to stop
        stop_state = stop.update(stop_state, 94.0)
        assert stop_state.is_triggered is True

        # Check limits
        is_halted, _ = manager.check_all_limits(
            current_drawdown=0.05,
            position_values={'TEST': position_size * portfolio_value},
            portfolio_value=portfolio_value
        )

        assert is_halted is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
