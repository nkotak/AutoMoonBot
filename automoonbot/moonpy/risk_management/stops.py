"""
Stop loss and take profit management for algorithmic trading.

This module implements various stop loss and take profit strategies
to manage risk and lock in profits automatically.

Stop Loss Strategies:
- Fixed Stop Loss: Fixed percentage or dollar amount
- Trailing Stop Loss: Follows price at fixed distance
- ATR-Based Stop: Based on Average True Range
- Volatility-Based Stop: Adjusted for asset volatility
- Time-Based Stop: Exit after fixed time period
- Chandelier Stop: Trailing stop based on ATR
- Parabolic SAR: Dynamic trailing stop

Take Profit Strategies:
- Fixed Take Profit: Fixed percentage or dollar target
- Trailing Take Profit: Locks in profits as price moves favorably
- Risk/Reward Ratio: Based on stop loss distance
- Scaled Exit: Partial profit taking at multiple levels
- Volatility-Based Target: Adjusted for volatility
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


class StopType(Enum):
    """Types of stop loss strategies."""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"
    CHANDELIER = "chandelier"
    PARABOLIC_SAR = "parabolic_sar"


class TakeProfitType(Enum):
    """Types of take profit strategies."""
    FIXED = "fixed"
    TRAILING = "trailing"
    RISK_REWARD = "risk_reward"
    SCALED = "scaled"
    VOLATILITY_BASED = "volatility_based"


@dataclass
class StopLossState:
    """
    State of a stop loss order.

    Attributes:
        entry_price: Price at which position was entered
        current_price: Current market price
        stop_price: Current stop loss price
        is_triggered: Whether stop has been triggered
        highest_price: Highest price since entry (for trailing stops)
        lowest_price: Lowest price since entry (for short positions)
        steps_held: Number of time steps position has been held
    """
    entry_price: float
    current_price: float
    stop_price: float
    is_triggered: bool = False
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    steps_held: int = 0

    def __post_init__(self):
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price


@dataclass
class TakeProfitState:
    """
    State of a take profit order.

    Attributes:
        entry_price: Entry price
        current_price: Current price
        target_price: Take profit target price
        is_triggered: Whether target has been hit
        partial_exits: List of partial exit levels
        remaining_position: Remaining position size (for scaled exits)
    """
    entry_price: float
    current_price: float
    target_price: float
    is_triggered: bool = False
    partial_exits: List[Tuple[float, float]] = None  # (price, size)
    remaining_position: float = 1.0

    def __post_init__(self):
        if self.partial_exits is None:
            self.partial_exits = []


class BaseStopLoss(ABC):
    """
    Abstract base class for stop loss strategies.

    Args:
        is_long: True for long positions, False for short
    """

    def __init__(self, is_long: bool = True):
        self.is_long = is_long

    @abstractmethod
    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            state: Current stop loss state
            **kwargs: Strategy-specific parameters

        Returns:
            Stop loss price
        """
        pass

    @abstractmethod
    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """
        Update stop loss state with new price.

        Args:
            state: Current state
            current_price: New market price
            **kwargs: Strategy-specific parameters

        Returns:
            Updated state
        """
        pass

    def is_stop_triggered(
        self,
        current_price: float,
        stop_price: float
    ) -> bool:
        """
        Check if stop loss is triggered.

        Args:
            current_price: Current market price
            stop_price: Stop loss price

        Returns:
            True if stop is triggered
        """
        if self.is_long:
            return current_price <= stop_price
        else:
            return current_price >= stop_price


class FixedStopLoss(BaseStopLoss):
    """
    Fixed stop loss at specified distance from entry.

    Args:
        stop_pct: Stop loss distance as percentage (default: 0.05 for 5%)
        stop_dollars: Stop loss distance in dollars (alternative to stop_pct)
    """

    def __init__(
        self,
        stop_pct: Optional[float] = 0.05,
        stop_dollars: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stop_pct = stop_pct
        self.stop_dollars = stop_dollars

    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """Calculate fixed stop price."""

        if self.stop_dollars is not None:
            if self.is_long:
                stop_price = state.entry_price - self.stop_dollars
            else:
                stop_price = state.entry_price + self.stop_dollars
        else:
            if self.is_long:
                stop_price = state.entry_price * (1 - self.stop_pct)
            else:
                stop_price = state.entry_price * (1 + self.stop_pct)

        return stop_price

    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """Update state (fixed stop doesn't change)."""

        state.current_price = current_price
        state.steps_held += 1

        # Update highest/lowest prices
        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)

        # Check if triggered
        state.is_triggered = self.is_stop_triggered(current_price, state.stop_price)

        return state


class TrailingStopLoss(BaseStopLoss):
    """
    Trailing stop loss that follows price.

    Args:
        trail_pct: Trailing distance as percentage (default: 0.05)
        trail_dollars: Trailing distance in dollars (alternative)
        activation_pct: Profit percentage before trailing activates (default: 0.0)
    """

    def __init__(
        self,
        trail_pct: Optional[float] = 0.05,
        trail_dollars: Optional[float] = None,
        activation_pct: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.trail_pct = trail_pct
        self.trail_dollars = trail_dollars
        self.activation_pct = activation_pct

    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """Calculate trailing stop price."""

        # Check if trailing should be activated
        if self.is_long:
            profit_pct = (state.highest_price - state.entry_price) / state.entry_price
            reference_price = state.highest_price
        else:
            profit_pct = (state.entry_price - state.lowest_price) / state.entry_price
            reference_price = state.lowest_price

        # If not enough profit, use fixed stop
        if profit_pct < self.activation_pct:
            if self.is_long:
                return state.entry_price * (1 - self.trail_pct)
            else:
                return state.entry_price * (1 + self.trail_pct)

        # Calculate trailing stop
        if self.trail_dollars is not None:
            if self.is_long:
                stop_price = reference_price - self.trail_dollars
            else:
                stop_price = reference_price + self.trail_dollars
        else:
            if self.is_long:
                stop_price = reference_price * (1 - self.trail_pct)
            else:
                stop_price = reference_price * (1 + self.trail_pct)

        return stop_price

    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """Update trailing stop."""

        state.current_price = current_price
        state.steps_held += 1

        # Update highest/lowest
        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)

        # Recalculate stop price (only trails upward for longs, downward for shorts)
        new_stop_price = self.calculate_stop_price(state)

        if self.is_long:
            state.stop_price = max(state.stop_price, new_stop_price)
        else:
            state.stop_price = min(state.stop_price, new_stop_price)

        # Check if triggered
        state.is_triggered = self.is_stop_triggered(current_price, state.stop_price)

        return state


class ATRBasedStopLoss(BaseStopLoss):
    """
    ATR-based stop loss.

    Stop distance is based on Average True Range, which adapts to
    market volatility.

    Args:
        atr_multiplier: Multiplier for ATR (default: 2.0)
        atr_period: Period for ATR calculation (default: 14)
        trailing: Whether stop should trail (default: False)
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        atr_period: int = 14,
        trailing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.trailing = trailing

    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """Calculate ATR-based stop price."""

        atr = kwargs.get('atr')
        if atr is None:
            raise ValueError("ATR value must be provided")

        reference_price = state.highest_price if (self.trailing and self.is_long) else state.entry_price
        if not self.is_long:
            reference_price = state.lowest_price if self.trailing else state.entry_price

        if self.is_long:
            stop_price = reference_price - (atr * self.atr_multiplier)
        else:
            stop_price = reference_price + (atr * self.atr_multiplier)

        return stop_price

    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """Update ATR-based stop."""

        state.current_price = current_price
        state.steps_held += 1

        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)

        # Recalculate stop price
        new_stop_price = self.calculate_stop_price(state, **kwargs)

        if self.trailing:
            if self.is_long:
                state.stop_price = max(state.stop_price, new_stop_price)
            else:
                state.stop_price = min(state.stop_price, new_stop_price)
        else:
            state.stop_price = new_stop_price

        state.is_triggered = self.is_stop_triggered(current_price, state.stop_price)

        return state


class VolatilityBasedStopLoss(BaseStopLoss):
    """
    Volatility-based stop loss.

    Stop distance adapts to asset volatility.

    Args:
        volatility_multiplier: Multiplier for volatility (default: 2.0)
        lookback_period: Period for volatility calculation (default: 20)
        trailing: Whether stop should trail (default: False)
    """

    def __init__(
        self,
        volatility_multiplier: float = 2.0,
        lookback_period: int = 20,
        trailing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.volatility_multiplier = volatility_multiplier
        self.lookback_period = lookback_period
        self.trailing = trailing

    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """Calculate volatility-based stop price."""

        volatility = kwargs.get('volatility')
        if volatility is None:
            returns = kwargs.get('returns')
            if returns is None:
                raise ValueError("Either 'volatility' or 'returns' must be provided")

            volatility = np.std(returns[-self.lookback_period:], ddof=1)

        reference_price = state.highest_price if (self.trailing and self.is_long) else state.entry_price
        if not self.is_long:
            reference_price = state.lowest_price if self.trailing else state.entry_price

        stop_distance = reference_price * volatility * self.volatility_multiplier

        if self.is_long:
            stop_price = reference_price - stop_distance
        else:
            stop_price = reference_price + stop_distance

        return stop_price

    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """Update volatility-based stop."""

        state.current_price = current_price
        state.steps_held += 1

        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)

        new_stop_price = self.calculate_stop_price(state, **kwargs)

        if self.trailing:
            if self.is_long:
                state.stop_price = max(state.stop_price, new_stop_price)
            else:
                state.stop_price = min(state.stop_price, new_stop_price)
        else:
            state.stop_price = new_stop_price

        state.is_triggered = self.is_stop_triggered(current_price, state.stop_price)

        return state


class TimeBasedStopLoss(BaseStopLoss):
    """
    Time-based stop loss.

    Exits position after specified time period regardless of price.

    Args:
        max_holding_periods: Maximum number of periods to hold (default: 100)
    """

    def __init__(
        self,
        max_holding_periods: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_holding_periods = max_holding_periods

    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """Return infinity (time-based, not price-based)."""
        return float('inf') if self.is_long else float('-inf')

    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """Update time-based stop."""

        state.current_price = current_price
        state.steps_held += 1

        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)

        # Trigger based on time
        state.is_triggered = state.steps_held >= self.max_holding_periods

        return state


class ChandelierStopLoss(BaseStopLoss):
    """
    Chandelier stop loss.

    Similar to ATR-based but always trails from the highest high (for longs)
    or lowest low (for shorts) over the lookback period.

    Args:
        atr_multiplier: Multiplier for ATR (default: 3.0)
        atr_period: Period for ATR (default: 22)
        lookback_period: Period for highest/lowest (default: 22)
    """

    def __init__(
        self,
        atr_multiplier: float = 3.0,
        atr_period: int = 22,
        lookback_period: int = 22,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.lookback_period = lookback_period

    def calculate_stop_price(
        self,
        state: StopLossState,
        **kwargs
    ) -> float:
        """Calculate Chandelier stop price."""

        atr = kwargs.get('atr')
        if atr is None:
            raise ValueError("ATR value must be provided")

        if self.is_long:
            stop_price = state.highest_price - (atr * self.atr_multiplier)
        else:
            stop_price = state.lowest_price + (atr * self.atr_multiplier)

        return stop_price

    def update(
        self,
        state: StopLossState,
        current_price: float,
        **kwargs
    ) -> StopLossState:
        """Update Chandelier stop."""

        state.current_price = current_price
        state.steps_held += 1

        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)

        state.stop_price = self.calculate_stop_price(state, **kwargs)
        state.is_triggered = self.is_stop_triggered(current_price, state.stop_price)

        return state


# Take Profit Classes

class BaseTakeProfit(ABC):
    """
    Abstract base class for take profit strategies.

    Args:
        is_long: True for long positions, False for short
    """

    def __init__(self, is_long: bool = True):
        self.is_long = is_long

    @abstractmethod
    def calculate_target_price(
        self,
        state: TakeProfitState,
        **kwargs
    ) -> float:
        """
        Calculate take profit target price.

        Args:
            state: Current take profit state
            **kwargs: Strategy-specific parameters

        Returns:
            Target price
        """
        pass

    @abstractmethod
    def update(
        self,
        state: TakeProfitState,
        current_price: float,
        **kwargs
    ) -> TakeProfitState:
        """
        Update take profit state.

        Args:
            state: Current state
            current_price: Current price
            **kwargs: Strategy-specific parameters

        Returns:
            Updated state
        """
        pass

    def is_target_hit(
        self,
        current_price: float,
        target_price: float
    ) -> bool:
        """Check if take profit target is hit."""

        if self.is_long:
            return current_price >= target_price
        else:
            return current_price <= target_price


class FixedTakeProfit(BaseTakeProfit):
    """
    Fixed take profit at specified distance from entry.

    Args:
        target_pct: Target profit percentage (default: 0.10 for 10%)
        target_dollars: Target profit in dollars (alternative)
    """

    def __init__(
        self,
        target_pct: Optional[float] = 0.10,
        target_dollars: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_pct = target_pct
        self.target_dollars = target_dollars

    def calculate_target_price(
        self,
        state: TakeProfitState,
        **kwargs
    ) -> float:
        """Calculate fixed target price."""

        if self.target_dollars is not None:
            if self.is_long:
                target_price = state.entry_price + self.target_dollars
            else:
                target_price = state.entry_price - self.target_dollars
        else:
            if self.is_long:
                target_price = state.entry_price * (1 + self.target_pct)
            else:
                target_price = state.entry_price * (1 - self.target_pct)

        return target_price

    def update(
        self,
        state: TakeProfitState,
        current_price: float,
        **kwargs
    ) -> TakeProfitState:
        """Update state (fixed target doesn't change)."""

        state.current_price = current_price
        state.is_triggered = self.is_target_hit(current_price, state.target_price)

        return state


class RiskRewardTakeProfit(BaseTakeProfit):
    """
    Take profit based on risk/reward ratio.

    Sets target at multiple of stop loss distance.

    Args:
        risk_reward_ratio: Reward to risk ratio (default: 2.0 for 2:1)
    """

    def __init__(
        self,
        risk_reward_ratio: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_reward_ratio = risk_reward_ratio

    def calculate_target_price(
        self,
        state: TakeProfitState,
        **kwargs
    ) -> float:
        """Calculate target based on risk/reward ratio."""

        stop_price = kwargs.get('stop_price')
        if stop_price is None:
            raise ValueError("stop_price must be provided for risk/reward calculation")

        stop_distance = abs(state.entry_price - stop_price)
        target_distance = stop_distance * self.risk_reward_ratio

        if self.is_long:
            target_price = state.entry_price + target_distance
        else:
            target_price = state.entry_price - target_distance

        return target_price

    def update(
        self,
        state: TakeProfitState,
        current_price: float,
        **kwargs
    ) -> TakeProfitState:
        """Update state."""

        state.current_price = current_price
        state.is_triggered = self.is_target_hit(current_price, state.target_price)

        return state


class ScaledTakeProfit(BaseTakeProfit):
    """
    Scaled take profit with multiple partial exit levels.

    Args:
        targets: List of (price_pct, exit_fraction) tuples
                 Example: [(0.05, 0.33), (0.10, 0.33), (0.15, 0.34)]
    """

    def __init__(
        self,
        targets: List[Tuple[float, float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.targets = targets or [(0.05, 0.50), (0.10, 0.50)]

    def calculate_target_price(
        self,
        state: TakeProfitState,
        **kwargs
    ) -> float:
        """Calculate next target price."""

        # Find next unexited target
        for target_pct, exit_fraction in self.targets:
            if self.is_long:
                target_price = state.entry_price * (1 + target_pct)
            else:
                target_price = state.entry_price * (1 - target_pct)

            # Check if this target hasn't been hit yet
            already_hit = any(price >= target_price if self.is_long else price <= target_price
                            for price, _ in state.partial_exits)

            if not already_hit:
                return target_price

        # All targets hit
        return float('inf') if self.is_long else float('-inf')

    def update(
        self,
        state: TakeProfitState,
        current_price: float,
        **kwargs
    ) -> TakeProfitState:
        """Update scaled exit state."""

        state.current_price = current_price

        # Check each target
        for target_pct, exit_fraction in self.targets:
            if self.is_long:
                target_price = state.entry_price * (1 + target_pct)
            else:
                target_price = state.entry_price * (1 - target_pct)

            # Check if target hit and not already exited
            already_hit = any(price >= target_price if self.is_long else price <= target_price
                            for price, _ in state.partial_exits)

            if not already_hit and self.is_target_hit(current_price, target_price):
                # Add partial exit
                exit_size = state.remaining_position * exit_fraction
                state.partial_exits.append((target_price, exit_size))
                state.remaining_position -= exit_size

        # Fully exited if no remaining position
        state.is_triggered = state.remaining_position <= 0.001

        return state


def create_stop_loss(
    stop_type: Union[str, StopType],
    **kwargs
) -> BaseStopLoss:
    """
    Factory function to create stop loss.

    Args:
        stop_type: Type of stop loss
        **kwargs: Strategy-specific parameters

    Returns:
        Stop loss instance
    """

    if isinstance(stop_type, str):
        stop_type = StopType(stop_type)

    stops = {
        StopType.FIXED: FixedStopLoss,
        StopType.TRAILING: TrailingStopLoss,
        StopType.ATR_BASED: ATRBasedStopLoss,
        StopType.VOLATILITY_BASED: VolatilityBasedStopLoss,
        StopType.TIME_BASED: TimeBasedStopLoss,
        StopType.CHANDELIER: ChandelierStopLoss,
    }

    stop_class = stops.get(stop_type)
    if stop_class is None:
        raise ValueError(f"Unknown stop type: {stop_type}")

    return stop_class(**kwargs)


def create_take_profit(
    tp_type: Union[str, TakeProfitType],
    **kwargs
) -> BaseTakeProfit:
    """
    Factory function to create take profit.

    Args:
        tp_type: Type of take profit
        **kwargs: Strategy-specific parameters

    Returns:
        Take profit instance
    """

    if isinstance(tp_type, str):
        tp_type = TakeProfitType(tp_type)

    take_profits = {
        TakeProfitType.FIXED: FixedTakeProfit,
        TakeProfitType.RISK_REWARD: RiskRewardTakeProfit,
        TakeProfitType.SCALED: ScaledTakeProfit,
    }

    tp_class = take_profits.get(tp_type)
    if tp_class is None:
        raise ValueError(f"Unknown take profit type: {tp_type}")

    return tp_class(**kwargs)
