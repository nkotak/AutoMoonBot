"""
Position sizing strategies for algorithmic trading.

This module implements various position sizing methods used in quantitative
trading to determine optimal trade sizes based on risk management principles.

Position Sizing Strategies:
- Kelly Criterion: Optimal bet sizing for maximum growth
- Fixed Fractional: Risk fixed percentage of capital per trade
- Volatility-Based: Size inversely proportional to volatility
- Risk Parity: Equal risk contribution across positions
- Fixed Dollar: Trade fixed dollar amount
- Percent of Portfolio: Fixed percentage allocation
- ATR-Based: Adjusted for Average True Range
- Optimal F: Larry Williams' fixed fractional method

All strategies support:
- Position limit enforcement
- Leverage constraints
- Minimum/maximum position sizes
- Dynamic adjustment based on market conditions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    KELLY_CRITERION = "kelly_criterion"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    FIXED_DOLLAR = "fixed_dollar"
    PERCENT_OF_PORTFOLIO = "percent_of_portfolio"
    ATR_BASED = "atr_based"
    OPTIMAL_F = "optimal_f"


class BasePositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.

    All position sizers must implement calculate_position_size() method.

    Args:
        max_position_size: Maximum position size as fraction of portfolio (default: 1.0)
        min_position_size: Minimum position size as fraction of portfolio (default: 0.0)
        max_leverage: Maximum leverage allowed (default: 1.0, no leverage)
    """

    def __init__(
        self,
        max_position_size: float = 1.0,
        min_position_size: float = 0.0,
        max_leverage: float = 1.0
    ):
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_leverage = max_leverage

    @abstractmethod
    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size.

        Args:
            portfolio_value: Current portfolio value
            asset_price: Current asset price
            **kwargs: Strategy-specific parameters

        Returns:
            Position size as fraction of portfolio value (0.0 to 1.0)
        """
        pass

    def _enforce_limits(self, position_size: float) -> float:
        """
        Enforce position size limits.

        Args:
            position_size: Calculated position size

        Returns:
            Position size after applying limits
        """
        # Enforce max leverage
        position_size = min(position_size, self.max_leverage)

        # Enforce max position size
        position_size = min(position_size, self.max_position_size)

        # Enforce min position size
        if position_size < self.min_position_size:
            position_size = 0.0

        return position_size


class KellyCriterionSizer(BasePositionSizer):
    """
    Kelly Criterion position sizing.

    Optimal bet sizing for maximizing long-term growth rate.
    Formula: f = (p * b - q) / b
    where:
        f = fraction to bet
        p = probability of win
        q = probability of loss (1 - p)
        b = odds received on the bet (win/loss ratio)

    Args:
        fractional_kelly: Fraction of Kelly to use (default: 0.5 for half-Kelly)
        win_rate: Historical win rate (default: 0.5)
        avg_win: Average win size (default: 0.02)
        avg_loss: Average loss size (default: 0.01)
        lookback_window: Window for calculating statistics (default: 100)
    """

    def __init__(
        self,
        fractional_kelly: float = 0.5,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        lookback_window: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fractional_kelly = fractional_kelly
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.lookback_window = lookback_window

        # Track trade results for dynamic calculation
        self.trade_results: List[float] = []

    def update_statistics(self, trade_result: float):
        """
        Update statistics with new trade result.

        Args:
            trade_result: Trade P&L as fraction (e.g., 0.05 for 5% gain)
        """
        self.trade_results.append(trade_result)
        if len(self.trade_results) > self.lookback_window:
            self.trade_results.pop(0)

    def _calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Calculate Kelly fraction.

        Args:
            win_rate: Win probability
            avg_win: Average win size
            avg_loss: Average loss size

        Returns:
            Kelly fraction
        """
        # Use provided values or calculate from history
        if win_rate is None:
            if len(self.trade_results) < 10:
                win_rate = self.win_rate if self.win_rate is not None else 0.5
            else:
                winning_trades = [r for r in self.trade_results if r > 0]
                win_rate = len(winning_trades) / len(self.trade_results)

        if avg_win is None or avg_loss is None:
            if len(self.trade_results) < 10:
                avg_win = self.avg_win if self.avg_win is not None else 0.02
                avg_loss = abs(self.avg_loss) if self.avg_loss is not None else 0.01
            else:
                winning_trades = [r for r in self.trade_results if r > 0]
                losing_trades = [r for r in self.trade_results if r < 0]

                avg_win = np.mean(winning_trades) if winning_trades else 0.02
                avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.01

        # Calculate Kelly fraction
        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Apply fractional Kelly (typically use 0.25 to 0.5 of full Kelly)
        kelly_fraction *= self.fractional_kelly

        return max(0.0, kelly_fraction)

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """Calculate position size using Kelly Criterion."""

        win_rate = kwargs.get('win_rate')
        avg_win = kwargs.get('avg_win')
        avg_loss = kwargs.get('avg_loss')

        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        return self._enforce_limits(kelly_fraction)


class FixedFractionalSizer(BasePositionSizer):
    """
    Fixed fractional position sizing.

    Risks a fixed percentage of capital on each trade.
    This is one of the simplest and most commonly used methods.

    Args:
        risk_per_trade: Fraction of portfolio to risk per trade (default: 0.02 for 2%)
        stop_loss_pct: Stop loss as percentage (default: 0.05 for 5%)
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size based on fixed risk per trade.

        Formula: position_size = (portfolio_value * risk_per_trade) /
                                 (asset_price * stop_loss_pct)
        """

        stop_loss_pct = kwargs.get('stop_loss_pct', self.stop_loss_pct)

        if stop_loss_pct == 0:
            return 0.0

        # Dollar amount to risk
        risk_amount = portfolio_value * self.risk_per_trade

        # Position size in dollars
        position_dollars = risk_amount / stop_loss_pct

        # Position size as fraction of portfolio
        position_fraction = position_dollars / portfolio_value

        return self._enforce_limits(position_fraction)


class VolatilityBasedSizer(BasePositionSizer):
    """
    Volatility-based position sizing.

    Sizes positions inversely proportional to volatility.
    Higher volatility = smaller position size.

    Args:
        target_volatility: Target portfolio volatility (default: 0.15 for 15% annual)
        lookback_window: Window for volatility calculation (default: 20)
        annualization_factor: Factor to annualize volatility (default: 252 for daily)
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        lookback_window: int = 20,
        annualization_factor: int = 252,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window
        self.annualization_factor = annualization_factor

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size inversely proportional to volatility.

        Formula: position_size = target_volatility / asset_volatility
        """

        asset_volatility = kwargs.get('asset_volatility')
        returns = kwargs.get('returns')

        if asset_volatility is None:
            if returns is None:
                raise ValueError("Either 'asset_volatility' or 'returns' must be provided")

            # Calculate volatility from returns
            if len(returns) < 2:
                return 0.0

            asset_volatility = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)

        if asset_volatility == 0:
            return 0.0

        # Position size inversely proportional to volatility
        position_fraction = self.target_volatility / asset_volatility

        return self._enforce_limits(position_fraction)


class RiskParitySizer(BasePositionSizer):
    """
    Risk parity position sizing.

    Allocates capital such that each position contributes equally to
    portfolio risk. Positions with higher volatility get smaller allocations.

    Args:
        num_assets: Number of assets in portfolio
        target_risk_contribution: Target risk contribution per asset (default: equal weight)
    """

    def __init__(
        self,
        num_assets: int = 1,
        target_risk_contribution: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_assets = num_assets
        self.target_risk_contribution = (
            target_risk_contribution if target_risk_contribution is not None
            else 1.0 / num_assets
        )

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size for risk parity.

        Args:
            portfolio_value: Current portfolio value
            asset_price: Asset price
            asset_volatility: Asset volatility
            portfolio_volatility: Current portfolio volatility
            correlation_matrix: Correlation with other assets (optional)

        Returns:
            Position size as fraction of portfolio
        """

        asset_volatility = kwargs.get('asset_volatility')
        if asset_volatility is None:
            raise ValueError("'asset_volatility' is required for risk parity sizing")

        portfolio_volatility = kwargs.get('portfolio_volatility', asset_volatility)

        if asset_volatility == 0:
            return 0.0

        # Inverse volatility weighting
        position_fraction = (
            self.target_risk_contribution * portfolio_volatility / asset_volatility
        )

        return self._enforce_limits(position_fraction)


class FixedDollarSizer(BasePositionSizer):
    """
    Fixed dollar amount position sizing.

    Trades a fixed dollar amount regardless of portfolio size.
    Useful for consistent position sizing across different capital levels.

    Args:
        dollar_amount: Fixed dollar amount per trade (default: 10000)
    """

    def __init__(
        self,
        dollar_amount: float = 10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dollar_amount = dollar_amount

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """Calculate fixed dollar position size as fraction of portfolio."""

        if portfolio_value == 0:
            return 0.0

        position_fraction = self.dollar_amount / portfolio_value

        return self._enforce_limits(position_fraction)


class PercentOfPortfolioSizer(BasePositionSizer):
    """
    Percent of portfolio position sizing.

    Allocates a fixed percentage of portfolio to each position.
    Simple and commonly used for diversified portfolios.

    Args:
        allocation_pct: Percentage of portfolio to allocate (default: 0.1 for 10%)
    """

    def __init__(
        self,
        allocation_pct: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.allocation_pct = allocation_pct

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """Calculate position size as fixed percentage of portfolio."""

        return self._enforce_limits(self.allocation_pct)


class ATRBasedSizer(BasePositionSizer):
    """
    ATR (Average True Range) based position sizing.

    Adjusts position size based on recent price volatility measured by ATR.
    Larger ATR = smaller position size.

    Args:
        risk_per_trade: Risk per trade as fraction of portfolio (default: 0.02)
        atr_multiplier: Multiplier for ATR stop distance (default: 2.0)
        atr_period: Period for ATR calculation (default: 14)
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        atr_multiplier: float = 2.0,
        atr_period: int = 14,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size based on ATR.

        Formula: position_size = (portfolio_value * risk_per_trade) /
                                 (ATR * atr_multiplier)
        """

        atr = kwargs.get('atr')
        if atr is None:
            high_prices = kwargs.get('high_prices')
            low_prices = kwargs.get('low_prices')
            close_prices = kwargs.get('close_prices')

            if high_prices is None or low_prices is None or close_prices is None:
                raise ValueError("Either 'atr' or OHLC prices must be provided")

            atr = self._calculate_atr(high_prices, low_prices, close_prices)

        if atr == 0 or asset_price == 0:
            return 0.0

        # Risk amount in dollars
        risk_amount = portfolio_value * self.risk_per_trade

        # Stop distance in price terms
        stop_distance = atr * self.atr_multiplier

        # Position size in dollars
        position_dollars = risk_amount / stop_distance * asset_price

        # Position size as fraction
        position_fraction = position_dollars / portfolio_value

        return self._enforce_limits(position_fraction)

    def _calculate_atr(
        self,
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float]
    ) -> float:
        """
        Calculate Average True Range.

        Args:
            high_prices: High prices
            low_prices: Low prices
            close_prices: Close prices

        Returns:
            ATR value
        """
        if len(high_prices) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(high_prices)):
            high = high_prices[i]
            low = low_prices[i]
            prev_close = close_prices[i-1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Use last N periods
        recent_trs = true_ranges[-self.atr_period:]
        atr = np.mean(recent_trs) if recent_trs else 0.0

        return atr


class OptimalFSizer(BasePositionSizer):
    """
    Optimal F position sizing (Larry Williams).

    Finds the fixed fraction that maximizes geometric growth
    based on historical trade results.

    Args:
        lookback_window: Number of trades to consider (default: 100)
    """

    def __init__(
        self,
        lookback_window: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lookback_window = lookback_window
        self.trade_results: List[float] = []

    def update_statistics(self, trade_result: float):
        """
        Update with new trade result.

        Args:
            trade_result: Trade P&L as fraction
        """
        self.trade_results.append(trade_result)
        if len(self.trade_results) > self.lookback_window:
            self.trade_results.pop(0)

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate Optimal F position size.

        Searches for the fraction that maximizes terminal wealth.
        """

        trade_results = kwargs.get('trade_results', self.trade_results)

        if len(trade_results) < 10:
            # Not enough data, use conservative sizing
            return self._enforce_limits(0.1)

        # Find largest loss (for normalization)
        largest_loss = abs(min(trade_results))
        if largest_loss == 0:
            return self._enforce_limits(0.5)

        # Search for optimal f
        best_f = 0.0
        best_twf = 0.0  # Terminal Wealth Factor

        for f in np.linspace(0.01, 1.0, 100):
            twf = 1.0
            for result in trade_results:
                # HPR = 1 + (result * f / largest_loss)
                hpr = 1.0 + (result * f / largest_loss)
                twf *= hpr

                if twf <= 0:
                    twf = 0.0
                    break

            if twf > best_twf:
                best_twf = twf
                best_f = f

        # Use half of optimal f for safety
        optimal_f = best_f * 0.5

        return self._enforce_limits(optimal_f)


class DynamicPositionSizer:
    """
    Dynamic position sizer that combines multiple strategies.

    Can switch between strategies or blend them based on market conditions.

    Args:
        primary_sizer: Primary position sizing strategy
        adjustment_factor: Dynamic adjustment factor (default: 1.0)
        regime_adjustments: Dict of regime-based adjustments
    """

    def __init__(
        self,
        primary_sizer: BasePositionSizer,
        adjustment_factor: float = 1.0,
        regime_adjustments: Optional[Dict[str, float]] = None
    ):
        self.primary_sizer = primary_sizer
        self.adjustment_factor = adjustment_factor
        self.regime_adjustments = regime_adjustments or {}
        self.current_regime = 'normal'

    def set_regime(self, regime: str):
        """
        Set current market regime.

        Args:
            regime: Market regime identifier
        """
        self.current_regime = regime

    def set_adjustment_factor(self, factor: float):
        """
        Set dynamic adjustment factor.

        Args:
            factor: Adjustment multiplier (1.0 = no adjustment)
        """
        self.adjustment_factor = factor

    def calculate_position_size(
        self,
        portfolio_value: float,
        asset_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size with dynamic adjustments.

        Args:
            portfolio_value: Portfolio value
            asset_price: Asset price
            **kwargs: Strategy-specific parameters

        Returns:
            Adjusted position size
        """

        # Get base position size from primary strategy
        base_size = self.primary_sizer.calculate_position_size(
            portfolio_value,
            asset_price,
            **kwargs
        )

        # Apply adjustment factor
        adjusted_size = base_size * self.adjustment_factor

        # Apply regime-based adjustment
        if self.current_regime in self.regime_adjustments:
            regime_factor = self.regime_adjustments[self.current_regime]
            adjusted_size *= regime_factor

        # Ensure within limits
        adjusted_size = min(adjusted_size, self.primary_sizer.max_position_size)
        adjusted_size = max(adjusted_size, 0.0)

        if adjusted_size < self.primary_sizer.min_position_size:
            adjusted_size = 0.0

        return adjusted_size


def create_position_sizer(
    method: Union[str, PositionSizingMethod],
    **kwargs
) -> BasePositionSizer:
    """
    Factory function to create position sizer.

    Args:
        method: Position sizing method
        **kwargs: Method-specific parameters

    Returns:
        Position sizer instance
    """

    if isinstance(method, str):
        method = PositionSizingMethod(method)

    sizers = {
        PositionSizingMethod.KELLY_CRITERION: KellyCriterionSizer,
        PositionSizingMethod.FIXED_FRACTIONAL: FixedFractionalSizer,
        PositionSizingMethod.VOLATILITY_BASED: VolatilityBasedSizer,
        PositionSizingMethod.RISK_PARITY: RiskParitySizer,
        PositionSizingMethod.FIXED_DOLLAR: FixedDollarSizer,
        PositionSizingMethod.PERCENT_OF_PORTFOLIO: PercentOfPortfolioSizer,
        PositionSizingMethod.ATR_BASED: ATRBasedSizer,
        PositionSizingMethod.OPTIMAL_F: OptimalFSizer,
    }

    sizer_class = sizers.get(method)
    if sizer_class is None:
        raise ValueError(f"Unknown position sizing method: {method}")

    return sizer_class(**kwargs)
