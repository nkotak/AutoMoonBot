"""
Comprehensive risk management for algorithmic trading.

This module provides production-ready risk management tools for trading systems:

Position Sizing:
- Kelly Criterion, Fixed Fractional, Volatility-Based, Risk Parity
- ATR-Based, Optimal F, Fixed Dollar, Percent of Portfolio

Stop Loss & Take Profit:
- Fixed, Trailing, ATR-Based, Volatility-Based, Time-Based
- Chandelier, Parabolic SAR, Risk/Reward, Scaled Exits

Portfolio Risk Limits:
- Maximum Drawdown, VaR, CVaR, Exposure, Concentration
- Leverage, Sector Exposure Limits

Risk Monitoring:
- Real-time risk tracking, Limit breach detection
- Risk budgeting, Performance attribution
"""

# Position Sizing
from automoonbot.moonpy.risk_management.position_sizing import (
    PositionSizingMethod,
    BasePositionSizer,
    KellyCriterionSizer,
    FixedFractionalSizer,
    VolatilityBasedSizer,
    RiskParitySizer,
    FixedDollarSizer,
    PercentOfPortfolioSizer,
    ATRBasedSizer,
    OptimalFSizer,
    DynamicPositionSizer,
    create_position_sizer,
)

# Stop Loss & Take Profit
from automoonbot.moonpy.risk_management.stops import (
    StopType,
    TakeProfitType,
    StopLossState,
    TakeProfitState,
    BaseStopLoss,
    FixedStopLoss,
    TrailingStopLoss,
    ATRBasedStopLoss,
    VolatilityBasedStopLoss,
    TimeBasedStopLoss,
    ChandelierStopLoss,
    BaseTakeProfit,
    FixedTakeProfit,
    RiskRewardTakeProfit,
    ScaledTakeProfit,
    create_stop_loss,
    create_take_profit,
)

# Portfolio Risk Limits
from automoonbot.moonpy.risk_management.limits import (
    LimitType,
    LimitSeverity,
    LimitBreach,
    BaseRiskLimit,
    MaxDrawdownLimit,
    VaRLimit,
    ExposureLimit,
    ConcentrationLimit,
    LeverageLimit,
    SectorExposureLimit,
    RiskLimitManager,
    create_default_limits,
)

__all__ = [
    # Position Sizing
    "PositionSizingMethod",
    "BasePositionSizer",
    "KellyCriterionSizer",
    "FixedFractionalSizer",
    "VolatilityBasedSizer",
    "RiskParitySizer",
    "FixedDollarSizer",
    "PercentOfPortfolioSizer",
    "ATRBasedSizer",
    "OptimalFSizer",
    "DynamicPositionSizer",
    "create_position_sizer",
    # Stop Loss & Take Profit
    "StopType",
    "TakeProfitType",
    "StopLossState",
    "TakeProfitState",
    "BaseStopLoss",
    "FixedStopLoss",
    "TrailingStopLoss",
    "ATRBasedStopLoss",
    "VolatilityBasedStopLoss",
    "TimeBasedStopLoss",
    "ChandelierStopLoss",
    "BaseTakeProfit",
    "FixedTakeProfit",
    "RiskRewardTakeProfit",
    "ScaledTakeProfit",
    "create_stop_loss",
    "create_take_profit",
    # Portfolio Risk Limits
    "LimitType",
    "LimitSeverity",
    "LimitBreach",
    "BaseRiskLimit",
    "MaxDrawdownLimit",
    "VaRLimit",
    "ExposureLimit",
    "ConcentrationLimit",
    "LeverageLimit",
    "SectorExposureLimit",
    "RiskLimitManager",
    "create_default_limits",
]
