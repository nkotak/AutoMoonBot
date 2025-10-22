"""
Reward functions and risk metrics for algorithmic trading.

This module provides comprehensive tools for reward function design
and risk measurement in reinforcement learning trading agents.
"""

# Risk metrics
from automoonbot.moonpy.reward.risk import (
    RiskMetrics,
    VolatilityEstimator,
    VaRMethod,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
)

# Reward functions
from automoonbot.moonpy.reward.reward import (
    BaseRewardFunction,
    RewardType,
    NormalizationMethod,
    SimpleReturnReward,
    RiskAdjustedReward,
    MultiComponentReward,
    AdaptiveReward,
    PotentialBasedReward,
    CuriosityReward,
    HierarchicalReward,
    SparseReward,
    RewardNormalizer,
    combine_rewards,
    create_default_reward,
)

# Market regime detection
from automoonbot.moonpy.reward.regime import (
    VolatilityRegime,
    TrendRegime,
    CorrelationRegime,
    VolumeRegime,
    MarketRegime,
    VolatilityRegimeDetector,
    TrendRegimeDetector,
    CorrelationRegimeDetector,
    VolumeRegimeDetector,
    MarketRegimeDetector,
    create_default_regime_detector,
    regime_to_risk_adjustment,
)

__all__ = [
    # Risk metrics
    "RiskMetrics",
    "VolatilityEstimator",
    "VaRMethod",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    # Reward functions
    "BaseRewardFunction",
    "RewardType",
    "NormalizationMethod",
    "SimpleReturnReward",
    "RiskAdjustedReward",
    "MultiComponentReward",
    "AdaptiveReward",
    "PotentialBasedReward",
    "CuriosityReward",
    "HierarchicalReward",
    "SparseReward",
    "RewardNormalizer",
    "combine_rewards",
    "create_default_reward",
    # Regime detection
    "VolatilityRegime",
    "TrendRegime",
    "CorrelationRegime",
    "VolumeRegime",
    "MarketRegime",
    "VolatilityRegimeDetector",
    "TrendRegimeDetector",
    "CorrelationRegimeDetector",
    "VolumeRegimeDetector",
    "MarketRegimeDetector",
    "create_default_regime_detector",
    "regime_to_risk_adjustment",
]
