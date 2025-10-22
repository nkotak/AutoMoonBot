"""
Advanced reward functions for reinforcement learning in trading.

This module provides a comprehensive framework for reward function design
in algorithmic trading environments. It includes:

Reward Function Types:
- SimpleReturnReward: Direct portfolio returns
- RiskAdjustedReward: Sharpe/Sortino/Calmar ratio-based rewards
- MultiComponentReward: Composable weighted sum of multiple objectives
- AdaptiveReward: Market regime-aware dynamic reward weighting
- PotentialBasedReward: Shaped rewards using potential functions
- CuriosityReward: Intrinsic motivation for exploration
- HierarchicalReward: Multi-level objectives (short/long term)
- SparseReward: Delayed rewards for episode milestones
- DenseReward: Step-by-step immediate feedback

Design Principles:
- All rewards are composable and can be combined
- Support for online (incremental) computation
- Configurable reward shaping and normalization
- Compatible with major RL algorithms (PPO, SAC, TD3, DQN)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from enum import Enum
import math

from automoonbot.moonpy.reward.risk import RiskMetrics


class RewardType(Enum):
    """Types of reward signals."""
    DENSE = "dense"  # Immediate reward at each step
    SPARSE = "sparse"  # Delayed reward at episode end or milestones
    SHAPED = "shaped"  # Potential-based reward shaping
    INTRINSIC = "intrinsic"  # Curiosity-driven exploration bonus
    EXTRINSIC = "extrinsic"  # Environment-based task reward


class NormalizationMethod(Enum):
    """Methods for reward normalization."""
    NONE = "none"  # No normalization
    STANDARDIZE = "standardize"  # Zero mean, unit variance
    CLIP = "clip"  # Clip to range
    SCALE = "scale"  # Scale to range
    RUNNING_MEAN = "running_mean"  # Running mean normalization


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions.

    All reward functions inherit from this and implement compute_reward().
    This ensures consistent interface for composability.

    Args:
        reward_type: Type of reward signal (dense, sparse, etc.)
        normalization: Normalization method
        clip_range: Range to clip rewards (min, max)
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.DENSE,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.reward_type = reward_type
        self.normalization = normalization
        self.clip_range = clip_range

        # For running statistics
        self.reward_history: List[float] = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    @abstractmethod
    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for state transition.

        Args:
            state: Current state information
            action: Action taken
            next_state: Next state information
            info: Additional information (portfolio metrics, etc.)

        Returns:
            reward: Scalar reward value
            components: Dictionary of reward components for analysis
        """
        pass

    def normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using configured method.

        Args:
            reward: Raw reward value

        Returns:
            Normalized reward
        """
        if self.normalization == NormalizationMethod.NONE:
            normalized = reward

        elif self.normalization == NormalizationMethod.STANDARDIZE:
            # Update running statistics
            self.count += 1
            delta = reward - self.running_mean
            self.running_mean += delta / self.count
            delta2 = reward - self.running_mean
            self.running_var += (delta * delta2 - self.running_var) / self.count

            std = math.sqrt(self.running_var) if self.running_var > 0 else 1.0
            normalized = (reward - self.running_mean) / (std + 1e-8)

        elif self.normalization == NormalizationMethod.CLIP:
            if self.clip_range is not None:
                normalized = np.clip(reward, self.clip_range[0], self.clip_range[1])
            else:
                normalized = reward

        elif self.normalization == NormalizationMethod.SCALE:
            if self.clip_range is not None:
                # Scale to [clip_range[0], clip_range[1]]
                # Assumes input roughly in [-1, 1]
                normalized = (reward + 1.0) / 2.0  # First scale to [0, 1]
                normalized = (normalized * (self.clip_range[1] - self.clip_range[0]) +
                            self.clip_range[0])
            else:
                normalized = reward

        elif self.normalization == NormalizationMethod.RUNNING_MEAN:
            self.count += 1
            delta = reward - self.running_mean
            self.running_mean += delta / self.count
            normalized = reward - self.running_mean

        else:
            normalized = reward

        return normalized

    def reset(self):
        """Reset internal state (for episodic rewards)."""
        self.reward_history.clear()
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0


class SimpleReturnReward(BaseRewardFunction):
    """
    Simple reward based on portfolio returns.

    Reward = portfolio_return * scale_factor

    This is the most basic reward: directly optimize returns without
    considering risk. Suitable for benchmarking or when combined with
    other reward components.

    Args:
        scale_factor: Multiplier for returns (default: 1.0)
        log_returns: If True, use log returns instead of simple returns
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        log_returns: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.log_returns = log_returns

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute simple return reward."""

        portfolio_return = info.get('portfolio_return', 0.0)

        if self.log_returns and portfolio_return > -1.0:
            portfolio_return = math.log(1.0 + portfolio_return)

        reward = self.scale_factor * portfolio_return
        reward = self.normalize_reward(reward)

        components = {
            'return': reward
        }

        return reward, components


class RiskAdjustedReward(BaseRewardFunction):
    """
    Risk-adjusted reward using Sharpe, Sortino, or Calmar ratio.

    Instead of raw returns, optimizes risk-adjusted returns which leads to
    more stable trading strategies. The reward at each step is based on the
    rolling risk metric.

    Args:
        metric: Risk metric to use ('sharpe', 'sortino', 'calmar', 'omega')
        window_size: Rolling window for metric calculation
        risk_free_rate: Annual risk-free rate
        scale_factor: Multiplier for the metric
        include_return: If True, add raw return as additional component
        return_weight: Weight for raw return component (if included)
    """

    def __init__(
        self,
        metric: str = 'sharpe',
        window_size: int = 30,
        risk_free_rate: float = 0.0,
        scale_factor: float = 1.0,
        include_return: bool = True,
        return_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.metric = metric
        self.scale_factor = scale_factor
        self.include_return = include_return
        self.return_weight = return_weight

        # Initialize risk metrics calculator
        self.risk_metrics = RiskMetrics(
            window_size=window_size,
            risk_free_rate=risk_free_rate,
            min_periods=10
        )

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute risk-adjusted reward."""

        portfolio_return = info.get('portfolio_return', 0.0)

        # Update risk metrics
        self.risk_metrics.update(portfolio_return)

        # Compute risk-adjusted metric
        if self.metric == 'sharpe':
            risk_metric = self.risk_metrics.sharpe_ratio(annualized=False)
        elif self.metric == 'sortino':
            risk_metric = self.risk_metrics.sortino_ratio(annualized=False)
        elif self.metric == 'calmar':
            risk_metric = self.risk_metrics.calmar_ratio(annualized=False)
        elif self.metric == 'omega':
            risk_metric = self.risk_metrics.omega_ratio()
        else:
            risk_metric = 0.0

        # Scale the metric
        reward = self.scale_factor * risk_metric

        # Optionally include raw return
        components = {
            'risk_adjusted': reward
        }

        if self.include_return:
            return_component = self.return_weight * portfolio_return
            reward += return_component
            components['return'] = return_component

        reward = self.normalize_reward(reward)

        return reward, components

    def reset(self):
        """Reset risk metrics for new episode."""
        super().reset()
        self.risk_metrics.reset()


class MultiComponentReward(BaseRewardFunction):
    """
    Composable multi-component reward function.

    Combines multiple objectives with configurable weights:
    - Returns (portfolio growth)
    - Risk penalties (volatility, drawdown)
    - Transaction costs
    - Diversification
    - Position holding (encourage stability)
    - Custom components

    This is the most flexible reward function and is similar to the
    RewardCalculator in the environment but with more features.

    Args:
        weights: Dictionary of component weights
        penalties: Dictionary of penalty configurations
        custom_components: List of custom reward functions
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        penalties: Optional[Dict[str, Dict]] = None,
        custom_components: Optional[List[BaseRewardFunction]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Default weights
        self.weights = weights or {
            'return': 1.0,
            'risk': 0.5,
            'transaction_cost': 0.1,
            'diversification': 0.05,
            'sharpe': 0.0,
            'drawdown': 0.5,
        }

        # Penalty configurations
        self.penalties = penalties or {
            'drawdown_threshold': 0.1,  # Start penalizing at 10% drawdown
            'concentration_power': 2.0,  # Quadratic penalty for concentration
        }

        self.custom_components = custom_components or []

        # Risk metrics for Sharpe ratio
        self.risk_metrics = RiskMetrics(window_size=30, min_periods=10)

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute multi-component reward."""

        components = {}

        # 1. Return component
        portfolio_return = info.get('portfolio_return', 0.0)
        components['return'] = self.weights.get('return', 0.0) * portfolio_return

        # 2. Risk component (volatility + drawdown)
        if 'portfolio_volatility' in info:
            volatility = info['portfolio_volatility']
            volatility_penalty = -self.weights.get('risk', 0.0) * volatility
            components['volatility'] = volatility_penalty

        # 3. Drawdown penalty
        if 'max_drawdown' in info:
            drawdown = info['max_drawdown']
            threshold = self.penalties.get('drawdown_threshold', 0.1)
            if drawdown > threshold:
                drawdown_penalty = -self.weights.get('drawdown', 0.0) * (drawdown - threshold)
                components['drawdown'] = drawdown_penalty
            else:
                components['drawdown'] = 0.0

        # 4. Transaction cost penalty
        if 'transaction_costs' in info:
            transaction_costs = info['transaction_costs']
            cost_penalty = -self.weights.get('transaction_cost', 0.0) * transaction_costs
            components['transaction_cost'] = cost_penalty

        # 5. Diversification (penalize concentration)
        if 'position_concentration' in info:
            concentration = info['position_concentration']
            power = self.penalties.get('concentration_power', 2.0)
            concentration_penalty = -self.weights.get('diversification', 0.0) * (concentration ** power)
            components['diversification'] = concentration_penalty

        # 6. Sharpe ratio component
        if self.weights.get('sharpe', 0.0) > 0:
            self.risk_metrics.update(portfolio_return)
            sharpe = self.risk_metrics.sharpe_ratio(annualized=False)
            components['sharpe'] = self.weights.get('sharpe', 0.0) * sharpe

        # 7. Custom components
        for i, custom_func in enumerate(self.custom_components):
            custom_reward, custom_comps = custom_func.compute_reward(
                state, action, next_state, info
            )
            components[f'custom_{i}'] = custom_reward

        # Total reward
        total_reward = sum(components.values())
        total_reward = self.normalize_reward(total_reward)

        return total_reward, components

    def reset(self):
        """Reset all components."""
        super().reset()
        self.risk_metrics.reset()
        for custom_func in self.custom_components:
            custom_func.reset()


class AdaptiveReward(BaseRewardFunction):
    """
    Adaptive reward function that adjusts based on market regime.

    Different market conditions require different trading strategies:
    - Bull markets: Favor momentum and returns
    - Bear markets: Favor risk management and capital preservation
    - High volatility: Reduce position sizes, favor Sharpe ratio
    - Low volatility: Can take more risk, favor absolute returns

    This reward function detects market regime and dynamically adjusts
    component weights.

    Args:
        base_weights: Default component weights
        regime_weights: Weight adjustments per regime
        regime_detector: Function to detect current regime
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        regime_detector: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.base_weights = base_weights or {
            'return': 1.0,
            'risk': 0.5,
            'transaction_cost': 0.1,
        }

        # Weight multipliers per regime
        self.regime_weights = regime_weights or {
            'bull': {'return': 1.5, 'risk': 0.5, 'transaction_cost': 1.0},
            'bear': {'return': 0.5, 'risk': 2.0, 'transaction_cost': 1.5},
            'high_vol': {'return': 0.8, 'risk': 1.5, 'transaction_cost': 1.2},
            'low_vol': {'return': 1.2, 'risk': 0.8, 'transaction_cost': 0.9},
        }

        self.regime_detector = regime_detector
        self.current_regime = 'neutral'

        # Multi-component reward for actual computation
        self.multi_component = MultiComponentReward(weights=self.base_weights)

    def detect_regime(self, info: Dict) -> str:
        """
        Detect market regime from information.

        Args:
            info: Environment info dict

        Returns:
            Regime string
        """
        if self.regime_detector is not None:
            return self.regime_detector(info)

        # Simple heuristic-based detection
        volatility = info.get('portfolio_volatility', 0.0)
        returns = info.get('portfolio_return', 0.0)

        # High/low volatility
        if volatility > 0.03:  # 3% daily vol is high
            regime = 'high_vol'
        elif volatility < 0.01:  # 1% daily vol is low
            regime = 'low_vol'
        # Bull/bear based on recent returns
        elif returns > 0.02:  # Strong positive return
            regime = 'bull'
        elif returns < -0.02:  # Strong negative return
            regime = 'bear'
        else:
            regime = 'neutral'

        return regime

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute adaptive reward based on market regime."""

        # Detect current regime
        self.current_regime = self.detect_regime(info)

        # Adjust weights based on regime
        if self.current_regime in self.regime_weights:
            regime_multipliers = self.regime_weights[self.current_regime]
            adjusted_weights = {
                k: self.base_weights[k] * regime_multipliers.get(k, 1.0)
                for k in self.base_weights
            }
        else:
            adjusted_weights = self.base_weights

        # Update multi-component weights
        self.multi_component.weights = adjusted_weights

        # Compute reward using adjusted weights
        reward, components = self.multi_component.compute_reward(
            state, action, next_state, info
        )

        # Add regime info to components
        components['regime'] = float(hash(self.current_regime) % 100) / 100.0  # For logging

        reward = self.normalize_reward(reward)

        return reward, components

    def reset(self):
        """Reset adaptive reward."""
        super().reset()
        self.multi_component.reset()
        self.current_regime = 'neutral'


class PotentialBasedReward(BaseRewardFunction):
    """
    Potential-based reward shaping.

    Reward shaping using potential functions maintains optimal policy while
    providing denser reward signal:

    R'(s, a, s') = R(s, a, s') + gamma * Phi(s') - Phi(s)

    where Phi is a potential function. Common potential functions:
    - Portfolio value
    - Distance to target allocation
    - Cumulative Sharpe ratio

    Args:
        base_reward: Base reward function
        potential_fn: Function that computes potential from state
        gamma: Discount factor
        potential_weight: Weight for potential difference
    """

    def __init__(
        self,
        base_reward: BaseRewardFunction,
        potential_fn: Optional[Callable] = None,
        gamma: float = 0.99,
        potential_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_reward = base_reward
        self.potential_fn = potential_fn or self._default_potential
        self.gamma = gamma
        self.potential_weight = potential_weight

    def _default_potential(self, state: Dict, info: Dict) -> float:
        """
        Default potential function based on portfolio value.

        Args:
            state: State dict
            info: Info dict

        Returns:
            Potential value
        """
        portfolio_value = info.get('portfolio_value', 1.0)
        # Use log to make potential bounded
        return math.log(max(portfolio_value, 1e-8))

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute shaped reward."""

        # Compute base reward
        base_reward, components = self.base_reward.compute_reward(
            state, action, next_state, info
        )

        # Compute potential difference
        phi_s = self.potential_fn(state, info)
        phi_next = self.potential_fn(next_state, info)

        potential_diff = self.gamma * phi_next - phi_s
        shaping_bonus = self.potential_weight * potential_diff

        # Shaped reward
        shaped_reward = base_reward + shaping_bonus

        components['base_reward'] = base_reward
        components['shaping'] = shaping_bonus

        shaped_reward = self.normalize_reward(shaped_reward)

        return shaped_reward, components

    def reset(self):
        """Reset shaping reward."""
        super().reset()
        self.base_reward.reset()


class CuriosityReward(BaseRewardFunction):
    """
    Intrinsic curiosity-driven reward for exploration.

    Adds exploration bonus based on state novelty or prediction error.
    Useful for encouraging the agent to explore diverse market conditions
    and trading strategies.

    Implements prediction-error based curiosity:
    - Forward model: predict next state given current state and action
    - Reward = prediction_error (novel states are rewarding)

    Args:
        base_reward: Base extrinsic reward function
        curiosity_weight: Weight for curiosity bonus
        prediction_error_scale: Scale factor for prediction error
    """

    def __init__(
        self,
        base_reward: BaseRewardFunction,
        curiosity_weight: float = 0.1,
        prediction_error_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_reward = base_reward
        self.curiosity_weight = curiosity_weight
        self.prediction_error_scale = prediction_error_scale

        # Simple state history for novelty detection
        self.state_history: List[float] = []

    def _compute_novelty(self, state: Dict, next_state: Dict) -> float:
        """
        Compute state novelty as distance from historical states.

        Args:
            state: Current state
            next_state: Next state

        Returns:
            Novelty score
        """
        # Simple novelty based on portfolio value change
        current_value = next_state.get('portfolio_value', 1.0)

        if not self.state_history:
            novelty = 1.0
        else:
            # Distance from recent states
            distances = [abs(current_value - v) for v in self.state_history[-10:]]
            novelty = np.mean(distances) if distances else 0.0

        # Add to history
        self.state_history.append(current_value)
        if len(self.state_history) > 100:
            self.state_history.pop(0)

        return novelty

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute curiosity-augmented reward."""

        # Compute base (extrinsic) reward
        extrinsic_reward, components = self.base_reward.compute_reward(
            state, action, next_state, info
        )

        # Compute intrinsic (curiosity) reward
        novelty = self._compute_novelty(state, next_state)
        intrinsic_reward = self.curiosity_weight * novelty * self.prediction_error_scale

        # Total reward
        total_reward = extrinsic_reward + intrinsic_reward

        components['extrinsic'] = extrinsic_reward
        components['intrinsic'] = intrinsic_reward

        total_reward = self.normalize_reward(total_reward)

        return total_reward, components

    def reset(self):
        """Reset curiosity reward."""
        super().reset()
        self.base_reward.reset()
        self.state_history.clear()


class HierarchicalReward(BaseRewardFunction):
    """
    Hierarchical reward function with multiple time scales.

    Combines short-term and long-term objectives:
    - Short-term: Step-by-step returns, risk management
    - Medium-term: Episode returns, Sharpe ratio
    - Long-term: Cumulative performance, drawdown recovery

    Args:
        short_term_reward: Reward for immediate objectives
        long_term_reward: Reward for long-term objectives
        short_term_weight: Weight for short-term component
        long_term_weight: Weight for long-term component
        long_term_horizon: Steps to accumulate for long-term reward
    """

    def __init__(
        self,
        short_term_reward: BaseRewardFunction,
        long_term_reward: BaseRewardFunction,
        short_term_weight: float = 0.7,
        long_term_weight: float = 0.3,
        long_term_horizon: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.short_term_reward = short_term_reward
        self.long_term_reward = long_term_reward
        self.short_term_weight = short_term_weight
        self.long_term_weight = long_term_weight
        self.long_term_horizon = long_term_horizon

        # Accumulate rewards over horizon
        self.accumulated_rewards: List[float] = []
        self.step_count = 0

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute hierarchical reward."""

        # Compute short-term reward (immediate)
        short_reward, short_components = self.short_term_reward.compute_reward(
            state, action, next_state, info
        )

        # Compute long-term reward
        long_reward, long_components = self.long_term_reward.compute_reward(
            state, action, next_state, info
        )

        # Accumulate long-term rewards
        self.accumulated_rewards.append(long_reward)
        if len(self.accumulated_rewards) > self.long_term_horizon:
            self.accumulated_rewards.pop(0)

        # Average long-term reward over horizon
        avg_long_reward = np.mean(self.accumulated_rewards) if self.accumulated_rewards else 0.0

        # Combine short and long term
        total_reward = (
            self.short_term_weight * short_reward +
            self.long_term_weight * avg_long_reward
        )

        components = {
            'short_term': short_reward,
            'long_term': avg_long_reward,
        }

        total_reward = self.normalize_reward(total_reward)

        self.step_count += 1

        return total_reward, components

    def reset(self):
        """Reset hierarchical reward."""
        super().reset()
        self.short_term_reward.reset()
        self.long_term_reward.reset()
        self.accumulated_rewards.clear()
        self.step_count = 0


class SparseReward(BaseRewardFunction):
    """
    Sparse reward only at episode end or milestones.

    Instead of rewarding every step, only provide reward when:
    - Episode ends
    - Milestone reached (e.g., 10% return)
    - Target achieved

    This can help with credit assignment in some scenarios but may
    make learning slower. Often combined with reward shaping.

    Args:
        milestone_fn: Function to check if milestone reached
        episode_end_reward_fn: Function to compute episode-end reward
        milestone_reward: Reward for reaching milestone
    """

    def __init__(
        self,
        milestone_fn: Optional[Callable] = None,
        episode_end_reward_fn: Optional[Callable] = None,
        milestone_reward: float = 1.0,
        **kwargs
    ):
        super().__init__(reward_type=RewardType.SPARSE, **kwargs)
        self.milestone_fn = milestone_fn
        self.episode_end_reward_fn = episode_end_reward_fn or self._default_episode_reward
        self.milestone_reward = milestone_reward

        self.milestones_reached = 0

    def _default_episode_reward(self, info: Dict) -> float:
        """
        Default episode-end reward based on final return.

        Args:
            info: Info dict

        Returns:
            Episode reward
        """
        total_return = info.get('total_return', 0.0)
        sharpe_ratio = info.get('sharpe_ratio', 0.0)

        # Combine return and Sharpe
        episode_reward = total_return + 0.5 * sharpe_ratio
        return episode_reward

    def compute_reward(
        self,
        state: Dict,
        action: Dict,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Compute sparse reward."""

        reward = 0.0
        components = {}

        # Check for milestone
        if self.milestone_fn is not None:
            if self.milestone_fn(info):
                reward += self.milestone_reward
                self.milestones_reached += 1
                components['milestone'] = self.milestone_reward

        # Check for episode end
        if info.get('done', False):
            episode_reward = self.episode_end_reward_fn(info)
            reward += episode_reward
            components['episode_end'] = episode_reward

        reward = self.normalize_reward(reward)

        return reward, components

    def reset(self):
        """Reset sparse reward."""
        super().reset()
        self.milestones_reached = 0


class RewardNormalizer:
    """
    Utility for normalizing rewards with running statistics.

    Implements various normalization methods with online updates.

    Args:
        method: Normalization method
        clip_range: Range to clip normalized rewards
        decay: Decay factor for running statistics (default: 0.99)
    """

    def __init__(
        self,
        method: NormalizationMethod = NormalizationMethod.STANDARDIZE,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        decay: float = 0.99
    ):
        self.method = method
        self.clip_range = clip_range
        self.decay = decay

        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def normalize(self, reward: float) -> float:
        """
        Normalize reward.

        Args:
            reward: Raw reward

        Returns:
            Normalized reward
        """
        if self.method == NormalizationMethod.STANDARDIZE:
            # Update running statistics with decay
            self.count += 1
            alpha = 1.0 / self.count if self.count < 100 else (1.0 - self.decay)

            delta = reward - self.running_mean
            self.running_mean += alpha * delta
            self.running_var = (1.0 - alpha) * (self.running_var + alpha * delta ** 2)

            std = math.sqrt(self.running_var) if self.running_var > 0 else 1.0
            normalized = (reward - self.running_mean) / (std + 1e-8)

        elif self.method == NormalizationMethod.CLIP:
            normalized = np.clip(reward, self.clip_range[0], self.clip_range[1])

        else:
            normalized = reward

        return normalized

    def reset(self):
        """Reset normalizer statistics."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0


# Utility functions

def combine_rewards(
    rewards: List[BaseRewardFunction],
    weights: Optional[List[float]] = None
) -> BaseRewardFunction:
    """
    Combine multiple reward functions with weights.

    Args:
        rewards: List of reward functions
        weights: Optional weights (default: equal weights)

    Returns:
        Combined multi-component reward function
    """
    if weights is None:
        weights_dict = {f'component_{i}': 1.0 / len(rewards) for i in range(len(rewards))}
    else:
        weights_dict = {f'component_{i}': w for i, w in enumerate(weights)}

    combined = MultiComponentReward(
        weights=weights_dict,
        custom_components=rewards
    )

    return combined


def create_default_reward(
    risk_adjusted: bool = True,
    adaptive: bool = False,
    curiosity: bool = False,
    **kwargs
) -> BaseRewardFunction:
    """
    Factory function to create recommended default reward configuration.

    Args:
        risk_adjusted: Use risk-adjusted returns (Sharpe ratio)
        adaptive: Use adaptive weights based on market regime
        curiosity: Add curiosity bonus for exploration
        **kwargs: Additional arguments passed to reward functions

    Returns:
        Configured reward function
    """
    if risk_adjusted:
        base = RiskAdjustedReward(
            metric='sharpe',
            window_size=30,
            include_return=True,
            **kwargs
        )
    else:
        base = MultiComponentReward(**kwargs)

    if adaptive:
        base = AdaptiveReward(
            base_weights={'return': 1.0, 'risk': 0.5, 'transaction_cost': 0.1},
            **kwargs
        )

    if curiosity:
        base = CuriosityReward(
            base_reward=base,
            curiosity_weight=0.1,
            **kwargs
        )

    return base
