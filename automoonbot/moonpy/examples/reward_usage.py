"""
Comprehensive usage examples for reward functions and risk metrics.

This file demonstrates how to use the reward module components:
1. Risk Metrics calculation
2. Simple return rewards
3. Risk-adjusted rewards
4. Multi-component rewards
5. Adaptive rewards with regime detection
6. Potential-based reward shaping
7. Curiosity-driven exploration
8. Hierarchical rewards
9. Sparse rewards
10. Market regime detection
11. Complete RL training integration
"""

import numpy as np
import torch
from typing import Dict, List

from automoonbot.moonpy.reward.risk import (
    RiskMetrics,
    VolatilityEstimator,
    VaRMethod,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
)

from automoonbot.moonpy.reward.reward import (
    SimpleReturnReward,
    RiskAdjustedReward,
    MultiComponentReward,
    AdaptiveReward,
    PotentialBasedReward,
    CuriosityReward,
    HierarchicalReward,
    SparseReward,
    RewardNormalizer,
    NormalizationMethod,
    combine_rewards,
    create_default_reward,
)

from automoonbot.moonpy.reward.regime import (
    VolatilityRegimeDetector,
    TrendRegimeDetector,
    CorrelationRegimeDetector,
    MarketRegimeDetector,
    create_default_regime_detector,
    regime_to_risk_adjustment,
)


def example_1_basic_risk_metrics():
    """
    Example 1: Computing basic risk metrics from returns.

    Shows how to use RiskMetrics to track and compute:
    - Sharpe ratio
    - Sortino ratio
    - Maximum drawdown
    - Volatility
    """
    print("\n" + "="*80)
    print("Example 1: Basic Risk Metrics")
    print("="*80)

    # Initialize risk metrics tracker
    risk = RiskMetrics(
        window_size=252,  # One year of daily data
        risk_free_rate=0.02,  # 2% annual risk-free rate
        annualization_factor=252
    )

    # Simulate a year of trading returns
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 252)  # Daily returns

    # Update risk metrics with each return
    for return_val in returns:
        risk.update(return_val)

    # Compute metrics
    sharpe = risk.sharpe_ratio(annualized=True)
    sortino = risk.sortino_ratio(annualized=True)
    calmar = risk.calmar_ratio(annualized=True)
    max_dd = risk.max_drawdown()
    volatility = risk.volatility(annualized=True)

    print(f"\nRisk Metrics (after 252 trading days):")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Sortino Ratio: {sortino:.4f}")
    print(f"  Calmar Ratio: {calmar:.4f}")
    print(f"  Max Drawdown: {max_dd:.4%}")
    print(f"  Volatility (annualized): {volatility:.4%}")
    print(f"  Mean Return (annualized): {risk.mean_return(annualized=True):.4%}")


def example_2_advanced_risk_metrics():
    """
    Example 2: Advanced risk metrics (VaR, CVaR, Omega).

    Shows how to compute:
    - Value at Risk (VaR)
    - Conditional VaR (CVaR)
    - Omega ratio
    - Different VaR methods
    """
    print("\n" + "="*80)
    print("Example 2: Advanced Risk Metrics")
    print("="*80)

    risk = RiskMetrics(window_size=100, confidence_level=0.95)

    # Generate returns with fat tails (more realistic)
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=100) * 0.01

    risk.update_batch(returns.tolist())

    # Compute VaR using different methods
    var_historical = risk.value_at_risk(method=VaRMethod.HISTORICAL)
    var_parametric = risk.value_at_risk(method=VaRMethod.PARAMETRIC)

    # Compute CVaR (Expected Shortfall)
    cvar = risk.conditional_var(method=VaRMethod.HISTORICAL)

    # Omega ratio
    omega = risk.omega_ratio(threshold=0.0)

    print(f"\nAdvanced Risk Metrics:")
    print(f"  VaR (95%, Historical): {var_historical:.4%}")
    print(f"  VaR (95%, Parametric): {var_parametric:.4%}")
    print(f"  CVaR (95%, Historical): {cvar:.4%}")
    print(f"  Omega Ratio: {omega:.4f}")

    print(f"\nInterpretation:")
    print(f"  VaR: There's a 5% chance of losing more than {var_historical:.2%}")
    print(f"  CVaR: If losses exceed VaR, expected loss is {cvar:.2%}")
    print(f"  Omega > 1 means gains outweigh losses: {omega > 1}")


def example_3_simple_return_reward():
    """
    Example 3: Simple return-based reward function.

    Basic reward function that directly uses portfolio returns.
    """
    print("\n" + "="*80)
    print("Example 3: Simple Return Reward")
    print("="*80)

    # Create simple return reward
    reward_fn = SimpleReturnReward(
        scale_factor=10.0,  # Amplify rewards for learning
        log_returns=False
    )

    # Simulate trading steps
    state = {}
    action = {}
    next_state = {}

    print("\nComputing rewards for different returns:")
    for portfolio_return in [0.05, 0.02, -0.01, 0.0, -0.03]:
        info = {'portfolio_return': portfolio_return}
        reward, components = reward_fn.compute_reward(state, action, next_state, info)

        print(f"  Return: {portfolio_return:+.2%} → Reward: {reward:+.4f}")


def example_4_risk_adjusted_reward():
    """
    Example 4: Risk-adjusted reward using Sharpe ratio.

    Rewards based on risk-adjusted performance rather than raw returns.
    """
    print("\n" + "="*80)
    print("Example 4: Risk-Adjusted Reward (Sharpe)")
    print("="*80)

    # Create risk-adjusted reward
    reward_fn = RiskAdjustedReward(
        metric='sharpe',
        window_size=30,
        risk_free_rate=0.02,
        scale_factor=1.0,
        include_return=True,
        return_weight=0.5
    )

    # Simulate episode with varying returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 50)

    print("\nRewards during episode (first 10 steps):")
    for i, ret in enumerate(returns[:10]):
        info = {'portfolio_return': ret}
        reward, components = reward_fn.compute_reward({}, {}, {}, info)

        if i >= 5:  # After some history is built
            print(f"  Step {i}: Return {ret:+.3%} → Reward: {reward:+.4f}")
            print(f"           Components: {components}")


def example_5_multi_component_reward():
    """
    Example 5: Multi-component reward with custom weights.

    Combines multiple objectives: returns, risk, costs, diversification.
    """
    print("\n" + "="*80)
    print("Example 5: Multi-Component Reward")
    print("="*80)

    # Create multi-component reward
    reward_fn = MultiComponentReward(
        weights={
            'return': 1.0,
            'risk': 0.5,
            'transaction_cost': 0.2,
            'diversification': 0.1,
            'drawdown': 0.3,
        },
        penalties={
            'drawdown_threshold': 0.10,
            'concentration_power': 2.0,
        }
    )

    # Simulate different trading scenarios
    scenarios = [
        {
            'name': 'Good Trade',
            'portfolio_return': 0.03,
            'portfolio_volatility': 0.01,
            'transaction_costs': 0.001,
            'position_concentration': 0.2,
            'max_drawdown': 0.05,
        },
        {
            'name': 'High Risk',
            'portfolio_return': 0.05,
            'portfolio_volatility': 0.05,
            'transaction_costs': 0.001,
            'position_concentration': 0.8,
            'max_drawdown': 0.15,
        },
        {
            'name': 'Expensive Trade',
            'portfolio_return': 0.02,
            'portfolio_volatility': 0.01,
            'transaction_costs': 0.01,
            'position_concentration': 0.3,
            'max_drawdown': 0.05,
        },
    ]

    print("\nReward breakdown for different scenarios:")
    for scenario in scenarios:
        name = scenario.pop('name')
        reward, components = reward_fn.compute_reward({}, {}, {}, scenario)

        print(f"\n{name}:")
        print(f"  Total Reward: {reward:+.4f}")
        print(f"  Components:")
        for comp_name, comp_value in components.items():
            print(f"    {comp_name}: {comp_value:+.4f}")


def example_6_adaptive_reward():
    """
    Example 6: Adaptive reward that changes with market regime.

    Adjusts reward weights based on detected market conditions.
    """
    print("\n" + "="*80)
    print("Example 6: Adaptive Reward with Regime Detection")
    print("="*80)

    # Create adaptive reward
    reward_fn = AdaptiveReward(
        base_weights={
            'return': 1.0,
            'risk': 0.5,
            'transaction_cost': 0.1,
        },
        regime_weights={
            'bull': {'return': 1.5, 'risk': 0.5, 'transaction_cost': 1.0},
            'bear': {'return': 0.5, 'risk': 2.0, 'transaction_cost': 1.5},
            'high_vol': {'return': 0.8, 'risk': 1.5, 'transaction_cost': 1.2},
        }
    )

    # Simulate different market conditions
    market_conditions = [
        ('Bull Market', {'portfolio_volatility': 0.01, 'portfolio_return': 0.03,
                        'transaction_costs': 0.001}),
        ('Bear Market', {'portfolio_volatility': 0.02, 'portfolio_return': -0.03,
                        'transaction_costs': 0.001}),
        ('High Volatility', {'portfolio_volatility': 0.05, 'portfolio_return': 0.01,
                            'transaction_costs': 0.001}),
    ]

    print("\nAdaptive rewards in different market regimes:")
    for condition_name, info in market_conditions:
        reward, components = reward_fn.compute_reward({}, {}, {}, info)

        print(f"\n{condition_name}:")
        print(f"  Detected Regime: {reward_fn.current_regime}")
        print(f"  Total Reward: {reward:+.4f}")
        print(f"  Active Weights: {reward_fn.multi_component.weights}")


def example_7_potential_based_shaping():
    """
    Example 7: Potential-based reward shaping.

    Uses potential functions to provide denser reward signal while
    maintaining optimal policy.
    """
    print("\n" + "="*80)
    print("Example 7: Potential-Based Reward Shaping")
    print("="*80)

    # Base reward
    base_reward = SimpleReturnReward()

    # Custom potential function based on portfolio value
    def portfolio_value_potential(state: Dict, info: Dict) -> float:
        portfolio_value = info.get('portfolio_value', 1.0)
        # Use log to bound potential
        import math
        return math.log(max(portfolio_value, 1e-8))

    # Create shaped reward
    shaped_reward = PotentialBasedReward(
        base_reward=base_reward,
        potential_fn=portfolio_value_potential,
        gamma=0.99,
        potential_weight=1.0
    )

    # Simulate episode
    print("\nReward comparison (Base vs Shaped):")
    portfolio_values = [1.0, 1.02, 1.05, 1.03, 1.08]

    for i in range(len(portfolio_values) - 1):
        state = {}
        next_state = {}
        info = {
            'portfolio_return': (portfolio_values[i+1] - portfolio_values[i]) / portfolio_values[i],
            'portfolio_value': portfolio_values[i+1]
        }

        base_r, _ = base_reward.compute_reward(state, {}, next_state, info)
        shaped_r, components = shaped_reward.compute_reward(state, {}, next_state, info)

        print(f"\nStep {i} → {i+1}:")
        print(f"  Portfolio: {portfolio_values[i]:.3f} → {portfolio_values[i+1]:.3f}")
        print(f"  Base Reward: {base_r:+.4f}")
        print(f"  Shaped Reward: {shaped_r:+.4f}")
        print(f"  Shaping Bonus: {components['shaping']:+.4f}")


def example_8_curiosity_reward():
    """
    Example 8: Curiosity-driven exploration bonus.

    Adds intrinsic motivation to encourage exploring novel states.
    """
    print("\n" + "="*80)
    print("Example 8: Curiosity-Driven Exploration")
    print("="*80)

    base_reward = SimpleReturnReward()
    curiosity_reward = CuriosityReward(
        base_reward=base_reward,
        curiosity_weight=0.2,
        prediction_error_scale=1.0
    )

    # Simulate exploration
    print("\nExploration rewards:")
    portfolio_values = [1.0, 1.01, 1.02, 1.50, 1.51]  # Large jump at step 3

    for i in range(len(portfolio_values) - 1):
        state = {'portfolio_value': portfolio_values[i]}
        next_state = {'portfolio_value': portfolio_values[i+1]}
        info = {'portfolio_return': (portfolio_values[i+1] - portfolio_values[i]) / portfolio_values[i]}

        reward, components = curiosity_reward.compute_reward(state, {}, next_state, info)

        print(f"\nStep {i}:")
        print(f"  Portfolio change: {portfolio_values[i]:.2f} → {portfolio_values[i+1]:.2f}")
        print(f"  Extrinsic reward: {components['extrinsic']:+.4f}")
        print(f"  Intrinsic reward: {components['intrinsic']:+.4f}")
        print(f"  Total reward: {reward:+.4f}")


def example_9_hierarchical_reward():
    """
    Example 9: Hierarchical reward with multiple time scales.

    Combines short-term and long-term objectives.
    """
    print("\n" + "="*80)
    print("Example 9: Hierarchical Reward (Multi-Time Scale)")
    print("="*80)

    # Short-term: immediate returns
    short_term = SimpleReturnReward(scale_factor=1.0)

    # Long-term: risk-adjusted returns
    long_term = RiskAdjustedReward(metric='sharpe', window_size=10)

    # Combine
    hierarchical = HierarchicalReward(
        short_term_reward=short_term,
        long_term_reward=long_term,
        short_term_weight=0.6,
        long_term_weight=0.4,
        long_term_horizon=10
    )

    # Simulate episode
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 20)

    print("\nHierarchical rewards over episode:")
    for i, ret in enumerate(returns):
        info = {'portfolio_return': ret}
        reward, components = hierarchical.compute_reward({}, {}, {}, info)

        if i % 5 == 0:
            print(f"\nStep {i}:")
            print(f"  Return: {ret:+.3%}")
            print(f"  Short-term component: {components['short_term']:+.4f}")
            print(f"  Long-term component: {components['long_term']:+.4f}")
            print(f"  Total reward: {reward:+.4f}")


def example_10_sparse_reward():
    """
    Example 10: Sparse rewards for milestone-based learning.

    Only provides rewards at episode end or milestones.
    """
    print("\n" + "="*80)
    print("Example 10: Sparse Rewards")
    print("="*80)

    # Milestone function: reward when return exceeds 10%
    def milestone_fn(info: Dict) -> bool:
        total_return = info.get('total_return', 0.0)
        return total_return > 0.10

    sparse_reward = SparseReward(
        milestone_fn=milestone_fn,
        milestone_reward=5.0
    )

    # Simulate episode
    cumulative_return = 0.0
    print("\nSparse rewards during episode:")

    for step in range(15):
        step_return = 0.01  # 1% per step
        cumulative_return = (1 + cumulative_return) * (1 + step_return) - 1

        info = {
            'done': step == 14,
            'total_return': cumulative_return,
            'sharpe_ratio': 1.2
        }

        reward, components = sparse_reward.compute_reward({}, {}, {}, info)

        if reward > 0:
            print(f"\nStep {step}:")
            print(f"  Cumulative return: {cumulative_return:.2%}")
            print(f"  Reward: {reward:+.4f}")
            print(f"  Components: {components}")


def example_11_market_regime_detection():
    """
    Example 11: Market regime detection for adaptive strategies.

    Detects volatility and trend regimes.
    """
    print("\n" + "="*80)
    print("Example 11: Market Regime Detection")
    print("="*80)

    # Create regime detector
    detector = MarketRegimeDetector()

    # Simulate different market phases
    phases = [
        ("Low Vol Bull", np.random.normal(0.002, 0.005, 30)),
        ("High Vol Bear", np.random.normal(-0.003, 0.03, 30)),
        ("Sideways", np.random.normal(0.0, 0.01, 30)),
    ]

    print("\nDetected regimes in different market phases:")
    for phase_name, returns in phases:
        detector.reset()

        # Feed data
        for i, ret in enumerate(returns):
            price = 100 * np.prod(1 + returns[:i+1])
            detector.update(ret, price)

        # Detect regime
        regime = detector.detect()

        print(f"\n{phase_name}:")
        print(f"  Volatility Regime: {regime.volatility.value}")
        print(f"  Trend Regime: {regime.trend.value}")
        print(f"  Confidence: {regime.confidence:.2f}")

        # Get risk adjustments
        adjustments = regime_to_risk_adjustment(regime)
        print(f"  Suggested Adjustments:")
        for key, value in adjustments.items():
            print(f"    {key}: {value:.2f}x")


def example_12_complete_training_integration():
    """
    Example 12: Complete integration with RL training loop.

    Shows how to use reward functions in actual training.
    """
    print("\n" + "="*80)
    print("Example 12: Complete RL Training Integration")
    print("="*80)

    # Create sophisticated reward function
    reward_fn = create_default_reward(
        risk_adjusted=True,
        adaptive=True,
        curiosity=False
    )

    # Simulate training episode
    print("\nSimulating training episode with adaptive risk-adjusted rewards:")

    episode_rewards = []
    np.random.seed(42)

    for step in range(50):
        # Simulate market data
        portfolio_return = np.random.normal(0.001, 0.015)
        portfolio_volatility = abs(np.random.normal(0.015, 0.005))
        transaction_costs = 0.0005

        # State information
        state = {}
        action = {}
        next_state = {}
        info = {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'transaction_costs': transaction_costs,
        }

        # Compute reward
        reward, components = reward_fn.compute_reward(state, action, next_state, info)
        episode_rewards.append(reward)

        # Log periodically
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Portfolio return: {portfolio_return:+.3%}")
            print(f"  Volatility: {portfolio_volatility:.3%}")
            print(f"  Reward: {reward:+.4f}")

    # Episode summary
    print(f"\nEpisode Summary:")
    print(f"  Total steps: {len(episode_rewards)}")
    print(f"  Average reward: {np.mean(episode_rewards):+.4f}")
    print(f"  Reward std: {np.std(episode_rewards):.4f}")
    print(f"  Max reward: {np.max(episode_rewards):+.4f}")
    print(f"  Min reward: {np.min(episode_rewards):+.4f}")


def example_13_reward_normalization():
    """
    Example 13: Reward normalization for stable training.

    Shows different normalization methods.
    """
    print("\n" + "="*80)
    print("Example 13: Reward Normalization")
    print("="*80)

    # Create normalizers with different methods
    standardizer = RewardNormalizer(
        method=NormalizationMethod.STANDARDIZE,
        clip_range=(-10.0, 10.0)
    )

    clipper = RewardNormalizer(
        method=NormalizationMethod.CLIP,
        clip_range=(-1.0, 1.0)
    )

    # Simulate rewards with high variance
    np.random.seed(42)
    raw_rewards = np.random.normal(0, 5.0, 100)

    print("\nNormalizing high-variance rewards:")
    print(f"  Raw rewards - Mean: {np.mean(raw_rewards):.2f}, Std: {np.std(raw_rewards):.2f}")

    # Normalize
    standardized = [standardizer.normalize(r) for r in raw_rewards]
    clipped = [clipper.normalize(r) for r in raw_rewards]

    print(f"  Standardized - Mean: {np.mean(standardized):.2f}, Std: {np.std(standardized):.2f}")
    print(f"  Clipped - Mean: {np.mean(clipped):.2f}, Range: [{np.min(clipped):.2f}, {np.max(clipped):.2f}]")


def example_14_combining_reward_functions():
    """
    Example 14: Combining multiple reward functions.

    Create complex reward by combining simpler ones.
    """
    print("\n" + "="*80)
    print("Example 14: Combining Reward Functions")
    print("="*80)

    # Create individual reward functions
    return_reward = SimpleReturnReward(scale_factor=1.0)
    sharpe_reward = RiskAdjustedReward(metric='sharpe', scale_factor=0.5)

    # Combine with custom weights
    combined = combine_rewards(
        rewards=[return_reward, sharpe_reward],
        weights=[0.6, 0.4]
    )

    # Test combined reward
    print("\nCombined reward function:")

    # Build some history first
    for ret in np.random.normal(0.001, 0.01, 20):
        info = {'portfolio_return': ret}
        combined.compute_reward({}, {}, {}, info)

    # Now test
    info = {'portfolio_return': 0.02}
    reward, components = combined.compute_reward({}, {}, {}, info)

    print(f"  Total reward: {reward:+.4f}")
    print(f"  Components: {list(components.keys())}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("REWARD FUNCTIONS AND RISK METRICS - USAGE EXAMPLES")
    print("="*80)

    examples = [
        example_1_basic_risk_metrics,
        example_2_advanced_risk_metrics,
        example_3_simple_return_reward,
        example_4_risk_adjusted_reward,
        example_5_multi_component_reward,
        example_6_adaptive_reward,
        example_7_potential_based_shaping,
        example_8_curiosity_reward,
        example_9_hierarchical_reward,
        example_10_sparse_reward,
        example_11_market_regime_detection,
        example_12_complete_training_integration,
        example_13_reward_normalization,
        example_14_combining_reward_functions,
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
    # Run all examples
    run_all_examples()

    # Or run individual examples:
    # example_1_basic_risk_metrics()
    # example_5_multi_component_reward()
    # example_11_market_regime_detection()
