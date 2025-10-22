"""
Trading Environment Usage Examples

This file demonstrates how to use the TradingEnvironment for reinforcement learning.
The environment provides a complete gym-like interface for training trading agents.

Author: AutoMoonBot Team
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple

from automoonbot.moonpy.environment import (
    TradingEnvironment,
    RewardCalculator,
    EpisodeMetrics,
    TerminationCondition,
)
from automoonbot.moonpy.model import Actor, Critic


def example_1_basic_environment_setup():
    """Example 1: Basic environment setup and configuration."""
    print("=" * 80)
    print("Example 1: Basic Environment Setup")
    print("=" * 80)

    # Create environment with configuration
    env = TradingEnvironment(
        tradable_symbols=["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"],
        initial_capital=1.0,              # Normalized starting capital
        fiat_currency="USD",
        max_steps_per_episode=252,        # ~1 trading year
        transaction_cost_pct=0.001,       # 0.1% per trade
        slippage_pct=0.0005,              # 0.05% slippage
        min_portfolio_value=0.1,          # Bankruptcy threshold
        max_drawdown_termination=0.5,     # Max 50% drawdown
        position_limit_pct=0.5,           # Max 50% per position
        lookback_window=20,               # Rolling stats window
    )

    print("Environment Configuration:")
    print(f"  Tradable symbols: {env.tradable_symbols}")
    print(f"  Initial capital: {env.initial_capital}")
    print(f"  Max episode length: {env.max_steps_per_episode} steps")
    print(f"  Transaction cost: {env.transaction_cost_pct * 100:.2f}%")
    print(f"  Position limit: {env.position_limit_pct * 100:.0f}%")
    print()


def example_2_simple_price_data():
    """Example 2: Using simple price data."""
    print("=" * 80)
    print("Example 2: Simple Price Data Interface")
    print("=" * 80)

    # Create price data (dict of timestep -> {symbol: price})
    price_data = {}
    np.random.seed(42)

    for t in range(100):
        prices = {}
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            # Simulate price with random walk
            if t == 0:
                prices[symbol] = 100.0
            else:
                prev_price = price_data[t-1][symbol]
                change = np.random.normal(0.001, 0.02)  # 0.1% drift, 2% vol
                prices[symbol] = prev_price * (1 + change)
        price_data[t] = prices

    print(f"Generated {len(price_data)} timesteps of price data")
    print(f"Symbols: {list(price_data[0].keys())}")
    print(f"Initial prices: {price_data[0]}")
    print(f"Final prices: {price_data[99]}")
    print()

    # Set price data
    env = TradingEnvironment(
        tradable_symbols=["AAPL", "GOOGL", "MSFT"],
        max_steps_per_episode=100,
    )
    env.set_price_data(price_data)

    print("Price data configured!")
    print()


def example_3_custom_data_provider():
    """Example 3: Custom data provider with complex graphs."""
    print("=" * 80)
    print("Example 3: Custom Market Data Provider")
    print("=" * 80)

    def custom_data_provider(timestep: int) -> Tuple[HeteroData, Dict[str, float]]:
        """
        Custom data provider that builds rich heterogeneous graphs.

        Returns:
            state: HeteroData with equities, news, correlations, etc.
            prices: Dict of current prices
        """
        data = HeteroData()

        # Create equity nodes with features
        symbols = ["AAPL", "GOOGL", "MSFT"]
        features = []
        prices = {}

        for i, symbol in enumerate(symbols):
            # Price with trend and noise
            price = 100.0 + timestep * 0.5 + np.random.randn() * 2.0
            prices[symbol] = max(price, 10.0)  # Floor at $10

            # Features: [price_normalized, momentum, volatility]
            features.append([
                price / 100.0,                    # Normalized price
                np.sin(timestep / 10.0) * 0.1,   # Momentum signal
                0.02 + np.random.rand() * 0.01,   # Volatility
            ])

        data["equity"].x = torch.tensor(features, dtype=torch.float32)
        data["equity"].symbol = symbols

        # Add correlation edges
        num_symbols = len(symbols)
        edge_index = []
        edge_attr = []
        for i in range(num_symbols):
            for j in range(num_symbols):
                if i != j:
                    edge_index.append([i, j])
                    # Random correlation
                    corr = np.random.uniform(-0.3, 0.8)
                    edge_attr.append([corr])

        data["equity", "correlated_with", "equity"].edge_index = torch.tensor(
            edge_index, dtype=torch.long
        ).t()
        data["equity", "correlated_with", "equity"].edge_attr = torch.tensor(
            edge_attr, dtype=torch.float32
        )

        return data, prices

    # Set custom provider
    env = TradingEnvironment(tradable_symbols=["AAPL", "GOOGL", "MSFT"])
    env.set_market_data_provider(custom_data_provider)

    print("Custom data provider configured!")
    print("Provider builds graphs with:")
    print("  - Equity nodes with price/momentum/volatility features")
    print("  - Correlation edges between equities")
    print()


def example_4_reset_and_episode():
    """Example 4: Resetting environment and running episode."""
    print("=" * 80)
    print("Example 4: Reset and Episode Execution")
    print("=" * 80)

    # Setup
    price_data = {t: {"AAPL": 100.0 + t * 0.5} for t in range(20)}
    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        max_steps_per_episode=20,
    )
    env.set_price_data(price_data)

    # Reset environment
    state, info = env.reset(seed=42)

    print("Environment Reset:")
    print(f"  Episode: {info['episode']}")
    print(f"  Initial portfolio value: {info['initial_portfolio_value']}")
    print(f"  Num tradable assets: {info['num_tradable_assets']}")
    print(f"  State type: {type(state)}")
    print()

    # Run a few steps
    print("Running 5 steps:")
    for step in range(5):
        # Simple hold action
        actions = {"AAPL": {"type": 0, "size": 0, "fraction": 0.0}}

        state, reward, done, info = env.step(actions)

        print(f"  Step {step + 1}:")
        print(f"    Portfolio value: {info['portfolio_value']:.4f}")
        print(f"    Reward: {reward:.4f}")
        print(f"    Done: {done}")

    print()


def example_5_buy_and_sell_actions():
    """Example 5: Executing buy and sell actions."""
    print("=" * 80)
    print("Example 5: Buy and Sell Actions")
    print("=" * 80)

    # Setup
    price_data = {
        0: {"AAPL": 100.0, "GOOGL": 1000.0},
        1: {"AAPL": 102.0, "GOOGL": 1010.0},
        2: {"AAPL": 104.0, "GOOGL": 1020.0},
        3: {"AAPL": 103.0, "GOOGL": 1015.0},
        4: {"AAPL": 105.0, "GOOGL": 1025.0},
    }

    env = TradingEnvironment(
        tradable_symbols=["AAPL", "GOOGL"],
        transaction_cost_pct=0.001,
    )
    env.set_price_data(price_data)

    state, _ = env.reset()

    print("Step 1: BUY 30% AAPL")
    buy_actions = {
        "AAPL": {"type": 1, "size": 2, "fraction": 0.3},  # BUY 30%
    }
    state, reward, done, info = env.step(buy_actions)
    print(f"  Trades executed: {info['num_trades_executed']}")
    print(f"  Transaction costs: {info['transaction_costs']:.6f}")
    print(f"  Positions: {info['positions']}")
    print()

    print("Step 2: BUY 40% GOOGL")
    buy_actions = {
        "GOOGL": {"type": 1, "size": 3, "fraction": 0.4},  # BUY 40%
    }
    state, reward, done, info = env.step(buy_actions)
    print(f"  Trades executed: {info['num_trades_executed']}")
    print(f"  Positions: {info['positions']}")
    print()

    print("Step 3: SELL 50% of AAPL position")
    sell_actions = {
        "AAPL": {"type": 2, "size": 2, "fraction": 0.5},  # SELL 50%
    }
    state, reward, done, info = env.step(sell_actions)
    print(f"  Trades executed: {info['num_trades_executed']}")
    print(f"  Positions: {info['positions']}")
    print()


def example_6_reward_components():
    """Example 6: Understanding reward components."""
    print("=" * 80)
    print("Example 6: Reward Component Breakdown")
    print("=" * 80)

    # Setup with rising prices
    price_data = {
        0: {"AAPL": 100.0},
        1: {"AAPL": 105.0},  # 5% gain
        2: {"AAPL": 110.0},  # Another 4.76% gain
    }

    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        transaction_cost_pct=0.001,
    )
    env.set_price_data(price_data)

    state, _ = env.reset()

    # Buy AAPL
    actions = {"AAPL": {"type": 1, "size": 4, "fraction": 1.0}}  # BUY 100%
    state, reward, done, info = env.step(actions)

    print("Reward Breakdown:")
    for component, value in info['reward_components'].items():
        print(f"  {component:20s}: {value:+.6f}")
    print(f"  {'Total Reward':20s}: {reward:+.6f}")
    print()

    # Hold and price goes up
    actions = {"AAPL": {"type": 0, "size": 0, "fraction": 0.0}}
    state, reward, done, info = env.step(actions)

    print("After price increase (hold position):")
    print(f"  Portfolio return: {info['portfolio_return']:+.2%}")
    for component, value in info['reward_components'].items():
        print(f"  {component:20s}: {value:+.6f}")
    print(f"  {'Total Reward':20s}: {reward:+.6f}")
    print()


def example_7_custom_reward_calculator():
    """Example 7: Custom reward function."""
    print("=" * 80)
    print("Example 7: Custom Reward Calculator")
    print("=" * 80)

    # Create custom reward calculator
    custom_reward = RewardCalculator(
        return_weight=2.0,              # Double weight on returns
        risk_weight=0.2,                # Low risk penalty
        transaction_cost_weight=0.5,    # High cost penalty
        diversification_weight=0.1,     # Encourage diversification
        sharpe_weight=0.3,              # Reward Sharpe ratio
    )

    print("Custom Reward Weights:")
    print(f"  Return: {custom_reward.return_weight}")
    print(f"  Risk: {custom_reward.risk_weight}")
    print(f"  Transaction costs: {custom_reward.transaction_cost_weight}")
    print(f"  Diversification: {custom_reward.diversification_weight}")
    print(f"  Sharpe: {custom_reward.sharpe_weight}")
    print()

    # Use in environment
    price_data = {t: {"AAPL": 100.0 + t} for t in range(10)}
    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        reward_calculator=custom_reward,
    )
    env.set_price_data(price_data)

    print("Environment uses custom reward calculator!")
    print()


def example_8_episode_metrics():
    """Example 8: Episode metrics and statistics."""
    print("=" * 80)
    print("Example 8: Episode Metrics")
    print("=" * 80)

    # Setup
    np.random.seed(42)
    price_data = {}
    price = 100.0
    for t in range(50):
        price *= (1 + np.random.normal(0.001, 0.02))
        price_data[t] = {"AAPL": price}

    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        max_steps_per_episode=50,
    )
    env.set_price_data(price_data)

    # Run episode
    state, _ = env.reset()
    done = False
    while not done:
        # Buy and hold strategy
        if env.current_step == 0:
            actions = {"AAPL": {"type": 1, "size": 4, "fraction": 1.0}}
        else:
            actions = {}

        state, reward, done, info = env.step(actions)

    # Print episode metrics
    print("Episode Metrics:")
    metrics = info['episode_metrics']
    print(f"  Total Return: {metrics['total_return']:+.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Num Trades: {metrics['num_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Transaction Costs: {metrics['total_transaction_costs']:.6f}")
    print(f"  Final Portfolio Value: {metrics['final_portfolio_value']:.4f}")
    print(f"  Volatility: {metrics['volatility']:.4f}")
    print()


def example_9_termination_conditions():
    """Example 9: Episode termination conditions."""
    print("=" * 80)
    print("Example 9: Termination Conditions")
    print("=" * 80)

    # Scenario 1: Max steps
    print("Scenario 1: MAX_STEPS termination")
    price_data = {t: {"AAPL": 100.0} for t in range(10)}
    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        max_steps_per_episode=5,  # Short episode
    )
    env.set_price_data(price_data)

    state, _ = env.reset()
    for _ in range(10):  # Try to run longer
        state, reward, done, info = env.step({})
        if done:
            print(f"  Terminated: {info['termination_reason']}")
            break
    print()

    # Scenario 2: Bankruptcy
    print("Scenario 2: BANKRUPTCY termination (simulated)")
    crash_data = {
        0: {"AAPL": 100.0},
        1: {"AAPL": 10.0},  # 90% crash
    }
    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        min_portfolio_value=0.5,  # 50% threshold
    )
    env.set_price_data(crash_data)

    state, _ = env.reset()
    # Buy all AAPL
    state, _, _, _ = env.step({"AAPL": {"type": 1, "size": 4, "fraction": 1.0}})
    # Price crashes
    state, reward, done, info = env.step({})
    print(f"  Portfolio value: {info['portfolio_value']:.4f}")
    print(f"  Terminated: {done}")
    print()


def example_10_actor_integration():
    """Example 10: Integration with Actor network."""
    print("=" * 80)
    print("Example 10: Actor Integration")
    print("=" * 80)

    # Setup environment
    price_data = {t: {"AAPL": 100.0 + t * 0.5, "GOOGL": 1000.0 + t * 5.0}
                  for t in range(20)}
    env = TradingEnvironment(
        tradable_symbols=["AAPL", "GOOGL"],
        max_steps_per_episode=20,
    )
    env.set_price_data(price_data)

    # Reset and get metadata
    state, _ = env.reset()
    metadata = env.get_metadata()

    print("Environment metadata:")
    print(f"  Node types: {metadata[0]}")
    print(f"  Edge types: {metadata[1][:3]}...")  # First 3
    print()

    # Create actor
    actor = Actor(metadata=metadata, gnn_embedding_dim=64)
    print(f"Actor created with {actor.get_num_parameters():,} parameters")
    print()

    # Run episode with actor
    print("Running episode with actor:")
    done = False
    total_reward = 0
    step = 0

    while not done and step < 5:
        # Get portfolio state for action masking
        portfolio_state = env.get_portfolio_state()

        # Get actions from actor
        actions = actor.get_action(
            state,
            portfolio_state=portfolio_state,
            deterministic=True,
        )

        # Step environment
        state, reward, done, info = env.step(actions)
        total_reward += reward

        print(f"  Step {step + 1}:")
        print(f"    Actions: {len(actions)} proposed")
        print(f"    Trades executed: {info['num_trades_executed']}")
        print(f"    Portfolio value: {info['portfolio_value']:.4f}")
        print(f"    Reward: {reward:.4f}")

        step += 1

    print(f"\nTotal reward: {total_reward:.4f}")
    print()


def example_11_actor_critic_training():
    """Example 11: Complete Actor-Critic training setup."""
    print("=" * 80)
    print("Example 11: Actor-Critic Training Setup")
    print("=" * 80)

    # Generate price data
    np.random.seed(42)
    price_data = {}
    prices = {"AAPL": 100.0, "GOOGL": 1000.0}

    for t in range(100):
        for symbol in prices.keys():
            prices[symbol] *= (1 + np.random.normal(0.001, 0.015))
        price_data[t] = prices.copy()

    # Create environment
    env = TradingEnvironment(
        tradable_symbols=["AAPL", "GOOGL"],
        max_steps_per_episode=100,
        transaction_cost_pct=0.001,
    )
    env.set_price_data(price_data)

    # Initialize networks
    state, _ = env.reset()
    metadata = env.get_metadata()

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)
    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    print("Training Setup:")
    print(f"  Actor parameters: {actor.get_num_parameters():,}")
    print(f"  Critic parameters: {critic.get_num_parameters():,}")
    print(f"  Environment episodes: 100 steps")
    print()

    # Training loop (simplified demonstration)
    print("Simulated Training Loop:")
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0
        trajectory = []

        done = False
        while not done:
            # Get actions from actor
            portfolio_state = env.get_portfolio_state()
            actions, log_probs, entropy = actor.get_action_and_log_prob(
                state,
                portfolio_state=portfolio_state,
            )

            # Step environment
            next_state, reward, done, info = env.step(actions)

            # Store transition
            trajectory.append({
                'state': state,
                'actions': actions,
                'reward': reward,
                'next_state': next_state,
                'done': done,
            })

            episode_reward += reward
            state = next_state

        print(f"  Episode {episode + 1}:")
        print(f"    Steps: {len(trajectory)}")
        print(f"    Total reward: {episode_reward:.4f}")
        print(f"    Final value: {info['portfolio_value']:.4f}")

    print("\n(In real training, would update actor and critic here)")
    print()


def example_12_position_limits_and_validation():
    """Example 12: Position limits and action validation."""
    print("=" * 80)
    print("Example 12: Position Limits and Validation")
    print("=" * 80)

    price_data = {
        0: {"AAPL": 100.0},
        1: {"AAPL": 101.0},
        2: {"AAPL": 102.0},
    }

    env = TradingEnvironment(
        tradable_symbols=["AAPL"],
        position_limit_pct=0.3,  # Max 30% per position
    )
    env.set_price_data(price_data)

    state, _ = env.reset()

    print(f"Position limit: {env.position_limit_pct * 100:.0f}%")
    print()

    # Try to buy 50% (should be capped to 30%)
    print("Attempting to BUY 50% (exceeds limit):")
    actions = {"AAPL": {"type": 1, "size": 2, "fraction": 0.5}}
    state, reward, done, info = env.step(actions)

    print(f"  Trade executed: {info['num_trades_executed'] > 0}")
    if "AAPL" in info['positions']:
        position_pct = info['positions']['AAPL'] / info['portfolio_value']
        print(f"  Actual position: {position_pct * 100:.1f}%")
    print()

    # Try to sell without position
    env.reset()
    print("Attempting to SELL without position:")
    actions = {"AAPL": {"type": 2, "size": 2, "fraction": 0.5}}
    state, reward, done, info = env.step(actions)

    print(f"  Trade executed: {info['num_trades_executed'] > 0}")
    print("  (Should be False - can't sell without position)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "TRADING ENVIRONMENT USAGE EXAMPLES" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    examples = [
        example_1_basic_environment_setup,
        example_2_simple_price_data,
        example_3_custom_data_provider,
        example_4_reset_and_episode,
        example_5_buy_and_sell_actions,
        example_6_reward_components,
        example_7_custom_reward_calculator,
        example_8_episode_metrics,
        example_9_termination_conditions,
        example_10_actor_integration,
        example_11_actor_critic_training,
        example_12_position_limits_and_validation,
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"Example {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
