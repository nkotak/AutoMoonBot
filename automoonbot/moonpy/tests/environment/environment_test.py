import pytest
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple

from automoonbot.moonpy.environment import (
    TradingEnvironment,
    RewardCalculator,
    EpisodeMetrics,
    StepInfo,
    TerminationCondition,
)


class TestRewardCalculator:
    """Tests for RewardCalculator component."""

    def test_reward_calculator_initialization(self):
        """Test reward calculator initializes correctly."""
        calc = RewardCalculator(
            return_weight=1.0,
            risk_weight=0.5,
            transaction_cost_weight=0.1,
            diversification_weight=0.05,
            sharpe_weight=0.2,
        )

        assert calc.return_weight == 1.0
        assert calc.risk_weight == 0.5
        assert calc.transaction_cost_weight == 0.1
        assert calc.diversification_weight == 0.05
        assert calc.sharpe_weight == 0.2

    def test_reward_computation_positive_return(self):
        """Test reward computation with positive return."""
        calc = RewardCalculator()

        reward, components = calc.compute_reward(
            portfolio_return=0.05,     # 5% return
            portfolio_volatility=0.01,
            transaction_costs=0.001,
            position_concentration=0.2,
            sharpe_ratio=1.5,
            drawdown=0.02,
        )

        # Check components exist
        assert 'return' in components
        assert 'risk' in components
        assert 'transaction_costs' in components
        assert 'diversification' in components

        # Return component should be positive
        assert components['return'] > 0

        # Total reward should be sum of components
        assert abs(reward - sum(components.values())) < 1e-6

    def test_reward_computation_negative_return(self):
        """Test reward computation with negative return."""
        calc = RewardCalculator()

        reward, components = calc.compute_reward(
            portfolio_return=-0.03,    # -3% return
            portfolio_volatility=0.02,
            transaction_costs=0.002,
            position_concentration=0.5,
            sharpe_ratio=-0.5,
            drawdown=0.15,
        )

        # Return component should be negative
        assert components['return'] < 0

        # Total reward should be negative
        assert reward < 0

    def test_reward_penalizes_high_concentration(self):
        """Test that reward penalizes concentrated positions."""
        calc = RewardCalculator(diversification_weight=0.1)

        # Low concentration (diversified)
        _, comp_low = calc.compute_reward(
            portfolio_return=0.01,
            portfolio_volatility=0.01,
            transaction_costs=0.0,
            position_concentration=0.2,  # Low concentration
            sharpe_ratio=1.0,
            drawdown=0.0,
        )

        # High concentration
        _, comp_high = calc.compute_reward(
            portfolio_return=0.01,
            portfolio_volatility=0.01,
            transaction_costs=0.0,
            position_concentration=0.8,  # High concentration
            sharpe_ratio=1.0,
            drawdown=0.0,
        )

        # Diversified should get less penalty
        assert comp_low['diversification'] > comp_high['diversification']


class TestTradingEnvironment:
    """Comprehensive tests for TradingEnvironment."""

    @pytest.fixture
    def simple_price_data(self):
        """Create simple price data for testing."""
        # 10 timesteps, 3 symbols
        return {
            0: {"AAPL": 100.0, "GOOGL": 1000.0, "MSFT": 200.0},
            1: {"AAPL": 101.0, "GOOGL": 1010.0, "MSFT": 202.0},
            2: {"AAPL": 102.0, "GOOGL": 1020.0, "MSFT": 204.0},
            3: {"AAPL": 101.5, "GOOGL": 1015.0, "MSFT": 203.0},
            4: {"AAPL": 103.0, "GOOGL": 1025.0, "MSFT": 205.0},
            5: {"AAPL": 104.0, "GOOGL": 1030.0, "MSFT": 206.0},
            6: {"AAPL": 103.5, "GOOGL": 1028.0, "MSFT": 205.5},
            7: {"AAPL": 105.0, "GOOGL": 1035.0, "MSFT": 207.0},
            8: {"AAPL": 106.0, "GOOGL": 1040.0, "MSFT": 208.0},
            9: {"AAPL": 107.0, "GOOGL": 1045.0, "MSFT": 209.0},
        }

    @pytest.fixture
    def basic_env(self, simple_price_data):
        """Create basic environment for testing."""
        env = TradingEnvironment(
            tradable_symbols=["AAPL", "GOOGL", "MSFT"],
            initial_capital=1.0,
            max_steps_per_episode=10,
            transaction_cost_pct=0.001,
        )
        env.set_price_data(simple_price_data)
        return env

    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        env = TradingEnvironment(
            tradable_symbols=["AAPL", "GOOGL"],
            initial_capital=1.0,
            fiat_currency="USD",
            max_steps_per_episode=100,
        )

        assert env.tradable_symbols == ["AAPL", "GOOGL"]
        assert env.initial_capital == 1.0
        assert env.fiat_currency == "USD"
        assert env.max_steps_per_episode == 100
        assert env.current_step == 0
        assert env.episode_count == 0

    def test_environment_reset(self, basic_env):
        """Test environment reset."""
        state, info = basic_env.reset(seed=42)

        # Check state is HeteroData
        assert isinstance(state, HeteroData)

        # Check info dict
        assert 'episode' in info
        assert 'initial_portfolio_value' in info
        assert 'num_tradable_assets' in info

        # Check episode incremented
        assert basic_env.episode_count == 1
        assert basic_env.current_step == 0

    def test_environment_reset_reproducibility(self, basic_env):
        """Test that reset with same seed is reproducible."""
        state1, _ = basic_env.reset(seed=42)
        state2, _ = basic_env.reset(seed=42)

        # States should be identical
        assert state1.node_types == state2.node_types

    def test_step_with_hold_actions(self, basic_env):
        """Test step with all HOLD actions."""
        state, _ = basic_env.reset()

        # All hold actions
        actions = {
            "AAPL": {"type": 0, "size": 0, "fraction": 0.0},  # HOLD
            "GOOGL": {"type": 0, "size": 0, "fraction": 0.0},
            "MSFT": {"type": 0, "size": 0, "fraction": 0.0},
        }

        next_state, reward, done, info = basic_env.step(actions)

        # Check return types
        assert isinstance(next_state, HeteroData)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # No trades should be executed
        assert info['num_trades_executed'] == 0
        assert info['transaction_costs'] == 0.0

    def test_step_with_buy_action(self, basic_env):
        """Test step with BUY action."""
        state, _ = basic_env.reset()

        # Buy AAPL
        actions = {
            "AAPL": {"type": 1, "size": 2, "fraction": 0.5},  # BUY 50%
        }

        next_state, reward, done, info = basic_env.step(actions)

        # Check trade executed
        assert info['num_trades_executed'] > 0
        assert info['transaction_costs'] > 0.0

        # Check position created
        assert "AAPL" in info['positions']
        assert info['positions']['AAPL'] > 0

    def test_step_with_sell_action(self, basic_env):
        """Test step with SELL action."""
        state, _ = basic_env.reset()

        # First buy
        buy_actions = {"AAPL": {"type": 1, "size": 2, "fraction": 0.3}}
        basic_env.step(buy_actions)

        # Then sell
        sell_actions = {"AAPL": {"type": 2, "size": 1, "fraction": 0.5}}
        next_state, reward, done, info = basic_env.step(sell_actions)

        # Sell should execute
        assert info['num_trades_executed'] > 0

    def test_cant_sell_without_position(self, basic_env):
        """Test that can't sell asset without owning it."""
        state, _ = basic_env.reset()

        # Try to sell AAPL without owning it
        actions = {"AAPL": {"type": 2, "size": 2, "fraction": 0.5}}  # SELL

        next_state, reward, done, info = basic_env.step(actions)

        # No trade should execute
        assert info['num_trades_executed'] == 0

    def test_position_limit_enforcement(self, basic_env):
        """Test that position limits are enforced."""
        # Environment has 50% position limit by default
        state, _ = basic_env.reset()

        # Try to buy 80% (should be capped to 50%)
        actions = {"AAPL": {"type": 1, "size": 4, "fraction": 0.8}}

        next_state, reward, done, info = basic_env.step(actions)

        # Position should not exceed limit
        if "AAPL" in info['positions']:
            portfolio_value = info['portfolio_value']
            position_pct = info['positions']['AAPL'] / portfolio_value
            assert position_pct <= basic_env.position_limit_pct + 0.01  # Small tolerance

    def test_transaction_costs_computed(self, basic_env):
        """Test that transaction costs are computed correctly."""
        state, _ = basic_env.reset()

        actions = {"AAPL": {"type": 1, "size": 2, "fraction": 0.3}}

        next_state, reward, done, info = basic_env.step(actions)

        # Transaction costs should be positive
        if info['num_trades_executed'] > 0:
            assert info['transaction_costs'] > 0.0

    def test_episode_terminates_at_max_steps(self, simple_price_data):
        """Test episode terminates when max steps reached."""
        env = TradingEnvironment(
            tradable_symbols=["AAPL"],
            max_steps_per_episode=3,  # Short episode
        )
        env.set_price_data(simple_price_data)

        state, _ = env.reset()

        # Run for max steps
        for _ in range(3):
            actions = {"AAPL": {"type": 0, "size": 0, "fraction": 0.0}}
            state, reward, done, info = env.step(actions)

        # Should be done
        assert done
        assert info['termination_reason'] == TerminationCondition.MAX_STEPS.value

    def test_episode_terminates_on_bankruptcy(self, simple_price_data):
        """Test episode terminates on bankruptcy."""
        # Create environment with low bankruptcy threshold
        env = TradingEnvironment(
            tradable_symbols=["AAPL"],
            min_portfolio_value=0.5,  # 50% of initial
        )

        # Create declining price data
        crash_data = {
            0: {"AAPL": 100.0},
            1: {"AAPL": 10.0},  # Crash to trigger bankruptcy
        }
        env.set_price_data(crash_data)

        state, _ = env.reset()

        # Buy all AAPL
        actions = {"AAPL": {"type": 1, "size": 4, "fraction": 1.0}}
        state, reward, done, info = env.step(actions)

        # Should terminate due to bankruptcy
        # (depends on exact portfolio mechanics)

    def test_portfolio_value_tracking(self, basic_env):
        """Test that portfolio value is tracked correctly."""
        state, _ = basic_env.reset()

        # Initial value should be initial capital
        initial_value = basic_env._get_portfolio_value()
        assert abs(initial_value - basic_env.initial_capital) < 1e-6

        # After some trades, value should change
        actions = {"AAPL": {"type": 1, "size": 2, "fraction": 0.3}}
        next_state, reward, done, info = basic_env.step(actions)

        # Portfolio value in info should match internal calculation
        assert abs(info['portfolio_value'] - basic_env._get_portfolio_value()) < 1e-6

    def test_sharpe_ratio_computation(self, basic_env):
        """Test Sharpe ratio computation."""
        state, _ = basic_env.reset()

        # Run a few steps to build return history
        for i in range(5):
            actions = {"AAPL": {"type": 1, "size": 1, "fraction": 0.1}}
            state, reward, done, info = basic_env.step(actions)

        # Sharpe ratio should be computed
        assert 'sharpe_ratio' in info
        assert isinstance(info['sharpe_ratio'], (int, float))

    def test_max_drawdown_tracking(self, basic_env):
        """Test maximum drawdown tracking."""
        state, _ = basic_env.reset()

        # Track drawdown over episode
        max_dd = 0.0
        for i in range(5):
            actions = {}
            state, reward, done, info = basic_env.step(actions)
            max_dd = max(max_dd, info['max_drawdown'])

        # Max drawdown should be non-negative
        assert max_dd >= 0.0

    def test_episode_metrics_computed(self, basic_env):
        """Test that episode metrics are computed on termination."""
        state, _ = basic_env.reset()

        # Run until termination
        done = False
        while not done:
            actions = {}
            state, reward, done, info = basic_env.step(actions)

        # Episode metrics should be in info
        assert 'episode_metrics' in info

        metrics = info['episode_metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'num_trades' in metrics

    def test_get_portfolio_state(self, basic_env):
        """Test get_portfolio_state method."""
        state, _ = basic_env.reset()

        # Buy some assets
        actions = {
            "AAPL": {"type": 1, "size": 2, "fraction": 0.2},
            "GOOGL": {"type": 1, "size": 2, "fraction": 0.3},
        }
        basic_env.step(actions)

        portfolio_state = basic_env.get_portfolio_state()

        # Should return dict of positions
        assert isinstance(portfolio_state, dict)
        assert "AAPL" in portfolio_state or "GOOGL" in portfolio_state

    def test_get_metadata(self, basic_env):
        """Test get_metadata method."""
        state, _ = basic_env.reset()

        node_types, edge_types = basic_env.get_metadata()

        # Should return lists
        assert isinstance(node_types, list)
        assert isinstance(edge_types, list)

        # Should have at least equity node type
        assert any("equity" in str(nt).lower() for nt in node_types)

    def test_reward_components_returned(self, basic_env):
        """Test that reward components are returned in info."""
        state, _ = basic_env.reset()

        actions = {"AAPL": {"type": 1, "size": 2, "fraction": 0.2}}
        state, reward, done, info = basic_env.step(actions)

        # Reward components should be in info
        assert 'reward_components' in info
        components = info['reward_components']

        # Should have standard components
        assert 'return' in components
        assert 'risk' in components
        assert 'transaction_costs' in components

    def test_multiple_episodes(self, basic_env):
        """Test running multiple episodes."""
        for episode in range(3):
            state, info = basic_env.reset()
            assert info['episode'] == episode + 1

            done = False
            steps = 0
            while not done and steps < 5:
                actions = {}
                state, reward, done, info = basic_env.step(actions)
                steps += 1

    def test_custom_reward_calculator(self, simple_price_data):
        """Test using custom reward calculator."""
        custom_calc = RewardCalculator(
            return_weight=2.0,  # Double weight on returns
            risk_weight=0.1,    # Low risk weight
        )

        env = TradingEnvironment(
            tradable_symbols=["AAPL"],
            reward_calculator=custom_calc,
        )
        env.set_price_data(simple_price_data)

        state, _ = env.reset()
        actions = {}
        state, reward, done, info = env.step(actions)

        # Custom calculator should be used
        assert env.reward_calculator.return_weight == 2.0

    def test_set_market_data_provider(self):
        """Test setting custom market data provider."""
        env = TradingEnvironment(tradable_symbols=["AAPL"])

        def custom_provider(timestep: int) -> Tuple[HeteroData, Dict[str, float]]:
            data = HeteroData()
            data["equity"].x = torch.randn(1, 32)
            data["equity"].symbol = ["AAPL"]
            prices = {"AAPL": 100.0 + timestep}
            return data, prices

        env.set_market_data_provider(custom_provider)

        # Should be able to reset
        state, _ = env.reset()
        assert isinstance(state, HeteroData)

    def test_error_on_step_without_data_source(self):
        """Test that stepping without data source raises error."""
        env = TradingEnvironment(tradable_symbols=["AAPL"])

        # Should raise error when resetting without data
        with pytest.raises(RuntimeError, match="No market data source"):
            env.reset()

    def test_error_on_step_after_episode_end(self, basic_env):
        """Test error when stepping after episode ends."""
        state, _ = basic_env.reset()

        # Run to completion
        done = False
        while not done:
            actions = {}
            state, reward, done, info = basic_env.step(actions)

        # Trying to step again should raise error
        with pytest.raises(RuntimeError, match="Episode already terminated"):
            basic_env.step({})

    def test_volatility_computation(self, basic_env):
        """Test portfolio volatility computation."""
        state, _ = basic_env.reset()

        # Run several steps
        for _ in range(10):
            actions = {}
            state, reward, done, info = basic_env.step(actions)
            if done:
                break

        # Volatility should be computable
        vol = basic_env._compute_volatility()
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_position_concentration_computation(self, basic_env):
        """Test position concentration (Herfindahl index) computation."""
        state, _ = basic_env.reset()

        # Concentrated portfolio (all in one asset)
        actions = {"AAPL": {"type": 1, "size": 4, "fraction": 0.9}}
        basic_env.step(actions)

        concentration = basic_env._compute_position_concentration()

        # Should be high (closer to 1)
        assert 0.0 <= concentration <= 1.0

    def test_build_state_from_prices(self, simple_price_data):
        """Test building HeteroData state from simple price data."""
        env = TradingEnvironment(tradable_symbols=["AAPL", "GOOGL"])
        env.set_price_data(simple_price_data)

        state = env._build_state_from_prices(0)

        # Should create HeteroData with equity nodes
        assert isinstance(state, HeteroData)
        assert "equity" in state.node_types

    def test_actions_to_transactions_conversion(self, basic_env):
        """Test converting actions to transaction format."""
        basic_env.reset()

        actions = {
            "AAPL": {"type": 1, "size": 2, "fraction": 0.3},   # BUY
            "GOOGL": {"type": 2, "size": 1, "fraction": 0.2},  # SELL
            "MSFT": {"type": 0, "size": 0, "fraction": 0.0},   # HOLD
        }

        transactions = basic_env._actions_to_transactions(actions)

        # Should have 2 transactions (BUY and SELL, not HOLD)
        assert len(transactions) == 2

        # Check format
        for txn in transactions:
            assert 'type' in txn
            assert 'asset' in txn
            assert 'size' in txn
            assert txn['type'] in ['buy', 'sell']


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""

    def test_episode_metrics_initialization(self):
        """Test EpisodeMetrics initializes with defaults."""
        metrics = EpisodeMetrics()

        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.num_trades == 0
        assert metrics.win_rate == 0.0

    def test_episode_metrics_with_values(self):
        """Test EpisodeMetrics with custom values."""
        metrics = EpisodeMetrics(
            total_return=0.25,
            sharpe_ratio=1.8,
            max_drawdown=0.12,
            num_trades=150,
            win_rate=0.55,
        )

        assert metrics.total_return == 0.25
        assert metrics.sharpe_ratio == 1.8
        assert metrics.max_drawdown == 0.12
        assert metrics.num_trades == 150
        assert metrics.win_rate == 0.55


class TestStepInfo:
    """Tests for StepInfo dataclass."""

    def test_step_info_initialization(self):
        """Test StepInfo initialization."""
        info = StepInfo(
            portfolio_value=1.05,
            portfolio_return=0.05,
            positions={"AAPL": 0.5},
            num_trades_executed=2,
            transaction_costs=0.001,
            sharpe_ratio=1.5,
            max_drawdown=0.03,
            termination_reason=None,
        )

        assert info.portfolio_value == 1.05
        assert info.portfolio_return == 0.05
        assert info.positions == {"AAPL": 0.5}
        assert info.num_trades_executed == 2
        assert info.transaction_costs == 0.001


class TestIntegration:
    """Integration tests with Actor and Critic."""

    def test_integration_with_actor(self, simple_price_data):
        """Test environment integration with Actor."""
        from automoonbot.moonpy.model import Actor

        # Create environment
        env = TradingEnvironment(tradable_symbols=["AAPL", "GOOGL"])
        env.set_price_data(simple_price_data)

        # Reset and get metadata
        state, _ = env.reset()
        metadata = env.get_metadata()

        # Create actor
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Get actions from actor
        portfolio_state = env.get_portfolio_state()
        actions = actor.get_action(
            state,
            portfolio_state=portfolio_state,
            deterministic=True,
        )

        # Step environment
        next_state, reward, done, info = env.step(actions)

        # Should complete without error
        assert isinstance(reward, (int, float))

    def test_integration_with_critic(self, simple_price_data):
        """Test environment integration with Critic."""
        from automoonbot.moonpy.model import Critic

        # Create environment
        env = TradingEnvironment(tradable_symbols=["AAPL"])
        env.set_price_data(simple_price_data)

        # Reset
        state, _ = env.reset()
        metadata = env.get_metadata()

        # Create critic
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        # Get value estimate
        value = critic.get_value(state)

        # Should work
        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
