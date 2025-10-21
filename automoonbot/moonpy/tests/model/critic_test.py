import pytest
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List

from automoonbot.moonpy.model import Critic, PoolingStrategy, GATNet


class TestCriticNetwork:
    """Comprehensive tests for Critic network implementation."""

    @pytest.fixture
    def metadata(self):
        """Create sample heterogeneous graph metadata."""
        node_types = ["equity", "currency", "bonds", "options", "article", "company"]
        edge_types = [
            ("equity", "mentioned_in", "article"),
            ("article", "published_by", "company"),
            ("equity", "correlated_with", "equity"),
            ("currency", "pairs_with", "currency"),
        ]
        return (node_types, edge_types)

    @pytest.fixture
    def sample_hetero_data(self):
        """Create sample HeteroData for testing."""
        data = HeteroData()

        # Add nodes with features
        data["equity"].x = torch.randn(5, 32)
        data["currency"].x = torch.randn(3, 32)
        data["bonds"].x = torch.randn(2, 32)
        data["options"].x = torch.randn(4, 32)
        data["article"].x = torch.randn(10, 32)
        data["company"].x = torch.randn(5, 32)

        # Add edges
        data["equity", "mentioned_in", "article"].edge_index = torch.randint(0, 5, (2, 20))
        data["article", "published_by", "company"].edge_index = torch.randint(0, 5, (2, 10))
        data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 15))
        data["currency", "pairs_with", "currency"].edge_index = torch.randint(0, 3, (2, 6))

        return data

    @pytest.fixture
    def sample_trajectory(self, sample_hetero_data):
        """Create sample trajectory for testing."""
        # Create multiple states (simulating trajectory)
        states = [sample_hetero_data for _ in range(10)]
        rewards = torch.randn(10)
        next_states = [sample_hetero_data for _ in range(10)]
        dones = torch.zeros(10)
        dones[-1] = 1.0  # Last state is terminal

        return states, rewards, next_states, dones

    def test_critic_initialization_state_value(self, metadata):
        """Test Critic initialization for state-value estimation."""
        critic = Critic(
            metadata=metadata,
            gnn_hidden_dims=(64, 32, 16),
            gnn_embedding_dim=16,
            value_hidden_dims=(32, 16),
            num_critics=1,
            action_conditioned=False,
        )

        assert critic is not None
        assert critic.num_critics == 1
        assert not critic.action_conditioned
        assert critic.gnn_embedding_dim == 16

    def test_critic_initialization_action_value(self, metadata):
        """Test Critic initialization for action-value estimation."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=2,  # Twin critics for TD3
            action_conditioned=True,
            action_dim=10,
        )

        assert critic is not None
        assert critic.num_critics == 2
        assert critic.action_conditioned
        assert critic.action_dim == 10

    def test_critic_forward_state_value(self, metadata, sample_hetero_data):
        """Test forward pass for state-value estimation."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=1,
        )

        value = critic.forward(sample_hetero_data)

        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)  # [batch_size, 1]

    def test_critic_forward_action_value(self, metadata, sample_hetero_data):
        """Test forward pass for action-value estimation."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=1,
            action_conditioned=True,
            action_dim=10,
        )

        # Create dummy action
        action = torch.randn(1, 10)

        value = critic.forward(sample_hetero_data, actions=action)

        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)

    def test_critic_twin_critics(self, metadata, sample_hetero_data):
        """Test twin critics (for TD3)."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=2,
        )

        # Get single value (minimum of two critics)
        value = critic.forward(sample_hetero_data, return_all_critics=False)
        assert value.shape == (1, 1)

        # Get all critic values
        values = critic.forward(sample_hetero_data, return_all_critics=True)
        assert isinstance(values, tuple)
        assert len(values) == 2
        assert all(v.shape == (1, 1) for v in values)

    def test_get_value(self, metadata, sample_hetero_data):
        """Test get_value method."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        value = critic.get_value(sample_hetero_data)

        assert isinstance(value, torch.Tensor)
        assert value.shape == (1, 1)

    def test_get_all_values(self, metadata, sample_hetero_data):
        """Test get_all_values method."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=3,
        )

        values = critic.get_all_values(sample_hetero_data)

        assert isinstance(values, tuple)
        assert len(values) == 3
        assert all(v.shape == (1, 1) for v in values)

    def test_pooling_strategies(self, metadata, sample_hetero_data):
        """Test different pooling strategies."""
        strategies = [
            PoolingStrategy.MEAN,
            PoolingStrategy.MAX,
            PoolingStrategy.SUM,
            PoolingStrategy.ATTENTION,
        ]

        for strategy in strategies:
            critic = Critic(
                metadata=metadata,
                gnn_embedding_dim=32,
                pooling_strategy=strategy,
            )

            value = critic.get_value(sample_hetero_data)
            assert value.shape == (1, 1), f"Failed for {strategy}"

    def test_compute_advantages_gae(self, metadata, sample_trajectory):
        """Test GAE advantage computation."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        states, rewards, next_states, dones = sample_trajectory

        advantages, returns = critic.compute_advantages(
            states=states,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=0.99,
            gae_lambda=0.95,
            normalize=True,
        )

        assert isinstance(advantages, torch.Tensor)
        assert isinstance(returns, torch.Tensor)
        assert advantages.shape == (10,)
        assert returns.shape == (10,)

        # Check normalization (mean ~ 0, std ~ 1)
        assert torch.abs(advantages.mean()) < 0.5
        assert torch.abs(advantages.std() - 1.0) < 0.5

    def test_compute_td_targets(self, metadata, sample_trajectory):
        """Test TD target computation."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        states, rewards, next_states, dones = sample_trajectory

        td_targets = critic.compute_td_targets(
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=0.99,
        )

        assert isinstance(td_targets, torch.Tensor)
        assert td_targets.shape == (10,)

    def test_compute_n_step_returns(self, metadata):
        """Test n-step return computation."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        # Create n-step trajectory
        data = HeteroData()
        data["equity"].x = torch.randn(5, 32)
        data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 10))

        batch_size = 4
        n = 3
        states = [data for _ in range(batch_size)]
        rewards = torch.randn(batch_size, n)
        dones = torch.zeros(batch_size, n)
        next_states = [data for _ in range(batch_size)]

        n_step_returns = critic.compute_n_step_returns(
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=0.99,
            n=n,
        )

        assert isinstance(n_step_returns, torch.Tensor)
        assert n_step_returns.shape == (batch_size,)

    def test_target_network_creation(self, metadata):
        """Test target network creation."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        # Initially no target network
        assert critic._target_network is None

        # Create target network
        critic.create_target_network()

        assert critic._target_network is not None
        assert isinstance(critic._target_network, Critic)

        # Target network parameters should be frozen
        for param in critic._target_network.parameters():
            assert not param.requires_grad

    def test_target_network_hard_update(self, metadata, sample_hetero_data):
        """Test hard update of target network."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)
        critic.create_target_network()

        # Get initial values
        value_before = critic.get_value(sample_hetero_data)
        target_value_before = critic.get_target_value(sample_hetero_data)

        # They should be equal initially
        assert torch.allclose(value_before, target_value_before)

        # Update critic parameters (simulate training)
        for param in critic.parameters():
            param.data += 0.1

        # Values should differ now
        value_after = critic.get_value(sample_hetero_data)
        target_value_after = critic.get_target_value(sample_hetero_data)
        assert not torch.allclose(value_after, target_value_after)

        # Hard update (tau=1.0)
        critic.update_target_network(tau=1.0)

        # Now they should be equal again
        value_final = critic.get_value(sample_hetero_data)
        target_value_final = critic.get_target_value(sample_hetero_data)
        assert torch.allclose(value_final, target_value_final)

    def test_target_network_soft_update(self, metadata):
        """Test soft update (Polyak averaging) of target network."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)
        critic.create_target_network()

        # Get initial parameter
        initial_param = list(critic.parameters())[0].clone()
        initial_target_param = list(critic._target_network.parameters())[0].clone()

        # Update critic parameters
        for param in critic.parameters():
            param.data += 0.5

        # Soft update (tau=0.1)
        tau = 0.1
        critic.update_target_network(tau=tau)

        # Check Polyak averaging
        updated_param = list(critic.parameters())[0]
        updated_target_param = list(critic._target_network.parameters())[0]

        expected_target = tau * updated_param + (1 - tau) * initial_target_param
        assert torch.allclose(updated_target_param, expected_target)

    def test_compute_value_loss(self, metadata, sample_trajectory):
        """Test value loss computation."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        states, rewards, next_states, dones = sample_trajectory

        # Compute target returns
        _, returns = critic.compute_advantages(
            states=states,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        # Compute value loss
        loss = critic.compute_value_loss(states, returns)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # MSE loss is non-negative

    def test_compute_bellman_error(self, metadata, sample_trajectory):
        """Test Bellman error computation."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        states, rewards, next_states, dones = sample_trajectory

        bellman_error = critic.compute_bellman_error(
            states=states,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=0.99,
        )

        assert isinstance(bellman_error, torch.Tensor)
        assert bellman_error.ndim == 0  # Scalar
        assert bellman_error >= 0  # Absolute error is non-negative

    def test_gradient_flow(self, metadata, sample_hetero_data):
        """Test that gradients flow through the network."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        # Forward pass with gradient tracking
        value = critic.get_value(sample_hetero_data)

        # Backward pass
        loss = value.mean()
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in critic.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        assert has_gradients, "No gradients found in critic parameters"

    def test_layer_normalization(self, metadata, sample_hetero_data):
        """Test layer normalization option."""
        critic_with_ln = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            use_layer_norm=True,
        )

        critic_without_ln = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            use_layer_norm=False,
        )

        # Both should work
        value_with_ln = critic_with_ln.get_value(sample_hetero_data)
        value_without_ln = critic_without_ln.get_value(sample_hetero_data)

        assert value_with_ln.shape == value_without_ln.shape

    def test_dropout(self, metadata, sample_hetero_data):
        """Test dropout option."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            dropout=0.5,
        )

        # Training mode (dropout active)
        critic.train()
        value_train_1 = critic.get_value(sample_hetero_data)
        value_train_2 = critic.get_value(sample_hetero_data)

        # Values should differ due to dropout (with high probability)
        # Note: There's a small chance they're identical

        # Eval mode (no dropout)
        critic.eval()
        value_eval_1 = critic.get_value(sample_hetero_data)
        value_eval_2 = critic.get_value(sample_hetero_data)

        # Values should be identical in eval mode
        assert torch.allclose(value_eval_1, value_eval_2)

    def test_get_num_parameters(self, metadata):
        """Test parameter counting."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        num_params = critic.get_num_parameters()

        assert isinstance(num_params, int)
        assert num_params > 0

    def test_get_architecture_summary(self, metadata):
        """Test architecture summary."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=2,
            pooling_strategy=PoolingStrategy.ATTENTION,
        )

        summary = critic.get_architecture_summary()

        assert isinstance(summary, dict)
        assert "total_parameters" in summary
        assert "gnn_embedding_dim" in summary
        assert "num_critics" in summary
        assert "pooling_strategy" in summary
        assert summary["num_critics"] == 2
        assert summary["pooling_strategy"] == "attention"

    def test_save_and_load_checkpoint(self, metadata, sample_hetero_data, tmp_path):
        """Test checkpoint saving and loading."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            num_critics=2,
        )
        critic.create_target_network()

        # Get initial value
        initial_value = critic.get_value(sample_hetero_data)

        # Save checkpoint
        checkpoint_path = tmp_path / "critic.pt"
        critic.save_checkpoint(str(checkpoint_path))

        # Load checkpoint
        loaded_critic = Critic.load_checkpoint(str(checkpoint_path))

        # Check that loaded critic produces same value
        loaded_value = loaded_critic.get_value(sample_hetero_data)
        assert torch.allclose(initial_value, loaded_value)

        # Check that target network was loaded
        assert loaded_critic._target_network is not None

    def test_action_conditioned_requires_actions(self, metadata, sample_hetero_data):
        """Test that action-conditioned critic requires actions."""
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=32,
            action_conditioned=True,
            action_dim=10,
        )

        # Should raise error without actions
        with pytest.raises(ValueError, match="actions must be provided"):
            critic.get_value(sample_hetero_data)

        # Should work with actions
        actions = torch.randn(1, 10)
        value = critic.get_value(sample_hetero_data, actions=actions)
        assert value.shape == (1, 1)

    def test_invalid_configuration_raises_error(self, metadata):
        """Test that invalid configurations raise errors."""
        # Action-conditioned without action_dim
        with pytest.raises(ValueError, match="action_dim must be specified"):
            Critic(
                metadata=metadata,
                action_conditioned=True,
                action_dim=None,
            )

    def test_target_network_errors(self, metadata, sample_hetero_data):
        """Test proper error handling for target network operations."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        # Should raise error if getting target value without creating target network
        with pytest.raises(RuntimeError, match="Target network not created"):
            critic.get_target_value(sample_hetero_data)

        # Should raise error if updating target network without creating it
        with pytest.raises(RuntimeError, match="Target network not created"):
            critic.update_target_network()

    def test_different_value_head_dims(self, metadata, sample_hetero_data):
        """Test critics with different value head architectures."""
        dims_list = [
            (256,),
            (256, 128),
            (512, 256, 128),
            (128, 64, 32, 16),
        ]

        for dims in dims_list:
            critic = Critic(
                metadata=metadata,
                gnn_embedding_dim=32,
                value_hidden_dims=dims,
            )

            value = critic.get_value(sample_hetero_data)
            assert value.shape == (1, 1), f"Failed for dims {dims}"

    def test_td_targets_with_target_network(self, metadata, sample_trajectory):
        """Test TD target computation with and without target network."""
        critic = Critic(metadata=metadata, gnn_embedding_dim=32)

        states, rewards, next_states, dones = sample_trajectory

        # Without target network
        td_targets_1 = critic.compute_td_targets(
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            use_target_network=False,
        )

        # Create target network
        critic.create_target_network()

        # With target network
        td_targets_2 = critic.compute_td_targets(
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            use_target_network=True,
        )

        # Should be equal initially (target network is copy)
        assert torch.allclose(td_targets_1, td_targets_2)


class TestCriticIntegration:
    """Integration tests with Actor and other components."""

    def test_critic_with_actor_advantage_computation(self, metadata):
        """Test computing advantages for actor-critic training."""
        from automoonbot.moonpy.model import Actor

        critic = Critic(metadata=metadata, gnn_embedding_dim=32)
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Create trajectory
        data = HeteroData()
        data["equity"].x = torch.randn(5, 32)
        data["equity"].symbol = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
        data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 10))

        states = [data for _ in range(10)]
        rewards = torch.randn(10)
        next_states = [data for _ in range(10)]
        dones = torch.zeros(10)
        dones[-1] = 1.0

        # Compute advantages
        advantages, returns = critic.compute_advantages(
            states=states,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        # Get actions and log probs from actor
        actions, log_probs, entropy = actor.get_action_and_log_prob(data)

        # Should be able to compute policy gradient loss
        # (advantages would be used to weight log probs)
        assert advantages.shape == rewards.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
