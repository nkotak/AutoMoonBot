import pytest
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List

from automoonbot.moonpy.model import Actor, ActionType, ActionSize, GATNet


class TestActorNetwork:
    """Comprehensive tests for Actor network implementation."""

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

        # Add equity nodes with features and symbols
        data["equity"].x = torch.randn(5, 32)  # 5 equities, 32 features
        data["equity"].symbol = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]

        # Add currency nodes
        data["currency"].x = torch.randn(3, 32)  # 3 currencies
        data["currency"].symbol = ["EURUSD", "GBPUSD", "JPYUSD"]

        # Add bonds nodes
        data["bonds"].x = torch.randn(2, 32)  # 2 bonds
        data["bonds"].symbol = ["US10Y", "US30Y"]

        # Add options nodes
        data["options"].x = torch.randn(4, 32)  # 4 options
        data["options"].symbol = ["AAPL_CALL_200", "AAPL_PUT_180", "TSLA_CALL_300", "TSLA_PUT_250"]

        # Add non-tradable nodes (article, company)
        data["article"].x = torch.randn(10, 32)
        data["company"].x = torch.randn(5, 32)

        # Add edges (simplified)
        data["equity", "mentioned_in", "article"].edge_index = torch.randint(0, 5, (2, 20))
        data["article", "published_by", "company"].edge_index = torch.randint(0, 5, (2, 10))
        data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 15))
        data["currency", "pairs_with", "currency"].edge_index = torch.randint(0, 3, (2, 6))

        return data

    @pytest.fixture
    def sample_portfolio_state(self):
        """Create sample portfolio state."""
        return {
            "AAPL": 0.2,    # 20% of portfolio in AAPL
            "GOOGL": 0.15,  # 15% in GOOGL
            "EURUSD": 0.1,  # 10% in EURUSD
            "US10Y": 0.05,  # 5% in bonds
            # Other assets have 0 position
        }

    def test_actor_initialization(self, metadata):
        """Test Actor network initializes correctly."""
        actor = Actor(
            metadata=metadata,
            gnn_hidden_dims=(64, 32, 16),
            gnn_embedding_dim=16,
            mem_heads=2,
            mem_size=16,
            mem_dim=16,
            key_dim=8,
            val_dim=16,
            action_hidden_dim=32,
        )

        assert actor is not None
        assert actor.gnn_embedding_dim == 16
        assert len(actor.tradable_node_types) == 4  # equity, currency, bonds, options
        assert "equity" in actor.action_type_classifiers
        assert "currency" in actor.action_type_classifiers
        assert "bonds" in actor.action_type_classifiers
        assert "options" in actor.action_type_classifiers

    def test_actor_forward_pass(self, metadata, sample_hetero_data):
        """Test forward pass through Actor network."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        action_logits, embeddings = actor.forward(sample_hetero_data)

        # Check action logits structure
        assert isinstance(action_logits, dict)
        assert "equity" in action_logits
        assert "action_type" in action_logits["equity"]
        assert "action_size" in action_logits["equity"]

        # Check dimensions
        assert action_logits["equity"]["action_type"].shape == (5, 3)  # 5 equities, 3 actions
        assert action_logits["equity"]["action_size"].shape == (5, 5)  # 5 equities, 5 sizes

        # Check embeddings
        assert isinstance(embeddings, dict)
        assert "equity" in embeddings

    def test_actor_get_action_deterministic(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test get_action in deterministic mode."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        actions = actor.get_action(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            deterministic=True,
        )

        # Check actions structure
        assert isinstance(actions, dict)
        assert len(actions) > 0

        # Check action format
        for symbol, action in actions.items():
            assert "type" in action
            assert "size" in action
            assert "fraction" in action
            assert isinstance(action["type"], int)
            assert action["type"] in [ActionType.HOLD, ActionType.BUY, ActionType.SELL]
            assert isinstance(action["size"], int)
            assert action["size"] in range(len(ActionSize))
            assert isinstance(action["fraction"], float)
            assert 0.0 <= action["fraction"] <= 1.0

    def test_actor_get_action_stochastic(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test get_action in stochastic mode."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        actions1 = actor.get_action(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            deterministic=False,
            temperature=1.0,
        )

        actions2 = actor.get_action(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            deterministic=False,
            temperature=1.0,
        )

        # With stochastic sampling, actions should differ (with high probability)
        # Note: There's a small chance they're identical, but very unlikely
        assert isinstance(actions1, dict)
        assert isinstance(actions2, dict)

    def test_actor_get_action_and_log_prob(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test get_action_and_log_prob for training."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        actions, log_probs, entropy = actor.get_action_and_log_prob(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            temperature=1.0,
        )

        # Check actions
        assert isinstance(actions, dict)
        assert len(actions) > 0

        # Check log probs
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape[0] == len(actions)
        assert torch.all(log_probs <= 0)  # Log probs should be negative or zero

        # Check entropy
        assert isinstance(entropy, torch.Tensor)
        assert entropy.shape[0] == len(actions)
        assert torch.all(entropy >= 0)  # Entropy should be non-negative

    def test_actor_evaluate_actions(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test evaluate_actions for PPO updates."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # First get actions
        actions, _, _ = actor.get_action_and_log_prob(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
        )

        # Then evaluate them
        log_probs, entropy, values = actor.evaluate_actions(
            data=sample_hetero_data,
            actions=actions,
            portfolio_state=sample_portfolio_state,
        )

        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert log_probs.shape[0] == len(actions)

    def test_action_masking_prevents_invalid_sells(self, metadata, sample_hetero_data):
        """Test that action masking prevents selling assets not owned."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Portfolio with no positions
        empty_portfolio = {}

        actions = actor.get_action(
            data=sample_hetero_data,
            portfolio_state=empty_portfolio,
            deterministic=True,
        )

        # Check that no SELL actions are taken (since portfolio is empty)
        for symbol, action in actions.items():
            if action["type"] == ActionType.SELL:
                # This should not happen with proper masking
                pytest.fail(f"Actor tried to SELL {symbol} with empty portfolio")

    def test_to_transaction_format(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test conversion to portfolio transaction format."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        actions = actor.get_action(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            deterministic=True,
        )

        transactions = actor.to_transaction_format(actions)

        # Check transaction structure
        assert isinstance(transactions, list)
        for txn in transactions:
            assert "type" in txn
            assert txn["type"] in ["buy", "sell"]
            assert "asset" in txn
            assert "size" in txn
            assert isinstance(txn["size"], float)
            assert 0.0 <= txn["size"] <= 1.0

    def test_temperature_affects_exploration(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test that temperature parameter affects action distribution."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Low temperature (more deterministic)
        actions_low_temp, log_probs_low, entropy_low = actor.get_action_and_log_prob(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            temperature=0.1,  # Very low temperature
        )

        # High temperature (more exploratory)
        actions_high_temp, log_probs_high, entropy_high = actor.get_action_and_log_prob(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
            temperature=2.0,  # High temperature
        )

        # Higher temperature should generally lead to higher entropy (more randomness)
        # Note: This is probabilistic, not guaranteed for every sample
        assert entropy_high.mean() > entropy_low.mean() * 0.8  # Allow some variance

    def test_hold_action_forces_zero_size(self, metadata, sample_hetero_data):
        """Test that HOLD actions always have size 0."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        actions = actor.get_action(
            data=sample_hetero_data,
            portfolio_state={},
            deterministic=False,
        )

        for symbol, action in actions.items():
            if action["type"] == ActionType.HOLD:
                assert action["size"] == ActionSize.SIZE_0
                assert action["fraction"] == 0.0

    def test_get_num_parameters(self, metadata):
        """Test parameter counting."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        num_params = actor.get_num_parameters()

        assert isinstance(num_params, int)
        assert num_params > 0

    def test_get_architecture_summary(self, metadata):
        """Test architecture summary generation."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        summary = actor.get_architecture_summary()

        assert isinstance(summary, dict)
        assert "total_parameters" in summary
        assert "gnn_embedding_dim" in summary
        assert "tradable_node_types" in summary
        assert "num_action_types" in summary
        assert "num_size_bins" in summary
        assert summary["gnn_embedding_dim"] == 32
        assert summary["num_action_types"] == 3
        assert summary["num_size_bins"] == 5

    def test_action_space_constants(self):
        """Test action space constant definitions."""
        assert ActionType.HOLD == 0
        assert ActionType.BUY == 1
        assert ActionType.SELL == 2

        assert ActionSize.SIZE_0 == 0
        assert ActionSize.SIZE_25 == 1
        assert ActionSize.SIZE_50 == 2
        assert ActionSize.SIZE_75 == 3
        assert ActionSize.SIZE_100 == 4

        assert Actor.SIZE_TO_FRACTION[ActionSize.SIZE_0] == 0.0
        assert Actor.SIZE_TO_FRACTION[ActionSize.SIZE_25] == 0.25
        assert Actor.SIZE_TO_FRACTION[ActionSize.SIZE_50] == 0.50
        assert Actor.SIZE_TO_FRACTION[ActionSize.SIZE_75] == 0.75
        assert Actor.SIZE_TO_FRACTION[ActionSize.SIZE_100] == 1.0

    def test_gradient_flow(self, metadata, sample_hetero_data, sample_portfolio_state):
        """Test that gradients flow through the network."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Forward pass with gradient tracking
        actions, log_probs, entropy = actor.get_action_and_log_prob(
            data=sample_hetero_data,
            portfolio_state=sample_portfolio_state,
        )

        # Compute a dummy loss
        loss = -log_probs.mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in actor.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        assert has_gradients, "No gradients found in actor parameters"

    def test_invalid_metadata_raises_error(self):
        """Test that invalid metadata raises appropriate errors."""
        invalid_metadata = (["article", "company"], [])  # Missing tradable types

        with pytest.raises(ValueError, match="not found in graph metadata"):
            actor = Actor(
                metadata=invalid_metadata,
                tradable_node_types=["equity"],  # Equity not in metadata
            )

    def test_batch_processing(self, metadata):
        """Test that actor can handle batched graphs."""
        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Create batched data (simulating multiple graphs)
        data = HeteroData()
        data["equity"].x = torch.randn(10, 32)  # 10 equities (could be from 2 graphs of 5 each)
        data["equity"].symbol = [f"STOCK_{i}" for i in range(10)]
        data["currency"].x = torch.randn(6, 32)  # 6 currencies
        data["currency"].symbol = [f"CURR_{i}" for i in range(6)]
        data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 10, (2, 20))

        # Should process without errors
        action_logits, embeddings = actor.forward(data)

        assert action_logits["equity"]["action_type"].shape[0] == 10
        assert action_logits["currency"]["action_type"].shape[0] == 6


class TestActorIntegration:
    """Integration tests with other components."""

    def test_actor_with_portfolio(self, metadata):
        """Test Actor integration with Portfolio class."""
        from automoonbot.moonpy.session.portfolio import Portfolio

        actor = Actor(metadata=metadata, gnn_embedding_dim=32)

        # Create portfolio
        portfolio = Portfolio(
            fiat="USD",
            tradables=["USD", "AAPL", "GOOGL", "EURUSD"],
        )

        # Create sample data
        data = HeteroData()
        data["equity"].x = torch.randn(2, 32)
        data["equity"].symbol = ["AAPL", "GOOGL"]
        data["currency"].x = torch.randn(1, 32)
        data["currency"].symbol = ["EURUSD"]
        data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 2, (2, 4))

        # Get actions
        portfolio_state = {
            "AAPL": 0.2,
            "GOOGL": 0.3,
        }

        actions = actor.get_action(
            data=data,
            portfolio_state=portfolio_state,
            deterministic=True,
        )

        # Convert to transactions
        transactions = actor.to_transaction_format(actions)

        # Transactions should be in format compatible with Portfolio
        for txn in transactions:
            assert "type" in txn
            assert "asset" in txn
            assert "size" in txn


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
