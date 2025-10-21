"""
Actor Network Usage Examples

This file demonstrates how to use the Actor network for trading in AutoMoonBot.
The Actor network processes heterogeneous market graphs and outputs trading actions.

Author: AutoMoonBot Team
"""

import torch
from torch_geometric.data import HeteroData
from automoonbot.moonpy.model import Actor, ActionType, ActionSize


def example_1_basic_usage():
    """Example 1: Basic Actor initialization and forward pass."""
    print("=" * 80)
    print("Example 1: Basic Actor Usage")
    print("=" * 80)

    # Define graph metadata (node types and edge types)
    node_types = ["equity", "currency", "bonds", "options", "article", "company"]
    edge_types = [
        ("equity", "mentioned_in", "article"),
        ("article", "published_by", "company"),
        ("equity", "correlated_with", "equity"),
    ]
    metadata = (node_types, edge_types)

    # Initialize Actor network
    actor = Actor(
        metadata=metadata,
        gnn_hidden_dims=(512, 256, 128),  # GNN layer dimensions
        gnn_embedding_dim=128,             # Final embedding size
        mem_heads=4,                       # Memory attention heads
        mem_size=32,                       # Memory bank size
        mem_dim=128,                       # Memory vector dimension
        key_dim=64,                        # Attention key dimension
        val_dim=128,                       # Attention value dimension
        action_hidden_dim=256,             # Action classifier hidden dim
    )

    print(f"Actor initialized with {actor.get_num_parameters():,} parameters")
    print(f"Tradable node types: {actor.tradable_node_types}")
    print()

    # Create sample market data (HeteroData graph)
    data = HeteroData()

    # Add equity nodes (stocks)
    data["equity"].x = torch.randn(5, 32)  # 5 stocks with 32 features each
    data["equity"].symbol = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]

    # Add currency nodes (forex)
    data["currency"].x = torch.randn(2, 32)
    data["currency"].symbol = ["EURUSD", "GBPUSD"]

    # Add edges (correlations, mentions, etc.)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 10))

    # Forward pass
    action_logits, embeddings = actor.forward(data)

    print("Forward pass output:")
    print(f"  Action logits keys: {list(action_logits.keys())}")
    print(f"  Equity action type logits shape: {action_logits['equity']['action_type'].shape}")
    print(f"  Equity action size logits shape: {action_logits['equity']['action_size'].shape}")
    print()


def example_2_inference_mode():
    """Example 2: Getting actions for inference/trading."""
    print("=" * 80)
    print("Example 2: Inference Mode (Deterministic Actions)")
    print("=" * 80)

    # Setup (same as example 1)
    node_types = ["equity", "currency"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)

    # Create market data
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity"].symbol = ["AAPL", "GOOGL", "MSFT"]
    data["currency"].x = torch.randn(2, 32)
    data["currency"].symbol = ["EURUSD", "GBPUSD"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    # Current portfolio state
    portfolio_state = {
        "AAPL": 0.2,   # Own 20% of portfolio in AAPL
        "GOOGL": 0.15, # Own 15% in GOOGL
        # MSFT, EURUSD, GBPUSD have 0 position
    }

    # Get deterministic actions (for inference)
    actions = actor.get_action(
        data=data,
        portfolio_state=portfolio_state,
        deterministic=True,  # Take argmax (no sampling)
        temperature=1.0,
    )

    print("Generated Actions:")
    for symbol, action in actions.items():
        action_type = ActionType(action['type'])
        action_name = action_type.name
        fraction = action['fraction']

        print(f"  {symbol:8s}: {action_name:4s} {fraction*100:5.1f}%")

    print()


def example_3_training_mode():
    """Example 3: Getting actions with log probabilities for training."""
    print("=" * 80)
    print("Example 3: Training Mode (Stochastic Actions with Log Probs)")
    print("=" * 80)

    # Setup
    node_types = ["equity", "currency"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)

    # Market data
    data = HeteroData()
    data["equity"].x = torch.randn(2, 32)
    data["equity"].symbol = ["AAPL", "GOOGL"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 2, (2, 3))

    portfolio_state = {"AAPL": 0.3}

    # Get actions with log probabilities (for training)
    actions, log_probs, entropy = actor.get_action_and_log_prob(
        data=data,
        portfolio_state=portfolio_state,
        temperature=1.0,  # Standard temperature
    )

    print("Training Mode Output:")
    print(f"  Actions: {list(actions.keys())}")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Log probs: {log_probs}")
    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Entropy: {entropy}")
    print(f"  Mean log prob: {log_probs.mean().item():.4f}")
    print(f"  Mean entropy: {entropy.mean().item():.4f}")
    print()


def example_4_ppo_update():
    """Example 4: Evaluating actions for PPO policy updates."""
    print("=" * 80)
    print("Example 4: PPO Policy Update (Action Evaluation)")
    print("=" * 80)

    # Setup
    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)

    # Market data
    data = HeteroData()
    data["equity"].x = torch.randn(2, 32)
    data["equity"].symbol = ["AAPL", "GOOGL"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 2, (2, 3))

    portfolio_state = {"AAPL": 0.2}

    # Step 1: Get actions (simulating agent's behavior)
    actions_taken = {
        "AAPL": {"type": ActionType.SELL, "size": ActionSize.SIZE_50},
        "GOOGL": {"type": ActionType.BUY, "size": ActionSize.SIZE_25},
    }

    # Step 2: Evaluate those actions (for computing PPO loss)
    log_probs, entropy, values = actor.evaluate_actions(
        data=data,
        actions=actions_taken,
        portfolio_state=portfolio_state,
        temperature=1.0,
    )

    print("Action Evaluation Output:")
    print(f"  Log probs: {log_probs}")
    print(f"  Entropy: {entropy}")
    print(f"  Values: {values}")
    print()

    # Simulate computing PPO loss
    # (In real training, you'd have returns and old_log_probs from replay buffer)
    returns = torch.tensor([0.05, -0.02])  # Example returns
    old_log_probs = log_probs.detach()     # Old policy log probs

    # PPO ratio
    ratio = (log_probs - old_log_probs).exp()
    print(f"PPO Ratio: {ratio}")
    print()


def example_5_transaction_conversion():
    """Example 5: Converting actions to portfolio transactions."""
    print("=" * 80)
    print("Example 5: Converting Actions to Portfolio Transactions")
    print("=" * 80)

    # Setup
    node_types = ["equity", "currency"]
    edge_types = []
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)

    # Simulated actions from actor
    actions = {
        "AAPL": {"type": ActionType.BUY, "size": ActionSize.SIZE_50, "fraction": 0.50},
        "GOOGL": {"type": ActionType.SELL, "size": ActionSize.SIZE_25, "fraction": 0.25},
        "MSFT": {"type": ActionType.HOLD, "size": ActionSize.SIZE_0, "fraction": 0.0},
        "EURUSD": {"type": ActionType.BUY, "size": ActionSize.SIZE_100, "fraction": 1.0},
    }

    # Convert to portfolio transaction format
    transactions = actor.to_transaction_format(actions)

    print("Actions:")
    for symbol, action in actions.items():
        print(f"  {symbol}: {ActionType(action['type']).name} {action['fraction']*100:.0f}%")

    print("\nPortfolio Transactions:")
    for txn in transactions:
        print(f"  {txn['type'].upper():4s} {txn['asset']:8s} {txn['size']*100:5.1f}%")

    print(f"\nTotal transactions: {len(transactions)} (HOLD actions filtered out)")
    print()


def example_6_temperature_exploration():
    """Example 6: Using temperature for exploration vs exploitation."""
    print("=" * 80)
    print("Example 6: Temperature Control for Exploration")
    print("=" * 80)

    # Setup
    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)

    # Market data
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity"].symbol = ["AAPL", "GOOGL", "MSFT"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    portfolio_state = {}

    # Low temperature (exploitation - more deterministic)
    print("Low Temperature (T=0.1) - Exploitation:")
    actions_low, log_probs_low, entropy_low = actor.get_action_and_log_prob(
        data=data,
        portfolio_state=portfolio_state,
        temperature=0.1,
    )
    print(f"  Mean entropy: {entropy_low.mean().item():.4f}")

    # Medium temperature (balanced)
    print("\nMedium Temperature (T=1.0) - Balanced:")
    actions_med, log_probs_med, entropy_med = actor.get_action_and_log_prob(
        data=data,
        portfolio_state=portfolio_state,
        temperature=1.0,
    )
    print(f"  Mean entropy: {entropy_med.mean().item():.4f}")

    # High temperature (exploration - more random)
    print("\nHigh Temperature (T=2.0) - Exploration:")
    actions_high, log_probs_high, entropy_high = actor.get_action_and_log_prob(
        data=data,
        portfolio_state=portfolio_state,
        temperature=2.0,
    )
    print(f"  Mean entropy: {entropy_high.mean().item():.4f}")

    print("\nNote: Higher temperature → Higher entropy → More exploration")
    print()


def example_7_action_masking():
    """Example 7: Action masking to prevent invalid actions."""
    print("=" * 80)
    print("Example 7: Action Masking (Prevent Invalid Trades)")
    print("=" * 80)

    # Setup
    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)

    # Market data
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity"].symbol = ["AAPL", "GOOGL", "MSFT"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    # Empty portfolio (no positions)
    empty_portfolio = {}

    print("Portfolio State: EMPTY (no positions)")
    print()

    # Get actions with action masking
    actions = actor.get_action(
        data=data,
        portfolio_state=empty_portfolio,
        deterministic=True,
    )

    print("Actions Generated:")
    sell_count = 0
    for symbol, action in actions.items():
        action_type = ActionType(action['type'])
        if action_type == ActionType.SELL:
            sell_count += 1
        print(f"  {symbol}: {action_type.name}")

    print(f"\nSELL actions: {sell_count} (should be 0 with proper masking)")
    print("Action masking prevents selling assets we don't own!")
    print()


def example_8_architecture_summary():
    """Example 8: Getting architecture summary and statistics."""
    print("=" * 80)
    print("Example 8: Actor Architecture Summary")
    print("=" * 80)

    node_types = ["equity", "currency", "bonds", "options"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(
        metadata=metadata,
        gnn_hidden_dims=(512, 256, 128),
        gnn_embedding_dim=128,
        mem_heads=8,
        mem_size=64,
        action_hidden_dim=256,
    )

    summary = actor.get_architecture_summary()

    print("Architecture Summary:")
    print(f"  Total Parameters: {summary['total_parameters']:,}")
    print(f"  GNN Embedding Dim: {summary['gnn_embedding_dim']}")
    print(f"  Tradable Node Types: {summary['tradable_node_types']}")
    print(f"  Action Types: {summary['num_action_types']} (HOLD, BUY, SELL)")
    print(f"  Size Bins: {summary['num_size_bins']} (0%, 25%, 50%, 75%, 100%)")
    print()

    print("Parameters per Classifier:")
    for node_type in summary['action_type_classifiers']:
        action_params = summary['action_type_classifiers'][node_type]
        size_params = summary['action_size_classifiers'][node_type]
        total = action_params + size_params
        print(f"  {node_type:8s}: {total:,} ({action_params:,} + {size_params:,})")
    print()


def example_9_gradient_computation():
    """Example 9: Computing gradients for training."""
    print("=" * 80)
    print("Example 9: Gradient Computation for Training")
    print("=" * 80)

    # Setup
    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    actor = Actor(metadata=metadata, gnn_embedding_dim=64)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    # Market data
    data = HeteroData()
    data["equity"].x = torch.randn(2, 32)
    data["equity"].symbol = ["AAPL", "GOOGL"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 2, (2, 3))

    portfolio_state = {"AAPL": 0.2}

    print("Training Step:")

    # Forward pass
    actions, log_probs, entropy = actor.get_action_and_log_prob(
        data=data,
        portfolio_state=portfolio_state,
    )

    # Dummy loss (negative log likelihood)
    # In real training, this would be PPO loss or policy gradient loss
    loss = -log_probs.mean() - 0.01 * entropy.mean()  # Entropy bonus

    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    print(f"  Gradient Norm: {grad_norm:.4f}")

    # Update parameters
    optimizer.step()

    print("  Parameters updated!")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ACTOR NETWORK USAGE EXAMPLES" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    examples = [
        example_1_basic_usage,
        example_2_inference_mode,
        example_3_training_mode,
        example_4_ppo_update,
        example_5_transaction_conversion,
        example_6_temperature_exploration,
        example_7_action_masking,
        example_8_architecture_summary,
        example_9_gradient_computation,
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"Example {i} failed with error: {e}")
            print()

    print("=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
