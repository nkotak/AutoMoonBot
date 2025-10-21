"""
Critic Network Usage Examples

This file demonstrates how to use the Critic network for value estimation in AutoMoonBot.
The Critic network estimates state values V(s) or action-values Q(s,a) for reinforcement learning.

Author: AutoMoonBot Team
"""

import torch
from torch_geometric.data import HeteroData
from automoonbot.moonpy.model import Critic, PoolingStrategy, Actor


def example_1_basic_usage():
    """Example 1: Basic Critic initialization and value estimation."""
    print("=" * 80)
    print("Example 1: Basic Critic Usage - State Value Estimation")
    print("=" * 80)

    # Define graph metadata
    node_types = ["equity", "currency", "bonds", "options", "article", "company"]
    edge_types = [
        ("equity", "mentioned_in", "article"),
        ("article", "published_by", "company"),
        ("equity", "correlated_with", "equity"),
    ]
    metadata = (node_types, edge_types)

    # Initialize Critic for state-value estimation V(s)
    critic = Critic(
        metadata=metadata,
        gnn_hidden_dims=(512, 256, 128),
        gnn_embedding_dim=128,
        value_hidden_dims=(256, 128),
        pooling_strategy=PoolingStrategy.MEAN,
        num_critics=1,
        action_conditioned=False,  # State value V(s)
    )

    print(f"Critic initialized with {critic.get_num_parameters():,} parameters")
    print(f"Value estimation mode: V(s) - State Values")
    print()

    # Create sample market state
    data = HeteroData()
    data["equity"].x = torch.randn(5, 32)
    data["currency"].x = torch.randn(2, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 10))

    # Get state value estimate
    value = critic.get_value(data)

    print(f"State value estimate: {value.item():.4f}")
    print()


def example_2_action_value_critic():
    """Example 2: Action-value (Q-value) estimation."""
    print("=" * 80)
    print("Example 2: Action-Value Critic Q(s,a)")
    print("=" * 80)

    node_types = ["equity", "currency"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    # Initialize action-conditioned critic (Q-value)
    critic = Critic(
        metadata=metadata,
        gnn_embedding_dim=64,
        num_critics=1,
        action_conditioned=True,  # Action-value Q(s,a)
        action_dim=15,  # Dimension of action representation
    )

    print("Critic initialized for Q(s,a) estimation")
    print(f"Action dimension: {critic.action_dim}")
    print()

    # Create state
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    # Create action
    action = torch.randn(1, 15)

    # Get Q-value
    q_value = critic.get_value(data, actions=action)

    print(f"Q(s,a) estimate: {q_value.item():.4f}")
    print()


def example_3_twin_critics_td3():
    """Example 3: Twin critics for TD3 algorithm."""
    print("=" * 80)
    print("Example 3: Twin Critics (TD3 Style)")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    # Initialize twin critics (reduces overestimation bias)
    critic = Critic(
        metadata=metadata,
        gnn_embedding_dim=64,
        num_critics=2,  # Two critics
        action_conditioned=True,
        action_dim=10,
    )

    print("Twin critics initialized")
    print(f"Number of critic heads: {critic.num_critics}")
    print()

    # Create state and action
    data = HeteroData()
    data["equity"].x = torch.randn(2, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 2, (2, 3))

    action = torch.randn(1, 10)

    # Get minimum Q-value (standard TD3 approach)
    q_min = critic.get_value(data, actions=action)
    print(f"Minimum Q-value: {q_min.item():.4f}")

    # Get both Q-values separately
    q1, q2 = critic.get_all_values(data, actions=action)
    print(f"Q1: {q1.item():.4f}")
    print(f"Q2: {q2.item():.4f}")
    print(f"Min(Q1, Q2): {min(q1.item(), q2.item()):.4f}")
    print()


def example_4_gae_advantage_computation():
    """Example 4: Computing advantages with GAE for PPO."""
    print("=" * 80)
    print("Example 4: GAE Advantage Computation for PPO")
    print("=" * 80)

    node_types = ["equity", "currency"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Create trajectory data
    data = HeteroData()
    data["equity"].x = torch.randn(5, 32)
    data["currency"].x = torch.randn(2, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 10))

    T = 10  # Trajectory length
    states = [data for _ in range(T)]
    rewards = torch.randn(T)
    next_states = [data for _ in range(T)]
    dones = torch.zeros(T)
    dones[-1] = 1.0  # Terminal state

    print(f"Trajectory length: {T}")
    print(f"Rewards: {rewards[:3].tolist()} ...")
    print()

    # Compute GAE advantages
    advantages, returns = critic.compute_advantages(
        states=states,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
        normalize=True,
    )

    print("GAE Computation Results:")
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Returns shape: {returns.shape}")
    print(f"  Advantages (first 5): {advantages[:5].tolist()}")
    print(f"  Advantages mean: {advantages.mean().item():.4f} (should be ~0)")
    print(f"  Advantages std: {advantages.std().item():.4f} (should be ~1)")
    print()


def example_5_td_target_computation():
    """Example 5: Computing TD targets for value function updates."""
    print("=" * 80)
    print("Example 5: TD Target Computation")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Create batch of transitions
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    batch_size = 5
    rewards = torch.randn(batch_size)
    next_states = [data for _ in range(batch_size)]
    dones = torch.zeros(batch_size)

    # Compute TD targets
    td_targets = critic.compute_td_targets(
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        gamma=0.99,
        use_target_network=False,
    )

    print(f"Batch size: {batch_size}")
    print(f"TD targets: {td_targets.tolist()}")
    print(f"Formula: r + γ * V(s')")
    print()


def example_6_target_network():
    """Example 6: Using target networks for stable learning."""
    print("=" * 80)
    print("Example 6: Target Network for Stable TD Learning")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Create state
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    print("Step 1: Create target network")
    critic.create_target_network()
    print("  Target network created (frozen copy)")
    print()

    # Get initial values
    value_main = critic.get_value(data)
    value_target = critic.get_target_value(data)

    print(f"Step 2: Initial values")
    print(f"  Main network value: {value_main.item():.4f}")
    print(f"  Target network value: {value_target.item():.4f}")
    print(f"  Difference: {abs(value_main.item() - value_target.item()):.6f}")
    print()

    # Simulate training step (update main network)
    print("Step 3: Simulate training (update main network)")
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    dummy_loss = value_main.pow(2)
    optimizer.zero_grad()
    dummy_loss.backward()
    optimizer.step()
    print("  Main network updated")
    print()

    # Values should now differ
    value_main_after = critic.get_value(data)
    value_target_after = critic.get_target_value(data)

    print(f"Step 4: After main network update")
    print(f"  Main network value: {value_main_after.item():.4f}")
    print(f"  Target network value: {value_target_after.item():.4f}")
    print(f"  Difference: {abs(value_main_after.item() - value_target_after.item()):.6f}")
    print()

    # Soft update (Polyak averaging)
    print("Step 5: Soft update target network (tau=0.1)")
    critic.update_target_network(tau=0.1)

    value_target_soft = critic.get_target_value(data)
    print(f"  Target network value: {value_target_soft.item():.4f}")
    print(f"  Formula: θ_target = 0.1*θ_main + 0.9*θ_target")
    print()


def example_7_pooling_strategies():
    """Example 7: Different graph pooling strategies."""
    print("=" * 80)
    print("Example 7: Graph Pooling Strategies")
    print("=" * 80)

    node_types = ["equity", "currency"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    # Create state
    data = HeteroData()
    data["equity"].x = torch.randn(5, 32)
    data["currency"].x = torch.randn(2, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 5, (2, 10))

    strategies = [
        PoolingStrategy.MEAN,
        PoolingStrategy.MAX,
        PoolingStrategy.SUM,
        PoolingStrategy.ATTENTION,
    ]

    print("Comparing pooling strategies:")
    print()

    for strategy in strategies:
        critic = Critic(
            metadata=metadata,
            gnn_embedding_dim=64,
            pooling_strategy=strategy,
        )

        value = critic.get_value(data)

        print(f"{strategy.value.upper():12s}: V(s) = {value.item():+.4f}")

    print()
    print("Note: Different pooling strategies capture different graph properties")
    print("  - MEAN: Average of all node embeddings (balanced)")
    print("  - MAX: Maximum values (captures extremes)")
    print("  - SUM: Sum of embeddings (magnitude-sensitive)")
    print("  - ATTENTION: Learned importance weights (most flexible)")
    print()


def example_8_value_loss_computation():
    """Example 8: Computing value function loss for training."""
    print("=" * 80)
    print("Example 8: Value Function Loss")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Create trajectory
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    T = 10
    states = [data for _ in range(T)]
    rewards = torch.randn(T)
    next_states = [data for _ in range(T)]
    dones = torch.zeros(T)
    dones[-1] = 1.0

    # Compute target returns using GAE
    _, returns = critic.compute_advantages(
        states=states,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    print(f"Trajectory: {T} steps")
    print(f"Returns shape: {returns.shape}")
    print()

    # Compute value loss
    loss = critic.compute_value_loss(states, returns)

    print(f"Value Loss: {loss.item():.4f}")
    print("This loss would be used to update critic parameters")
    print()


def example_9_n_step_returns():
    """Example 9: N-step returns for improved learning."""
    print("=" * 80)
    print("Example 9: N-step Returns")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Create n-step trajectory
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    batch_size = 4
    n = 3  # 3-step returns
    rewards = torch.randn(batch_size, n)
    next_states = [data for _ in range(batch_size)]
    dones = torch.zeros(batch_size, n)

    print(f"Batch size: {batch_size}")
    print(f"N-step: {n}")
    print()

    # Compute n-step returns
    n_step_returns = critic.compute_n_step_returns(
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        gamma=0.99,
        n=n,
    )

    print(f"N-step returns: {n_step_returns.tolist()}")
    print(f"Formula: R^(n) = Σ(γ^i * r_i) + γ^n * V(s_n)")
    print()


def example_10_actor_critic_training():
    """Example 10: Complete Actor-Critic training loop."""
    print("=" * 80)
    print("Example 10: Actor-Critic Training Loop")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    # Initialize actor and critic
    actor = Actor(metadata=metadata, gnn_embedding_dim=64)
    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    print("Actor-Critic Setup:")
    print(f"  Actor parameters: {actor.get_num_parameters():,}")
    print(f"  Critic parameters: {critic.get_num_parameters():,}")
    print()

    # Create trajectory
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity"].symbol = ["AAPL", "GOOGL", "MSFT"]
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    T = 10
    states = [data for _ in range(T)]
    rewards = torch.randn(T)
    next_states = [data for _ in range(T)]
    dones = torch.zeros(T)
    dones[-1] = 1.0

    portfolio_state = {"AAPL": 0.2}

    print("Training Step:")
    print()

    # Step 1: Compute advantages (critic's role)
    print("1. Compute advantages using critic")
    advantages, returns = critic.compute_advantages(
        states=states,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        gamma=0.99,
        gae_lambda=0.95,
        normalize=True,
    )
    print(f"   Advantages computed: {advantages.shape}")
    print()

    # Step 2: Update critic (value function)
    print("2. Update critic (value function)")
    critic_loss = critic.compute_value_loss(states, returns)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    print(f"   Critic loss: {critic_loss.item():.4f}")
    print()

    # Step 3: Update actor (policy)
    print("3. Update actor (policy)")
    # Get actions and log probs
    actions, log_probs, entropy = actor.get_action_and_log_prob(
        data, portfolio_state=portfolio_state
    )

    # Policy gradient loss (simplified)
    # In PPO, this would use clipped objective
    actor_loss = -(log_probs[:len(advantages)] * advantages.detach()).mean()
    actor_loss -= 0.01 * entropy[:len(advantages)].mean()  # Entropy bonus

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    print(f"   Actor loss: {actor_loss.item():.4f}")
    print()

    print("Training step complete!")
    print()


def example_11_bellman_error_monitoring():
    """Example 11: Monitoring Bellman error for debugging."""
    print("=" * 80)
    print("Example 11: Bellman Error Monitoring")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    critic = Critic(metadata=metadata, gnn_embedding_dim=64)

    # Create trajectory
    data = HeteroData()
    data["equity"].x = torch.randn(3, 32)
    data["equity", "correlated_with", "equity"].edge_index = torch.randint(0, 3, (2, 5))

    T = 10
    states = [data for _ in range(T)]
    rewards = torch.randn(T)
    next_states = [data for _ in range(T)]
    dones = torch.zeros(T)

    # Compute Bellman error
    bellman_error = critic.compute_bellman_error(
        states=states,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        gamma=0.99,
    )

    print(f"Bellman Error: {bellman_error.item():.4f}")
    print()
    print("Bellman error = |V(s) - (r + γV(s'))|")
    print("Lower is better - indicates how well Bellman equation is satisfied")
    print("Useful for:")
    print("  - Monitoring learning progress")
    print("  - Detecting training instabilities")
    print("  - Debugging value function approximation")
    print()


def example_12_checkpoint_save_load():
    """Example 12: Saving and loading critic checkpoints."""
    print("=" * 80)
    print("Example 12: Checkpoint Save/Load")
    print("=" * 80)

    node_types = ["equity"]
    edge_types = [("equity", "correlated_with", "equity")]
    metadata = (node_types, edge_types)

    # Create and train critic
    critic = Critic(metadata=metadata, gnn_embedding_dim=64, num_critics=2)
    critic.create_target_network()

    print("Original critic:")
    summary = critic.get_architecture_summary()
    print(f"  Parameters: {summary['total_parameters']:,}")
    print(f"  Num critics: {summary['num_critics']}")
    print(f"  Has target network: {summary['has_target_network']}")
    print()

    # Save checkpoint
    checkpoint_path = "/tmp/critic_checkpoint.pt"
    critic.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
    print()

    # Load checkpoint
    loaded_critic = Critic.load_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully!")

    loaded_summary = loaded_critic.get_architecture_summary()
    print(f"  Parameters: {loaded_summary['total_parameters']:,}")
    print(f"  Num critics: {loaded_summary['num_critics']}")
    print(f"  Has target network: {loaded_summary['has_target_network']}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 19 + "CRITIC NETWORK USAGE EXAMPLES" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    examples = [
        example_1_basic_usage,
        example_2_action_value_critic,
        example_3_twin_critics_td3,
        example_4_gae_advantage_computation,
        example_5_td_target_computation,
        example_6_target_network,
        example_7_pooling_strategies,
        example_8_value_loss_computation,
        example_9_n_step_returns,
        example_10_actor_critic_training,
        example_11_bellman_error_monitoring,
        example_12_checkpoint_save_load,
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
