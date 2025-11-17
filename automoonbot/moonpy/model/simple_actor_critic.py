"""
Simplified Actor-Critic Networks for Single-Stock Trading

These are simplified versions designed for:
- Single stock trading (not multi-asset portfolio)
- Continuous state space (price, volume, indicators)
- Discrete + continuous action space (BUY/SELL/HOLD + position size)
- PPO (Proximal Policy Optimization) training

Use these instead of the complex graph-based Actor for simple trading tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Tuple, Optional
import numpy as np


class SimpleActor(nn.Module):
    """
    Simple Actor network for single-stock trading.

    Architecture:
        Input (state) → FC layers → Action head + Size head

    Action Space:
        - Action: Discrete (HOLD=0, BUY=1, SELL=2)
        - Size: Continuous [0.0, 1.0] (fraction of portfolio/position)

    Args:
        state_dim: Dimension of state vector (e.g., 20 features)
        hidden_dims: List of hidden layer dimensions [256, 128]
        action_dim: Number of discrete actions (default: 3 = HOLD/BUY/SELL)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 128],
        action_dim: int = 3,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Action head (discrete: HOLD/BUY/SELL)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Position size head (continuous: 0.0 to 1.0)
        # Uses Beta distribution parameters (alpha, beta)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output: (alpha, beta) for Beta distribution
            nn.Softplus()  # Ensure positive values
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            action_logits: Logits for action distribution [batch_size, action_dim]
            alpha: Alpha parameter for Beta distribution [batch_size, 1]
            beta: Beta parameter for Beta distribution [batch_size, 1]
        """
        features = self.shared_net(state)

        # Action logits
        action_logits = self.action_head(features)

        # Beta distribution parameters for position size
        size_params = self.size_head(features)
        alpha = size_params[:, 0:1] + 1.0  # Add 1 to avoid alpha/beta = 0
        beta = size_params[:, 1:2] + 1.0

        return action_logits, alpha, beta

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, return most likely action (no sampling)

        Returns:
            action: Sampled action [batch_size]
            position_size: Sampled position size [batch_size]
            action_log_prob: Log probability of action [batch_size]
            size_log_prob: Log probability of size [batch_size]
        """
        action_logits, alpha, beta = self.forward(state)

        # Action distribution
        action_dist = Categorical(logits=action_logits)

        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()

        action_log_prob = action_dist.log_prob(action)

        # Position size distribution (Beta distribution)
        size_dist = Beta(alpha.squeeze(-1), beta.squeeze(-1))

        if deterministic:
            # Use mean of Beta distribution
            position_size = alpha / (alpha + beta)
            position_size = position_size.squeeze(-1)
        else:
            position_size = size_dist.sample()

        size_log_prob = size_dist.log_prob(position_size)

        return action, position_size, action_log_prob, size_log_prob

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        position_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.

        Used during training to compute PPO loss.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size]
            position_size: Position size tensor [batch_size]

        Returns:
            action_log_prob: Log probability of actions [batch_size]
            size_log_prob: Log probability of sizes [batch_size]
            entropy: Total entropy (action + size) [batch_size]
        """
        action_logits, alpha, beta = self.forward(state)

        # Action distribution
        action_dist = Categorical(logits=action_logits)
        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()

        # Size distribution
        size_dist = Beta(alpha.squeeze(-1), beta.squeeze(-1))
        size_log_prob = size_dist.log_prob(position_size)
        size_entropy = size_dist.entropy()

        # Total entropy
        entropy = action_entropy + size_entropy

        return action_log_prob, size_log_prob, entropy


class SimpleCritic(nn.Module):
    """
    Simple Critic network for value function estimation.

    Architecture:
        Input (state) → FC layers → Value output

    Args:
        state_dim: Dimension of state vector
        hidden_dims: List of hidden layer dimensions [256, 128]
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 128],
    ):
        super().__init__()

        self.state_dim = state_dim

        # Value network
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim

        # Final value head
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.value_net = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            value: State value [batch_size, 1]
        """
        return self.value_net(state)


class PPOBuffer:
    """
    Experience buffer for PPO training.

    Stores trajectories and computes advantages using GAE (Generalized Advantage Estimation).
    """

    def __init__(
        self,
        state_dim: int,
        buffer_size: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.state_dim = state_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.position_sizes = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.action_log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.size_log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def store(
        self,
        state: np.ndarray,
        action: int,
        position_size: float,
        reward: float,
        value: float,
        action_log_prob: float,
        size_log_prob: float,
        done: bool,
    ):
        """Store one timestep of experience."""
        assert self.ptr < self.buffer_size

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.position_sizes[self.ptr] = position_size
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.action_log_probs[self.ptr] = action_log_prob
        self.size_log_probs[self.ptr] = size_log_prob
        self.dones[self.ptr] = done

        self.ptr += 1

    def finish_path(self, last_value: float = 0.0):
        """
        Calculate advantages and returns for a finished trajectory.

        Uses GAE (Generalized Advantage Estimation).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)

        # Compute advantages using GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        advantages = np.zeros_like(deltas)
        last_gae = 0
        for t in reversed(range(len(deltas))):
            last_gae = deltas[t] + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        # Store in buffer (will be added as new attribute)
        if not hasattr(self, 'advantages'):
            self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
            self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + self.values[path_slice]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data from buffer and reset.

        Returns dict with all experience.
        """
        assert self.ptr == self.buffer_size

        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

        data = dict(
            states=self.states,
            actions=self.actions,
            position_sizes=self.position_sizes,
            returns=self.returns,
            advantages=self.advantages,
            action_log_probs=self.action_log_probs,
            size_log_probs=self.size_log_probs,
        )

        # Reset
        self.ptr = 0
        self.path_start_idx = 0

        return data


if __name__ == "__main__":
    # Test the networks
    print("Testing SimpleActor and SimpleCritic...")

    state_dim = 20
    batch_size = 32

    actor = SimpleActor(state_dim=state_dim)
    critic = SimpleCritic(state_dim=state_dim)

    # Random state
    state = torch.randn(batch_size, state_dim)

    # Test actor
    action, pos_size, action_log_prob, size_log_prob = actor.get_action(state)
    print(f"Actor output shapes:")
    print(f"  action: {action.shape}")
    print(f"  position_size: {pos_size.shape}")
    print(f"  action_log_prob: {action_log_prob.shape}")
    print(f"  size_log_prob: {size_log_prob.shape}")

    # Test critic
    value = critic(state)
    print(f"\nCritic output shape: {value.shape}")

    print("\n✓ Networks working correctly!")
