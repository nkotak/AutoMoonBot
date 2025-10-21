import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import to_hetero, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import copy

from automoonbot.moonpy.model import GATNet


class PoolingStrategy(Enum):
    """Graph pooling strategies for aggregating node embeddings."""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    ATTENTION = "attention"


class Critic(nn.Module):
    """
    Critic network for value function estimation in reinforcement learning.

    This critic processes market data as a heterogeneous graph and estimates:
    1. State values V(s) - Expected return from a state (for PPO, A2C)
    2. Action values Q(s,a) - Expected return from state-action pair (for SAC, TD3, DQN)

    The critic uses the same GNN backbone as the Actor for shared representations,
    then aggregates graph embeddings and passes through value head(s).

    Architecture:
        1. GATNet backbone for heterogeneous graph processing
        2. Graph pooling to aggregate node embeddings into global state
        3. Value head(s) for scalar value prediction
        4. Optional action conditioning for Q-value estimation
        5. Target network support for stable TD learning

    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        gnn_hidden_dims: Hidden dimensions for GNN layers [h1, h2, out]
        gnn_embedding_dim: Final embedding dimension from GNN
        value_hidden_dims: Hidden dimensions for value head layers
        pooling_strategy: Strategy for aggregating node embeddings
        num_critics: Number of parallel critics (e.g., 2 for TD3)
        action_conditioned: If True, critic estimates Q(s,a) instead of V(s)
        action_dim: Dimension of action input (required if action_conditioned=True)
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout probability for value head
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        gnn_hidden_dims: Tuple[int, int, int] = (512, 256, 128),
        gnn_embedding_dim: int = 128,
        value_hidden_dims: Tuple[int, ...] = (256, 128),
        pooling_strategy: Union[str, PoolingStrategy] = PoolingStrategy.MEAN,
        num_critics: int = 1,
        action_conditioned: bool = False,
        action_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Store configuration
        self.metadata = metadata
        self.gnn_embedding_dim = gnn_embedding_dim
        self.num_critics = num_critics
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm

        # Validate configuration
        if action_conditioned and action_dim is None:
            raise ValueError("action_dim must be specified when action_conditioned=True")

        # Parse pooling strategy
        if isinstance(pooling_strategy, str):
            self.pooling_strategy = PoolingStrategy(pooling_strategy)
        else:
            self.pooling_strategy = pooling_strategy

        # Graph Neural Network backbone
        self.gnn = GATNet(
            h1_dim=gnn_hidden_dims[0],
            h2_dim=gnn_hidden_dims[1],
            out_dim=gnn_hidden_dims[2],
        )
        self.gnn = to_hetero(self.gnn, metadata, aggr="sum")

        # Attention pooling (if using attention strategy)
        if self.pooling_strategy == PoolingStrategy.ATTENTION:
            self.attention_weights = nn.ModuleDict()
            for node_type in metadata[0]:
                self.attention_weights[node_type] = nn.Sequential(
                    nn.Linear(gnn_embedding_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 1),
                )

        # Calculate input dimension for value head
        # Global state = concatenation of pooled embeddings from all node types
        # We'll estimate this based on metadata (assume all node types present)
        num_node_types = len(metadata[0])
        global_state_dim = num_node_types * gnn_embedding_dim

        # Add action dimension if action-conditioned
        if action_conditioned:
            global_state_dim += action_dim

        # Build value heads (multiple for ensemble critics like TD3)
        self.value_heads = nn.ModuleList()
        for _ in range(num_critics):
            layers = []
            input_dim = global_state_dim

            # Hidden layers
            for hidden_dim in value_hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            # Output layer (scalar value)
            layers.append(nn.Linear(input_dim, 1))

            self.value_heads.append(nn.Sequential(*layers))

        # Target network (optional, for DQN/DDPG/TD3/SAC)
        self._target_network: Optional[nn.Module] = None

    def forward(
        self,
        data: HeteroData,
        actions: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
        return_all_critics: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Forward pass through the critic network.

        Args:
            data: HeteroData graph containing market state
            actions: Optional actions for Q-value estimation (required if action_conditioned=True)
            return_all_critics: If True and num_critics > 1, return all critic values separately

        Returns:
            values: Estimated state/action values
                   - If num_critics=1: Tensor [batch_size, 1]
                   - If num_critics>1 and return_all_critics=False: Tensor [batch_size, 1] (minimum)
                   - If num_critics>1 and return_all_critics=True: Tuple of Tensors
        """
        # Validate action input
        if self.action_conditioned and actions is None:
            raise ValueError("actions must be provided when action_conditioned=True")

        # Process graph through GNN to get node embeddings
        embeddings = self.gnn(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None,
        )

        # Pool embeddings to get global state representation
        global_state = self._pool_embeddings(embeddings, data)

        # Concatenate action if action-conditioned
        if self.action_conditioned:
            # Convert action dict to tensor if needed
            if isinstance(actions, dict):
                action_tensor = self._dict_to_tensor(actions, data)
            else:
                action_tensor = actions

            global_state = torch.cat([global_state, action_tensor], dim=-1)

        # Pass through value head(s)
        if self.num_critics == 1:
            value = self.value_heads[0](global_state)
            return value
        else:
            values = tuple(head(global_state) for head in self.value_heads)
            if return_all_critics:
                return values
            else:
                # Return minimum value (standard for TD3 to reduce overestimation)
                return torch.min(torch.stack(values, dim=0), dim=0)[0]

    def get_value(
        self,
        data: HeteroData,
        actions: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    ) -> Tensor:
        """
        Get value estimate V(s) or Q(s,a).

        Args:
            data: HeteroData graph containing market state
            actions: Optional actions (required if action_conditioned=True)

        Returns:
            value: Estimated value [batch_size, 1]
        """
        return self.forward(data, actions, return_all_critics=False)

    def get_all_values(
        self,
        data: HeteroData,
        actions: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, ...]:
        """
        Get values from all critic heads (for ensemble critics).

        Args:
            data: HeteroData graph containing market state
            actions: Optional actions (required if action_conditioned=True)

        Returns:
            values: Tuple of value estimates from each critic head
        """
        if self.num_critics == 1:
            return (self.forward(data, actions),)
        else:
            return self.forward(data, actions, return_all_critics=True)

    def compute_advantages(
        self,
        states: List[HeteroData],
        rewards: Tensor,
        next_states: List[HeteroData],
        dones: Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        GAE(�, �) = �(��)^t �_t where �_t = r_t + �V(s_{t+1}) - V(s_t)

        Args:
            states: List of HeteroData states [T]
            rewards: Rewards [T]
            next_states: List of next HeteroData states [T]
            dones: Done flags [T]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (trades off bias vs variance)
            normalize: Whether to normalize advantages (recommended for PPO)

        Returns:
            advantages: Computed advantages [T]
            returns: Computed returns (advantages + values) [T]
        """
        T = len(states)
        assert T == len(rewards) == len(next_states) == len(dones), \
            "All inputs must have same length"

        # Get values for all states
        with torch.no_grad():
            values = torch.stack([self.get_value(state).squeeze(-1) for state in states])
            next_values = torch.stack([self.get_value(state).squeeze(-1) for state in next_states])

        # Compute TD errors: delta_t = r_t + gamma*V(s_{t+1})(1 - done) - V(s_t)
        td_errors = rewards + gamma * next_values * (1 - dones) - values

        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        # Backward pass to compute GAE
        for t in reversed(range(T)):
            gae = td_errors[t] + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns (for value function update)
        returns = advantages + values

        # Normalize advantages (standard practice for PPO)
        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_td_targets(
        self,
        rewards: Tensor,
        next_states: List[HeteroData],
        dones: Tensor,
        gamma: float = 0.99,
        use_target_network: bool = True,
    ) -> Tensor:
        """
        Compute temporal difference (TD) targets.

        TD target: y = r + �V(s') * (1 - done)

        Args:
            rewards: Rewards [batch_size]
            next_states: List of next states
            dones: Done flags [batch_size]
            gamma: Discount factor
            use_target_network: Whether to use target network (if available)

        Returns:
            td_targets: Computed TD targets [batch_size]
        """
        with torch.no_grad():
            # Use target network if available and requested
            if use_target_network and self._target_network is not None:
                next_values = torch.stack([
                    self._target_network.get_value(state).squeeze(-1)
                    for state in next_states
                ])
            else:
                next_values = torch.stack([
                    self.get_value(state).squeeze(-1)
                    for state in next_states
                ])

            # TD target: r + �V(s')(1 - done)
            td_targets = rewards + gamma * next_values * (1 - dones)

        return td_targets

    def compute_n_step_returns(
        self,
        rewards: Tensor,
        next_states: List[HeteroData],
        dones: Tensor,
        gamma: float = 0.99,
        n: int = 3,
    ) -> Tensor:
        """
        Compute n-step returns for more stable learning.

        n-step return: R^(n) = �(�^i * r_{t+i}) + �^n * V(s_{t+n})

        Args:
            rewards: Rewards [batch_size, n]
            next_states: List of n-step next states
            dones: Done flags [batch_size, n]
            gamma: Discount factor
            n: Number of steps

        Returns:
            n_step_returns: Computed n-step returns [batch_size]
        """
        batch_size = rewards.shape[0]

        # Compute discounted sum of rewards
        discounted_rewards = torch.zeros(batch_size, device=rewards.device)
        gamma_pow = 1.0

        for i in range(n):
            discounted_rewards += gamma_pow * rewards[:, i] * (1 - dones[:, i])
            gamma_pow *= gamma

        # Add bootstrapped value from n-step next state
        with torch.no_grad():
            final_values = torch.stack([
                self.get_value(state).squeeze(-1)
                for state in next_states
            ])

            # Only add if episode didn't terminate
            final_term_mask = 1 - dones[:, -1]
            n_step_returns = discounted_rewards + (gamma ** n) * final_values * final_term_mask

        return n_step_returns

    def _pool_embeddings(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
    ) -> Tensor:
        """
        Pool node embeddings to create global state representation.

        Args:
            embeddings: Dict mapping node_type -> node_embeddings
            data: Original HeteroData (for attention pooling)

        Returns:
            global_state: Pooled global state representation [batch_size, global_dim]
        """
        pooled_embeddings = []

        for node_type in self.metadata[0]:
            if node_type not in embeddings or embeddings[node_type].size(0) == 0:
                # No nodes of this type - use zero vector
                pooled = torch.zeros(
                    1, self.gnn_embedding_dim,
                    device=next(self.parameters()).device
                )
            else:
                node_emb = embeddings[node_type]  # [num_nodes, embedding_dim]

                if self.pooling_strategy == PoolingStrategy.MEAN:
                    pooled = node_emb.mean(dim=0, keepdim=True)

                elif self.pooling_strategy == PoolingStrategy.MAX:
                    pooled = node_emb.max(dim=0, keepdim=True)[0]

                elif self.pooling_strategy == PoolingStrategy.SUM:
                    pooled = node_emb.sum(dim=0, keepdim=True)

                elif self.pooling_strategy == PoolingStrategy.ATTENTION:
                    # Attention-based pooling
                    attention_logits = self.attention_weights[node_type](node_emb)  # [num_nodes, 1]
                    attention_weights = torch.softmax(attention_logits, dim=0)
                    pooled = (node_emb * attention_weights).sum(dim=0, keepdim=True)

            pooled_embeddings.append(pooled)

        # Concatenate all pooled embeddings
        global_state = torch.cat(pooled_embeddings, dim=-1)  # [1, num_types * embedding_dim]

        return global_state

    def _dict_to_tensor(
        self,
        action_dict: Dict[str, Dict[str, Union[int, float]]],
        data: HeteroData,
    ) -> Tensor:
        """
        Convert action dictionary to tensor.

        Args:
            action_dict: Dict mapping symbols to action dicts
            data: HeteroData for reference

        Returns:
            action_tensor: Flattened action representation
        """
        # This is a simplified version - in practice, you'd want a more
        # sophisticated action encoding based on your action space
        action_values = []

        for symbol, action in action_dict.items():
            action_values.extend([
                float(action.get('type', 0)),
                float(action.get('size', 0)),
                float(action.get('fraction', 0.0)),
            ])

        # Pad to action_dim if needed
        action_tensor = torch.tensor(action_values, dtype=torch.float32).unsqueeze(0)

        if action_tensor.size(-1) < self.action_dim:
            padding = torch.zeros(1, self.action_dim - action_tensor.size(-1))
            action_tensor = torch.cat([action_tensor, padding], dim=-1)
        elif action_tensor.size(-1) > self.action_dim:
            action_tensor = action_tensor[:, :self.action_dim]

        return action_tensor

    def create_target_network(self) -> None:
        """
        Create a target network (deep copy of current network).

        Used in DQN, DDPG, TD3, SAC for stable TD learning.
        """
        self._target_network = copy.deepcopy(self)
        # Freeze target network parameters
        for param in self._target_network.parameters():
            param.requires_grad = False

    def update_target_network(self, tau: float = 1.0) -> None:
        """
        Update target network using Polyak averaging.

        �_target = � * �_current + (1 - �) * �_target

        Args:
            tau: Polyak averaging coefficient
                 �=1.0: Hard update (copy)
                 �<1.0: Soft update (exponential moving average)
        """
        if self._target_network is None:
            raise RuntimeError("Target network not created. Call create_target_network() first.")

        with torch.no_grad():
            for param, target_param in zip(
                self.parameters(),
                self._target_network.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    def get_target_value(
        self,
        data: HeteroData,
        actions: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
    ) -> Tensor:
        """
        Get value from target network.

        Args:
            data: HeteroData graph containing market state
            actions: Optional actions (required if action_conditioned=True)

        Returns:
            target_value: Value from target network
        """
        if self._target_network is None:
            raise RuntimeError("Target network not created. Call create_target_network() first.")

        return self._target_network.get_value(data, actions)

    def compute_value_loss(
        self,
        states: List[HeteroData],
        returns: Tensor,
        loss_fn: Optional[Callable] = None,
    ) -> Tensor:
        """
        Compute value function loss.

        Args:
            states: List of states
            returns: Target returns [batch_size]
            loss_fn: Loss function (default: MSE)

        Returns:
            loss: Value function loss
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Get predicted values
        predicted_values = torch.stack([
            self.get_value(state).squeeze(-1)
            for state in states
        ])

        # Compute loss
        loss = loss_fn(predicted_values, returns)

        return loss

    def compute_bellman_error(
        self,
        states: List[HeteroData],
        rewards: Tensor,
        next_states: List[HeteroData],
        dones: Tensor,
        gamma: float = 0.99,
    ) -> Tensor:
        """
        Compute Bellman error (for debugging/monitoring).

        Bellman error = |V(s) - (r + �V(s'))|

        Args:
            states: List of states
            rewards: Rewards [batch_size]
            next_states: List of next states
            dones: Done flags [batch_size]
            gamma: Discount factor

        Returns:
            bellman_error: Mean absolute Bellman error
        """
        with torch.no_grad():
            values = torch.stack([self.get_value(state).squeeze(-1) for state in states])
            next_values = torch.stack([self.get_value(state).squeeze(-1) for state in next_states])

            # TD targets
            targets = rewards + gamma * next_values * (1 - dones)

            # Bellman error
            bellman_error = torch.abs(values - targets).mean()

        return bellman_error

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> Dict[str, any]:
        """Get summary of critic architecture."""
        return {
            'total_parameters': self.get_num_parameters(),
            'gnn_embedding_dim': self.gnn_embedding_dim,
            'num_critics': self.num_critics,
            'action_conditioned': self.action_conditioned,
            'action_dim': self.action_dim if self.action_conditioned else None,
            'pooling_strategy': self.pooling_strategy.value,
            'use_layer_norm': self.use_layer_norm,
            'has_target_network': self._target_network is not None,
            'value_head_parameters': [
                sum(p.numel() for p in head.parameters())
                for head in self.value_heads
            ],
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save critic checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'metadata': self.metadata,
                'gnn_embedding_dim': self.gnn_embedding_dim,
                'num_critics': self.num_critics,
                'action_conditioned': self.action_conditioned,
                'action_dim': self.action_dim,
                'pooling_strategy': self.pooling_strategy.value,
                'use_layer_norm': self.use_layer_norm,
            }
        }

        if self._target_network is not None:
            checkpoint['target_state_dict'] = self._target_network.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> 'Critic':
        """
        Load critic from checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            critic: Loaded critic network
        """
        checkpoint = torch.load(path)
        config = checkpoint['config']

        # Parse pooling strategy
        config['pooling_strategy'] = PoolingStrategy(config['pooling_strategy'])

        # Create critic
        critic = cls(**config)

        # Load state dict
        critic.load_state_dict(checkpoint['state_dict'])

        # Load target network if present
        if 'target_state_dict' in checkpoint:
            critic.create_target_network()
            critic._target_network.load_state_dict(checkpoint['target_state_dict'])

        return critic
