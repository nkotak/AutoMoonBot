import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union
from enum import IntEnum

from automoonbot.moonpy.model import GATNet, MemoryClassifier


class ActionType(IntEnum):
    """Enumeration of possible trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


class ActionSize(IntEnum):
    """Discrete size bins for position sizing as percentages."""
    SIZE_0 = 0    # 0% - effectively HOLD
    SIZE_25 = 1   # 25% of available capital/position
    SIZE_50 = 2   # 50% of available capital/position
    SIZE_75 = 3   # 75% of available capital/position
    SIZE_100 = 4  # 100% of available capital/position


class Actor(nn.Module):
    """
    Actor network for trading agent using heterogeneous graph neural networks.

    This actor processes market data as a heterogeneous graph and outputs trading
    actions for multiple asset types (Equity, Currency, Bonds, Options).

    Action Space:
        - For each tradable asset:
            * Action Type: HOLD (0), BUY (1), SELL (2)
            * Action Size: 0%, 25%, 50%, 75%, 100% of capital/position

    Architecture:
        1. GATNet backbone for heterogeneous graph processing
        2. Separate action heads for each tradable node type
        3. Action masking for invalid actions (can't sell without position, etc.)
        4. Memory-augmented classifiers for action and size prediction

    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        gnn_hidden_dims: Hidden dimensions for GNN layers [h1, h2, out]
        gnn_embedding_dim: Final embedding dimension from GNN
        mem_heads: Number of attention heads in memory module
        mem_size: Size of learnable memory bank
        mem_dim: Dimension of memory vectors
        key_dim: Dimension of memory keys for attention
        val_dim: Dimension of memory values
        action_hidden_dim: Hidden dimension for action classifiers
        tradable_node_types: List of node types that can be traded
    """

    # Default tradable asset types in the graph
    TRADABLE_TYPES = ["equity", "currency", "bonds", "options"]

    # Action space dimensions
    NUM_ACTION_TYPES = len(ActionType)
    NUM_SIZE_BINS = len(ActionSize)

    # Size bin to actual fraction mapping
    SIZE_TO_FRACTION = {
        ActionSize.SIZE_0: 0.0,
        ActionSize.SIZE_25: 0.25,
        ActionSize.SIZE_50: 0.50,
        ActionSize.SIZE_75: 0.75,
        ActionSize.SIZE_100: 1.0,
    }

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        gnn_hidden_dims: Tuple[int, int, int] = (512, 256, 128),
        gnn_embedding_dim: int = 128,
        mem_heads: int = 4,
        mem_size: int = 32,
        mem_dim: int = 128,
        key_dim: int = 64,
        val_dim: int = 128,
        action_hidden_dim: int = 256,
        tradable_node_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        # Store configuration
        self.metadata = metadata
        self.gnn_embedding_dim = gnn_embedding_dim
        self.tradable_node_types = tradable_node_types or self.TRADABLE_TYPES

        # Validate tradable node types exist in metadata
        available_node_types = [nt.lower() for nt in metadata[0]]
        for node_type in self.tradable_node_types:
            if node_type.lower() not in available_node_types:
                raise ValueError(
                    f"Tradable node type '{node_type}' not found in graph metadata. "
                    f"Available types: {available_node_types}"
                )

        # Graph Neural Network backbone
        self.gnn = GATNet(
            h1_dim=gnn_hidden_dims[0],
            h2_dim=gnn_hidden_dims[1],
            out_dim=gnn_hidden_dims[2],
        )
        self.gnn = to_hetero(self.gnn, metadata, aggr="sum")

        # Action type classifiers for each tradable node type
        # These predict: HOLD, BUY, or SELL
        self.action_type_classifiers = nn.ModuleDict()
        for node_type in self.tradable_node_types:
            self.action_type_classifiers[node_type] = MemoryClassifier(
                inp_dim=gnn_embedding_dim,
                hdn_dim=action_hidden_dim,
                out_dim=self.NUM_ACTION_TYPES,
                mem_heads=mem_heads,
                mem_size=mem_size,
                mem_dim=mem_dim,
                key_dim=key_dim,
                val_dim=val_dim,
            )

        # Action size classifiers for each tradable node type
        # These predict: 0%, 25%, 50%, 75%, or 100%
        self.action_size_classifiers = nn.ModuleDict()
        for node_type in self.tradable_node_types:
            self.action_size_classifiers[node_type] = MemoryClassifier(
                inp_dim=gnn_embedding_dim,
                hdn_dim=action_hidden_dim,
                out_dim=self.NUM_SIZE_BINS,
                mem_heads=mem_heads,
                mem_size=mem_size,
                mem_dim=mem_dim,
                key_dim=key_dim,
                val_dim=val_dim,
            )

    def forward(
        self,
        data: HeteroData,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Forward pass through the actor network.

        Args:
            data: HeteroData graph containing market state
            deterministic: If True, take argmax actions (inference mode)
                          If False, sample from distributions (training mode)
            temperature: Temperature for action distribution sampling
                        Higher = more exploration, Lower = more exploitation

        Returns:
            action_logits: Dict mapping node_type -> (action_type_logits, action_size_logits)
            embeddings: Dict mapping node_type -> node_embeddings (for debugging/analysis)
        """
        # Process graph through GNN to get node embeddings
        embeddings = self.gnn(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None,
        )

        # Generate action logits for each tradable node type
        action_logits = {}

        for node_type in self.tradable_node_types:
            # Skip if this node type has no nodes in the current graph
            if node_type not in embeddings or embeddings[node_type].size(0) == 0:
                continue

            # Get embeddings for this node type
            node_emb = embeddings[node_type]  # [num_nodes, embedding_dim]

            # Predict action types (HOLD/BUY/SELL)
            action_type_logits = self.action_type_classifiers[node_type](node_emb)
            # Apply temperature scaling
            action_type_logits = action_type_logits / temperature

            # Predict action sizes (0%/25%/50%/75%/100%)
            action_size_logits = self.action_size_classifiers[node_type](node_emb)
            # Apply temperature scaling
            action_size_logits = action_size_logits / temperature

            action_logits[node_type] = {
                'action_type': action_type_logits,  # [num_nodes, 3]
                'action_size': action_size_logits,  # [num_nodes, 5]
            }

        return action_logits, embeddings

    def get_action(
        self,
        data: HeteroData,
        portfolio_state: Optional[Dict[str, float]] = None,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get actions for inference/execution.

        Args:
            data: HeteroData graph containing market state
            portfolio_state: Dict mapping asset symbols to current holdings (for action masking)
            deterministic: If True, take argmax actions. If False, sample.
            temperature: Temperature for sampling (ignored if deterministic=True)

        Returns:
            actions: Dict mapping asset symbols to action dicts:
                    {
                        'symbol': {
                            'type': int (0=HOLD, 1=BUY, 2=SELL),
                            'size': int (0-4, index into SIZE_TO_FRACTION),
                            'fraction': float (actual fraction 0.0-1.0),
                        }
                    }
        """
        # Get action logits from forward pass
        action_logits, embeddings = self.forward(data, deterministic=False, temperature=temperature)

        # Build action masks if portfolio state provided
        action_masks = self._create_action_masks(
            data, portfolio_state, action_logits
        ) if portfolio_state is not None else None

        # Convert logits to actions
        actions = {}

        for node_type, logits in action_logits.items():
            # Get node identifiers (symbols) for this node type
            if not hasattr(data[node_type], 'symbol'):
                continue  # Skip if no symbol attribute

            symbols = data[node_type].symbol  # List or tensor of symbols
            num_nodes = logits['action_type'].size(0)

            for i in range(num_nodes):
                symbol = symbols[i] if isinstance(symbols, list) else symbols[i].item()

                # Get logits for this specific node
                type_logits = logits['action_type'][i]  # [3]
                size_logits = logits['action_size'][i]  # [5]

                # Apply action mask if available
                if action_masks is not None and node_type in action_masks:
                    type_mask = action_masks[node_type]['action_type'][i]
                    size_mask = action_masks[node_type]['action_size'][i]

                    # Mask invalid actions by setting logits to -inf
                    type_logits = type_logits.masked_fill(~type_mask, float('-inf'))
                    size_logits = size_logits.masked_fill(~size_mask, float('-inf'))

                # Select action type
                if deterministic:
                    action_type = torch.argmax(type_logits).item()
                else:
                    type_dist = Categorical(logits=type_logits)
                    action_type = type_dist.sample().item()

                # Select action size (only relevant for BUY/SELL, but predict anyway)
                if deterministic:
                    action_size = torch.argmax(size_logits).item()
                else:
                    size_dist = Categorical(logits=size_logits)
                    action_size = size_dist.sample().item()

                # If action is HOLD, force size to 0
                if action_type == ActionType.HOLD:
                    action_size = ActionSize.SIZE_0

                # Convert to transaction format
                actions[symbol] = {
                    'type': action_type,
                    'size': action_size,
                    'fraction': self.SIZE_TO_FRACTION[ActionSize(action_size)],
                }

        return actions

    def get_action_and_log_prob(
        self,
        data: HeteroData,
        portfolio_state: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Dict[str, Union[int, float]]], Tensor, Tensor]:
        """
        Get actions and their log probabilities for training.

        Args:
            data: HeteroData graph containing market state
            portfolio_state: Dict mapping asset symbols to current holdings
            temperature: Temperature for action distribution sampling

        Returns:
            actions: Dict mapping asset symbols to action dicts
            log_probs: Tensor of log probabilities for selected actions [num_actions]
            entropy: Tensor of policy entropy [num_actions]
        """
        # Get action logits
        action_logits, embeddings = self.forward(data, deterministic=False, temperature=temperature)

        # Build action masks
        action_masks = self._create_action_masks(
            data, portfolio_state, action_logits
        ) if portfolio_state is not None else None

        # Sample actions and compute log probabilities
        actions = {}
        log_probs_list = []
        entropy_list = []

        for node_type, logits in action_logits.items():
            # Get node symbols
            if not hasattr(data[node_type], 'symbol'):
                continue

            symbols = data[node_type].symbol
            num_nodes = logits['action_type'].size(0)

            for i in range(num_nodes):
                symbol = symbols[i] if isinstance(symbols, list) else symbols[i].item()

                # Get logits for this node
                type_logits = logits['action_type'][i]  # [3]
                size_logits = logits['action_size'][i]  # [5]

                # Apply action masks
                if action_masks is not None and node_type in action_masks:
                    type_mask = action_masks[node_type]['action_type'][i]
                    size_mask = action_masks[node_type]['action_size'][i]

                    type_logits = type_logits.masked_fill(~type_mask, float('-inf'))
                    size_logits = size_logits.masked_fill(~size_mask, float('-inf'))

                # Create distributions
                type_dist = Categorical(logits=type_logits)
                size_dist = Categorical(logits=size_logits)

                # Sample actions
                action_type = type_dist.sample()
                action_size = size_dist.sample()

                # Force size to 0 if HOLD
                if action_type == ActionType.HOLD:
                    action_size = torch.tensor(ActionSize.SIZE_0, device=action_size.device)

                # Compute log probabilities
                type_log_prob = type_dist.log_prob(action_type)
                size_log_prob = size_dist.log_prob(action_size)

                # Joint log probability
                joint_log_prob = type_log_prob + size_log_prob
                log_probs_list.append(joint_log_prob)

                # Compute entropy for exploration bonus
                type_entropy = type_dist.entropy()
                size_entropy = size_dist.entropy()
                entropy_list.append(type_entropy + size_entropy)

                # Store action
                actions[symbol] = {
                    'type': action_type.item(),
                    'size': action_size.item(),
                    'fraction': self.SIZE_TO_FRACTION[ActionSize(action_size.item())],
                }

        # Stack log probs and entropy
        log_probs = torch.stack(log_probs_list) if log_probs_list else torch.tensor([])
        entropy = torch.stack(entropy_list) if entropy_list else torch.tensor([])

        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        data: HeteroData,
        actions: Dict[str, Dict[str, int]],
        portfolio_state: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate log probabilities and entropy of given actions (for PPO updates).

        Args:
            data: HeteroData graph containing market state
            actions: Dict mapping asset symbols to action dicts with 'type' and 'size'
            portfolio_state: Dict mapping asset symbols to current holdings
            temperature: Temperature for action distributions

        Returns:
            log_probs: Log probabilities of the provided actions [num_actions]
            entropy: Entropy of action distributions [num_actions]
            values: Action values (for advantage estimation) [num_actions]
        """
        # Get action logits
        action_logits, embeddings = self.forward(data, deterministic=False, temperature=temperature)

        # Build action masks
        action_masks = self._create_action_masks(
            data, portfolio_state, action_logits
        ) if portfolio_state is not None else None

        # Evaluate log probabilities for provided actions
        log_probs_list = []
        entropy_list = []
        values_list = []

        for node_type, logits in action_logits.items():
            if not hasattr(data[node_type], 'symbol'):
                continue

            symbols = data[node_type].symbol
            num_nodes = logits['action_type'].size(0)

            for i in range(num_nodes):
                symbol = symbols[i] if isinstance(symbols, list) else symbols[i].item()

                # Skip if no action for this symbol
                if symbol not in actions:
                    continue

                # Get logits
                type_logits = logits['action_type'][i]
                size_logits = logits['action_size'][i]

                # Apply masks
                if action_masks is not None and node_type in action_masks:
                    type_mask = action_masks[node_type]['action_type'][i]
                    size_mask = action_masks[node_type]['action_size'][i]

                    type_logits = type_logits.masked_fill(~type_mask, float('-inf'))
                    size_logits = size_logits.masked_fill(~size_mask, float('-inf'))

                # Create distributions
                type_dist = Categorical(logits=type_logits)
                size_dist = Categorical(logits=size_logits)

                # Get provided actions
                action_type = torch.tensor(actions[symbol]['type'], device=type_logits.device)
                action_size = torch.tensor(actions[symbol]['size'], device=size_logits.device)

                # Evaluate log probabilities
                type_log_prob = type_dist.log_prob(action_type)
                size_log_prob = size_dist.log_prob(action_size)
                joint_log_prob = type_log_prob + size_log_prob

                log_probs_list.append(joint_log_prob)

                # Compute entropy
                type_entropy = type_dist.entropy()
                size_entropy = size_dist.entropy()
                entropy_list.append(type_entropy + size_entropy)

                # Action value (max log prob)
                max_type_logit = type_logits.max()
                max_size_logit = size_logits.max()
                values_list.append(max_type_logit + max_size_logit)

        log_probs = torch.stack(log_probs_list) if log_probs_list else torch.tensor([])
        entropy = torch.stack(entropy_list) if entropy_list else torch.tensor([])
        values = torch.stack(values_list) if values_list else torch.tensor([])

        return log_probs, entropy, values

    def _create_action_masks(
        self,
        data: HeteroData,
        portfolio_state: Dict[str, float],
        action_logits: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Create action masks to prevent invalid actions.

        Rules:
            - Cannot SELL if position is 0
            - Cannot BUY with size > 0 if no capital available
            - Cannot use size > current position when SELLING

        Args:
            data: HeteroData graph
            portfolio_state: Dict mapping symbols to position sizes (fraction of portfolio)
            action_logits: Action logits from forward pass

        Returns:
            action_masks: Dict mapping node_type -> {
                'action_type': BoolTensor [num_nodes, 3],
                'action_size': BoolTensor [num_nodes, 5]
            }
        """
        action_masks = {}

        for node_type, logits in action_logits.items():
            if not hasattr(data[node_type], 'symbol'):
                continue

            symbols = data[node_type].symbol
            num_nodes = logits['action_type'].size(0)
            device = logits['action_type'].device

            # Initialize masks (True = allowed, False = forbidden)
            type_mask = torch.ones((num_nodes, self.NUM_ACTION_TYPES), dtype=torch.bool, device=device)
            size_mask = torch.ones((num_nodes, self.NUM_SIZE_BINS), dtype=torch.bool, device=device)

            for i in range(num_nodes):
                symbol = symbols[i] if isinstance(symbols, list) else symbols[i].item()

                # Get current position (default to 0 if not in portfolio)
                current_position = portfolio_state.get(symbol, 0.0)

                # Rule 1: Cannot SELL if position is 0
                if current_position <= 0.0:
                    type_mask[i, ActionType.SELL] = False
                    # If can't sell, all non-zero sizes are invalid when SELL is selected
                    # (this is handled by forcing size=0 when action=HOLD)

                # Rule 2: Cannot BUY if portfolio is fully allocated
                # (This requires knowing total portfolio allocation, simplified here)
                # For now, always allow BUY (environment will handle capital constraints)

                # Rule 3: HOLD action should always be valid
                type_mask[i, ActionType.HOLD] = True

                # Rule 4: SIZE_0 should always be valid (represents no action)
                size_mask[i, ActionSize.SIZE_0] = True

                # Rule 5: If position is small, cannot sell 100%
                # (Simplified: allow all sizes, environment handles constraints)

            action_masks[node_type] = {
                'action_type': type_mask,
                'action_size': size_mask,
            }

        return action_masks

    def to_transaction_format(
        self,
        actions: Dict[str, Dict[str, Union[int, float]]],
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Convert actions to Portfolio transaction format.

        Args:
            actions: Dict mapping symbols to action dicts

        Returns:
            transactions: List of transaction dicts for Portfolio._build_transaction()
                         [{'type': 'buy'/'sell', 'asset': symbol, 'size': fraction}, ...]
        """
        transactions = []

        for symbol, action in actions.items():
            action_type = ActionType(action['type'])
            fraction = action['fraction']

            # Skip HOLD actions
            if action_type == ActionType.HOLD or fraction == 0.0:
                continue

            # Convert to transaction format
            transaction = {
                'type': 'buy' if action_type == ActionType.BUY else 'sell',
                'asset': symbol,
                'size': fraction,
            }

            transactions.append(transaction)

        return transactions

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> Dict[str, any]:
        """Get summary of actor architecture."""
        return {
            'total_parameters': self.get_num_parameters(),
            'gnn_embedding_dim': self.gnn_embedding_dim,
            'tradable_node_types': self.tradable_node_types,
            'num_action_types': self.NUM_ACTION_TYPES,
            'num_size_bins': self.NUM_SIZE_BINS,
            'action_type_classifiers': {
                nt: sum(p.numel() for p in clf.parameters())
                for nt, clf in self.action_type_classifiers.items()
            },
            'action_size_classifiers': {
                nt: sum(p.numel() for p in clf.parameters())
                for nt, clf in self.action_size_classifiers.items()
            },
        }
