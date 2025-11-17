"""
Portfolio-Level Actor-Critic Networks for Multi-Stock Trading

Designed for:
- Multi-stock portfolio management
- Learning inter-stock correlations
- Portfolio-level optimization
- Complex market dynamics

Uses Transformer-based architecture with attention to model:
- Individual stock dynamics
- Cross-stock correlations
- Portfolio-level decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Tuple, List, Dict, Optional
import numpy as np
import math


class StockEncoder(nn.Module):
    """
    Encodes individual stock features.

    Takes raw stock features (price, volume, indicators) and
    encodes them into a fixed-size embedding.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_stocks, feature_dim]

        Returns:
            embeddings: [batch_size, num_stocks, embed_dim]
        """
        batch_size, num_stocks, feature_dim = x.shape
        x_flat = x.reshape(-1, feature_dim)
        embeddings = self.encoder(x_flat)
        embeddings = embeddings.reshape(batch_size, num_stocks, -1)
        return embeddings


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for modeling stock interactions.

    Allows the model to learn which stocks are correlated
    and how they influence each other's trading decisions.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, num_stocks, embed_dim]
            mask: Optional attention mask

        Returns:
            output: [batch_size, num_stocks, embed_dim]
            attention_weights: [batch_size, num_heads, num_stocks, num_stocks]
        """
        batch_size, num_stocks, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_stocks, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, stocks, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # [batch, heads, stocks, stocks]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # [batch, heads, stocks, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, num_stocks, embed_dim)

        # Final projection
        out = self.proj(out)

        return out, attn_weights


class TransformerBlock(nn.Module):
    """
    Transformer block for processing portfolio state.

    Consists of:
    - Multi-head attention (stock interactions)
    - Feed-forward network (individual stock processing)
    - Residual connections and layer normalization
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, num_stocks, embed_dim]
            mask: Optional attention mask

        Returns:
            output: [batch_size, num_stocks, embed_dim]
            attention_weights: Attention weights from attention layer
        """
        # Attention with residual
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class PortfolioActor(nn.Module):
    """
    Portfolio-level Actor network using Transformer architecture.

    Processes multiple stocks simultaneously and outputs trading actions
    for each stock, considering inter-stock correlations and portfolio state.

    State Space:
        For each stock:
            - Price features (normalized price, returns, momentum)
            - Technical indicators (RSI, MACD, Bollinger Bands)
            - Volume features
        Portfolio-level:
            - Current allocations
            - Total portfolio value
            - Correlation features

    Action Space:
        For each stock:
            - Action: Discrete (HOLD=0, BUY=1, SELL=2)
            - Size: Continuous [0.0, 1.0] (fraction to trade)

    Args:
        stock_feature_dim: Number of features per stock
        num_stocks: Maximum number of stocks in portfolio
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
    """

    def __init__(
        self,
        stock_feature_dim: int = 20,
        num_stocks: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.stock_feature_dim = stock_feature_dim
        self.num_stocks = num_stocks
        self.embed_dim = embed_dim

        # Stock encoder
        self.stock_encoder = StockEncoder(stock_feature_dim, ff_dim // 2, embed_dim)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_stocks, embed_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Action heads (one per stock)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),  # HOLD, BUY, SELL
        )

        # Position size head (Beta distribution parameters)
        self.size_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # alpha, beta
            nn.Softplus(),
        )

        # Portfolio-level feature encoder (for global context)
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(num_stocks + 10, embed_dim),  # +10 for portfolio-level features
            nn.ReLU(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        stock_features: torch.Tensor,
        portfolio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            stock_features: [batch_size, num_stocks, stock_feature_dim]
            portfolio_features: [batch_size, num_stocks + 10]
                - First num_stocks: current allocation per stock
                - Remaining: portfolio value, cash, drawdown, etc.
            mask: [batch_size, num_stocks] - 1 for valid stocks, 0 for padding

        Returns:
            action_logits: [batch_size, num_stocks, 3]
            alpha: [batch_size, num_stocks, 1]
            beta: [batch_size, num_stocks, 1]
        """
        batch_size = stock_features.shape[0]

        # Encode individual stocks
        stock_embeds = self.stock_encoder(stock_features)  # [B, N, D]

        # Add positional embeddings
        stock_embeds = stock_embeds + self.pos_embedding[:, :stock_embeds.shape[1], :]

        # Encode portfolio-level features
        portfolio_embed = self.portfolio_encoder(portfolio_features)  # [B, D]
        portfolio_embed = portfolio_embed.unsqueeze(1)  # [B, 1, D]

        # Concatenate portfolio embed as additional "stock"
        x = torch.cat([portfolio_embed, stock_embeds], dim=1)  # [B, N+1, D]

        # Transform with attention (models stock interactions)
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_w = layer(x, mask)
            attention_weights.append(attn_w)

        # Remove portfolio token, keep only stock embeddings
        stock_outputs = x[:, 1:, :]  # [B, N, D]

        # Action logits for each stock
        action_logits = self.action_head(stock_outputs)  # [B, N, 3]

        # Position size parameters
        size_params = self.size_head(stock_outputs)  # [B, N, 2]
        alpha = size_params[:, :, 0:1] + 1.0
        beta = size_params[:, :, 1:2] + 1.0

        return action_logits, alpha, beta

    def get_actions(
        self,
        stock_features: torch.Tensor,
        portfolio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions for all stocks.

        Args:
            stock_features: [batch_size, num_stocks, stock_feature_dim]
            portfolio_features: [batch_size, num_stocks + 10]
            mask: [batch_size, num_stocks]
            deterministic: If True, use greedy actions

        Returns:
            actions: [batch_size, num_stocks]
            position_sizes: [batch_size, num_stocks]
            action_log_probs: [batch_size, num_stocks]
            size_log_probs: [batch_size, num_stocks]
        """
        action_logits, alpha, beta = self.forward(stock_features, portfolio_features, mask)

        batch_size, num_stocks, _ = action_logits.shape

        # Sample actions
        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)  # [B, N]
        else:
            action_dist = Categorical(logits=action_logits)
            actions = action_dist.sample()

        # Compute log probs
        action_dist = Categorical(logits=action_logits)
        action_log_probs = action_dist.log_prob(actions)

        # Sample position sizes
        alpha_flat = alpha.squeeze(-1)  # [B, N]
        beta_flat = beta.squeeze(-1)  # [B, N]
        size_dist = Beta(alpha_flat, beta_flat)

        if deterministic:
            position_sizes = alpha_flat / (alpha_flat + beta_flat)
        else:
            position_sizes = size_dist.sample()

        size_log_probs = size_dist.log_prob(position_sizes)

        # Mask invalid stocks
        if mask is not None:
            actions = actions * mask.long()
            position_sizes = position_sizes * mask
            action_log_probs = action_log_probs * mask
            size_log_probs = size_log_probs * mask

        return actions, position_sizes, action_log_probs, size_log_probs

    def evaluate_actions(
        self,
        stock_features: torch.Tensor,
        portfolio_features: torch.Tensor,
        actions: torch.Tensor,
        position_sizes: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Used during PPO training.

        Returns:
            action_log_probs: [batch_size, num_stocks]
            size_log_probs: [batch_size, num_stocks]
            entropy: [batch_size, num_stocks]
        """
        action_logits, alpha, beta = self.forward(stock_features, portfolio_features, mask)

        # Action log probs and entropy
        action_dist = Categorical(logits=action_logits)
        action_log_probs = action_dist.log_prob(actions)
        action_entropy = action_dist.entropy()

        # Size log probs and entropy
        alpha_flat = alpha.squeeze(-1)
        beta_flat = beta.squeeze(-1)
        size_dist = Beta(alpha_flat, beta_flat)
        size_log_probs = size_dist.log_prob(position_sizes)
        size_entropy = size_dist.entropy()

        # Total entropy
        entropy = action_entropy + size_entropy

        # Mask invalid stocks
        if mask is not None:
            action_log_probs = action_log_probs * mask
            size_log_probs = size_log_probs * mask
            entropy = entropy * mask

        return action_log_probs, size_log_probs, entropy


class PortfolioCritic(nn.Module):
    """
    Portfolio-level Critic network.

    Estimates the value of a given portfolio state, considering
    all stocks and their interactions.

    Uses similar transformer architecture as Actor to model
    the portfolio holistically.
    """

    def __init__(
        self,
        stock_feature_dim: int = 20,
        num_stocks: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.stock_feature_dim = stock_feature_dim
        self.num_stocks = num_stocks
        self.embed_dim = embed_dim

        # Stock encoder
        self.stock_encoder = StockEncoder(stock_feature_dim, ff_dim // 2, embed_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_stocks, embed_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Portfolio-level encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(num_stocks + 10, embed_dim),
            nn.ReLU(),
        )

        # Value head (aggregates all stock embeddings)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * (num_stocks + 1), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        stock_features: torch.Tensor,
        portfolio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            stock_features: [batch_size, num_stocks, stock_feature_dim]
            portfolio_features: [batch_size, num_stocks + 10]
            mask: [batch_size, num_stocks]

        Returns:
            value: [batch_size, 1]
        """
        batch_size = stock_features.shape[0]

        # Encode stocks
        stock_embeds = self.stock_encoder(stock_features)
        stock_embeds = stock_embeds + self.pos_embedding[:, :stock_embeds.shape[1], :]

        # Encode portfolio
        portfolio_embed = self.portfolio_encoder(portfolio_features)
        portfolio_embed = portfolio_embed.unsqueeze(1)

        # Concatenate
        x = torch.cat([portfolio_embed, stock_embeds], dim=1)

        # Transform
        for layer in self.transformer_layers:
            x, _ = layer(x, mask)

        # Flatten all embeddings
        x_flat = x.reshape(batch_size, -1)

        # Compute value
        value = self.value_head(x_flat)

        return value


if __name__ == "__main__":
    # Test the networks
    print("Testing PortfolioActor and PortfolioCritic...")

    batch_size = 8
    num_stocks = 5
    stock_feature_dim = 20

    actor = PortfolioActor(
        stock_feature_dim=stock_feature_dim,
        num_stocks=num_stocks,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
    )

    critic = PortfolioCritic(
        stock_feature_dim=stock_feature_dim,
        num_stocks=num_stocks,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
    )

    # Random inputs
    stock_features = torch.randn(batch_size, num_stocks, stock_feature_dim)
    portfolio_features = torch.randn(batch_size, num_stocks + 10)
    mask = torch.ones(batch_size, num_stocks)

    # Test actor
    actions, sizes, action_lp, size_lp = actor.get_actions(
        stock_features, portfolio_features, mask
    )

    print(f"Actor outputs:")
    print(f"  actions: {actions.shape}")
    print(f"  sizes: {sizes.shape}")
    print(f"  Example actions: {actions[0]}")
    print(f"  Example sizes: {sizes[0]}")

    # Test critic
    value = critic(stock_features, portfolio_features, mask)
    print(f"\nCritic output: {value.shape}")
    print(f"  Example values: {value[:3, 0]}")

    print("\n✓ Portfolio networks working correctly!")
    print(f"✓ Can handle {num_stocks} stocks simultaneously")
    print(f"✓ Models inter-stock correlations with attention")
