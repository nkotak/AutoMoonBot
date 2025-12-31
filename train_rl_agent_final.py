#!/usr/bin/env python3
"""
AutoMoonBot: RL Agent Training Script (Production - Hybrid Best of Both)

Features:
- Proper PPO with GAE (Generalized Advantage Estimation)
- Multi-epoch updates with mini-batch SGD
- Full 20-dimensional state features
- Multi-ticker support
- MPS/CUDA/CPU device support (with macOS deadlock fix)
- Risk-aware rewards with drawdown penalty
- Vectorized environment for performance

Usage:
    python train_rl_agent_final.py --tickers HESM AAPL --episodes 1000
    python train_rl_agent_final.py --tickers HESM --episodes 500 --lr 0.0003
"""

import sys
import os

# ============================================================================
# STEP 1: AGGRESSIVE MPS Deadlock Prevention
# ============================================================================
# Set EVERY possible environment variable to prevent MPS initialization
print("Initializing environment...", flush=True)
os.environ['PYTORCH_MPS_ENABLED'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'  # Don't fall back either
os.environ['DISABLE_MPS'] = '1'
os.environ['PYTORCH_DISABLE_MPS'] = '1'
# Also prevent multithreading during import
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
print("✓ MPS aggressively disabled during imports", flush=True)

import argparse
from datetime import datetime
from pathlib import Path

print("Loading libraries...", flush=True)
import numpy as np
print("  ✓ numpy", flush=True)

import pandas as pd
print("  ✓ pandas", flush=True)

# Import torch and IMMEDIATELY force CPU mode before it can initialize MPS
import torch
torch.set_default_device('cpu')  # Force CPU mode
print("  ✓ torch (forced to CPU during imports)", flush=True)

import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
print("  ✓ tensorboard", flush=True)

from automoonbot.moonpy.model.simple_actor_critic import SimpleActor, SimpleCritic, PPOBuffer
print("  ✓ AutoMoonBot modules", flush=True)

# ============================================================================
# STEP 2: Re-enable MPS After Imports
# ============================================================================
print("\nRe-enabling MPS for training...", flush=True)
del os.environ['PYTORCH_MPS_ENABLED']
del os.environ['PYTORCH_ENABLE_MPS_FALLBACK']
del os.environ['DISABLE_MPS']
del os.environ['PYTORCH_DISABLE_MPS']

# Reset threading
del os.environ['OMP_NUM_THREADS']
del os.environ['MKL_NUM_THREADS']

# Detect best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (GPU acceleration enabled!)", flush=True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU", flush=True)
else:
    device = torch.device("cpu")
    print("✓ Using CPU", flush=True)

print(f"  Device: {device}\n", flush=True)


# ============================================================================
# Data Download & Technical Indicators
# ============================================================================

def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download stock data with caching."""
    import yfinance as yf

    # Simple file-based caching
    cache_file = Path(f".cache_{ticker}_{period}.csv")
    if cache_file.exists():
        print(f"  Loading {ticker} from cache...", flush=True)
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"  ✓ {ticker}: {len(df)} days (cached)", flush=True)
        return df

    print(f"  Downloading {ticker} ({period})...", flush=True)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")

    # Save to cache
    df.to_csv(cache_file)
    print(f"  ✓ {ticker}: {len(df)} days", flush=True)
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all 20 technical indicators."""
    # Returns
    df['returns'] = df['Close'].pct_change()

    # Moving averages
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()

    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    return df.dropna()


def prepare_state(df: pd.DataFrame, idx: int, position: float = 0.0,
                 unrealized_pnl: float = 0.0, days_held: int = 0) -> np.ndarray:
    """
    Prepare full 20-dimensional state vector.

    Features:
    [0-10]: Market features (price, volume, indicators)
    [11-14]: Price history (last 4 returns)
    [15-16]: Position info (size, unrealized PnL)
    [17]: Days held
    [18-19]: Reserved for future use
    """
    row = df.iloc[idx]
    close = row['Close']
    state = np.zeros(20, dtype=np.float32)

    # Market features
    state[0] = row['returns'] if not pd.isna(row['returns']) else 0.0
    state[1] = np.clip(row['volume_ratio'], 0, 5) / 5 if not pd.isna(row['volume_ratio']) else 0.5
    state[2] = df['Close'].pct_change(5).iloc[idx] if idx >= 5 else 0.0
    state[3] = df['Close'].pct_change(20).iloc[idx] if idx >= 20 else 0.0
    state[4] = row['rsi'] / 100 if not pd.isna(row['rsi']) else 0.5
    state[5] = np.clip(row['macd'] / close, -0.1, 0.1) * 10 if not pd.isna(row['macd']) else 0.0
    state[6] = np.clip(row['bb_position'], 0, 1) if not pd.isna(row['bb_position']) else 0.5
    state[7] = np.clip(row['volatility_20'], 0, 0.1) * 10 if not pd.isna(row['volatility_20']) else 0.0
    state[8] = (close - row['ma_20']) / close if not pd.isna(row['ma_20']) else 0.0
    state[9] = (close - row['ma_50']) / close if not pd.isna(row['ma_50']) else 0.0
    state[10] = np.clip(row['volume_ratio'], 0, 3) / 3 if not pd.isna(row['volume_ratio']) else 0.5

    # Price history (last 4 returns)
    for i in range(4):
        if idx >= i + 1:
            state[11 + i] = df['returns'].iloc[idx - i]

    # Position features
    state[15] = position
    state[16] = np.clip(unrealized_pnl, -1, 1)
    state[17] = min(days_held / 100.0, 1.0)

    return state


# ============================================================================
# Trading Environment
# ============================================================================

class TradingEnvironment:
    """
    Vectorized trading environment with risk management.

    Features:
    - Pre-calculated technical indicators
    - Transaction costs
    - Drawdown penalty
    - Position holding penalty
    """

    def __init__(self, df: pd.DataFrame, ticker: str, initial_cash: float = 100000.0,
                 transaction_cost: float = 0.001, max_steps: int = 252):
        self.df = df
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps

        # Pre-calculate price array for faster access
        self.prices = df['Close'].values
        self.n_samples = len(df)

        self.reset()

    def reset(self, start_idx: int = None):
        """Reset environment to random starting point."""
        if start_idx is None:
            max_start = self.n_samples - self.max_steps - 1
            self.start_idx = np.random.randint(50, max(51, max_start))
        else:
            self.start_idx = start_idx

        self.current_idx = self.start_idx
        self.cash = self.initial_cash
        self.position = 0.0
        self.entry_price = 0.0
        self.days_held = 0
        self.portfolio_value = self.cash
        self.peak_value = self.initial_cash
        self.step_count = 0

        return self._get_state()

    def _get_state(self):
        """Get current state vector."""
        unrealized_pnl = 0.0
        if self.position > 0:
            current_price = self.prices[self.current_idx]
            unrealized_pnl = (current_price - self.entry_price) * self.position / self.initial_cash

        state = prepare_state(
            self.df,
            self.current_idx,
            position=self.position / (self.initial_cash / self.prices[self.current_idx]),
            unrealized_pnl=unrealized_pnl,
            days_held=self.days_held
        )

        return state

    def step(self, action: int, position_size: float):
        """
        Execute one trading step.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY
            position_size: Fraction (0.0 to 1.0) of cash/position to use

        Returns:
            next_state, reward, done, info
        """
        current_price = self.prices[self.current_idx]
        reward = 0.0

        # Execute action
        if action == 2 and self.cash > 0:  # BUY
            invest_amount = self.cash * position_size
            shares = invest_amount / current_price
            cost = shares * current_price * (1 + self.transaction_cost)

            if cost <= self.cash:
                if self.position == 0:
                    self.entry_price = current_price
                else:
                    # Average entry price for additional buys
                    total_value = self.position * self.entry_price + shares * current_price
                    self.entry_price = total_value / (self.position + shares)

                self.position += shares
                self.cash -= cost
                self.days_held = 0

        elif action == 0 and self.position > 0:  # SELL
            sell_amount = self.position * position_size
            proceeds = sell_amount * current_price * (1 - self.transaction_cost)
            pnl = (current_price - self.entry_price) * sell_amount

            self.cash += proceeds
            self.position -= sell_amount
            reward = pnl / self.initial_cash  # Immediate reward for realized PnL

            if self.position < 0.01:
                self.position = 0
                self.entry_price = 0
                self.days_held = 0

        elif action == 1:  # HOLD
            if self.position > 0:
                self.days_held += 1
                # Penalty for holding too long
                if self.days_held > 50:
                    reward -= 0.001

        # Calculate portfolio value
        position_value = self.position * current_price
        self.portfolio_value = self.cash + position_value

        # Update peak and calculate drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Drawdown penalty (risk management)
        if current_drawdown > 0.10:
            reward -= current_drawdown * 0.5

        # Portfolio return reward
        portfolio_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        reward += portfolio_return * 0.01

        # Move to next timestep
        self.current_idx += 1
        self.step_count += 1

        # Check if episode is done
        done = (
            self.step_count >= self.max_steps or
            self.current_idx >= self.n_samples - 1 or
            self.portfolio_value < self.initial_cash * 0.5  # 50% loss = terminal
        )

        next_state = self._get_state()

        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'return': portfolio_return,
        }

        return next_state, reward, done, info


# ============================================================================
# PPO Training Loop
# ============================================================================

def ppo_update(
    actor: SimpleActor,
    critic: SimpleCritic,
    optimizer: optim.Optimizer,
    buffer: PPOBuffer,
    batch_size: int = 64,
    n_epochs: int = 10,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: torch.device = torch.device('cpu'),
):
    """
    Perform PPO update with mini-batch SGD.

    This implements the proper PPO algorithm:
    1. Get data from buffer (with GAE-computed advantages)
    2. For multiple epochs:
        - Shuffle data
        - Update on mini-batches
        - Clip policy ratio for stability
        - Include entropy bonus for exploration
    """
    # Get data from buffer
    data = buffer.get()

    # Convert to tensors
    states = torch.FloatTensor(data['states']).to(device)
    actions = torch.LongTensor(data['actions']).to(device)
    position_sizes = torch.FloatTensor(data['position_sizes']).to(device)
    old_action_log_probs = torch.FloatTensor(data['action_log_probs']).to(device)
    old_size_log_probs = torch.FloatTensor(data['size_log_probs']).to(device)
    returns = torch.FloatTensor(data['returns']).to(device)
    advantages = torch.FloatTensor(data['advantages']).to(device)

    n_samples = len(states)

    # Training metrics
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0

    # Multiple epochs of SGD
    for epoch in range(n_epochs):
        # Shuffle indices
        indices = torch.randperm(n_samples)

        # Mini-batch updates
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            mb_indices = indices[start:end]

            # Get mini-batch
            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_sizes = position_sizes[mb_indices]
            mb_old_action_log_probs = old_action_log_probs[mb_indices]
            mb_old_size_log_probs = old_size_log_probs[mb_indices]
            mb_returns = returns[mb_indices]
            mb_advantages = advantages[mb_indices]

            # Evaluate actions with current policy
            new_action_log_probs, new_size_log_probs, entropy = actor.evaluate_actions(
                mb_states, mb_actions, mb_sizes
            )

            # Policy loss (PPO clipped objective)
            # Separate clipping for action and size
            ratio_action = (new_action_log_probs - mb_old_action_log_probs).exp()
            ratio_size = (new_size_log_probs - mb_old_size_log_probs).exp()

            # Combined ratio (product of independent probabilities)
            ratio = ratio_action * ratio_size

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = critic(mb_states).squeeze()
            value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

            optimizer.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    # Return average losses
    return {
        'policy_loss': total_policy_loss / n_updates,
        'value_loss': total_value_loss / n_updates,
        'entropy': total_entropy / n_updates,
    }


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train RL trading agent with PPO')
    parser.add_argument('--tickers', nargs='+', required=True, help='Stock tickers (e.g., HESM AAPL)')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='PPO mini-batch size')
    parser.add_argument('--buffer-size', type=int, default=2048, help='Rollout buffer size')
    parser.add_argument('--ppo-epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--save-freq', type=int, default=50, help='Save every N episodes')
    parser.add_argument('--data-period', type=str, default='2y', help='Data period (e.g., 2y, 5y)')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("AUTOMOONBOT: PPO TRAINING (HYBRID VERSION)")
    print("=" * 80)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log_dir = output_dir / 'logs' / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Configuration:")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  PPO Epochs: {args.ppo_epochs}")
    print(f"  Buffer Size: {args.buffer_size}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Logs: {log_dir}")
    print()

    # Download data
    print("Downloading market data...")
    ticker_data = {}
    for ticker in args.tickers:
        try:
            df = download_stock_data(ticker, period=args.data_period)
            df = calculate_technical_indicators(df)
            ticker_data[ticker] = df
        except Exception as e:
            print(f"  ✗ Failed: {ticker}: {e}", flush=True)

    if len(ticker_data) == 0:
        print("\n✗ No data downloaded. Exiting.", flush=True)
        sys.exit(1)

    print(f"\n✓ Downloaded {len(ticker_data)} tickers!\n", flush=True)

    # Initialize networks
    print("Initializing networks...", flush=True)
    actor = SimpleActor(state_dim=20).to(device)
    critic = SimpleCritic(state_dim=20).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Actor: {actor_params:,} parameters", flush=True)
    print(f"  Critic: {critic_params:,} parameters", flush=True)
    print(f"  Models on: {device}", flush=True)

    # Single optimizer for both networks
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr
    )

    start_episode = 0
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}", flush=True)
        checkpoint = torch.load(args.resume)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint.get('episode', 0)
        print(f"✓ Resumed from episode {start_episode}", flush=True)

    print(f"\n✓ Ready to train!\n", flush=True)

    # Training loop
    print("=" * 80)
    print("TRAINING WITH PPO + GAE")
    print("=" * 80)
    print()

    buffer = PPOBuffer(state_dim=20, buffer_size=args.buffer_size)

    for episode in range(start_episode, start_episode + args.episodes):
        # Select random ticker
        ticker = np.random.choice(list(ticker_data.keys()))
        df = ticker_data[ticker]

        # Create environment
        env = TradingEnvironment(df=df, ticker=ticker)
        state = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        # Rollout: Collect experience
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, position_size, action_log_prob, size_log_prob = actor.get_action(state_tensor)
                value = critic(state_tensor)

            next_state, reward, done, info = env.step(action.item(), position_size.item())

            buffer.store(
                state, action.item(), position_size.item(), reward,
                value.item(), action_log_prob.item(), size_log_prob.item(), done
            )

            episode_reward += reward
            episode_steps += 1
            state = next_state

            # Perform PPO update when buffer is full
            if buffer.ptr == args.buffer_size:
                # Bootstrap value if not done
                last_val = 0.0
                if not done:
                    with torch.no_grad():
                        next_val_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                        last_val = critic(next_val_tensor).item()
                
                buffer.finish_path(last_val)
                
                losses = ppo_update(
                    actor, critic, optimizer, buffer,
                    batch_size=args.batch_size,
                    n_epochs=args.ppo_epochs,
                    device=device
                )

                # Log training metrics
                writer.add_scalar('train/policy_loss', losses['policy_loss'], episode)
                writer.add_scalar('train/value_loss', losses['value_loss'], episode)
                writer.add_scalar('train/entropy', losses['entropy'], episode)

            if done:
                # Only finish path if buffer has data (might have just been reset)
                if buffer.ptr > 0:
                    buffer.finish_path(0.0)
                break

        # Log episode metrics
        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/portfolio_value', info['portfolio_value'], episode)
        writer.add_scalar('episode/return', info['return'], episode)
        writer.add_scalar('episode/steps', episode_steps, episode)

        # Print progress
        if episode % 10 == 0 or episode == start_episode:
            print(f"Episode {episode}/{start_episode + args.episodes}")
            print(f"  Ticker: {ticker}")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Portfolio: ${info['portfolio_value']:,.0f} ({info['return']:+.2%})")
            print(f"  Steps: {episode_steps}")
            print(flush=True)

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0 and episode > start_episode:
            checkpoint_path = output_dir / f'checkpoint_ep{episode+1}.pth'
            torch.save({
                'episode': episode + 1,
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'optimizer': optimizer.state_dict(),
                'tickers': list(ticker_data.keys()),
            }, checkpoint_path)
            print(f"  ✓ Saved: {checkpoint_path}\n", flush=True)

    # Save final model
    final_path = output_dir / 'trading_agent_final.pth'
    torch.save({
        'episode': start_episode + args.episodes,
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'tickers': list(ticker_data.keys()),
        'state_dim': 20,
        'timestamp': datetime.now().isoformat(),
    }, final_path)

    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Final model: {final_path}")
    print(f"✓ Logs: tensorboard --logdir {log_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted (Ctrl+C)", flush=True)
        print("Checkpoints saved in models/", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
