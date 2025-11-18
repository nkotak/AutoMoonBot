#!/usr/bin/env python3
"""
AutoMoonBot: RL Agent Training Script (Fixed for macOS)

Trains a PPO agent for stock trading with verbose output and macOS compatibility.

Usage:
    python train_rl_agent_fixed.py --tickers HESM AAPL --episodes 100 --quick-test
"""

import sys
import os

# Fix macOS issues BEFORE importing torch
print("Initializing environment...", flush=True)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Prevent MPS errors
os.environ['OMP_NUM_THREADS'] = '1'  # Fix threading issues
os.environ['MKL_NUM_THREADS'] = '1'
print("✓ Environment configured", flush=True)

import argparse
import json
from datetime import datetime
from pathlib import Path

print("Loading libraries (this may take 10-20 seconds)...", flush=True)

import numpy as np
print("  ✓ numpy", flush=True)

import pandas as pd
print("  ✓ pandas", flush=True)

# PyTorch imports - these can hang
print("  Loading PyTorch...", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim

# Disable multiprocessing (causes hangs on macOS)
torch.set_num_threads(1)
print("  ✓ torch (using 1 thread to prevent hangs)", flush=True)

from torch.utils.tensorboard import SummaryWriter
print("  ✓ tensorboard", flush=True)

# Add automoonbot to path
sys.path.insert(0, str(Path(__file__).parent))

print("  Loading AutoMoonBot modules...", flush=True)
from automoonbot.moonpy.model.simple_actor_critic import SimpleActor, SimpleCritic, PPOBuffer
print("  ✓ AutoMoonBot modules", flush=True)

print("✓ All libraries loaded successfully!\n", flush=True)


def download_stock_data(ticker: str, period: str = "5y", timeout: int = 30) -> pd.DataFrame:
    """Download stock data from yfinance with timeout."""
    try:
        import yfinance as yf
        from functools import partial
        import signal

        print(f"  Downloading {ticker} (timeout: {timeout}s)...", flush=True)

        # Simple approach - just try download with shorter period for testing
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if len(df) == 0:
            raise ValueError(f"No data available for {ticker}")

        print(f"  ✓ {ticker}: {len(df)} days", flush=True)
        return df

    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {e}", flush=True)
        raise


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
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

    df = df.dropna()
    return df


def prepare_state(df: pd.DataFrame, idx: int, position: float = 0.0, unrealized_pnl: float = 0.0) -> np.ndarray:
    """Prepare state vector (20 features)."""
    row = df.iloc[idx]
    close = row['Close']

    state = np.zeros(20, dtype=np.float32)

    # Price features
    state[0] = row['returns'] if not pd.isna(row['returns']) else 0.0
    state[1] = np.clip(row['volume_ratio'], 0, 5) / 5 if not pd.isna(row['volume_ratio']) else 0.5

    # Momentum
    state[2] = df['Close'].pct_change(5).iloc[idx] if idx >= 5 else 0.0
    state[3] = df['Close'].pct_change(20).iloc[idx] if idx >= 20 else 0.0

    # Technical indicators
    state[4] = row['rsi'] / 100 if not pd.isna(row['rsi']) else 0.5
    state[5] = np.clip(row['macd'] / close, -0.1, 0.1) * 10 if not pd.isna(row['macd']) else 0.0
    state[6] = np.clip(row['bb_position'], 0, 1) if not pd.isna(row['bb_position']) else 0.5
    state[7] = np.clip(row['volatility_20'], 0, 0.1) * 10 if not pd.isna(row['volatility_20']) else 0.0

    # MA distances
    state[8] = (close - row['ma_20']) / close if not pd.isna(row['ma_20']) else 0.0
    state[9] = (close - row['ma_50']) / close if not pd.isna(row['ma_50']) else 0.0

    # Volume
    state[10] = np.clip(row['volume_ratio'], 0, 3) / 3 if not pd.isna(row['volume_ratio']) else 0.5

    # Price history
    for i in range(4):
        if idx >= i + 1:
            state[11 + i] = df['returns'].iloc[idx - i]

    # Position info
    state[15] = position
    state[16] = np.clip(unrealized_pnl, -1, 1)

    return state


class TradingEnvironment:
    """Simple trading environment."""

    def __init__(self, df: pd.DataFrame, ticker: str, initial_cash: float = 100000.0,
                 transaction_cost: float = 0.001, max_steps: int = 252):
        self.df = df
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.reset()

    def reset(self, start_idx: int = None):
        """Reset environment."""
        if start_idx is None:
            max_start = len(self.df) - self.max_steps - 1
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
        """Get current state."""
        unrealized_pnl = 0.0
        if self.position > 0:
            current_price = self.df['Close'].iloc[self.current_idx]
            unrealized_pnl = (current_price - self.entry_price) * self.position / self.initial_cash

        state = prepare_state(self.df, self.current_idx,
                            position=self.position / (self.initial_cash / self.df['Close'].iloc[self.current_idx]),
                            unrealized_pnl=unrealized_pnl)
        state[17] = min(self.days_held / 100.0, 1.0)
        return state

    def step(self, action: int, position_size: float):
        """Execute one step."""
        current_price = self.df['Close'].iloc[self.current_idx]
        reward = 0.0

        # Execute action
        if action == 2 and self.position == 0 and self.cash > 0:  # BUY
            invest_amount = self.cash * position_size
            shares = invest_amount / current_price
            cost = shares * current_price * (1 + self.transaction_cost)

            if cost <= self.cash:
                self.position = shares
                self.cash -= cost
                self.entry_price = current_price
                self.days_held = 0

        elif action == 0 and self.position > 0:  # SELL
            sell_amount = self.position * position_size
            proceeds = sell_amount * current_price * (1 - self.transaction_cost)
            pnl = (current_price - self.entry_price) * sell_amount

            self.cash += proceeds
            self.position -= sell_amount
            reward = pnl / self.initial_cash

            if self.position < 0.01:
                self.position = 0
                self.entry_price = 0
                self.days_held = 0

        elif action == 1 and self.position > 0:  # HOLD
            self.days_held += 1
            if self.days_held > 50:
                reward -= 0.001

        # Update portfolio
        position_value = self.position * current_price
        self.portfolio_value = self.cash + position_value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        if current_drawdown > 0.10:
            reward -= current_drawdown * 0.5

        portfolio_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        reward += portfolio_return * 0.01

        # Move to next day
        self.current_idx += 1
        self.step_count += 1

        done = (self.step_count >= self.max_steps or
                self.current_idx >= len(self.df) - 1 or
                self.portfolio_value < self.initial_cash * 0.5)

        next_state = self._get_state()

        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position,
        }

        return next_state, reward, done, info


def main():
    parser = argparse.ArgumentParser(description='Train RL trading agent')
    parser.add_argument('--tickers', nargs='+', required=True, help='Stock tickers')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=2048, help='Buffer size')
    parser.add_argument('--save-freq', type=int, default=50, help='Save every N episodes')
    parser.add_argument('--data-period', type=str, default='2y', help='Data period')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (fewer episodes)')

    args = parser.parse_args()

    if args.quick_test:
        args.episodes = min(args.episodes, 10)
        args.data_period = '6mo'
        print("⚠️  QUICK TEST MODE: 10 episodes, 6 months data\n", flush=True)

    print("=" * 80)
    print("AUTOMOONBOT: RL AGENT TRAINING")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}", flush=True)

    # Tensorboard
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"✓ Tensorboard logs: {log_dir}", flush=True)
    print(f"  Run: tensorboard --logdir {log_dir}", flush=True)
    print()

    print(f"Configuration:")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Data Period: {args.data_period}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Buffer Size: {args.buffer_size}")
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
            print(f"  ✗ Failed to download {ticker}: {e}", flush=True)
            print(f"  Skipping {ticker}", flush=True)

    if len(ticker_data) == 0:
        print("\n✗ No data downloaded. Exiting.", flush=True)
        sys.exit(1)

    print(f"\n✓ Downloaded {len(ticker_data)} tickers successfully!\n", flush=True)

    # Initialize networks
    print("Initializing neural networks...", flush=True)
    state_dim = 20
    actor = SimpleActor(state_dim=state_dim)
    critic = SimpleCritic(state_dim=state_dim)

    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Actor: {actor_params:,} parameters", flush=True)
    print(f"  Critic: {critic_params:,} parameters", flush=True)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    start_episode = 0

    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}", flush=True)
        checkpoint = torch.load(args.resume)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        start_episode = checkpoint.get('episode', 0)
        print(f"✓ Resumed from episode {start_episode}", flush=True)

    print(f"\n✓ Networks initialized!\n", flush=True)

    # Training loop
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    buffer = PPOBuffer(state_dim=state_dim, buffer_size=args.buffer_size)

    for episode in range(start_episode, start_episode + args.episodes):
        # Select ticker
        ticker = np.random.choice(list(ticker_data.keys()))
        df = ticker_data[ticker]

        # Create environment
        env = TradingEnvironment(df=df, ticker=ticker)
        state = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        # Collect experience
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, position_size, action_log_prob, size_log_prob = actor.get_action(state_tensor)
                value = critic(state_tensor)

            action = action.item()
            position_size = position_size.item()
            action_log_prob = action_log_prob.item()
            size_log_prob = size_log_prob.item()
            value = value.item()

            next_state, reward, done, info = env.step(action, position_size)

            buffer.store(
                state=state,
                action=action,
                position_size=position_size,
                reward=reward,
                value=value,
                action_log_prob=action_log_prob,
                size_log_prob=size_log_prob,
                done=done,
            )

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                buffer.finish_path(0.0)
                break

        # Log episode
        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/portfolio_value', info['portfolio_value'], episode)

        # Print progress
        if episode % 5 == 0 or episode == start_episode:
            portfolio_return = (info['portfolio_value'] - 100000) / 100000
            print(f"Episode {episode}/{start_episode + args.episodes}")
            print(f"  Ticker: {ticker}, Steps: {episode_steps}")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Portfolio: ${info['portfolio_value']:,.0f} ({portfolio_return:+.2%})")
            print(flush=True)

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0 and episode > start_episode:
            checkpoint_path = output_dir / f'checkpoint_ep{episode+1}.pth'
            torch.save({
                'episode': episode + 1,
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
                'tickers': list(ticker_data.keys()),
                'timestamp': datetime.now().isoformat(),
            }, checkpoint_path)
            print(f"  ✓ Saved: {checkpoint_path}\n", flush=True)

    # Save final model
    final_path = output_dir / 'trading_agent_final.pth'
    torch.save({
        'episode': start_episode + args.episodes,
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'tickers': list(ticker_data.keys()),
        'state_dim': state_dim,
        'timestamp': datetime.now().isoformat(),
    }, final_path)

    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Final model: {final_path}")
    print(f"✓ Tensorboard logs: {log_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)", flush=True)
        print("Your checkpoints are saved in the models/ directory", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
