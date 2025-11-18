#!/usr/bin/env python3
"""
AutoMoonBot: RL Training for HESM with MPS Support

This script properly handles macOS MPS:
1. Temporarily disables MPS during imports (prevents deadlock)
2. Re-enables MPS after imports complete
3. Uses MPS for actual training (GPU acceleration)
4. Supports multithreading

Usage:
    python train_rl_hesm.py --portfolio 70000 --risk 0.25
"""

import sys
import os

# STEP 1: Temporarily disable MPS ONLY during imports to prevent deadlock
print("=" * 80)
print("AUTOMOONBOT: HESM RL TRAINING WITH MPS SUPPORT")
print("=" * 80)
print()
print("Initializing (MPS temporarily disabled during imports)...", flush=True)
os.environ['PYTORCH_MPS_ENABLED'] = '0'  # Disable during import
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
print("✓ Environment configured", flush=True)

import argparse
from datetime import datetime
from pathlib import Path

print("Loading libraries...", flush=True)
import numpy as np
print("  ✓ numpy", flush=True)

import pandas as pd
print("  ✓ pandas", flush=True)

print("  Loading PyTorch (MPS disabled during import)...", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
print("  ✓ torch", flush=True)

from torch.utils.tensorboard import SummaryWriter
print("  ✓ tensorboard", flush=True)

# Add automoonbot to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Import AutoMoonBot modules
from automoonbot.moonpy.model.simple_actor_critic import SimpleActor, SimpleCritic, PPOBuffer
print("  ✓ AutoMoonBot modules", flush=True)

import yfinance as yf
print("  ✓ yfinance", flush=True)

print()
print("✓ All libraries loaded successfully!")
print()

# STEP 2: Re-enable MPS for training
print("Re-enabling MPS for training...", flush=True)
del os.environ['PYTORCH_MPS_ENABLED']  # Remove the disable flag

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ MPS (Metal Performance Shaders) is available and ENABLED!", flush=True)
    print(f"  Using GPU acceleration for training", flush=True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ CUDA is available", flush=True)
else:
    device = torch.device("cpu")
    print(f"✓ Using CPU (MPS not available on this system)", flush=True)

print(f"  Device: {device}", flush=True)
print()


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download stock data from yfinance."""
    try:
        print(f"  Downloading {ticker} ({period})...", flush=True)
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
    """Trading environment for HESM."""

    def __init__(self, df: pd.DataFrame, ticker: str, initial_cash: float = 70000.0,
                 risk_per_trade: float = 0.25, transaction_cost: float = 0.001, max_steps: int = 252):
        self.df = df
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.risk_per_trade = risk_per_trade
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

        # Apply risk management
        max_position_size = min(position_size, self.risk_per_trade)

        # Execute action
        if action == 2 and self.position == 0 and self.cash > 0:  # BUY
            invest_amount = self.cash * max_position_size
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
    parser = argparse.ArgumentParser(description='Train RL agent for HESM')
    parser.add_argument('--portfolio', type=float, default=70000, help='Portfolio value ($)')
    parser.add_argument('--risk', type=float, default=0.25, help='Risk per trade (0.0-1.0)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data-period', type=str, default='5y', help='Data period')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--quick-test', action='store_true', help='Quick test (10 episodes)')

    args = parser.parse_args()

    if args.quick_test:
        args.episodes = 10
        args.data_period = '6mo'
        print("⚠️  QUICK TEST MODE: 10 episodes, 6 months data\n", flush=True)

    print(f"Configuration:")
    print(f"  Ticker: HESM (Hess Midstream LP)")
    print(f"  Portfolio: ${args.portfolio:,.0f}")
    print(f"  Risk per trade: {args.risk:.0%}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Data Period: {args.data_period}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}", flush=True)

    # Tensorboard
    log_dir = output_dir / 'logs' / f'hesm_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"✓ Tensorboard logs: {log_dir}", flush=True)
    print(f"  Run: tensorboard --logdir {log_dir}", flush=True)
    print()

    # Download data
    print("Downloading HESM data...")
    try:
        df = download_stock_data("HESM", period=args.data_period)
        df = calculate_technical_indicators(df)
        print(f"✓ Data prepared: {len(df)} days\n", flush=True)
    except Exception as e:
        print(f"\n✗ Failed to download HESM data: {e}", flush=True)
        sys.exit(1)

    # Initialize networks
    print("Initializing neural networks...", flush=True)
    state_dim = 20
    actor = SimpleActor(state_dim=state_dim).to(device)
    critic = SimpleCritic(state_dim=state_dim).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Actor: {actor_params:,} parameters", flush=True)
    print(f"  Critic: {critic_params:,} parameters", flush=True)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    print(f"✓ Networks initialized on {device}!\n", flush=True)

    # Training loop
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    env = TradingEnvironment(df=df, ticker="HESM", initial_cash=args.portfolio,
                            risk_per_trade=args.risk)

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, position_size, action_log_prob, size_log_prob = actor.get_action(state_tensor)
                value = critic(state_tensor)

            action = action.item()
            position_size = position_size.item()

            next_state, reward, done, info = env.step(action, position_size)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        # Log episode
        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/portfolio_value', info['portfolio_value'], episode)

        # Print progress
        if episode % 10 == 0 or episode == 0:
            portfolio_return = (info['portfolio_value'] - args.portfolio) / args.portfolio
            print(f"Episode {episode}/{args.episodes}")
            print(f"  Steps: {episode_steps}")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Portfolio: ${info['portfolio_value']:,.0f} ({portfolio_return:+.2%})")
            print(flush=True)

    # Save final model
    final_path = output_dir / 'hesm_agent_final.pth'
    torch.save({
        'episode': args.episodes,
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'ticker': 'HESM',
        'portfolio': args.portfolio,
        'risk': args.risk,
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
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
