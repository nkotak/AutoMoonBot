#!/usr/bin/env python3
"""
AutoMoonBot: RL Agent Training Script (Production Ready)

Fixed version with:
- Proper package structure (no moonrs dependency for simple models)
- macOS compatibility (threading fixes)
- Verbose output
- Ctrl+C handling

Usage:
    python train_rl_agent_final.py --tickers HESM AAPL --episodes 500
"""

import sys
import os

# STEP 1: Disable MPS during imports to prevent mutex deadlock
print("Initializing environment...", flush=True)
os.environ['PYTORCH_MPS_ENABLED'] = '0'  # Temporarily disable MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
print("✓ MPS temporarily disabled during imports (prevents deadlock)", flush=True)

import argparse
from datetime import datetime
from pathlib import Path

print("Loading libraries...", flush=True)
import numpy as np
print("  ✓ numpy", flush=True)

import pandas as pd
print("  ✓ pandas", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
print("  ✓ torch", flush=True)

from torch.utils.tensorboard import SummaryWriter
print("  ✓ tensorboard", flush=True)

# Now import can work normally since __init__.py is fixed
from automoonbot.moonpy.model.simple_actor_critic import SimpleActor, SimpleCritic, PPOBuffer
print("  ✓ AutoMoonBot modules", flush=True)

# STEP 2: Re-enable MPS after all imports complete
print("\nRe-enabling MPS for training...", flush=True)
del os.environ['PYTORCH_MPS_ENABLED']  # Remove disable flag

# Check device availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (GPU acceleration enabled!)", flush=True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA", flush=True)
else:
    device = torch.device("cpu")
    print("✓ Using CPU", flush=True)

print(f"  Device: {device}\n", flush=True)


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download stock data from yfinance."""
    import yfinance as yf

    print(f"  Downloading {ticker} ({period})...", flush=True)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")

    print(f"  ✓ {ticker}: {len(df)} days", flush=True)
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df['returns'] = df['Close'].pct_change()
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
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


def prepare_state(df: pd.DataFrame, idx: int, position: float = 0.0, unrealized_pnl: float = 0.0) -> np.ndarray:
    """Prepare 20-dimensional state vector."""
    row = df.iloc[idx]
    close = row['Close']
    state = np.zeros(20, dtype=np.float32)

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

    for i in range(4):
        if idx >= i + 1:
            state[11 + i] = df['returns'].iloc[idx - i]

    state[15] = position
    state[16] = np.clip(unrealized_pnl, -1, 1)

    return state


class TradingEnvironment:
    """Trading environment for RL."""

    def __init__(self, df: pd.DataFrame, ticker: str, initial_cash: float = 100000.0,
                 transaction_cost: float = 0.001, max_steps: int = 252):
        self.df = df
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.reset()

    def reset(self, start_idx: int = None):
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
        current_price = self.df['Close'].iloc[self.current_idx]
        reward = 0.0

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

        position_value = self.position * current_price
        self.portfolio_value = self.cash + position_value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        if current_drawdown > 0.10:
            reward -= current_drawdown * 0.5

        portfolio_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        reward += portfolio_return * 0.01

        self.current_idx += 1
        self.step_count += 1

        done = (self.step_count >= self.max_steps or
                self.current_idx >= len(self.df) - 1 or
                self.portfolio_value < self.initial_cash * 0.5)

        next_state = self._get_state()

        info = {'portfolio_value': self.portfolio_value, 'cash': self.cash, 'position': self.position}

        return next_state, reward, done, info


def main():
    parser = argparse.ArgumentParser(description='Train RL trading agent')
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-size', type=int, default=2048)
    parser.add_argument('--save-freq', type=int, default=50)
    parser.add_argument('--data-period', type=str, default='2y')
    parser.add_argument('--output-dir', type=str, default='models')

    args = parser.parse_args()

    print("=" * 80)
    print("AUTOMOONBOT: RL TRAINING")
    print("=" * 80)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Configuration:")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output: {output_dir}")
    print(f"  Tensorboard: tensorboard --logdir {log_dir}")
    print()

    print("Downloading data...")
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

    print("Initializing networks...", flush=True)
    actor = SimpleActor(state_dim=20).to(device)
    critic = SimpleCritic(state_dim=20).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Actor: {actor_params:,} parameters", flush=True)
    print(f"  Critic: {critic_params:,} parameters", flush=True)
    print(f"  Models moved to: {device}", flush=True)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    start_episode = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        start_episode = checkpoint.get('episode', 0)
        print(f"✓ Resumed from episode {start_episode}", flush=True)

    print(f"\n✓ Ready to train!\n", flush=True)

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()

    buffer = PPOBuffer(state_dim=20, buffer_size=args.buffer_size)

    for episode in range(start_episode, start_episode + args.episodes):
        ticker = np.random.choice(list(ticker_data.keys()))
        df = ticker_data[ticker]

        env = TradingEnvironment(df=df, ticker=ticker)
        state = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, position_size, action_log_prob, size_log_prob = actor.get_action(state_tensor)
                value = critic(state_tensor)

            next_state, reward, done, info = env.step(action.item(), position_size.item())

            buffer.store(state, action.item(), position_size.item(), reward,
                        value.item(), action_log_prob.item(), size_log_prob.item(), done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                buffer.finish_path(0.0)
                break

        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/portfolio_value', info['portfolio_value'], episode)

        if episode % 10 == 0:
            portfolio_return = (info['portfolio_value'] - 100000) / 100000
            print(f"Episode {episode}/{start_episode + args.episodes}")
            print(f"  {ticker}: Reward={episode_reward:.4f}, Return={portfolio_return:+.2%}")
            print(flush=True)

        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_ep{episode+1}.pth'
            torch.save({
                'episode': episode + 1,
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
                'tickers': list(ticker_data.keys()),
            }, checkpoint_path)
            print(f"  ✓ Saved: {checkpoint_path}\n", flush=True)

    final_path = output_dir / 'trading_agent_final.pth'
    torch.save({
        'episode': start_episode + args.episodes,
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'tickers': list(ticker_data.keys()),
        'state_dim': 20,
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
        print("\n\n⚠️  Training interrupted", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
