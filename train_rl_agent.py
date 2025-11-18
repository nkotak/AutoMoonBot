#!/usr/bin/env python3
"""
AutoMoonBot: RL Agent Training Script

Trains a PPO (Proximal Policy Optimization) agent for stock trading using:
- Real market data from yfinance
- SimpleActor/SimpleCritic networks
- Multi-ticker training
- Incremental learning (resume from checkpoint)

Usage:
    # Train from scratch
    python train_rl_agent.py --tickers HESM AAPL TSLA --episodes 5000

    # Resume training
    python train_rl_agent.py --tickers MSFT GOOGL --resume models/checkpoint.pth

    # Custom configuration
    python train_rl_agent.py --tickers HESM --episodes 10000 --lr 0.0001 --batch-size 64
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add automoonbot to path
sys.path.insert(0, str(Path(__file__).parent))

from automoonbot.moonpy.model.simple_actor_critic import SimpleActor, SimpleCritic, PPOBuffer


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download stock data from yfinance.

    Args:
        ticker: Stock symbol
        period: Time period (1y, 2y, 5y, max)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf

        print(f"  Downloading {ticker} data (period={period})...")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if len(df) == 0:
            raise ValueError(f"No data available for {ticker}")

        print(f"  ✓ Downloaded {len(df)} days of data")
        return df

    except ImportError:
        print("ERROR: yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading {ticker}: {e}")
        sys.exit(1)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
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

    # Volume indicators
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    # Drop NaN rows
    df = df.dropna()

    return df


def prepare_state(df: pd.DataFrame, idx: int, position: float = 0.0, unrealized_pnl: float = 0.0) -> np.ndarray:
    """
    Prepare state vector for RL agent.

    State vector (20 dimensions):
        [0] = Normalized close price (current / previous)
        [1] = Normalized volume
        [2] = Returns (5-day momentum)
        [3] = Returns (20-day momentum)
        [4] = RSI / 100
        [5] = MACD / price
        [6] = BB position
        [7] = Volatility
        [8] = Distance from MA20
        [9] = Distance from MA50
        [10] = Volume ratio
        [11-14] = Price history (last 4 days, normalized)
        [15] = Current position
        [16] = Unrealized P&L
        [17] = Days held
        [18-19] = Reserved for future features

    Args:
        df: DataFrame with technical indicators
        idx: Current index in dataframe
        position: Current position size (0.0 to 1.0)
        unrealized_pnl: Unrealized profit/loss

    Returns:
        State vector as numpy array
    """
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

    # Moving average distances
    state[8] = (close - row['ma_20']) / close if not pd.isna(row['ma_20']) else 0.0
    state[9] = (close - row['ma_50']) / close if not pd.isna(row['ma_50']) else 0.0

    # Volume
    state[10] = np.clip(row['volume_ratio'], 0, 3) / 3 if not pd.isna(row['volume_ratio']) else 0.5

    # Price history (last 4 days normalized)
    for i in range(4):
        if idx >= i + 1:
            state[11 + i] = df['returns'].iloc[idx - i]

    # Position info
    state[15] = position
    state[16] = np.clip(unrealized_pnl, -1, 1)  # Clip to [-1, 1]
    state[17] = 0.0  # Days held (will be updated in environment)

    return state


class TradingEnvironment:
    """
    Simple trading environment for RL training.

    Actions:
        0 = SELL/CLOSE position
        1 = HOLD (do nothing)
        2 = BUY/OPEN position

    Position Size:
        Continuous value [0.0, 1.0] representing fraction of capital to use
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ticker: str,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1%
        max_steps: int = 252,  # 1 year of trading days
    ):
        self.df = df
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps

        self.reset()

    def reset(self, start_idx: int = None):
        """Reset environment to initial state."""
        # Random start index if not specified
        if start_idx is None:
            # Leave room for max_steps
            max_start = len(self.df) - self.max_steps - 1
            self.start_idx = np.random.randint(50, max(51, max_start))
        else:
            self.start_idx = start_idx

        self.current_idx = self.start_idx
        self.cash = self.initial_cash
        self.position = 0.0  # Shares held
        self.position_value = 0.0
        self.entry_price = 0.0
        self.days_held = 0
        self.portfolio_value = self.cash
        self.peak_value = self.initial_cash
        self.current_drawdown = 0.0

        self.step_count = 0
        self.trades = []

        return self._get_state()

    def _get_state(self):
        """Get current state."""
        unrealized_pnl = 0.0
        if self.position > 0:
            current_price = self.df['Close'].iloc[self.current_idx]
            unrealized_pnl = (current_price - self.entry_price) * self.position / self.initial_cash

        state = prepare_state(
            self.df,
            self.current_idx,
            position=self.position / (self.initial_cash / self.df['Close'].iloc[self.current_idx]),
            unrealized_pnl=unrealized_pnl
        )

        state[17] = min(self.days_held / 100.0, 1.0)  # Normalize days held

        return state

    def step(self, action: int, position_size: float):
        """
        Execute one step.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY
            position_size: Fraction of capital/position to use [0.0, 1.0]

        Returns:
            next_state, reward, done, info
        """
        current_price = self.df['Close'].iloc[self.current_idx]

        # Execute action
        reward = 0.0
        trade_info = None

        if action == 2:  # BUY
            if self.position == 0 and self.cash > 0:
                # Open new position
                invest_amount = self.cash * position_size
                shares = invest_amount / current_price
                cost = shares * current_price * (1 + self.transaction_cost)

                if cost <= self.cash:
                    self.position = shares
                    self.cash -= cost
                    self.entry_price = current_price
                    self.days_held = 0
                    trade_info = f"BUY {shares:.2f} @ ${current_price:.2f}"

        elif action == 0:  # SELL
            if self.position > 0:
                # Close position
                sell_amount = self.position * position_size
                proceeds = sell_amount * current_price * (1 - self.transaction_cost)
                pnl = (current_price - self.entry_price) * sell_amount

                self.cash += proceeds
                self.position -= sell_amount

                # Calculate reward as profit/loss percentage
                reward = pnl / self.initial_cash

                if self.position < 0.01:  # Close remaining dust
                    self.position = 0
                    self.entry_price = 0
                    self.days_held = 0

                trade_info = f"SELL {sell_amount:.2f} @ ${current_price:.2f}, P&L=${pnl:.2f}"

        elif action == 1:  # HOLD
            # Small penalty for holding too long
            if self.position > 0:
                self.days_held += 1
                if self.days_held > 50:
                    reward -= 0.001  # Encourage closing positions

        # Update portfolio value
        self.position_value = self.position * current_price
        self.portfolio_value = self.cash + self.position_value

        # Track drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Drawdown penalty
        if self.current_drawdown > 0.10:
            reward -= self.current_drawdown * 0.5

        # Portfolio growth reward
        portfolio_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        reward += portfolio_return * 0.01  # Small continuous reward for growth

        # Move to next day
        self.current_idx += 1
        self.step_count += 1

        # Check if done
        done = (
            self.step_count >= self.max_steps or
            self.current_idx >= len(self.df) - 1 or
            self.portfolio_value < self.initial_cash * 0.5  # Stop if lost 50%
        )

        next_state = self._get_state()

        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'drawdown': self.current_drawdown,
            'trade': trade_info,
        }

        if trade_info:
            self.trades.append(trade_info)

        return next_state, reward, done, info


def train_ppo(
    actor: SimpleActor,
    critic: SimpleCritic,
    buffer: PPOBuffer,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    clip_epsilon: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 64,
    writer: SummaryWriter = None,
    global_step: int = 0,
):
    """
    PPO training update.

    Args:
        actor: Actor network
        critic: Critic network
        buffer: Experience buffer
        actor_optimizer: Actor optimizer
        critic_optimizer: Critic optimizer
        clip_epsilon: PPO clip parameter
        n_epochs: Number of optimization epochs
        batch_size: Mini-batch size
        writer: Tensorboard writer
        global_step: Global step for logging

    Returns:
        Training metrics dict
    """
    data = buffer.get()

    # Convert to tensors
    states = torch.FloatTensor(data['states'])
    actions = torch.LongTensor(data['actions'])
    position_sizes = torch.FloatTensor(data['position_sizes'])
    returns = torch.FloatTensor(data['returns'])
    advantages = torch.FloatTensor(data['advantages'])
    old_action_log_probs = torch.FloatTensor(data['action_log_probs'])
    old_size_log_probs = torch.FloatTensor(data['size_log_probs'])

    metrics = {
        'actor_loss': 0.0,
        'critic_loss': 0.0,
        'entropy': 0.0,
        'approx_kl': 0.0,
    }

    # Multiple epochs of optimization
    for epoch in range(n_epochs):
        # Random mini-batches
        indices = torch.randperm(len(states))

        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_sizes = position_sizes[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_old_action_log_probs = old_action_log_probs[batch_indices]
            batch_old_size_log_probs = old_size_log_probs[batch_indices]

            # Evaluate actions
            action_log_probs, size_log_probs, entropy = actor.evaluate_actions(
                batch_states, batch_actions, batch_sizes
            )

            # PPO actor loss
            total_log_prob = action_log_probs + size_log_probs
            old_total_log_prob = batch_old_action_log_probs + batch_old_size_log_probs

            ratio = torch.exp(total_log_prob - old_total_log_prob)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()

            total_actor_loss = actor_loss + 0.01 * entropy_loss

            # Update actor
            actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()

            # Critic loss
            values = critic(batch_states).squeeze()
            critic_loss = ((values - batch_returns) ** 2).mean()

            # Update critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()

            # Track metrics
            metrics['actor_loss'] += actor_loss.item()
            metrics['critic_loss'] += critic_loss.item()
            metrics['entropy'] += entropy.mean().item()

            # Approximate KL divergence
            approx_kl = (old_total_log_prob - total_log_prob).mean().item()
            metrics['approx_kl'] += abs(approx_kl)

    # Average metrics
    n_updates = n_epochs * (len(states) // batch_size)
    for key in metrics:
        metrics[key] /= n_updates

    # Log to tensorboard
    if writer:
        writer.add_scalar('train/actor_loss', metrics['actor_loss'], global_step)
        writer.add_scalar('train/critic_loss', metrics['critic_loss'], global_step)
        writer.add_scalar('train/entropy', metrics['entropy'], global_step)
        writer.add_scalar('train/approx_kl', metrics['approx_kl'], global_step)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train RL trading agent')
    parser.add_argument('--tickers', nargs='+', required=True, help='Stock tickers to train on')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=2048, help='Buffer size')
    parser.add_argument('--save-freq', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--eval-freq', type=int, default=50, help='Evaluate every N episodes')
    parser.add_argument('--data-period', type=str, default='5y', help='Data period (1y, 2y, 5y, max)')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("AUTOMOONBOT: RL AGENT TRAINING")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'logs')

    print(f"Configuration:")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Buffer Size: {args.buffer_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Output Dir: {output_dir}")
    print()

    # Download data for all tickers
    print("Downloading data...")
    ticker_data = {}
    for ticker in args.tickers:
        df = download_stock_data(ticker, period=args.data_period)
        df = calculate_technical_indicators(df)
        ticker_data[ticker] = df

    print(f"✓ Downloaded data for {len(ticker_data)} tickers")
    print()

    # Initialize networks
    state_dim = 20
    actor = SimpleActor(state_dim=state_dim)
    critic = SimpleCritic(state_dim=state_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    start_episode = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        start_episode = checkpoint.get('episode', 0)
        print(f"✓ Resumed from episode {start_episode}")
        print()

    # Training loop
    print("Starting training...")
    print()

    buffer = PPOBuffer(state_dim=state_dim, buffer_size=args.buffer_size)
    global_step = start_episode

    for episode in range(start_episode, start_episode + args.episodes):
        # Randomly select ticker
        ticker = np.random.choice(args.tickers)
        df = ticker_data[ticker]

        # Create environment
        env = TradingEnvironment(df=df, ticker=ticker)
        state = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        # Collect experience
        while True:
            # Get action from actor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, position_size, action_log_prob, size_log_prob = actor.get_action(state_tensor)
                value = critic(state_tensor)

            action = action.item()
            position_size = position_size.item()
            action_log_prob = action_log_prob.item()
            size_log_prob = size_log_prob.item()
            value = value.item()

            # Step environment
            next_state, reward, done, info = env.step(action, position_size)

            # Store in buffer
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

            # If buffer is full, train
            if buffer.ptr >= args.buffer_size:
                # Finish path with last value
                if not done:
                    with torch.no_grad():
                        last_value = critic(torch.FloatTensor(next_state).unsqueeze(0)).item()
                else:
                    last_value = 0.0

                buffer.finish_path(last_value)

                # Train
                metrics = train_ppo(
                    actor, critic, buffer,
                    actor_optimizer, critic_optimizer,
                    batch_size=args.batch_size,
                    writer=writer,
                    global_step=global_step
                )

                # Reset buffer
                buffer = PPOBuffer(state_dim=state_dim, buffer_size=args.buffer_size)

            if done:
                if buffer.ptr > 0:
                    buffer.finish_path(0.0)
                break

        # Log episode
        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/steps', episode_steps, episode)
        writer.add_scalar('episode/final_portfolio_value', info['portfolio_value'], episode)
        writer.add_scalar('episode/return', (info['portfolio_value'] - 100000) / 100000, episode)

        global_step += 1

        # Print progress
        if episode % 10 == 0:
            portfolio_return = (info['portfolio_value'] - 100000) / 100000
            print(f"Episode {episode}/{start_episode + args.episodes}")
            print(f"  Ticker: {ticker}")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Final Value: ${info['portfolio_value']:,.2f} ({portfolio_return:+.2%})")
            print(f"  Trades: {len(env.trades)}")
            print()

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_ep{episode+1}.pth'
            torch.save({
                'episode': episode + 1,
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
                'tickers': args.tickers,
                'timestamp': datetime.now().isoformat(),
            }, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")
            print()

    # Save final model
    final_path = output_dir / 'trading_agent_final.pth'
    torch.save({
        'episode': start_episode + args.episodes,
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'tickers': args.tickers,
        'state_dim': state_dim,
        'timestamp': datetime.now().isoformat(),
    }, final_path)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final model saved to: {final_path}")
    print()


if __name__ == "__main__":
    main()
