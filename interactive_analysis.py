#!/usr/bin/env python3
"""
AutoMoonBot: Interactive Stock Analysis with RL Agent

Features:
- Loads your trained PPO agent (if available)
- Uses RL model for BUY/SELL/HOLD recommendations
- Falls back to technical analysis if no model
- Combines RL + technical signals for best results

USAGE:
    python interactive_analysis.py

    # Or specify model path:
    python interactive_analysis.py --model models/trading_agent_final.pth
"""

import sys
import os

# MPS deadlock prevention (before any torch imports)
os.environ['PYTORCH_MPS_ENABLED'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

def get_input(prompt, default=None, input_type=str, validator=None):
    """Get user input with validation"""
    while True:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
            if not user_input:
                print("  ⚠ This field is required. Please try again.")
                continue

        try:
            value = input_type(user_input)
            if validator and not validator(value):
                continue
            return value
        except ValueError:
            print(f"  ⚠ Invalid input. Expected {input_type.__name__}. Please try again.")

def get_yes_no(prompt, default="y"):
    """Get yes/no input"""
    while True:
        response = input(f"{prompt} [y/n, default={default}]: ").strip().lower()
        if not response:
            response = default
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("  ⚠ Please enter 'y' or 'n'")

print("=" * 80)
print("AUTOMOONBOT: INTERACTIVE ANALYSIS WITH RL AGENT")
print("=" * 80)
print()
print("This tool will analyze stocks using:")
print("  • Trained RL Agent (PPO) - if model is available")
print("  • Technical Analysis (moving averages, volatility)")
print("  • Market Regime Detection")
print("  • Position Sizing & Risk Management")
print()

# ============================================================================
# Load RL Model (if available)
# ============================================================================

RL_MODEL_AVAILABLE = False
actor = None
device = None

try:
    import torch
    import torch.nn as nn
    torch.set_default_device('cpu')

    from automoonbot.moonpy.model.simple_actor_critic import SimpleActor
    from pathlib import Path

    # Re-enable MPS after imports
    if 'PYTORCH_MPS_ENABLED' in os.environ:
        del os.environ['PYTORCH_MPS_ENABLED']
    if 'PYTORCH_ENABLE_MPS_FALLBACK' in os.environ:
        del os.environ['PYTORCH_ENABLE_MPS_FALLBACK']

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Look for trained model
    model_path = Path("models/trading_agent_final.pth")
    if not model_path.exists():
        # Try checkpoint
        checkpoints = list(Path("models").glob("checkpoint_ep*.pth")) if Path("models").exists() else []
        if checkpoints:
            model_path = sorted(checkpoints)[-1]  # Most recent

    if model_path.exists():
        print(f"✓ Found RL model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        actor = SimpleActor(state_dim=20).to(device)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()  # Set to evaluation mode

        RL_MODEL_AVAILABLE = True
        print(f"✓ RL Agent loaded on {device}")
        print(f"  Trained on: {', '.join(checkpoint.get('tickers', ['unknown']))}")
        print(f"  Episodes: {checkpoint.get('episode', 'unknown')}")
    else:
        print("⚠ No trained RL model found")
        print("  Will use technical analysis only")
        print(f"  To train a model, run: python train_rl_agent_final.py --tickers AAPL --episodes 1000")

except Exception as e:
    print(f"⚠ Could not load RL model: {e}")
    print("  Will use technical analysis only")

print()
print("Let's get started!")
print()

# ============================================================================
# Collect User Inputs
# ============================================================================
print("-" * 80)
print("STEP 1: STOCK SELECTION")
print("-" * 80)

ticker = get_input(
    "Enter stock ticker symbol (e.g., AAPL, TSLA, HESM)",
    input_type=str
).upper()

print(f"  ✓ Will analyze: {ticker}")
print()

print("-" * 80)
print("STEP 2: PORTFOLIO CONFIGURATION")
print("-" * 80)

portfolio_value = get_input(
    "Enter your portfolio value in dollars",
    default=100000,
    input_type=float,
    validator=lambda x: x > 0 or print("  ⚠ Portfolio value must be positive")
)

print(f"  ✓ Portfolio: ${portfolio_value:,.2f}")
print()

print("-" * 80)
print("STEP 3: RISK PARAMETERS")
print("-" * 80)
print()
print("Professional risk guidelines:")
print("  • Conservative: 0.5-1% per trade")
print("  • Moderate: 1-2% per trade")
print("  • Aggressive: 2-5% per trade")
print("  • Very Aggressive: 5-10% per trade")
print()

risk_pct = get_input(
    "Enter risk per trade as percentage (e.g., 2 for 2%)",
    default=2.0,
    input_type=float,
    validator=lambda x: 0 < x <= 100 or print("  ⚠ Risk must be between 0 and 100")
)

risk_per_trade = risk_pct / 100

if risk_per_trade > 0.10:
    print(f"  ⚠ WARNING: {risk_pct}% is EXTREMELY aggressive!")
    confirm = get_yes_no("  Are you sure?", default="n")
    if not confirm:
        risk_pct = 2.0
        risk_per_trade = 0.02
        print(f"  ✓ Reset to 2% risk")
else:
    print(f"  ✓ {risk_pct}% risk")

print()

stop_loss_pct = get_input(
    "Enter stop loss percentage (e.g., 5 for 5%)",
    default=5.0,
    input_type=float,
    validator=lambda x: 0 < x <= 50 or print("  ⚠ Stop loss must be between 0 and 50")
) / 100

print(f"  ✓ Stop loss: {stop_loss_pct*100:.1f}%")
print()

max_position = get_input(
    "Enter maximum position size as percentage (e.g., 25 for 25%)",
    default=25.0,
    input_type=float,
    validator=lambda x: 0 < x <= 100 or print("  ⚠ Max position must be between 0 and 100")
) / 100

print(f"  ✓ Max position: {max_position*100:.0f}%")
print()

# ============================================================================
# Fetch Market Data
# ============================================================================

print("=" * 80)
print("FETCHING MARKET DATA")
print("=" * 80)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np

    print(f"Downloading {ticker} from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")  # Need more data for RL features

    if len(hist) == 0:
        print(f"✗ No data available for {ticker}")
        sys.exit(1)

    # Calculate technical indicators (needed for RL state)
    hist['returns'] = hist['Close'].pct_change()
    hist['ma_5'] = hist['Close'].rolling(5).mean()
    hist['ma_20'] = hist['Close'].rolling(20).mean()
    hist['ma_50'] = hist['Close'].rolling(50).mean()
    hist['volatility_20'] = hist['returns'].rolling(20).std()

    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    hist['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = hist['Close'].ewm(span=12).mean()
    ema_26 = hist['Close'].ewm(span=26).mean()
    hist['macd'] = ema_12 - ema_26

    # Bollinger Bands
    hist['bb_middle'] = hist['Close'].rolling(20).mean()
    bb_std = hist['Close'].rolling(20).std()
    hist['bb_upper'] = hist['bb_middle'] + (bb_std * 2)
    hist['bb_lower'] = hist['bb_middle'] - (bb_std * 2)
    hist['bb_position'] = (hist['Close'] - hist['bb_lower']) / (hist['bb_upper'] - hist['bb_lower'])

    # Volume
    hist['volume_ma'] = hist['Volume'].rolling(20).mean()
    hist['volume_ratio'] = hist['Volume'] / hist['volume_ma']

    hist = hist.dropna()

    # Current metrics
    current_price = float(hist['Close'].iloc[-1])
    prev_close = float(hist['Close'].iloc[-2])
    daily_change = (current_price - prev_close) / prev_close

    volatility = float(hist['returns'].std() * np.sqrt(252))

    info = stock.info
    company_name = info.get('longName', ticker)

    print(f"✓ Data fetched: {len(hist)} days")
    print()
    print(f"Company: {company_name}")
    print(f"Current Price: ${current_price:.2f} ({daily_change:+.2%})")
    print(f"Volatility (annualized): {volatility:.1%}")
    print()

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# RL Agent Prediction (if available)
# ============================================================================

rl_action = None
rl_position_size = None
rl_confidence = None

if RL_MODEL_AVAILABLE and actor is not None:
    print("-" * 80)
    print("RL AGENT ANALYSIS")
    print("-" * 80)

    try:
        # Prepare state vector (same as training)
        idx = len(hist) - 1
        row = hist.iloc[idx]
        close = row['Close']

        state = np.zeros(20, dtype=np.float32)
        state[0] = row['returns'] if not pd.isna(row['returns']) else 0.0
        state[1] = np.clip(row['volume_ratio'], 0, 5) / 5 if not pd.isna(row['volume_ratio']) else 0.5
        state[2] = hist['Close'].pct_change(5).iloc[idx] if idx >= 5 else 0.0
        state[3] = hist['Close'].pct_change(20).iloc[idx] if idx >= 20 else 0.0
        state[4] = row['rsi'] / 100 if not pd.isna(row['rsi']) else 0.5
        state[5] = np.clip(row['macd'] / close, -0.1, 0.1) * 10 if not pd.isna(row['macd']) else 0.0
        state[6] = np.clip(row['bb_position'], 0, 1) if not pd.isna(row['bb_position']) else 0.5
        state[7] = np.clip(row['volatility_20'], 0, 0.1) * 10 if not pd.isna(row['volatility_20']) else 0.0
        state[8] = (close - row['ma_20']) / close if not pd.isna(row['ma_20']) else 0.0
        state[9] = (close - row['ma_50']) / close if not pd.isna(row['ma_50']) else 0.0
        state[10] = np.clip(row['volume_ratio'], 0, 3) / 3 if not pd.isna(row['volume_ratio']) else 0.5

        for i in range(4):
            if idx >= i + 1:
                state[11 + i] = hist['returns'].iloc[idx - i]

        # No position currently (state[15-17] = 0)

        # Get RL agent's recommendation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action, position_size, action_log_prob, size_log_prob = actor.get_action(state_tensor)

            rl_action = action.item()  # 0=SELL, 1=HOLD, 2=BUY
            rl_position_size = position_size.item()

            # Confidence based on log probabilities
            rl_confidence = min(torch.exp(action_log_prob).item(), 1.0)

        action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

        print(f"RL Recommendation: {action_names[rl_action]}")
        print(f"Position Size: {rl_position_size:.1%}")
        print(f"Confidence: {rl_confidence:.0%}")
        print()

    except Exception as e:
        print(f"⚠ RL prediction failed: {e}")
        RL_MODEL_AVAILABLE = False

# ============================================================================
# Technical Analysis
# ============================================================================

print("-" * 80)
print("TECHNICAL ANALYSIS")
print("-" * 80)

ma_20 = float(hist['ma_20'].iloc[-1])
ma_50 = float(hist['ma_50'].iloc[-1])
price_vs_ma20 = (current_price - ma_20) / ma_20
price_vs_ma50 = (current_price - ma_50) / ma_50

# Trend
if price_vs_ma20 > 0.05 and price_vs_ma50 > 0.05:
    trend = "STRONG UPTREND"
    trend_signal = "BULLISH"
elif price_vs_ma20 > 0 and price_vs_ma50 > 0:
    trend = "UPTREND"
    trend_signal = "BULLISH"
elif price_vs_ma20 < -0.05 and price_vs_ma50 < -0.05:
    trend = "STRONG DOWNTREND"
    trend_signal = "BEARISH"
elif price_vs_ma20 < 0 and price_vs_ma50 < 0:
    trend = "DOWNTREND"
    trend_signal = "BEARISH"
else:
    trend = "SIDEWAYS"
    trend_signal = "NEUTRAL"

print(f"Trend: {trend}")
print(f"Price vs 20-MA: {price_vs_ma20:+.1%}")
print(f"Price vs 50-MA: {price_vs_ma50:+.1%}")
print(f"RSI: {hist['rsi'].iloc[-1]:.1f}")
print()

# ============================================================================
# Combined Signal (RL + Technical)
# ============================================================================

print("=" * 80)
print("TRADING SIGNAL")
print("=" * 80)

if RL_MODEL_AVAILABLE and rl_action is not None:
    # Use RL agent as primary signal
    if rl_action == 2:  # BUY
        signal = "BUY"
        recommended_size = min(rl_position_size, max_position)
    elif rl_action == 0:  # SELL
        signal = "SELL/AVOID"
        recommended_size = 0
    else:  # HOLD
        signal = "HOLD"
        recommended_size = 0.1  # Small position

    # Check if technical agrees
    technical_agrees = (
        (rl_action == 2 and trend_signal == "BULLISH") or
        (rl_action == 0 and trend_signal == "BEARISH") or
        (rl_action == 1 and trend_signal == "NEUTRAL")
    )

    if technical_agrees:
        print(f"Signal: {signal} ✓")
        print(f"  RL Agent: {signal}")
        print(f"  Technical: {trend_signal} (confirms)")
        print(f"  Confidence: HIGH ({rl_confidence:.0%})")
    else:
        print(f"Signal: {signal} ⚠")
        print(f"  RL Agent: {signal}")
        print(f"  Technical: {trend_signal} (conflicts)")
        print(f"  Confidence: MEDIUM ({rl_confidence * 0.7:.0%})")
else:
    # Fall back to technical only
    if trend_signal == "BULLISH":
        signal = "BUY"
        recommended_size = 0.15
    elif trend_signal == "BEARISH":
        signal = "AVOID"
        recommended_size = 0
    else:
        signal = "HOLD"
        recommended_size = 0.05

    print(f"Signal: {signal} (Technical Analysis)")
    print(f"  Trend: {trend_signal}")

print()

# ============================================================================
# Position Sizing & Targets
# ============================================================================

if signal in ["BUY", "HOLD"] and recommended_size > 0:
    position_value = recommended_size * portfolio_value
    shares = int(position_value / current_price)
    actual_invested = shares * current_price
    actual_position_pct = actual_invested / portfolio_value

    stop_price = current_price * (1 - stop_loss_pct)
    max_risk_dollars = shares * current_price * stop_loss_pct

    risk_per_share = current_price - stop_price
    target_2_0 = current_price + (risk_per_share * 2.0)
    target_3_0 = current_price + (risk_per_share * 3.0)

    print("-" * 80)
    print("POSITION DETAILS")
    print("-" * 80)
    print(f"Recommended Shares: {shares:,}")
    print(f"@ ${current_price:.2f} = ${actual_invested:,.2f}")
    print(f"Position Size: {actual_position_pct:.1%} of portfolio")
    print(f"Max Risk: ${max_risk_dollars:,.2f}")
    print()
    print(f"Entry: ${current_price:.2f}")
    print(f"Stop Loss: ${stop_price:.2f} (-{stop_loss_pct*100:.0f}%)")
    print(f"Target 1 (2R): ${target_2_0:.2f} (+{((target_2_0/current_price-1)*100):.0f}%)")
    print(f"Target 2 (3R): ${target_3_0:.2f} (+{((target_3_0/current_price-1)*100):.0f}%)")
else:
    print("-" * 80)
    print("RECOMMENDATION")
    print("-" * 80)
    print(f"{signal}: Do not enter position at this time")
    print(f"Reason: {trend}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()

if RL_MODEL_AVAILABLE:
    print("Note: This analysis used your trained RL agent")
else:
    print("Note: To use RL agent, train a model first:")
    print("  python train_rl_agent_final.py --tickers AAPL --episodes 1000")

print()
print("DISCLAIMER: This is analysis only - NOT financial advice")
print("=" * 80)
