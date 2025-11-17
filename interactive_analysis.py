#!/usr/bin/env python3
"""
AutoMoonBot: Interactive Stock Analysis
Asks for all inputs interactively, then runs analysis

USAGE:
    python interactive_analysis.py
"""

import sys
import os

# Add AutoMoonBot to path if running from repo
if os.path.exists('/home/user/AutoMoonBot'):
    sys.path.insert(0, '/home/user/AutoMoonBot')

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
print("AUTOMOONBOT: INTERACTIVE STOCK ANALYSIS")
print("=" * 80)
print()
print("This tool will analyze any stock and provide:")
print("  • Market regime analysis")
print("  • Position sizing recommendations")
print("  • Entry/exit price targets")
print("  • Risk assessment")
print("  • Trading signals")
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
    confirm = get_yes_no("  Are you sure you want to continue with this risk level?", default="n")
    if not confirm:
        risk_pct = 2.0
        risk_per_trade = 0.02
        print(f"  ✓ Reset to recommended 2% risk")
    else:
        print(f"  ⚠ Proceeding with {risk_pct}% risk - BE CAREFUL!")
elif risk_per_trade > 0.05:
    print(f"  ⚠ {risk_pct}% is aggressive (higher than typical)")
else:
    print(f"  ✓ {risk_pct}% is a reasonable risk level")

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
# Summary and Confirmation
# ============================================================================
print("=" * 80)
print("CONFIGURATION SUMMARY")
print("=" * 80)
print(f"  Ticker: {ticker}")
print(f"  Portfolio: ${portfolio_value:,.2f}")
print(f"  Risk per Trade: {risk_per_trade*100:.1f}%")
print(f"  Stop Loss: {stop_loss_pct*100:.1f}%")
print(f"  Max Position: {max_position*100:.0f}%")
print("=" * 80)
print()

proceed = get_yes_no("Proceed with analysis?", default="y")

if not proceed:
    print("Analysis cancelled.")
    sys.exit(0)

print()
print("Starting analysis...")
print()

# ============================================================================
# Run Analysis
# ============================================================================

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np

    print("-" * 80)
    print("FETCHING MARKET DATA")
    print("-" * 80)
    print(f"Downloading {ticker} from Yahoo Finance...")

    # Fetch stock data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    if len(hist) == 0:
        print(f"✗ No data available for {ticker}")
        print("  Please verify the ticker symbol is correct")
        sys.exit(1)

    # Current data
    current_price = float(hist['Close'].iloc[-1])
    prev_close = float(hist['Close'].iloc[-2])
    daily_change = (current_price - prev_close) / prev_close

    # Volume
    current_volume = int(hist['Volume'].iloc[-1])
    avg_volume = int(hist['Volume'].mean())
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    # Volatility
    returns = hist['Close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252))

    # Moving averages
    ma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
    ma_50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else ma_20

    price_vs_ma20 = (current_price - ma_20) / ma_20 if ma_20 > 0 else 0
    price_vs_ma50 = (current_price - ma_50) / ma_50 if ma_50 > 0 else 0

    # Get company info
    info = stock.info
    company_name = info.get('longName', ticker)

    print(f"✓ Data fetched successfully")
    print()
    print(f"Company: {company_name}")
    print(f"Current Price: ${current_price:.2f} ({daily_change:+.2%})")
    print(f"Volatility: {volatility:.1%}")
    print(f"Volume: {current_volume:,} ({volume_ratio:.2f}x avg)")
    print()

except ImportError:
    print("✗ ERROR: yfinance not installed")
    print("  Install with: pip install yfinance pandas numpy")
    sys.exit(1)

except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# Market regime analysis
if volatility < 0.20:
    vol_regime = "LOW"
    vol_adjustment = 1.2
elif volatility < 0.40:
    vol_regime = "NORMAL"
    vol_adjustment = 1.0
else:
    vol_regime = "HIGH"
    vol_adjustment = 0.8

# Trend analysis
if price_vs_ma20 > 0.05 and price_vs_ma50 > 0.05:
    trend = "STRONG UPTREND"
    trend_signal = "BULLISH"
    trend_score = 0.9
elif price_vs_ma20 > 0 and price_vs_ma50 > 0:
    trend = "UPTREND"
    trend_signal = "BULLISH"
    trend_score = 0.7
elif price_vs_ma20 < -0.05 and price_vs_ma50 < -0.05:
    trend = "STRONG DOWNTREND"
    trend_signal = "BEARISH"
    trend_score = 0.3
elif price_vs_ma20 < 0 and price_vs_ma50 < 0:
    trend = "DOWNTREND"
    trend_signal = "BEARISH"
    trend_score = 0.4
else:
    trend = "SIDEWAYS"
    trend_signal = "NEUTRAL"
    trend_score = 0.5

# Volume score
if volume_ratio > 1.5:
    volume_score = 0.9
elif volume_ratio > 1.2:
    volume_score = 0.7
else:
    volume_score = 0.5

print("-" * 80)
print("MARKET ANALYSIS")
print("-" * 80)
print(f"Volatility: {vol_regime} ({volatility:.1%})")
print(f"Trend: {trend} ({trend_signal})")
print(f"Price vs 20-MA: {price_vs_ma20:+.1%}")
print(f"Price vs 50-MA: {price_vs_ma50:+.1%}")
print()

# Position sizing
try:
    from automoonbot.moonpy.risk_management import FixedFractionalSizer
    sizer = FixedFractionalSizer(
        risk_per_trade=risk_per_trade,
        stop_loss_pct=stop_loss_pct,
        max_position_size=max_position,
        min_position_size=0.01
    )
    position_size = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        asset_price=current_price
    )
    print("✓ Using AutoMoonBot FixedFractionalSizer")
except:
    # Fallback
    risk_amount = portfolio_value * risk_per_trade
    position_dollars = risk_amount / stop_loss_pct
    position_size = position_dollars / portfolio_value
    position_size = min(position_size, max_position)
    position_size = max(position_size, 0.01)
    print("✓ Using FixedFractionalSizer algorithm")

# Apply adjustments
adjusted_position = position_size * vol_adjustment
final_position = min(adjusted_position, max_position)

position_value = final_position * portfolio_value
shares = int(position_value / current_price)
actual_invested = shares * current_price
actual_position_pct = actual_invested / portfolio_value

max_risk_dollars = shares * current_price * stop_loss_pct
max_risk_pct = max_risk_dollars / portfolio_value

print()
print("-" * 80)
print("POSITION SIZING")
print("-" * 80)
print(f"Recommended Shares: {shares:,}")
print(f"@ ${current_price:.2f} = ${actual_invested:,.2f}")
print(f"Position Size: {actual_position_pct:.1%} of portfolio")
print(f"Max Risk: ${max_risk_dollars:,.2f} ({max_risk_pct:.2%})")
print()

# Entry/Exit
entry_price = current_price
stop_price = entry_price * (1 - stop_loss_pct)
risk_per_share = entry_price - stop_price

target_2_0 = entry_price + (risk_per_share * 2.0)
gain_2_0 = shares * (target_2_0 - entry_price)

print("-" * 80)
print("ENTRY/EXIT TARGETS")
print("-" * 80)
print(f"Entry: ${entry_price:.2f}")
print(f"Stop Loss: ${stop_price:.2f} (-{stop_loss_pct*100:.0f}%)")
print(f"Take Profit: ${target_2_0:.2f} (+{((target_2_0/entry_price-1)*100):.0f}%)")
print(f"Target Gain: ${gain_2_0:+,.0f}")
print()

# Signal generation
risk_warnings = []
if risk_per_trade > 0.10:
    risk_warnings.append(f"EXTREME risk ({risk_per_trade*100:.0f}%)")
elif risk_per_trade > 0.05:
    risk_warnings.append(f"High risk ({risk_per_trade*100:.0f}%)")

if volatility > 0.50:
    risk_warnings.append("Extreme volatility")

confidence = (trend_score * 0.4 + volume_score * 0.3 + (1 - min(len(risk_warnings) * 0.15, 0.4)) * 0.3)

if trend_signal == "BULLISH" and len(risk_warnings) <= 2:
    signal = "BUY"
elif trend_signal == "BULLISH":
    signal = "CONDITIONAL BUY"
elif trend_signal == "BEARISH":
    signal = "AVOID"
else:
    signal = "HOLD"

print("=" * 80)
print("TRADING SIGNAL")
print("=" * 80)
print(f"Signal: {signal}")
print(f"Confidence: {confidence:.0%}")
print()

if signal in ["BUY", "CONDITIONAL BUY"]:
    print(f"RECOMMENDATION: {signal} {shares:,} shares @ ${entry_price:.2f}")
    print(f"  Investment: ${actual_invested:,.2f}")
    print(f"  Stop: ${stop_price:.2f}")
    print(f"  Target: ${target_2_0:.2f}")
    if risk_warnings:
        print()
        print("  WARNINGS:")
        for w in risk_warnings:
            print(f"    • {w}")
else:
    print(f"RECOMMENDATION: {signal}")
    print(f"  Reason: {trend} - wait for better conditions")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("DISCLAIMER: This is analysis only - NOT financial advice")
print("=" * 80)
