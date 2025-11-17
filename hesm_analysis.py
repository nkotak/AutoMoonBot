#!/usr/bin/env python3
"""
AutoMoonBot: HESM Analysis Script
Hess Midstream LP (NYSE: HESM)

Your Parameters:
- Portfolio: $70,000
- Risk: 25% per trade
- Stock: HESM (Hess Midstream LP)

INSTRUCTIONS TO RUN:
1. Save this file: hesm_analysis.py
2. Install dependencies: pip install yfinance pandas numpy
3. Run: python hesm_analysis.py
"""

import sys
import os

print("=" * 80)
print("AUTOMOONBOT: HESM (HESS MIDSTREAM LP) ANALYSIS")
print("=" * 80)
print()

# Add AutoMoonBot to path if running from repo
if os.path.exists('/home/user/AutoMoonBot'):
    sys.path.insert(0, '/home/user/AutoMoonBot')

# ============================================================================
# YOUR CONFIGURATION
# ============================================================================
print("YOUR CONFIGURATION")
print("-" * 80)

TICKER = "HESM"
COMPANY_NAME = "Hess Midstream LP"
EXCHANGE = "NYSE"
PORTFOLIO_VALUE = 70000
RISK_PER_TRADE = 0.25        # 25% risk per trade (VERY AGGRESSIVE!)
STOP_LOSS_PCT = 0.05         # 5% stop loss
MAX_POSITION = 0.50          # Max 50% in any position

print(f"Ticker: {TICKER} ({COMPANY_NAME})")
print(f"Exchange: {EXCHANGE}")
print(f"Portfolio Value: ${PORTFOLIO_VALUE:,}")
print(f"Risk per Trade: {RISK_PER_TRADE:.1%} ⚠ VERY AGGRESSIVE")
print(f"Stop Loss: {STOP_LOSS_PCT:.1%}")
print(f"Max Position: {MAX_POSITION:.1%}")
print()

# ============================================================================
# STEP 1: Import and Fetch Real Data
# ============================================================================
print("STEP 1: FETCHING REAL MARKET DATA")
print("-" * 80)

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np

    print(f"Downloading {TICKER} from Yahoo Finance...")

    # Fetch stock data
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="6mo")  # 6 months of data

    if len(hist) == 0:
        raise ValueError(f"No data received for {TICKER}")

    # Current data
    current_price = float(hist['Close'].iloc[-1])
    prev_close = float(hist['Close'].iloc[-2])
    daily_change = (current_price - prev_close) / prev_close

    # Intraday range
    high_today = float(hist['High'].iloc[-1])
    low_today = float(hist['Low'].iloc[-1])

    # Historical statistics
    high_52w = float(hist['High'].max())
    low_52w = float(hist['Low'].min())

    # Volume
    current_volume = int(hist['Volume'].iloc[-1])
    avg_volume = int(hist['Volume'].mean())
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    # Volatility (annualized)
    returns = hist['Close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252))

    # Moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()

    ma_20 = float(hist['MA20'].iloc[-1])
    ma_50 = float(hist['MA50'].iloc[-1]) if len(hist) >= 50 else ma_20

    # Price vs MAs
    price_vs_ma20 = (current_price - ma_20) / ma_20 if ma_20 > 0 else 0
    price_vs_ma50 = (current_price - ma_50) / ma_50 if ma_50 > 0 else 0

    # Get company info
    info = stock.info
    market_cap = info.get('marketCap', 0)

    print(f"✓ Data fetched successfully")
    print()
    print(f"Company: {COMPANY_NAME}")
    if market_cap > 0:
        print(f"Market Cap: ${market_cap:,.0f}")
    print()
    print(f"CURRENT PRICE:")
    print(f"  Last: ${current_price:.2f}")
    print(f"  Change: {daily_change:+.2%}")
    print(f"  Today's Range: ${low_today:.2f} - ${high_today:.2f}")
    print()
    print(f"52-WEEK RANGE:")
    print(f"  High: ${high_52w:.2f} ({((current_price-high_52w)/high_52w):.1%} from high)")
    print(f"  Low: ${low_52w:.2f} ({((current_price-low_52w)/low_52w):.1%} from low)")
    print()
    print(f"TECHNICAL INDICATORS:")
    print(f"  20-Day MA: ${ma_20:.2f} ({price_vs_ma20:+.1%})")
    print(f"  50-Day MA: ${ma_50:.2f} ({price_vs_ma50:+.1%})")
    print(f"  Volatility: {volatility:.1%} annualized")
    print()
    print(f"VOLUME:")
    print(f"  Current: {current_volume:,}")
    print(f"  Average: {avg_volume:,}")
    print(f"  Ratio: {volume_ratio:.2f}x")
    print()

    data_fetched = True

except ImportError:
    print("✗ ERROR: yfinance not installed")
    print()
    print("To install:")
    print("  pip install yfinance pandas numpy")
    print()
    print("Then run this script again.")
    sys.exit(1)

except Exception as e:
    print(f"✗ ERROR: {e}")
    print()
    print("Possible issues:")
    print("  - No internet connection")
    print("  - Yahoo Finance is temporarily down")
    print("  - Ticker symbol issue")
    print()
    sys.exit(1)

# ============================================================================
# STEP 2: Market Regime Analysis
# ============================================================================
print("STEP 2: MARKET REGIME ANALYSIS")
print("-" * 80)

# Volatility regime
if volatility < 0.20:
    vol_regime = "LOW"
    vol_adjustment = 1.2
    vol_desc = "Favorable for larger positions"
elif volatility < 0.40:
    vol_regime = "NORMAL"
    vol_adjustment = 1.0
    vol_desc = "Standard position sizing"
elif volatility < 0.60:
    vol_regime = "HIGH"
    vol_adjustment = 0.8
    vol_desc = "Reduce position sizes"
else:
    vol_regime = "EXTREME"
    vol_adjustment = 0.5
    vol_desc = "Significantly reduce positions"

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

# Volume analysis
if volume_ratio > 1.5:
    volume_signal = "VERY HIGH"
    volume_desc = "Strong institutional interest"
    volume_score = 0.9
elif volume_ratio > 1.2:
    volume_signal = "HIGH"
    volume_desc = "Above average participation"
    volume_score = 0.7
elif volume_ratio > 0.8:
    volume_signal = "NORMAL"
    volume_desc = "Average participation"
    volume_score = 0.5
else:
    volume_signal = "LOW"
    volume_desc = "Below average - be cautious"
    volume_score = 0.3

print(f"Volatility: {vol_regime} ({volatility:.1%})")
print(f"  → {vol_desc}")
print(f"  → Position adjustment: {vol_adjustment:.0%}")
print()
print(f"Trend: {trend} ({trend_signal})")
print(f"  → Price vs 20-MA: {price_vs_ma20:+.1%}")
print(f"  → Price vs 50-MA: {price_vs_ma50:+.1%}")
print()
print(f"Volume: {volume_signal} ({volume_ratio:.2f}x average)")
print(f"  → {volume_desc}")
print()

# ============================================================================
# STEP 3: Position Sizing (AutoMoonBot Algorithm)
# ============================================================================
print("STEP 3: POSITION SIZING")
print("-" * 80)

# Try to import production code
try:
    from automoonbot.moonpy.risk_management import FixedFractionalSizer

    sizer = FixedFractionalSizer(
        risk_per_trade=RISK_PER_TRADE,
        stop_loss_pct=STOP_LOSS_PCT,
        max_position_size=MAX_POSITION,
        min_position_size=0.01
    )

    position_size = sizer.calculate_position_size(
        portfolio_value=PORTFOLIO_VALUE,
        asset_price=current_price,
        stop_loss_pct=STOP_LOSS_PCT
    )

    print("✓ Using AutoMoonBot FixedFractionalSizer (Production Code)")

except:
    # Fallback: manual calculation
    risk_amount = PORTFOLIO_VALUE * RISK_PER_TRADE
    position_dollars = risk_amount / STOP_LOSS_PCT
    position_size = position_dollars / PORTFOLIO_VALUE
    position_size = min(position_size, MAX_POSITION)
    position_size = max(position_size, 0.01)

    print("✓ Using FixedFractionalSizer Algorithm")

# Apply regime adjustment
adjusted_position = position_size * vol_adjustment
final_position = min(adjusted_position, MAX_POSITION)

# Calculate investment
position_value = final_position * PORTFOLIO_VALUE
shares = int(position_value / current_price)
actual_invested = shares * current_price
actual_position_pct = actual_invested / PORTFOLIO_VALUE

# Calculate risk
max_risk_dollars = shares * current_price * STOP_LOSS_PCT
max_risk_pct = max_risk_dollars / PORTFOLIO_VALUE

print()
print(f"Calculation:")
print(f"  Risk Amount: ${PORTFOLIO_VALUE * RISK_PER_TRADE:,.0f} ({RISK_PER_TRADE:.0%} of ${PORTFOLIO_VALUE:,})")
print(f"  ÷ Stop Loss: {STOP_LOSS_PCT:.0%}")
print(f"  = Position: ${position_value:,.0f} ({position_size:.0%})")
print()
print(f"  Volatility Adjustment: ×{vol_adjustment:.0%}")
print(f"  = Adjusted Position: {adjusted_position:.0%}")
print(f"  Capped at Max: {MAX_POSITION:.0%}")
print(f"  = Final: {final_position:.0%}")
print()
print(f"RECOMMENDED INVESTMENT:")
print(f"  Shares: {shares:,}")
print(f"  @ ${current_price:.2f} per share")
print(f"  = ${actual_invested:,.2f} ({actual_position_pct:.1%} of portfolio)")
print()
print(f"RISK EXPOSURE:")
print(f"  Max Loss: ${max_risk_dollars:,.2f}")
print(f"  As % of Portfolio: {max_risk_pct:.1%}")
print()

# ============================================================================
# STEP 4: Entry/Exit Targets
# ============================================================================
print("STEP 4: ENTRY/EXIT TARGETS")
print("-" * 80)

entry_price = current_price
stop_price = entry_price * (1 - STOP_LOSS_PCT)
risk_per_share = entry_price - stop_price

# Profit targets
target_1_5 = entry_price + (risk_per_share * 1.5)
target_2_0 = entry_price + (risk_per_share * 2.0)
target_3_0 = entry_price + (risk_per_share * 3.0)

# Dollar gains
gain_1_5 = shares * (target_1_5 - entry_price)
gain_2_0 = shares * (target_2_0 - entry_price)
gain_3_0 = shares * (target_3_0 - entry_price)

print(f"Entry: ${entry_price:.2f}")
print(f"Stop Loss: ${stop_price:.2f} (-{STOP_LOSS_PCT:.0%})")
print()
print(f"Take Profit Targets:")
print(f"  T1 (1.5:1): ${target_1_5:.2f} (+{((target_1_5/entry_price-1)*100):.1f}%) = ${gain_1_5:+,.0f} profit")
print(f"  T2 (2.0:1): ${target_2_0:.2f} (+{((target_2_0/entry_price-1)*100):.1f}%) = ${gain_2_0:+,.0f} profit")
print(f"  T3 (3.0:1): ${target_3_0:.2f} (+{((target_3_0/entry_price-1)*100):.1f}%) = ${gain_3_0:+,.0f} profit")
print()

# ============================================================================
# STEP 5: Risk Assessment
# ============================================================================
print("STEP 5: RISK ASSESSMENT")
print("-" * 80)

risk_warnings = []

print("RISK CHECKS:")

# Concentration
if actual_position_pct > MAX_POSITION:
    print(f"  ✗ CONCENTRATION: {actual_position_pct:.0%} exceeds max")
    risk_warnings.append("Position too large")
elif actual_position_pct > 0.30:
    print(f"  ⚠ CONCENTRATION: {actual_position_pct:.0%} is high")
    risk_warnings.append("High concentration")
else:
    print(f"  ✓ CONCENTRATION: {actual_position_pct:.0%} OK")

# Risk level
if RISK_PER_TRADE > 0.10:
    print(f"  ⚠ RISK: {RISK_PER_TRADE:.0%} is EXTREMELY HIGH")
    risk_warnings.append(f"EXTREME risk - professionals use 1-2%")
elif RISK_PER_TRADE > 0.05:
    print(f"  ⚠ RISK: {RISK_PER_TRADE:.0%} is HIGH")
    risk_warnings.append("Aggressive risk level")
else:
    print(f"  ✓ RISK: {RISK_PER_TRADE:.0%} reasonable")

# Volatility
if volatility > 0.50:
    print(f"  ⚠ VOLATILITY: {volatility:.0%} EXTREME")
    risk_warnings.append("Extreme volatility")
elif volatility > 0.35:
    print(f"  ⚠ VOLATILITY: {volatility:.0%} HIGH")
    risk_warnings.append("High volatility")
else:
    print(f"  ✓ VOLATILITY: {volatility:.0%} OK")

# Liquidity
if volume_ratio < 0.5:
    print(f"  ⚠ LIQUIDITY: {volume_ratio:.2f}x - LOW")
    risk_warnings.append("Low trading volume")
else:
    print(f"  ✓ LIQUIDITY: {volume_ratio:.2f}x OK")

print()

if risk_warnings:
    print("⚠ WARNINGS:")
    for i, warning in enumerate(risk_warnings, 1):
        print(f"  {i}. {warning}")
    print()

# Confidence calculation
confidence = (trend_score * 0.4 + volume_score * 0.3 +
              (1 - min(len(risk_warnings) * 0.15, 0.4)) * 0.3)
confidence = max(0.3, min(0.95, confidence))

# ============================================================================
# STEP 6: Trading Signal
# ============================================================================
print("STEP 6: TRADING SIGNAL")
print("-" * 80)

if trend_signal == "BULLISH" and volume_score >= 0.5 and len(risk_warnings) <= 2:
    signal = "BUY"
    signal_strength = "STRONG" if confidence > 0.7 else "MODERATE"
elif trend_signal == "BULLISH":
    signal = "CONDITIONAL BUY"
    signal_strength = "WEAK"
elif trend_signal == "BEARISH":
    signal = "AVOID"
    signal_strength = "N/A"
else:
    signal = "HOLD"
    signal_strength = "N/A"

print(f"SIGNAL: {signal}")
print(f"Strength: {signal_strength}")
print(f"Confidence: {confidence:.0%}")
print()

if signal in ["BUY", "CONDITIONAL BUY"]:
    print("═" * 80)
    print("RECOMMENDED ACTION")
    print("═" * 80)
    print()
    print(f"  BUY {shares:,} shares of {TICKER} @ ${entry_price:.2f}")
    print()
    print(f"  Total Investment: ${actual_invested:,.2f}")
    print(f"  Portfolio Allocation: {actual_position_pct:.0%}")
    print()
    print(f"  RISK MANAGEMENT:")
    print(f"    • Stop Loss: ${stop_price:.2f} (-{STOP_LOSS_PCT:.0%})")
    print(f"    • Take Profit 1: ${target_1_5:.2f} (1.5:1)")
    print(f"    • Take Profit 2: ${target_2_0:.2f} (2.0:1)")
    print(f"    • Take Profit 3: ${target_3_0:.2f} (3.0:1)")
    print()
    print(f"  POTENTIAL OUTCOMES:")
    print(f"    • Max Loss: ${max_risk_dollars:,.0f} ({max_risk_pct:.1%})")
    print(f"    • Target Gain: ${gain_2_0:,.0f} ({(gain_2_0/PORTFOLIO_VALUE)*100:.1f}%)")
    print()

    if len(risk_warnings) > 0:
        print(f"  ⚠ CAUTIONS:")
        for warning in risk_warnings:
            print(f"      • {warning}")
        print()

    print(f"  EXECUTION PLAN:")
    print(f"    1. Place buy order for {shares:,} shares")
    print(f"    2. Immediately set stop loss at ${stop_price:.2f}")
    print(f"    3. Set take profit orders at targets above")
    print(f"    4. Monitor closely - each $1 move = ${shares:,.0f}")
    print()

elif signal == "AVOID":
    print("═" * 80)
    print("RECOMMENDATION: DO NOT BUY")
    print("═" * 80)
    print()
    print(f"  Market conditions are BEARISH")
    print(f"  Trend: {trend}")
    print(f"  Wait for trend reversal")
    print()

else:
    print("═" * 80)
    print("RECOMMENDATION: WAIT")
    print("═" * 80)
    print()
    print(f"  Market conditions unclear")
    print(f"  Wait for stronger signal")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Stock: {TICKER} - {COMPANY_NAME}")
print(f"Price: ${current_price:.2f} ({daily_change:+.2%})")
print(f"Signal: {signal} ({signal_strength})")
print()
print("CRITICAL NOTES:")
print(f"  ⚠ Your {RISK_PER_TRADE:.0%} risk setting is EXTREMELY aggressive")
print(f"  ⚠ Professional traders typically risk 1-2% per trade")
print(f"  ⚠ 4 consecutive losses = {(1-(1-RISK_PER_TRADE)**4)*100:.0f}% account loss")
print()
print("DISCLAIMER:")
print("  • This is analysis only - NOT financial advice")
print("  • NO TRADES are executed automatically")
print("  • All decisions are YOUR responsibility")
print("  • Past performance ≠ future results")
print("=" * 80)
