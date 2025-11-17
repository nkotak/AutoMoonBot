# AutoMoonBot Quick Start Guide

## What You Need to Provide

### 1. **Stock Ticker Symbol**
```python
ticker = "AAPL"  # Or any stock: TSLA, MSFT, GOOGL, etc.
```

### 2. **Your Portfolio Information**
```python
portfolio_value = 100000   # How much capital you have
risk_per_trade = 0.02      # How much to risk per trade (2%)
max_position = 0.25        # Maximum position size (25%)
```

### 3. **Market Data** (Automatically Fetched)
Install yfinance to fetch real market data:
```bash
pip install yfinance pandas numpy torch
```

## What You Get (OUTPUT)

AutoMoonBot provides **analysis and recommendations** - it does NOT execute trades.

### Output Format:

```
SIGNAL: BUY AAPL
Confidence: 75%

RECOMMENDATION:
  BUY 166 shares @ $150.00
  Total Investment: $25,000 (25% of portfolio)

  SET Stop Loss: $142.50 (-5%)
  SET Take Profit: $165.00 (+10%)

  Max Risk: $1,245 (1.24% of portfolio)
  Target Gain: $2,490
  Risk/Reward: 2.0:1

RISK ANALYSIS:
  ✓ Position Concentration: 25% (within limits)
  ✓ Risk Amount: 1.24% (within 2% target)
  ✓ Volatility: 25% (acceptable)

MARKET REGIME:
  Trend: BULLISH (weak uptrend)
  Volatility: HIGH (25% annualized)
  Volume: ABOVE AVERAGE
```

## How to Use It

### Step 1: Install Dependencies
```bash
cd /home/user/AutoMoonBot
pip install yfinance pandas numpy torch
```

### Step 2: Run Analysis Script
```bash
python /tmp/automoonbot_integration_example.py
```

Or modify the ticker in the script:
```python
# Open the file and change this line:
TICKER = "TSLA"  # Change to your ticker
PORTFOLIO_VALUE = 50000  # Change to your capital
```

### Step 3: Review Output
The script will output:
- **Trading Signal** (BUY/HOLD/SELL)
- **Position Size** (how many shares)
- **Entry Price** (current market price)
- **Stop Loss** (where to exit if losing)
- **Take Profit** (where to exit if winning)
- **Risk Analysis** (exposure, limits, validation)
- **Market Regime** (volatility, trend, volume)

### Step 4: Make Your Decision
**YOU** decide whether to execute the trade:
- Review the recommendation
- Validate with your own analysis
- Manually place the trade in your broker if you agree
- Set the stop loss and take profit levels as recommended

## What AutoMoonBot Does NOT Do

❌ Does NOT automatically execute trades
❌ Does NOT connect to your broker
❌ Does NOT manage your positions
❌ Does NOT require real-time data streaming

## What AutoMoonBot DOES Do

✅ Fetches real market data (Yahoo Finance)
✅ Analyzes market conditions using RL algorithms
✅ Calculates optimal position sizes
✅ Determines entry/exit prices
✅ Validates against risk limits
✅ Provides actionable recommendations

## Example Workflow

1. **You provide**: "AAPL" ticker, $100,000 portfolio
2. **AutoMoonBot fetches**: Current AAPL price, volume, volatility
3. **AutoMoonBot analyzes**: Market regime, trend, risk
4. **AutoMoonBot calculates**: Position size (25%), stop ($142.50), target ($165)
5. **AutoMoonBot validates**: Risk limits, concentration, drawdown
6. **AutoMoonBot outputs**: "BUY 166 shares @ $150.00"
7. **You decide**: Review recommendation and execute manually (or not)

## Real Data Sources

AutoMoonBot can fetch data from:
- **Yahoo Finance** (via yfinance - FREE)
- **Alpha Vantage** (API key required)
- **IEX Cloud** (API key required)
- **Alpaca** (requires account)

Default: Yahoo Finance (no API key needed)

## No Simulations

When you use real tickers, AutoMoonBot uses **actual market data**:
- Real prices
- Real volumes
- Real volatility
- Real market conditions

The only "example" data is used when:
- yfinance is not installed, OR
- The ticker symbol doesn't exist

Otherwise, all data is REAL.

## Quick Test

Try this right now:

```bash
# Install dependencies
pip install yfinance

# Run analysis on Apple stock
python -c "
import yfinance as yf
stock = yf.Ticker('AAPL')
hist = stock.history(period='1d')
price = hist['Close'].iloc[-1]
print(f'AAPL Current Price: \${price:.2f}')
print('Data is REAL - fetched from Yahoo Finance')
"
```

## Files You Can Run

1. **Real Market Analysis** (basic):
   ```bash
   python /tmp/real_market_analysis.py
   ```

2. **Full Integration** (complete):
   ```bash
   python /tmp/automoonbot_integration_example.py
   ```

3. **Production Demo** (risk management):
   ```bash
   python /tmp/demo_actual_risk_management.py
   ```

## Customization

Edit these variables in the scripts:

```python
TICKER = "AAPL"           # ← Your stock
PORTFOLIO_VALUE = 100000  # ← Your capital
RISK_PER_TRADE = 0.02     # ← Your risk (2%)
STOP_LOSS_PCT = 0.05      # ← Your stop (5%)
```

## Next Steps

1. Install dependencies: `pip install yfinance pandas numpy torch`
2. Run example script: `python /tmp/automoonbot_integration_example.py`
3. Review the output and recommendations
4. Modify ticker and parameters to your needs
5. Use recommendations to inform your manual trading decisions

## Summary

**Input**: Stock ticker + Portfolio info
**Process**: Real market data → RL analysis → Risk management
**Output**: Trading recommendations with entry/exit/risk info
**Action**: YOU manually execute trades (or not)

No simulations. No automatic trading. Just analysis and recommendations.
