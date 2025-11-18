# AutoMoonBot: RL Training Architecture Plan

## Your Questions Answered

### 1. **What Model Will We Use?**

**ANSWER: PPO (Proximal Policy Optimization)**

**Why PPO and not GRPO?**
- **PPO** is proven, stable, and industry-standard for continuous control
- **GRPO** (Group Relative Policy Optimization) is newer, designed for LLM fine-tuning
- For financial RL, PPO is better because:
  - ✅ Handles continuous action spaces (position sizes)
  - ✅ Stable training with clipped objectives
  - ✅ Works well with sparse rewards (trading)
  - ✅ Well-documented and battle-tested

**PPO Components:**
```
Actor Network (Policy):
  Input: Market state (prices, volume, indicators)
  Output: Action distribution (BUY/SELL/HOLD + position size)

Critic Network (Value):
  Input: Market state
  Output: Expected future return

Training:
  Uses clipped surrogate objective to prevent large policy updates
  Balances exploration vs exploitation
```

---

### 2. **Incremental Training (Not Retraining from Scratch)**

**ANSWER: Model Checkpointing + Incremental Learning**

**How it works:**
```python
# First time: Train from scratch
python train_rl.py --tickers HESM AAPL TSLA --episodes 5000
# Saves: models/trading_agent_checkpoint.pth

# Later: Continue training with new data
python train_rl.py --tickers MSFT GOOGL --resume models/trading_agent_checkpoint.pth
# Loads existing weights, trains on new data, saves updated model
```

**Key Features:**
- ✅ Saves model weights after each epoch
- ✅ Loads previous weights before training
- ✅ Updates existing knowledge (doesn't forget)
- ✅ Tracks training history across sessions

**Implementation:**
```python
class TrainingManager:
    def __init__(self, resume_from=None):
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()

        if resume_from:
            # Load existing weights
            checkpoint = torch.load(resume_from)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.episode_count = checkpoint['episode']
            print(f"Resumed from episode {self.episode_count}")
        else:
            self.episode_count = 0

    def save_checkpoint(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'episode': self.episode_count,
            'timestamp': datetime.now()
        }, path)
```

---

### 3. **Training on Multiple Tickers**

**ANSWER: Multi-Ticker Training Pipeline**

**How it works:**
```python
# Train on multiple tickers simultaneously
tickers = ["HESM", "AAPL", "TSLA", "MSFT", "GOOGL"]

for ticker in tickers:
    # Download data from yfinance
    data = yf.download(ticker, period="5y")

    # Train agent on this ticker's data
    for episode in range(1000):
        env = TradingEnvironment(data=data, ticker=ticker)
        # Agent learns patterns specific to this ticker
        # BUT keeps general trading knowledge from other tickers
```

**Benefits:**
- ✅ Learns general market patterns (volatility, trends, volume)
- ✅ Learns ticker-specific patterns (HESM behavior vs AAPL behavior)
- ✅ Better generalization (doesn't overfit to one stock)

---

### 4. **What Data Structure from yfinance?**

**ANSWER: OHLCV + Technical Indicators**

**Data Pipeline:**
```python
# Step 1: Download from yfinance
df = yf.download("HESM", period="5y", interval="1d")

# df has:
# - Open, High, Low, Close, Volume
# - Date index

# Step 2: Calculate features
df['returns'] = df['Close'].pct_change()
df['volatility'] = df['returns'].rolling(20).std()
df['ma_20'] = df['Close'].rolling(20).mean()
df['ma_50'] = df['Close'].rolling(50).mean()
df['rsi'] = calculate_rsi(df['Close'])
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

# Step 3: Create state vector for RL
state = [
    df['Close'].iloc[-1] / df['Close'].iloc[-2],  # Normalized price
    df['Volume'].iloc[-1] / df['Volume'].mean(),  # Volume ratio
    df['ma_20'].iloc[-1] / df['Close'].iloc[-1],  # Price vs MA20
    df['ma_50'].iloc[-1] / df['Close'].iloc[-1],  # Price vs MA50
    df['volatility'].iloc[-1],                    # Current volatility
    df['rsi'].iloc[-1] / 100,                     # RSI (normalized)
    # ... more features
]

# Step 4: Feed to neural network
action = actor_network(state)
```

**State Space Design (What the model sees):**
```
State Vector (20 dimensions):
  [0] = Normalized close price (current / previous)
  [1] = Normalized volume (current / average)
  [2] = Price momentum (5-day)
  [3] = Price momentum (20-day)
  [4] = RSI (Relative Strength Index)
  [5] = MACD
  [6] = Bollinger Band position
  [7] = Volatility (20-day)
  [8] = Distance from 20-day MA
  [9] = Distance from 50-day MA
  [10] = Volume trend
  [11] = Intraday range (high-low)
  [12-15] = Price history (last 4 days, normalized)
  [16] = Current position (if holding)
  [17] = Unrealized P&L
  [18] = Portfolio value change
  [19] = Days held
```

**Action Space:**
```
Discrete Actions (3 options):
  0 = SELL/CLOSE position
  1 = HOLD (do nothing)
  2 = BUY/OPEN position

Continuous Parameter:
  position_size = [0.0, 1.0]  # What % of portfolio to use
```

---

### 5. **How Training Works (Step by Step)**

**Training Loop:**
```python
# Initialize
actor = ActorNetwork(state_dim=20, action_dim=3)
critic = CriticNetwork(state_dim=20)
env = TradingEnvironment()

# Training loop
for episode in range(10000):
    # Reset environment (start new trading period)
    state = env.reset()  # Random start date in historical data
    episode_reward = 0

    for step in range(252):  # 252 trading days per year
        # Actor decides action
        action, position_size = actor.get_action(state)

        # Execute trade in environment
        next_state, reward, done, info = env.step(action, position_size)

        # Store experience
        buffer.store(state, action, reward, next_state, done)

        # Update networks every N steps
        if len(buffer) > batch_size:
            actor_loss = actor.update(buffer)
            critic_loss = critic.update(buffer)

        state = next_state
        episode_reward += reward

        if done:
            break

    # Log progress
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    # Save checkpoint every 100 episodes
    if episode % 100 == 0:
        save_checkpoint(f"models/checkpoint_{episode}.pth")
```

---

### 6. **Reward Function Design**

**What the model learns to maximize:**
```python
def calculate_reward(self, action, position_size, state, next_state):
    reward = 0

    # 1. Profit/Loss (main signal)
    if self.position > 0:
        pnl = (next_state['price'] - self.entry_price) * self.position
        reward += pnl / self.portfolio_value  # Normalize by portfolio

    # 2. Risk penalty (discourage large positions in high volatility)
    risk_penalty = -position_size * state['volatility'] * 0.1
    reward += risk_penalty

    # 3. Transaction cost penalty
    if action != HOLD:
        reward -= 0.001  # 0.1% transaction cost

    # 4. Drawdown penalty (discourage large losses)
    if self.current_drawdown > 0.10:
        reward -= self.current_drawdown * 10

    # 5. Sharpe ratio bonus (reward risk-adjusted returns)
    if len(self.returns_history) > 20:
        sharpe = np.mean(self.returns_history) / np.std(self.returns_history)
        reward += sharpe * 0.01

    return reward
```

**What this teaches the model:**
- ✅ Maximize profits
- ✅ Minimize risk
- ✅ Avoid excessive trading (transaction costs)
- ✅ Prevent large drawdowns
- ✅ Optimize risk-adjusted returns

---

### 7. **Integration with interactive_analysis.py**

**After training, use trained model:**
```python
# In interactive_analysis.py

# Load trained model
actor = ActorNetwork(state_dim=20, action_dim=3)
actor.load_state_dict(torch.load('models/trading_agent.pth'))
actor.eval()

# Prepare state from current market data
state = prepare_state(current_price, hist_data, indicators)

# Get AI decision
with torch.no_grad():
    action_probs, position_size = actor(state)
    action = torch.argmax(action_probs).item()

# Convert to signal
if action == 2:  # BUY
    signal = "BUY"
    confidence = action_probs[2].item()  # Probability of BUY action
    recommended_size = position_size.item()  # AI's position size
elif action == 0:  # SELL/AVOID
    signal = "AVOID"
    confidence = action_probs[0].item()
else:  # HOLD
    signal = "HOLD"
    confidence = action_probs[1].item()

print(f"AI Signal: {signal} (confidence: {confidence:.1%})")
print(f"AI Recommended Size: {recommended_size:.1%}")
```

---

## Training Pipeline Summary

```
Step 1: Collect Data
  ├─ Download from yfinance (HESM, AAPL, TSLA, etc.)
  ├─ Calculate technical indicators
  └─ Prepare state vectors

Step 2: Train Model
  ├─ Initialize Actor/Critic networks (or load checkpoint)
  ├─ For each ticker:
  │   ├─ Create trading environment with historical data
  │   ├─ Train for N episodes
  │   └─ Evaluate performance
  ├─ Save checkpoint every 100 episodes
  └─ Save final model

Step 3: Use Trained Model
  ├─ Load model weights
  ├─ Fetch current market data
  ├─ Prepare state vector
  ├─ Get AI action/confidence
  └─ Execute trade (or show recommendation)

Step 4: Incremental Training (Later)
  ├─ Load existing checkpoint
  ├─ Add new tickers or more recent data
  ├─ Continue training (doesn't forget previous knowledge)
  └─ Save updated checkpoint
```

---

## Expected Training Time & Requirements

**Hardware:**
- CPU: ~6-12 hours for 10,000 episodes
- GPU (CUDA): ~2-4 hours for 10,000 episodes

**Data Requirements:**
- 5 years of daily data per ticker
- ~1,260 trading days × 5 tickers = 6,300 data points
- Storage: ~50MB for model + data

**Training Metrics:**
- Episode reward should increase over time
- Portfolio value should trend upward
- Sharpe ratio should improve
- Win rate should stabilize above 50%

---

## Next Steps

1. **I'll create**: `train_rl_agent.py` - Full training script
2. **I'll create**: `models/` directory structure
3. **I'll modify**: `interactive_analysis.py` - Add AI mode
4. **I'll create**: `test_rl_agent.py` - Evaluate trained model

**You'll be able to**:
```bash
# Train from scratch
python train_rl_agent.py --tickers HESM AAPL TSLA --episodes 5000

# Continue training later
python train_rl_agent.py --tickers MSFT GOOGL --resume models/checkpoint.pth

# Use trained model
python interactive_analysis.py --use-ai
```

Ready to implement?
