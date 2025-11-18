# Building moonrs: Rust Extension for Graph-Based RL

## What is moonrs?

`moonrs` is a high-performance Rust extension that provides heterogeneous graph data structures for AutoMoonBot's graph-based reinforcement learning architecture. It enables modeling multi-asset portfolios as graphs where:

- **Nodes**: Represent different asset types (equity, currency, bonds, options, news, macro data)
- **Edges**: Represent relationships and correlations between assets
- **Performance**: Rust implementation provides 10-100x speed improvements over pure Python

## When do you need moonrs?

### ✅ You NEED moonrs for:
- **Graph-based Actor** (`automoonbot/moonpy/model/actor.py`) - Uses heterogeneous GNNs for multi-asset portfolio management
- **Advanced graph features** in `automoonbot/moonpy/data/wrapper.py`

### ❌ You DON'T need moonrs for:
- **Simple Actor-Critic** (`automoonbot/moonpy/model/simple_actor_critic.py`) - MLP-based, works for single stock trading
- **Portfolio Actor-Critic** (`automoonbot/moonpy/model/portfolio_actor_critic.py`) - Transformer-based, works without graphs
- Basic RL training with `train_rl_agent_final.py`

---

## Building moonrs on macOS (ARM64)

### Prerequisites

1. **Rust toolchain**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Python development headers**:
   ```bash
   brew install python@3.11
   ```

3. **maturin** (Rust-Python build tool):
   ```bash
   pip install maturin
   ```

---

## Build Methods

### Method 1: maturin build (Recommended - No virtualenv required)

This method builds a wheel file that you can install anywhere:

```bash
# Navigate to moonrs directory
cd /path/to/AutoMoonBot/automoonbot/moonrs

# Build the wheel (release mode for performance)
# IMPORTANT: Use --bindings pyo3 to avoid cffi detection issues
maturin build --release --bindings pyo3 --features python

# The wheel will be created in target/wheels/
# Install it with pip
pip install target/wheels/moonrs-0.1.0-*.whl
```

**Common Issues**:
- If maturin says "Found cffi bindings", add `--bindings pyo3` flag
- If it uses the wrong Python version, specify with `--interpreter python3.11`
- If wheel installation fails with "not a supported wheel on this platform", rebuild for your Python version:
  ```bash
  # Check your Python version
  python --version
  # Build for that specific version (e.g., python3.11)
  maturin build --release --bindings pyo3 --features python --interpreter python3.11
  ```

**Advantages**:
- No virtualenv required
- Creates portable wheel file
- Can distribute to other machines
- Clean separation between build and install

**Verify installation**:
```bash
python -c "import moonrs; print('moonrs imported successfully!')"
```

---

### Method 2: maturin develop (For active development)

This method installs directly into your Python environment for rapid iteration:

```bash
# Navigate to project root
cd /path/to/AutoMoonBot

# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Navigate to moonrs and build
cd automoonbot/moonrs
maturin develop --release --bindings pyo3 --features python

# moonrs is now installed in the virtualenv
```

**Advantages**:
- Faster iteration during development
- Automatic reinstall with code changes
- Integrated with Python development workflow

**Disadvantages**:
- Requires virtualenv
- Must rebuild after every change
- Only available in that specific virtualenv

---

### Method 3: cargo build (Not recommended - Complex Python linking)

Direct cargo build requires manual Python linking configuration:

```bash
cd automoonbot/moonrs

# This will likely fail with linking errors on macOS:
cargo build --release --features python
```

**Why it fails**:
- cargo doesn't automatically link against Python's C API on macOS
- Requires manual configuration of `RUSTFLAGS` with Python library paths
- maturin handles this automatically

**Only use if**: You need fine-grained control over the build process and understand Rust-Python FFI

---

## Troubleshooting

### Error: "Found cffi bindings" / "ModuleNotFoundError: No module named 'cffi'"

This happens when maturin incorrectly detects cffi bindings instead of PyO3.

**Solution**: Explicitly specify PyO3 bindings:
```bash
maturin build --release --bindings pyo3 --features python
pip install target/wheels/moonrs-*.whl
```

**Alternative**: If you want to use cffi mode, install cffi for the Python version maturin is using:
```bash
# Check which Python version maturin is using (shown in error message)
python3.13 -m pip install cffi
# Then retry: maturin build --release
```

### Error: "Couldn't find a virtualenv or conda environment"

**Solution**: Use `maturin build` instead of `maturin develop`, or create a virtualenv first.

```bash
# Quick fix - use build instead
maturin build --release --bindings pyo3 --features python
pip install target/wheels/moonrs-*.whl
```

### Error: "is not a supported wheel on this platform"

This happens when the wheel was built for a different Python version than the one you're using with pip.

**Example**: Wheel built for Python 3.13 (`cp313`) but pip is using Python 3.11.

**Solution**: Build the wheel for your specific Python version:
```bash
# Check which Python version pip is using
python --version
pip --version

# Build for that specific version (e.g., 3.11)
maturin build --release --bindings pyo3 --features python --interpreter python3.11

# Or find the correct wheel in target/wheels/
ls -lh target/wheels/
pip install target/wheels/moonrs-0.1.0-cp311-*.whl
```

**Alternative**: Use the Python version the wheel was built for:
```bash
# If wheel is cp313 (Python 3.13)
python3.13 -m pip install target/wheels/moonrs-0.1.0-cp313-*.whl
```

### Error: "Undefined symbols for architecture arm64"

This happens when using `cargo build` directly. **Solution**: Use maturin instead.

### Error: "ModuleNotFoundError: No module named 'moonrs'" after building

**Check installation**:
```bash
pip list | grep moonrs
```

If not listed:
```bash
# Find the wheel and install it
pip install automoonbot/moonrs/target/wheels/moonrs-*.whl
```

### Build is slow

**Expected**: First build compiles all dependencies (~5-10 minutes). Subsequent builds are much faster.

**Speed up**:
```bash
# Use more CPU cores
export CARGO_BUILD_JOBS=8
maturin build --release
```

---

## Verifying the Build

### Test 1: Import moonrs
```bash
python -c "import moonrs; print(dir(moonrs))"
```

Expected output:
```
['HeteroGraph', '__doc__', '__file__', ...]
```

### Test 2: Import graph-based Actor
```bash
python -c "from automoonbot.moonpy.model.actor import Actor; print('Graph Actor imported!')"
```

### Test 3: Create a HeteroGraph
```python
from moonrs import HeteroGraph
from automoonbot.moonpy.data.wrapper import HeteroGraphWrapper

# Create graph
graph = HeteroGraphWrapper()
print(f"Graph created: {graph}")
```

---

## Using the Graph-Based System

Once moonrs is built, you can use the advanced graph-based Actor:

```python
from automoonbot.moonpy.model.actor import Actor
from automoonbot.moonpy.data.wrapper import HeteroGraphWrapper

# Create graph structure for multi-asset portfolio
graph = HeteroGraphWrapper()

# Add nodes for different asset types
graph.add_nodes("equity", ["AAPL", "TSLA", "MSFT"])
graph.add_nodes("currency", ["USD", "EUR"])
graph.add_nodes("bonds", ["TLT"])

# Add edges for correlations
graph.add_edges("equity", "equity", "correlation")
graph.add_edges("equity", "currency", "forex_impact")

# Initialize graph-based Actor
actor = Actor(
    node_types=["equity", "currency", "bonds"],
    edge_types=[("equity", "correlation", "equity"),
                ("equity", "forex_impact", "currency")],
    hidden_channels=128,
    num_heads=4
)

# Use with RL training
state_graph = prepare_graph_state(market_data)
action = actor(state_graph)
```

---

## Architecture: Graph vs Simple vs Portfolio

### Simple Actor-Critic (No moonrs needed)
- **Input**: 20-dim state vector (single stock)
- **Architecture**: MLP (256 → 128)
- **Use case**: Single stock trading
- **File**: `automoonbot/moonpy/model/simple_actor_critic.py`

### Portfolio Actor-Critic (No moonrs needed)
- **Input**: Sequence of multiple stock states
- **Architecture**: Transformer with cross-attention
- **Use case**: Multi-stock portfolio (treats stocks independently)
- **File**: `automoonbot/moonpy/model/portfolio_actor_critic.py`

### Graph-Based Actor (Requires moonrs)
- **Input**: Heterogeneous graph with multiple asset types
- **Architecture**: Graph Attention Network (GAT) with heterogeneous support
- **Use case**: Multi-asset portfolio with explicit relationship modeling
- **File**: `automoonbot/moonpy/model/actor.py`
- **Advantage**: Learns correlations, sector relationships, forex impacts, etc.

**Key insight**: The graph-based approach can model complex relationships like:
- Tech stocks moving together (sector correlation)
- Currency fluctuations affecting international stocks
- Bond yields affecting equity valuations
- News sentiment propagating through supply chains

---

## Next Steps After Building moonrs

1. **Test the installation**:
   ```bash
   python -c "from automoonbot.moonpy.model.actor import Actor; print('Success!')"
   ```

2. **Create a graph-based training script**:
   - Similar to `train_rl_agent_final.py`
   - But uses graph-based Actor and HeteroGraphWrapper
   - Models multi-asset portfolios as graphs

3. **Collect multi-modal data**:
   - Stock prices (yfinance)
   - Currency rates
   - Bond yields
   - News sentiment
   - Macro indicators

4. **Train with graph structure**:
   - Edges between correlated stocks
   - Edges between stocks and currencies
   - Edges between bonds and equities

---

## Quick Reference

| Task | Command |
|------|---------|
| Build wheel | `maturin build --release --bindings pyo3 --features python` |
| Install wheel | `pip install target/wheels/moonrs-*.whl` |
| Dev install | `maturin develop --release --bindings pyo3 --features python` (requires venv) |
| Check install | `python -c "import moonrs"` |
| Clean build | `cargo clean && maturin build --release --bindings pyo3 --features python` |

---

## Support

If you encounter issues:

1. Check Python version: `python --version` (should be 3.8+)
2. Check Rust version: `rustc --version` (should be 1.70+)
3. Check maturin version: `maturin --version` (should be 1.0+)
4. Try clean build: `cargo clean && maturin build --release`
5. Check build logs in `target/` directory

---

**Built with ❤️ using PyO3 and maturin**
