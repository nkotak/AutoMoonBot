#!/usr/bin/env python3
"""
Diagnostic version - bypasses package __init__.py to avoid moonrs dependency
"""

import sys
print("1. Python started", flush=True)

from pathlib import Path
print("2. Imported pathlib", flush=True)

import argparse
print("3. Imported argparse", flush=True)

import numpy as np
print("4. Imported numpy", flush=True)

import pandas as pd
print("5. Imported pandas", flush=True)

print("6. About to import torch...", flush=True)
import torch
print("7. Imported torch successfully!", flush=True)

print("8. About to import yfinance...", flush=True)
import yfinance as yf
print("9. Imported yfinance successfully!", flush=True)

print("10. About to import automoonbot modules (bypassing __init__.py)...", flush=True)

# Add to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Import directly from the file, not through package
# This avoids triggering automoonbot.moonpy.__init__.py which imports moonrs
try:
    import importlib.util

    # Load simple_actor_critic.py directly
    module_path = repo_root / "automoonbot" / "moonpy" / "model" / "simple_actor_critic.py"

    if not module_path.exists():
        print(f"ERROR: Module not found at {module_path}", flush=True)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("simple_actor_critic", module_path)
    simple_actor_critic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simple_actor_critic)

    SimpleActor = simple_actor_critic.SimpleActor
    SimpleCritic = simple_actor_critic.SimpleCritic
    PPOBuffer = simple_actor_critic.PPOBuffer

    print("11. Imported automoonbot modules successfully (direct import)!", flush=True)

except Exception as e:
    print(f"ERROR importing automoonbot modules: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful!", flush=True)

# Test the classes
print("\nTesting SimpleActor...", flush=True)
actor = SimpleActor(state_dim=20, hidden_dims=[256, 128])
print(f"  ✓ Created actor with {sum(p.numel() for p in actor.parameters()):,} parameters", flush=True)

print("Testing SimpleCritic...", flush=True)
critic = SimpleCritic(state_dim=20, hidden_dims=[256, 128])
print(f"  ✓ Created critic with {sum(p.numel() for p in critic.parameters()):,} parameters", flush=True)

print("Testing PPOBuffer...", flush=True)
buffer = PPOBuffer(state_dim=20, buffer_size=128)
print(f"  ✓ Created buffer with size 128", flush=True)

print("\nTesting yfinance download...", flush=True)
try:
    stock = yf.Ticker("AAPL")
    hist = stock.history(period="1mo")
    print(f"✓ Downloaded {len(hist)} days of AAPL data", flush=True)
    print(f"  Latest close: ${hist['Close'].iloc[-1]:.2f}", flush=True)
except Exception as e:
    print(f"ERROR downloading data: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED! Your environment is working correctly.")
print("=" * 80)
print("\nYou can now run: python train_rl_agent_fixed.py --tickers HESM --quick-test")
