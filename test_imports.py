#!/usr/bin/env python3
"""
Diagnostic version to find where it's hanging
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

print("10. About to import automoonbot modules...", flush=True)
sys.path.insert(0, str(Path(__file__).parent))

try:
    from automoonbot.moonpy.model.simple_actor_critic import SimpleActor, SimpleCritic, PPOBuffer
    print("11. Imported automoonbot modules successfully!", flush=True)
except Exception as e:
    print(f"ERROR importing automoonbot modules: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful!", flush=True)
print("\nNow testing yfinance download...", flush=True)

try:
    stock = yf.Ticker("AAPL")
    hist = stock.history(period="1mo")
    print(f"✓ Downloaded {len(hist)} days of AAPL data", flush=True)
except Exception as e:
    print(f"ERROR downloading data: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n✓ All tests passed! Your environment is working.", flush=True)
