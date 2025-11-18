"""
AutoMoonBot MoonPy Module

This package contains trading RL components.

Submodules:
  - data: Graph data structures (requires moonrs Rust extension)
  - environment: Trading environment for RL
  - model: Neural network architectures
  - reward: Reward functions
  - risk_management: Position sizing, stops, limits

Import submodules explicitly:
  from automoonbot.moonpy.model.simple_actor_critic import SimpleActor
  from automoonbot.moonpy.environment import TradingEnvironment
  etc.
"""

# Empty __init__.py - import submodules explicitly as needed
# This avoids forcing moonrs dependency on all imports
