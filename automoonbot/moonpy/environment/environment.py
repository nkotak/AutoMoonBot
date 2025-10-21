import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass
from collections import deque
import copy

from automoonbot.moonpy.session.portfolio import Portfolio


class TerminationCondition(Enum):
    """Conditions that can terminate an episode."""
    MAX_STEPS = "max_steps"
    BANKRUPTCY = "bankruptcy"
    TARGET_RETURN = "target_return"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class EpisodeMetrics:
    """Metrics tracked during an episode."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    total_transaction_costs: float = 0.0
    final_portfolio_value: float = 1.0
    num_steps: int = 0
    avg_position_concentration: float = 0.0
    volatility: float = 0.0


@dataclass
class StepInfo:
    """Information returned after each environment step."""
    portfolio_value: float
    portfolio_return: float
    positions: Dict[str, float]
    num_trades_executed: int
    transaction_costs: float
    sharpe_ratio: float
    max_drawdown: float
    termination_reason: Optional[str]


class RewardCalculator:
    """
    Modular reward calculator for trading environment.

    Computes reward as weighted sum of components:
    - Returns (portfolio growth)
    - Risk penalties (volatility, drawdown)
    - Transaction cost penalties
    - Diversification rewards
    - Position holding rewards
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        risk_weight: float = 0.5,
        transaction_cost_weight: float = 0.1,
        diversification_weight: float = 0.05,
        sharpe_weight: float = 0.0,
    ):
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.transaction_cost_weight = transaction_cost_weight
        self.diversification_weight = diversification_weight
        self.sharpe_weight = sharpe_weight

    def compute_reward(
        self,
        portfolio_return: float,
        portfolio_volatility: float,
        transaction_costs: float,
        position_concentration: float,
        sharpe_ratio: float,
        drawdown: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward and component breakdown.

        Args:
            portfolio_return: Portfolio return for this step
            portfolio_volatility: Recent portfolio volatility
            transaction_costs: Transaction costs incurred this step
            position_concentration: Herfindahl index of positions (0-1)
            sharpe_ratio: Rolling Sharpe ratio
            drawdown: Current drawdown from peak

        Returns:
            total_reward: Weighted sum of components
            components: Dict of individual reward components
        """
        components = {}

        # Return component (primary signal)
        components['return'] = self.return_weight * portfolio_return

        # Risk penalty (volatility and drawdown)
        volatility_penalty = -self.risk_weight * portfolio_volatility
        drawdown_penalty = -self.risk_weight * max(0, drawdown - 0.1)  # Penalty if >10% drawdown
        components['risk'] = volatility_penalty + drawdown_penalty

        # Transaction cost penalty
        components['transaction_costs'] = -self.transaction_cost_weight * transaction_costs

        # Diversification reward (penalize concentration)
        # Herfindahl index: 0 = fully diversified, 1 = concentrated
        concentration_penalty = -self.diversification_weight * position_concentration
        components['diversification'] = concentration_penalty

        # Sharpe ratio reward (risk-adjusted returns)
        components['sharpe'] = self.sharpe_weight * sharpe_ratio

        # Total reward
        total_reward = sum(components.values())

        return total_reward, components


class TradingEnvironment:
    """
    Complete trading environment for reinforcement learning.

    This environment simulates a trading system where an agent can:
    - Observe market state as a heterogeneous graph
    - Execute buy/sell/hold actions on multiple assets
    - Receive rewards based on portfolio performance
    - Manage risk and transaction costs

    The environment follows a gym-like interface:
    - reset(): Initialize new episode
    - step(actions): Execute actions and get next state, reward, done, info

    Args:
        tradable_symbols: List of symbols that can be traded
        initial_capital: Starting capital (default: 1.0 normalized)
        fiat_currency: Base currency for portfolio (default: 'USD')
        max_steps_per_episode: Maximum steps before episode terminates
        transaction_cost_pct: Transaction cost as percentage of trade value
        slippage_pct: Slippage as percentage of trade value
        reward_calculator: Custom reward calculator (optional)
        min_portfolio_value: Minimum portfolio value before bankruptcy (default: 0.1)
        max_drawdown_termination: Max drawdown before episode terminates (default: 0.5)
        position_limit_pct: Maximum position size as % of portfolio (default: 0.5)
        enable_shorting: Whether to allow short positions (default: False)
        lookback_window: Window for computing rolling statistics (default: 20)
    """

    def __init__(
        self,
        tradable_symbols: List[str],
        initial_capital: float = 1.0,
        fiat_currency: str = 'USD',
        max_steps_per_episode: int = 252,  # ~1 trading year
        transaction_cost_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,  # 0.05% slippage
        reward_calculator: Optional[RewardCalculator] = None,
        min_portfolio_value: float = 0.1,
        max_drawdown_termination: float = 0.5,
        position_limit_pct: float = 0.5,
        enable_shorting: bool = False,
        lookback_window: int = 20,
    ):
        # Configuration
        self.tradable_symbols = tradable_symbols
        self.initial_capital = initial_capital
        self.fiat_currency = fiat_currency
        self.max_steps_per_episode = max_steps_per_episode
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.min_portfolio_value = min_portfolio_value
        self.max_drawdown_termination = max_drawdown_termination
        self.position_limit_pct = position_limit_pct
        self.enable_shorting = enable_shorting
        self.lookback_window = lookback_window

        # Reward calculator
        self.reward_calculator = reward_calculator or RewardCalculator()

        # Portfolio setup
        all_assets = [fiat_currency] + tradable_symbols
        self.portfolio = Portfolio(fiat=fiat_currency, tradables=all_assets)

        # State tracking
        self.current_step = 0
        self.episode_count = 0
        self.current_state: Optional[HeteroData] = None

        # Historical tracking for metrics
        self.portfolio_values: deque = deque(maxlen=lookback_window)
        self.returns: deque = deque(maxlen=lookback_window)
        self.portfolio_values.append(initial_capital)

        # Peak tracking for drawdown
        self.peak_value = initial_capital
        self.current_drawdown = 0.0

        # Transaction tracking
        self.total_transaction_costs = 0.0
        self.num_trades = 0
        self.profitable_trades = 0

        # Episode metrics
        self.episode_metrics = EpisodeMetrics()

        # Market data provider (to be set externally)
        self._market_data_provider: Optional[Callable] = None
        self._price_data: Optional[Dict[int, Dict[str, float]]] = None

    def set_market_data_provider(
        self,
        data_provider: Callable[[int], Tuple[HeteroData, Dict[str, float]]],
    ) -> None:
        """
        Set the market data provider function.

        Args:
            data_provider: Function that takes timestep and returns (state, prices)
                          - state: HeteroData graph representing market at timestep
                          - prices: Dict mapping symbols to current prices
        """
        self._market_data_provider = data_provider

    def set_price_data(
        self,
        price_data: Dict[int, Dict[str, float]],
    ) -> None:
        """
        Set price data directly (alternative to data provider).

        Args:
            price_data: Dict mapping timestep -> {symbol: price}
        """
        self._price_data = price_data

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[HeteroData, Dict]:
        """
        Reset environment to initial state for new episode.

        Args:
            seed: Random seed for reproducibility (optional)

        Returns:
            initial_state: HeteroData graph of initial market state
            info: Dictionary with episode initialization info
        """
        # Reset random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Reset step counter
        self.current_step = 0
        self.episode_count += 1

        # Reset portfolio
        all_assets = [self.fiat_currency] + self.tradable_symbols
        self.portfolio = Portfolio(fiat=self.fiat_currency, tradables=all_assets)

        # Reset tracking
        self.portfolio_values = deque(maxlen=self.lookback_window)
        self.returns = deque(maxlen=self.lookback_window)
        self.portfolio_values.append(self.initial_capital)

        self.peak_value = self.initial_capital
        self.current_drawdown = 0.0

        self.total_transaction_costs = 0.0
        self.num_trades = 0
        self.profitable_trades = 0

        # Reset episode metrics
        self.episode_metrics = EpisodeMetrics()

        # Get initial market state
        if self._market_data_provider is not None:
            self.current_state, prices = self._market_data_provider(self.current_step)
            # Initialize portfolio quotes
            self.portfolio.update_quotes(prices)
        elif self._price_data is not None:
            # Build simple state from price data
            self.current_state = self._build_state_from_prices(self.current_step)
            prices = self._price_data[self.current_step]
            self.portfolio.update_quotes(prices)
        else:
            raise RuntimeError(
                "No market data source configured. Call set_market_data_provider() "
                "or set_price_data() before reset()."
            )

        info = {
            'episode': self.episode_count,
            'initial_portfolio_value': self.initial_capital,
            'num_tradable_assets': len(self.tradable_symbols),
        }

        return self.current_state, info

    def step(
        self,
        actions: Dict[str, Dict[str, Union[int, float]]],
    ) -> Tuple[HeteroData, float, bool, Dict]:
        """
        Execute actions and advance environment by one timestep.

        Args:
            actions: Dict mapping symbols to action dicts
                    {symbol: {'type': int, 'size': int, 'fraction': float}}

        Returns:
            next_state: HeteroData graph of next market state
            reward: Scalar reward for this step
            done: Whether episode has terminated
            info: Dictionary with step information
        """
        # Validate we're not past episode end
        if self.current_step >= self.max_steps_per_episode:
            raise RuntimeError(
                f"Episode already terminated at step {self.current_step}. "
                f"Call reset() to start new episode."
            )

        # 1. Validate and preprocess actions
        valid_actions, num_trades_attempted = self._validate_actions(actions)

        # 2. Convert actions to portfolio transactions
        transactions = self._actions_to_transactions(valid_actions)

        # 3. Compute transaction costs
        transaction_costs = self._compute_transaction_costs(transactions)
        self.total_transaction_costs += transaction_costs

        # 4. Apply transactions to portfolio
        if transactions:
            transaction_matrix = self.portfolio._build_transaction(transactions)
            self.portfolio.apply_transaction(transaction_matrix)
            self.num_trades += len(transactions)

        # 5. Advance to next timestep
        self.current_step += 1

        # 6. Get new market state and prices
        if self._market_data_provider is not None:
            next_state, prices = self._market_data_provider(self.current_step)
        elif self._price_data is not None:
            next_state = self._build_state_from_prices(self.current_step)
            prices = self._price_data[self.current_step]
        else:
            raise RuntimeError("No market data source configured.")

        # 7. Update portfolio with new prices
        self.portfolio.update_quotes(prices)

        # 8. Compute portfolio value and return
        portfolio_value = self._get_portfolio_value()
        portfolio_return = (portfolio_value - self.portfolio_values[-1]) / self.portfolio_values[-1]

        self.portfolio_values.append(portfolio_value)
        self.returns.append(portfolio_return)

        # 9. Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value

        # 10. Compute metrics for reward
        volatility = self._compute_volatility()
        sharpe_ratio = self._compute_sharpe_ratio()
        position_concentration = self._compute_position_concentration()

        # 11. Compute reward
        reward, reward_components = self.reward_calculator.compute_reward(
            portfolio_return=portfolio_return,
            portfolio_volatility=volatility,
            transaction_costs=transaction_costs,
            position_concentration=position_concentration,
            sharpe_ratio=sharpe_ratio,
            drawdown=self.current_drawdown,
        )

        # 12. Check termination conditions
        done, termination_reason = self._check_termination(portfolio_value)

        # 13. Update state
        self.current_state = next_state

        # 14. Build info dict
        info = StepInfo(
            portfolio_value=portfolio_value,
            portfolio_return=portfolio_return,
            positions=self._get_positions(),
            num_trades_executed=len(transactions),
            transaction_costs=transaction_costs,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.current_drawdown,
            termination_reason=termination_reason if done else None,
        )

        # Convert to dict and add reward components
        info_dict = {
            'portfolio_value': info.portfolio_value,
            'portfolio_return': info.portfolio_return,
            'positions': info.positions,
            'num_trades_executed': info.num_trades_executed,
            'transaction_costs': info.transaction_costs,
            'sharpe_ratio': info.sharpe_ratio,
            'max_drawdown': info.max_drawdown,
            'termination_reason': info.termination_reason,
            'reward_components': reward_components,
            'step': self.current_step,
        }

        # 15. Update episode metrics if done
        if done:
            self._finalize_episode_metrics()
            info_dict['episode_metrics'] = self._get_episode_metrics_dict()

        return next_state, reward, done, info_dict

    def _validate_actions(
        self,
        actions: Dict[str, Dict[str, Union[int, float]]],
    ) -> Tuple[Dict[str, Dict[str, Union[int, float]]], int]:
        """
        Validate and filter actions to ensure they're legal.

        Returns:
            valid_actions: Filtered actions
            num_attempted: Number of actions attempted
        """
        valid_actions = {}
        num_attempted = len(actions)

        current_positions = self._get_positions()
        current_value = self._get_portfolio_value()

        for symbol, action in actions.items():
            # Check symbol is tradable
            if symbol not in self.tradable_symbols:
                continue

            action_type = action['type']
            fraction = action['fraction']

            # Skip hold actions
            if action_type == 0 or fraction == 0.0:  # HOLD
                continue

            # Validate sell action
            if action_type == 2:  # SELL
                current_position = current_positions.get(symbol, 0.0)
                if current_position <= 0.0:
                    continue  # Can't sell what we don't own
                # Limit sell to current position
                max_sell_fraction = current_position / current_value
                if fraction > max_sell_fraction:
                    action['fraction'] = max_sell_fraction

            # Validate buy action
            elif action_type == 1:  # BUY
                # Check position limits
                would_be_position = current_positions.get(symbol, 0.0) + fraction * current_value
                if would_be_position / current_value > self.position_limit_pct:
                    # Reduce to position limit
                    max_additional = self.position_limit_pct * current_value - current_positions.get(symbol, 0.0)
                    if max_additional <= 0:
                        continue  # Already at limit
                    action['fraction'] = max_additional / current_value

            valid_actions[symbol] = action

        return valid_actions, num_attempted

    def _actions_to_transactions(
        self,
        actions: Dict[str, Dict[str, Union[int, float]]],
    ) -> List[Dict[str, Union[str, float]]]:
        """Convert validated actions to portfolio transaction format."""
        transactions = []

        for symbol, action in actions.items():
            action_type = action['type']
            fraction = action['fraction']

            if action_type == 1:  # BUY
                transactions.append({
                    'type': 'buy',
                    'asset': symbol,
                    'size': fraction,
                })
            elif action_type == 2:  # SELL
                transactions.append({
                    'type': 'sell',
                    'asset': symbol,
                    'size': fraction,
                })

        return transactions

    def _compute_transaction_costs(
        self,
        transactions: List[Dict[str, Union[str, float]]],
    ) -> float:
        """Compute total transaction costs including fees and slippage."""
        total_cost = 0.0
        current_value = self._get_portfolio_value()

        for txn in transactions:
            trade_value = txn['size'] * current_value
            # Transaction fee
            fee = trade_value * self.transaction_cost_pct
            # Slippage
            slippage = trade_value * self.slippage_pct
            total_cost += fee + slippage

        return total_cost

    def _get_portfolio_value(self) -> float:
        """Get current total portfolio value."""
        # Sum all asset values
        values = self.portfolio._portfolio[:, Portfolio.ColAttr.Value.value]
        return float(np.sum(values))

    def _get_positions(self) -> Dict[str, float]:
        """Get current positions (value in each asset)."""
        positions = {}
        values = self.portfolio._portfolio[:, Portfolio.ColAttr.Value.value]

        for symbol, idx in self.portfolio.index_map.items():
            if symbol != self.fiat_currency:
                positions[symbol] = float(values[idx])

        return positions

    def _compute_volatility(self) -> float:
        """Compute rolling portfolio volatility."""
        if len(self.returns) < 2:
            return 0.0
        returns_array = np.array(list(self.returns))
        return float(np.std(returns_array))

    def _compute_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Compute rolling Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0
        returns_array = np.array(list(self.returns))
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        return float(sharpe)

    def _compute_position_concentration(self) -> float:
        """
        Compute Herfindahl index of position concentration.

        Returns value in [0, 1] where:
        - 0: Perfectly diversified
        - 1: Fully concentrated in one asset
        """
        positions = self._get_positions()
        total_value = self._get_portfolio_value()

        if total_value == 0:
            return 1.0  # Fully concentrated (bankruptcy)

        # Compute squared weights
        herfindahl = sum((v / total_value) ** 2 for v in positions.values())
        return float(herfindahl)

    def _check_termination(
        self,
        portfolio_value: float,
    ) -> Tuple[bool, Optional[str]]:
        """Check if episode should terminate."""
        # Max steps reached
        if self.current_step >= self.max_steps_per_episode:
            return True, TerminationCondition.MAX_STEPS.value

        # Bankruptcy
        if portfolio_value < self.min_portfolio_value:
            return True, TerminationCondition.BANKRUPTCY.value

        # Max drawdown
        if self.current_drawdown > self.max_drawdown_termination:
            return True, TerminationCondition.MAX_DRAWDOWN.value

        return False, None

    def _build_state_from_prices(
        self,
        timestep: int,
    ) -> HeteroData:
        """
        Build simple HeteroData state from price data.

        This is a fallback when no custom data provider is set.
        Creates a minimal graph with just price information.
        """
        data = HeteroData()

        if self._price_data is None or timestep not in self._price_data:
            # Return empty state
            return data

        prices = self._price_data[timestep]

        # Create equity nodes with simple features
        equity_symbols = [s for s in self.tradable_symbols if s in prices]
        if equity_symbols:
            # Features: [normalized_price, return, log_return]
            features = []
            for symbol in equity_symbols:
                price = prices[symbol]

                # Get previous price for return calculation
                if timestep > 0 and timestep - 1 in self._price_data:
                    prev_price = self._price_data[timestep - 1].get(symbol, price)
                else:
                    prev_price = price

                ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
                log_ret = np.log(price / prev_price) if prev_price > 0 else 0.0

                features.append([price / 100.0, ret, log_ret])  # Normalize price

            data["equity"].x = torch.tensor(features, dtype=torch.float32)
            data["equity"].symbol = equity_symbols

            # Add self-loop edges (minimal graph structure)
            num_equities = len(equity_symbols)
            edge_index = torch.stack([
                torch.arange(num_equities),
                torch.arange(num_equities),
            ])
            data["equity", "self_loop", "equity"].edge_index = edge_index

        return data

    def _finalize_episode_metrics(self) -> None:
        """Compute final episode metrics."""
        self.episode_metrics.total_return = (
            self._get_portfolio_value() - self.initial_capital
        ) / self.initial_capital
        self.episode_metrics.sharpe_ratio = self._compute_sharpe_ratio()
        self.episode_metrics.max_drawdown = max(self.current_drawdown, self.episode_metrics.max_drawdown)
        self.episode_metrics.num_trades = self.num_trades
        self.episode_metrics.win_rate = (
            self.profitable_trades / self.num_trades if self.num_trades > 0 else 0.0
        )
        self.episode_metrics.total_transaction_costs = self.total_transaction_costs
        self.episode_metrics.final_portfolio_value = self._get_portfolio_value()
        self.episode_metrics.num_steps = self.current_step

        # Compute average position concentration
        if len(self.portfolio_values) > 0:
            self.episode_metrics.avg_position_concentration = self._compute_position_concentration()

        # Compute volatility
        self.episode_metrics.volatility = self._compute_volatility()

    def _get_episode_metrics_dict(self) -> Dict:
        """Get episode metrics as dictionary."""
        return {
            'total_return': self.episode_metrics.total_return,
            'sharpe_ratio': self.episode_metrics.sharpe_ratio,
            'max_drawdown': self.episode_metrics.max_drawdown,
            'num_trades': self.episode_metrics.num_trades,
            'win_rate': self.episode_metrics.win_rate,
            'total_transaction_costs': self.episode_metrics.total_transaction_costs,
            'final_portfolio_value': self.episode_metrics.final_portfolio_value,
            'num_steps': self.episode_metrics.num_steps,
            'avg_position_concentration': self.episode_metrics.avg_position_concentration,
            'volatility': self.episode_metrics.volatility,
        }

    def get_state(self) -> HeteroData:
        """Get current environment state."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.current_state

    def get_portfolio_state(self) -> Dict[str, float]:
        """Get current portfolio state for action masking."""
        return self._get_positions()

    def get_metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Get graph metadata for initializing Actor/Critic networks.

        Returns:
            node_types: List of node type names
            edge_types: List of (src, relation, dst) tuples
        """
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        node_types = list(self.current_state.node_types)
        edge_types = list(self.current_state.edge_types)

        return node_types, edge_types
