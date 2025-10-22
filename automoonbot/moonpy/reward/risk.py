"""
Advanced risk metrics for portfolio evaluation and reward calculation.

This module provides comprehensive risk measurement tools used in quantitative
finance and algorithmic trading:

Risk-Adjusted Performance:
- Sharpe Ratio: Excess return per unit of total risk
- Sortino Ratio: Excess return per unit of downside risk
- Calmar Ratio: Return over maximum drawdown
- Information Ratio: Active return per unit of tracking error
- Omega Ratio: Probability-weighted ratio of gains vs losses
- Treynor Ratio: Excess return per unit of systematic risk (beta)

Risk Metrics:
- Maximum Drawdown: Largest peak-to-trough decline
- Value at Risk (VaR): Maximum expected loss at confidence level
- Conditional VaR (CVaR): Expected loss beyond VaR threshold
- Volatility: Standard deviation of returns (multiple estimation methods)
- Beta: Sensitivity to market movements
- Alpha: Excess return above market-adjusted expectations

All metrics support both online (incremental) and batch computation modes
for efficiency in RL training loops.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, List
from enum import Enum
import math


class VolatilityEstimator(Enum):
    """Methods for estimating volatility."""
    STANDARD = "standard"  # Classical standard deviation
    EWMA = "ewma"  # Exponentially weighted moving average
    PARKINSON = "parkinson"  # High-low range estimator
    GARMAN_KLASS = "garman_klass"  # OHLC estimator
    ROGERS_SATCHELL = "rogers_satchell"  # OHLC drift-independent estimator
    YANG_ZHANG = "yang_zhang"  # OHLC with overnight jumps


class VaRMethod(Enum):
    """Methods for computing Value at Risk."""
    HISTORICAL = "historical"  # Historical simulation
    PARAMETRIC = "parametric"  # Gaussian assumption
    CORNISH_FISHER = "cornish_fisher"  # Modified parametric with skew/kurtosis


class RiskMetrics:
    """
    Comprehensive risk metrics calculator with online update capabilities.

    This class maintains rolling statistics and can compute risk metrics
    incrementally as new returns arrive, making it efficient for RL training
    where we need to compute metrics at every step.

    Args:
        window_size: Number of periods for rolling calculations (default: 252 for daily data)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        confidence_level: Confidence level for VaR/CVaR (default: 0.95)
        annualization_factor: Factor to annualize metrics (252 for daily, 52 for weekly)
        min_periods: Minimum periods before computing metrics (default: 30)

    Example:
        >>> risk = RiskMetrics(window_size=252, risk_free_rate=0.02)
        >>> for return_t in returns:
        ...     risk.update(return_t)
        ...     sharpe = risk.sharpe_ratio()
        ...     drawdown = risk.max_drawdown()
    """

    def __init__(
        self,
        window_size: int = 252,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
        annualization_factor: int = 252,
        min_periods: int = 30,
    ):
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.annualization_factor = annualization_factor
        self.min_periods = min_periods

        # Rolling buffers for returns
        self.returns_history: List[float] = []

        # For drawdown calculation
        self.cumulative_returns = 1.0
        self.peak_value = 1.0
        self.current_drawdown = 0.0
        self.max_drawdown_value = 0.0

        # For online variance calculation (Welford's algorithm)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean

        # For downside deviation (Sortino)
        self.downside_count = 0
        self.downside_mean = 0.0
        self.downside_m2 = 0.0

        # EWMA parameters
        self.ewma_lambda = 0.94  # Standard RiskMetrics decay factor
        self.ewma_variance = 0.0
        self.ewma_initialized = False

    def reset(self):
        """Reset all statistics to initial state."""
        self.returns_history.clear()
        self.cumulative_returns = 1.0
        self.peak_value = 1.0
        self.current_drawdown = 0.0
        self.max_drawdown_value = 0.0
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.downside_count = 0
        self.downside_mean = 0.0
        self.downside_m2 = 0.0
        self.ewma_variance = 0.0
        self.ewma_initialized = False

    def update(self, return_value: float):
        """
        Update rolling statistics with new return.

        Uses Welford's online algorithm for numerically stable variance computation.

        Args:
            return_value: Period return (e.g., 0.01 for 1% return)
        """
        # Add to history
        self.returns_history.append(return_value)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Update cumulative returns and drawdown
        self.cumulative_returns *= (1.0 + return_value)
        if self.cumulative_returns > self.peak_value:
            self.peak_value = self.cumulative_returns

        self.current_drawdown = (self.peak_value - self.cumulative_returns) / self.peak_value
        self.max_drawdown_value = max(self.max_drawdown_value, self.current_drawdown)

        # Update online variance using Welford's algorithm
        self.count += 1
        delta = return_value - self.mean
        self.mean += delta / self.count
        delta2 = return_value - self.mean
        self.m2 += delta * delta2

        # Update downside deviation (only negative returns)
        if return_value < 0:
            self.downside_count += 1
            delta_down = return_value - self.downside_mean
            self.downside_mean += delta_down / self.downside_count
            delta2_down = return_value - self.downside_mean
            self.downside_m2 += delta_down * delta2_down

        # Update EWMA variance
        if not self.ewma_initialized:
            self.ewma_variance = return_value ** 2
            self.ewma_initialized = True
        else:
            self.ewma_variance = (
                self.ewma_lambda * self.ewma_variance +
                (1 - self.ewma_lambda) * return_value ** 2
            )

    def update_batch(self, returns: Union[List[float], np.ndarray, torch.Tensor]):
        """
        Update with multiple returns at once.

        Args:
            returns: Sequence of returns
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, np.ndarray):
            returns = returns.tolist()

        for ret in returns:
            self.update(ret)

    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data to compute metrics."""
        return len(self.returns_history) >= self.min_periods

    def mean_return(self, annualized: bool = True) -> float:
        """
        Compute mean return.

        Args:
            annualized: If True, annualize the return

        Returns:
            Mean return (annualized or per-period)
        """
        if not self._has_sufficient_data():
            return 0.0

        mean_ret = self.mean
        if annualized:
            mean_ret *= self.annualization_factor
        return mean_ret

    def volatility(
        self,
        method: VolatilityEstimator = VolatilityEstimator.STANDARD,
        annualized: bool = True
    ) -> float:
        """
        Compute volatility using specified method.

        Args:
            method: Volatility estimation method
            annualized: If True, annualize the volatility

        Returns:
            Volatility (standard deviation)
        """
        if not self._has_sufficient_data():
            return 0.0

        if method == VolatilityEstimator.STANDARD:
            # Standard deviation from Welford's algorithm
            if self.count < 2:
                return 0.0
            variance = self.m2 / (self.count - 1)
            vol = math.sqrt(variance)

        elif method == VolatilityEstimator.EWMA:
            # Exponentially weighted moving average
            vol = math.sqrt(self.ewma_variance)

        else:
            # For OHLC-based estimators, fall back to standard
            # (would need OHLC data, not just returns)
            variance = self.m2 / (self.count - 1) if self.count >= 2 else 0.0
            vol = math.sqrt(variance)

        if annualized:
            vol *= math.sqrt(self.annualization_factor)

        return vol

    def downside_deviation(self, annualized: bool = True, target: float = 0.0) -> float:
        """
        Compute downside deviation (semi-deviation below target).

        Only considers returns below the target (typically 0 or risk-free rate).
        Used in Sortino ratio.

        Args:
            annualized: If True, annualize the deviation
            target: Target return threshold (default: 0)

        Returns:
            Downside deviation
        """
        if not self._has_sufficient_data():
            return 0.0

        if self.downside_count < 2:
            return 0.0

        # Compute downside variance from negative returns only
        downside_variance = self.downside_m2 / (self.downside_count - 1)
        downside_dev = math.sqrt(downside_variance)

        if annualized:
            downside_dev *= math.sqrt(self.annualization_factor)

        return downside_dev

    def max_drawdown(self) -> float:
        """
        Get maximum drawdown observed so far.

        Returns:
            Maximum drawdown as fraction (0.0 to 1.0)
        """
        return self.max_drawdown_value

    def current_dd(self) -> float:
        """
        Get current drawdown from peak.

        Returns:
            Current drawdown as fraction (0.0 to 1.0)
        """
        return self.current_drawdown

    def sharpe_ratio(
        self,
        annualized: bool = True,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Compute Sharpe ratio: (Return - RiskFree) / Volatility.

        The Sharpe ratio measures excess return per unit of total risk.
        Higher is better. Typical interpretation:
        - < 0: Negative excess returns
        - 0-1: Sub-optimal
        - 1-2: Good
        - 2-3: Very good
        - > 3: Excellent

        Args:
            annualized: If True, compute annualized Sharpe
            risk_free_rate: Override default risk-free rate

        Returns:
            Sharpe ratio
        """
        if not self._has_sufficient_data():
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        mean_ret = self.mean_return(annualized=annualized)
        vol = self.volatility(annualized=annualized)

        if vol < 1e-8:
            return 0.0

        sharpe = (mean_ret - rf) / vol
        return sharpe

    def sortino_ratio(
        self,
        annualized: bool = True,
        risk_free_rate: Optional[float] = None,
        target_return: float = 0.0
    ) -> float:
        """
        Compute Sortino ratio: (Return - RiskFree) / DownsideDeviation.

        Similar to Sharpe but only penalizes downside volatility, not upside.
        Better for strategies with asymmetric return distributions.

        Args:
            annualized: If True, compute annualized Sortino
            risk_free_rate: Override default risk-free rate
            target_return: Target return for downside calculation

        Returns:
            Sortino ratio
        """
        if not self._has_sufficient_data():
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        mean_ret = self.mean_return(annualized=annualized)
        downside_dev = self.downside_deviation(annualized=annualized, target=target_return)

        if downside_dev < 1e-8:
            return 0.0

        sortino = (mean_ret - rf) / downside_dev
        return sortino

    def calmar_ratio(self, annualized: bool = True) -> float:
        """
        Compute Calmar ratio: AnnualizedReturn / MaxDrawdown.

        Measures return relative to worst drawdown. Higher is better.
        Useful for evaluating risk of ruin.

        Args:
            annualized: If True, use annualized return

        Returns:
            Calmar ratio
        """
        if not self._has_sufficient_data():
            return 0.0

        mean_ret = self.mean_return(annualized=annualized)
        max_dd = self.max_drawdown()

        if max_dd < 1e-8:
            return 0.0

        calmar = mean_ret / max_dd
        return calmar

    def value_at_risk(
        self,
        confidence_level: Optional[float] = None,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> float:
        """
        Compute Value at Risk (VaR).

        VaR is the maximum expected loss at a given confidence level.
        For example, 95% VaR of 0.05 means there's a 5% chance of losing
        more than 5% in a period.

        Args:
            confidence_level: Confidence level (default: 0.95)
            method: VaR computation method

        Returns:
            VaR as positive number (loss magnitude)
        """
        if not self._has_sufficient_data():
            return 0.0

        conf = confidence_level if confidence_level is not None else self.confidence_level
        alpha = 1.0 - conf

        if method == VaRMethod.HISTORICAL:
            # Historical simulation: use empirical quantile
            returns_array = np.array(self.returns_history)
            var = -np.percentile(returns_array, alpha * 100)
            return max(0.0, var)

        elif method == VaRMethod.PARAMETRIC:
            # Parametric (Gaussian) VaR
            from scipy import stats
            z_score = stats.norm.ppf(alpha)
            vol = self.volatility(annualized=False)
            mean_ret = self.mean_return(annualized=False)
            var = -(mean_ret + z_score * vol)
            return max(0.0, var)

        elif method == VaRMethod.CORNISH_FISHER:
            # Modified VaR with skewness and kurtosis
            from scipy import stats as scipy_stats
            returns_array = np.array(self.returns_history)

            mean_ret = self.mean_return(annualized=False)
            vol = self.volatility(annualized=False)
            skew = scipy_stats.skew(returns_array)
            kurt = scipy_stats.kurtosis(returns_array)

            # Cornish-Fisher expansion
            z = scipy_stats.norm.ppf(alpha)
            z_cf = (z +
                    (z**2 - 1) * skew / 6 +
                    (z**3 - 3*z) * kurt / 24 -
                    (2*z**3 - 5*z) * skew**2 / 36)

            var = -(mean_ret + z_cf * vol)
            return max(0.0, var)

        return 0.0

    def conditional_var(
        self,
        confidence_level: Optional[float] = None,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> float:
        """
        Compute Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that we've exceeded VaR.
        It's a more comprehensive risk measure than VaR as it considers
        the tail distribution.

        Args:
            confidence_level: Confidence level (default: 0.95)
            method: CVaR computation method

        Returns:
            CVaR as positive number (expected loss in tail)
        """
        if not self._has_sufficient_data():
            return 0.0

        conf = confidence_level if confidence_level is not None else self.confidence_level
        alpha = 1.0 - conf

        if method == VaRMethod.HISTORICAL:
            # Historical simulation: average of worst alpha% returns
            returns_array = np.array(self.returns_history)
            sorted_returns = np.sort(returns_array)
            cutoff_idx = max(1, int(len(sorted_returns) * alpha))
            tail_losses = -sorted_returns[:cutoff_idx]
            cvar = np.mean(tail_losses)
            return max(0.0, cvar)

        elif method == VaRMethod.PARAMETRIC:
            # Parametric CVaR for Gaussian distribution
            from scipy import stats
            z_score = stats.norm.ppf(alpha)
            vol = self.volatility(annualized=False)
            mean_ret = self.mean_return(annualized=False)

            # CVaR = � - � * �(z) / �  where � is standard normal PDF
            phi_z = stats.norm.pdf(z_score)
            cvar = -(mean_ret - vol * phi_z / alpha)
            return max(0.0, cvar)

        else:
            # For Cornish-Fisher, fall back to historical
            returns_array = np.array(self.returns_history)
            sorted_returns = np.sort(returns_array)
            cutoff_idx = max(1, int(len(sorted_returns) * alpha))
            tail_losses = -sorted_returns[:cutoff_idx]
            cvar = np.mean(tail_losses)
            return max(0.0, cvar)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Compute Omega ratio: probability-weighted gains over losses.

        Omega = (Probability-weighted gains above threshold) /
                (Probability-weighted losses below threshold)

        Considers the entire return distribution, not just mean and variance.
        Omega > 1 indicates positive expected value above threshold.

        Args:
            threshold: Return threshold (typically 0 or risk-free rate)

        Returns:
            Omega ratio
        """
        if not self._has_sufficient_data():
            return 1.0

        returns_array = np.array(self.returns_history)

        gains = returns_array[returns_array > threshold] - threshold
        losses = threshold - returns_array[returns_array < threshold]

        total_gains = np.sum(gains) if len(gains) > 0 else 0.0
        total_losses = np.sum(losses) if len(losses) > 0 else 1e-8

        omega = total_gains / total_losses
        return omega

    def information_ratio(
        self,
        benchmark_returns: Union[List[float], np.ndarray],
        annualized: bool = True
    ) -> float:
        """
        Compute Information Ratio: ActiveReturn / TrackingError.

        Measures risk-adjusted active return relative to a benchmark.
        Used to evaluate skill in active management.

        Args:
            benchmark_returns: Returns of the benchmark
            annualized: If True, annualize the ratio

        Returns:
            Information ratio
        """
        if not self._has_sufficient_data():
            return 0.0

        if isinstance(benchmark_returns, np.ndarray):
            benchmark_returns = benchmark_returns.tolist()

        # Ensure same length
        min_len = min(len(self.returns_history), len(benchmark_returns))
        portfolio_rets = np.array(self.returns_history[-min_len:])
        benchmark_rets = np.array(benchmark_returns[-min_len:])

        # Active returns
        active_returns = portfolio_rets - benchmark_rets

        # Tracking error (volatility of active returns)
        tracking_error = np.std(active_returns, ddof=1)
        mean_active = np.mean(active_returns)

        if tracking_error < 1e-8:
            return 0.0

        if annualized:
            mean_active *= self.annualization_factor
            tracking_error *= math.sqrt(self.annualization_factor)

        ir = mean_active / tracking_error
        return ir

    def beta(self, market_returns: Union[List[float], np.ndarray]) -> float:
        """
        Compute beta: sensitivity to market movements.

        Beta = Cov(portfolio, market) / Var(market)

        Beta interpretation:
        - beta = 1: Moves with market
        - beta > 1: More volatile than market
        - beta < 1: Less volatile than market
        - beta < 0: Moves opposite to market

        Args:
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        if not self._has_sufficient_data():
            return 1.0

        if isinstance(market_returns, np.ndarray):
            market_returns = market_returns.tolist()

        # Ensure same length
        min_len = min(len(self.returns_history), len(market_returns))
        portfolio_rets = np.array(self.returns_history[-min_len:])
        market_rets = np.array(market_returns[-min_len:])

        # Compute covariance and variance
        covariance = np.cov(portfolio_rets, market_rets)[0, 1]
        market_variance = np.var(market_rets, ddof=1)

        if market_variance < 1e-8:
            return 1.0

        beta_value = covariance / market_variance
        return beta_value

    def alpha(
        self,
        market_returns: Union[List[float], np.ndarray],
        risk_free_rate: Optional[float] = None,
        annualized: bool = True
    ) -> float:
        """
        Compute Jensen's alpha: excess return above CAPM prediction.

        Alpha = PortfolioReturn - [RiskFree + Beta * (MarketReturn - RiskFree)]

        Positive alpha indicates outperformance relative to risk taken.

        Args:
            market_returns: Market returns
            risk_free_rate: Risk-free rate
            annualized: If True, compute annualized alpha

        Returns:
            Alpha (excess return)
        """
        if not self._has_sufficient_data():
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        if isinstance(market_returns, np.ndarray):
            market_returns = market_returns.tolist()

        # Ensure same length
        min_len = min(len(self.returns_history), len(market_returns))
        market_rets = np.array(market_returns[-min_len:])

        # Compute beta
        beta_value = self.beta(market_returns)

        # Compute mean returns
        portfolio_return = self.mean_return(annualized=annualized)
        market_return = np.mean(market_rets)

        if annualized:
            market_return *= self.annualization_factor

        # Jensen's alpha
        expected_return = rf + beta_value * (market_return - rf)
        alpha_value = portfolio_return - expected_return

        return alpha_value

    def treynor_ratio(
        self,
        market_returns: Union[List[float], np.ndarray],
        risk_free_rate: Optional[float] = None,
        annualized: bool = True
    ) -> float:
        """
        Compute Treynor ratio: (Return - RiskFree) / Beta.

        Similar to Sharpe but uses systematic risk (beta) instead of total risk.
        Useful for well-diversified portfolios where systematic risk dominates.

        Args:
            market_returns: Market returns
            risk_free_rate: Risk-free rate
            annualized: If True, use annualized return

        Returns:
            Treynor ratio
        """
        if not self._has_sufficient_data():
            return 0.0

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        beta_value = self.beta(market_returns)

        if abs(beta_value) < 1e-8:
            return 0.0

        portfolio_return = self.mean_return(annualized=annualized)
        treynor = (portfolio_return - rf) / beta_value

        return treynor

    def get_all_metrics(
        self,
        market_returns: Optional[Union[List[float], np.ndarray]] = None
    ) -> dict:
        """
        Compute all available risk metrics.

        Args:
            market_returns: Optional market returns for beta/alpha/Treynor

        Returns:
            Dictionary of all computed metrics
        """
        metrics = {
            'mean_return': self.mean_return(annualized=True),
            'volatility': self.volatility(annualized=True),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'max_drawdown': self.max_drawdown(),
            'current_drawdown': self.current_dd(),
            'var_95': self.value_at_risk(confidence_level=0.95),
            'cvar_95': self.conditional_var(confidence_level=0.95),
            'omega_ratio': self.omega_ratio(),
            'downside_deviation': self.downside_deviation(),
        }

        # Add market-relative metrics if benchmark provided
        if market_returns is not None:
            metrics.update({
                'beta': self.beta(market_returns),
                'alpha': self.alpha(market_returns),
                'treynor_ratio': self.treynor_ratio(market_returns),
                'information_ratio': self.information_ratio(market_returns),
            })

        return metrics


def compute_sharpe_ratio(
    returns: Union[List[float], np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Standalone function to compute Sharpe ratio from returns array.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize (252 for daily data)

    Returns:
        Sharpe ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    if isinstance(returns, list):
        returns = np.array(returns)

    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns) * annualization_factor
    volatility = np.std(returns, ddof=1) * np.sqrt(annualization_factor)

    if volatility < 1e-8:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / volatility
    return sharpe


def compute_sortino_ratio(
    returns: Union[List[float], np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Standalone function to compute Sortino ratio from returns array.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        target_return: Target return for downside calculation
        annualization_factor: Factor to annualize

    Returns:
        Sortino ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    if isinstance(returns, list):
        returns = np.array(returns)

    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns) * annualization_factor

    # Downside deviation
    downside_returns = returns[returns < target_return]
    if len(downside_returns) < 2:
        return 0.0

    downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(annualization_factor)

    if downside_deviation < 1e-8:
        return 0.0

    sortino = (mean_return - risk_free_rate) / downside_deviation
    return sortino


def compute_max_drawdown(
    returns: Union[List[float], np.ndarray, torch.Tensor]
) -> float:
    """
    Standalone function to compute maximum drawdown from returns array.

    Args:
        returns: Array of returns

    Returns:
        Maximum drawdown as fraction (0.0 to 1.0)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.cpu().numpy()
    if isinstance(returns, list):
        returns = np.array(returns)

    if len(returns) == 0:
        return 0.0

    # Compute cumulative returns
    cumulative = np.cumprod(1 + returns)

    # Compute running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Compute drawdown
    drawdown = (running_max - cumulative) / running_max

    max_dd = np.max(drawdown)
    return max_dd
