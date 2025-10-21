"""
Comprehensive unit tests for reward functions and risk metrics.

Tests cover:
- Risk metrics (Sharpe, Sortino, Calmar, VaR, CVaR, etc.)
- Reward functions (simple, risk-adjusted, multi-component, adaptive)
- Market regime detection
- Reward normalization and composition
"""

import pytest
import numpy as np
import torch
from typing import Dict, List

from automoonbot.moonpy.reward.risk import (
    RiskMetrics,
    VolatilityEstimator,
    VaRMethod,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
)

from automoonbot.moonpy.reward.reward import (
    BaseRewardFunction,
    RewardType,
    NormalizationMethod,
    SimpleReturnReward,
    RiskAdjustedReward,
    MultiComponentReward,
    AdaptiveReward,
    PotentialBasedReward,
    CuriosityReward,
    HierarchicalReward,
    SparseReward,
    RewardNormalizer,
    combine_rewards,
    create_default_reward,
)

from automoonbot.moonpy.reward.regime import (
    VolatilityRegime,
    TrendRegime,
    CorrelationRegime,
    VolumeRegime,
    MarketRegime,
    VolatilityRegimeDetector,
    TrendRegimeDetector,
    CorrelationRegimeDetector,
    VolumeRegimeDetector,
    MarketRegimeDetector,
    create_default_regime_detector,
    regime_to_risk_adjustment,
)


# ============================================================================
# Risk Metrics Tests
# ============================================================================

class TestRiskMetrics:
    """Test suite for RiskMetrics class."""

    def test_initialization(self):
        """Test RiskMetrics initialization."""
        risk = RiskMetrics(
            window_size=252,
            risk_free_rate=0.02,
            confidence_level=0.95
        )
        assert risk.window_size == 252
        assert risk.risk_free_rate == 0.02
        assert risk.confidence_level == 0.95
        assert risk.count == 0

    def test_update_single_return(self):
        """Test updating with single return."""
        risk = RiskMetrics()
        risk.update(0.01)
        assert risk.count == 1
        assert len(risk.returns_history) == 1
        assert risk.cumulative_returns == pytest.approx(1.01)

    def test_update_batch_returns(self):
        """Test updating with batch of returns."""
        risk = RiskMetrics()
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        risk.update_batch(returns)
        assert risk.count == 5
        assert len(risk.returns_history) == 5

    def test_mean_return(self):
        """Test mean return calculation."""
        risk = RiskMetrics(window_size=10, annualization_factor=252)
        returns = [0.01] * 10
        risk.update_batch(returns)

        mean_ret = risk.mean_return(annualized=False)
        assert mean_ret == pytest.approx(0.01, abs=1e-6)

        mean_ret_ann = risk.mean_return(annualized=True)
        assert mean_ret_ann == pytest.approx(0.01 * 252, abs=1e-6)

    def test_volatility_standard(self):
        """Test volatility calculation using standard method."""
        risk = RiskMetrics(window_size=10, annualization_factor=252)
        returns = [0.01, -0.01, 0.02, -0.02, 0.015, -0.015, 0.01, -0.01, 0.005, -0.005]
        risk.update_batch(returns)

        vol = risk.volatility(method=VolatilityEstimator.STANDARD, annualized=False)
        assert vol > 0

        # Annualized should be higher
        vol_ann = risk.volatility(method=VolatilityEstimator.STANDARD, annualized=True)
        assert vol_ann > vol

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        risk = RiskMetrics(window_size=50, risk_free_rate=0.02)
        # Generate positive returns
        returns = np.random.normal(0.05/252, 0.15/np.sqrt(252), 50)
        risk.update_batch(returns.tolist())

        sharpe = risk.sharpe_ratio(annualized=True)
        assert isinstance(sharpe, float)
        # With positive mean returns, Sharpe should be positive
        assert sharpe > 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        risk = RiskMetrics(window_size=50)
        # Returns with some downside
        returns = [0.01, -0.02, 0.015, -0.01, 0.02] * 10
        risk.update_batch(returns)

        sortino = risk.sortino_ratio(annualized=False)
        assert isinstance(sortino, float)

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        risk = RiskMetrics(window_size=50)
        # Generate returns with drawdown
        returns = [0.01] * 30 + [-0.03] * 10 + [0.01] * 10
        risk.update_batch(returns)

        calmar = risk.calmar_ratio(annualized=False)
        assert isinstance(calmar, float)
        # Should be positive since we have positive mean return
        assert calmar > 0

    def test_max_drawdown(self):
        """Test maximum drawdown tracking."""
        risk = RiskMetrics()
        # Create a drawdown scenario: go up, then down
        returns = [0.05, 0.05, 0.05, -0.10, -0.05, 0.02]
        risk.update_batch(returns)

        max_dd = risk.max_drawdown()
        assert max_dd > 0
        assert max_dd <= 1.0

    def test_value_at_risk_historical(self):
        """Test VaR calculation using historical method."""
        risk = RiskMetrics(window_size=100, confidence_level=0.95)
        # Generate returns
        returns = np.random.normal(0, 0.01, 100)
        risk.update_batch(returns.tolist())

        var = risk.value_at_risk(method=VaRMethod.HISTORICAL)
        assert isinstance(var, float)
        assert var >= 0

    def test_value_at_risk_parametric(self):
        """Test VaR calculation using parametric method."""
        risk = RiskMetrics(window_size=100, confidence_level=0.95)
        returns = np.random.normal(0, 0.01, 100)
        risk.update_batch(returns.tolist())

        var = risk.value_at_risk(method=VaRMethod.PARAMETRIC)
        assert isinstance(var, float)
        assert var >= 0

    def test_conditional_var(self):
        """Test CVaR calculation."""
        risk = RiskMetrics(window_size=100, confidence_level=0.95)
        returns = np.random.normal(0, 0.01, 100)
        risk.update_batch(returns.tolist())

        cvar = risk.conditional_var(method=VaRMethod.HISTORICAL)
        assert isinstance(cvar, float)
        assert cvar >= 0

        # CVaR should be >= VaR
        var = risk.value_at_risk(method=VaRMethod.HISTORICAL)
        assert cvar >= var

    def test_omega_ratio(self):
        """Test Omega ratio calculation."""
        risk = RiskMetrics(window_size=50)
        # Positive skewed returns (more gains than losses)
        returns = [0.02, 0.01, -0.005, 0.015, 0.01] * 10
        risk.update_batch(returns)

        omega = risk.omega_ratio(threshold=0.0)
        assert isinstance(omega, float)
        # Should be > 1 for positive returns
        assert omega > 1.0

    def test_beta(self):
        """Test beta calculation."""
        risk = RiskMetrics(window_size=50)
        # Generate correlated returns
        market_returns = np.random.normal(0.001, 0.01, 50)
        portfolio_returns = 0.5 + 1.2 * market_returns + np.random.normal(0, 0.005, 50)

        risk.update_batch(portfolio_returns.tolist())
        beta = risk.beta(market_returns)

        assert isinstance(beta, float)
        # Beta should be around 1.2 (our true beta)
        assert 0.8 < beta < 1.6

    def test_alpha(self):
        """Test alpha calculation."""
        risk = RiskMetrics(window_size=50, risk_free_rate=0.02)
        market_returns = np.random.normal(0.001, 0.01, 50)
        # Portfolio with positive alpha
        portfolio_returns = market_returns + 0.002 + np.random.normal(0, 0.005, 50)

        risk.update_batch(portfolio_returns.tolist())
        alpha = risk.alpha(market_returns)

        assert isinstance(alpha, float)

    def test_information_ratio(self):
        """Test information ratio calculation."""
        risk = RiskMetrics(window_size=50)
        benchmark_returns = np.random.normal(0.001, 0.01, 50)
        # Portfolio that outperforms benchmark
        portfolio_returns = benchmark_returns + 0.001 + np.random.normal(0, 0.005, 50)

        risk.update_batch(portfolio_returns.tolist())
        ir = risk.information_ratio(benchmark_returns)

        assert isinstance(ir, float)

    def test_reset(self):
        """Test reset functionality."""
        risk = RiskMetrics()
        risk.update_batch([0.01, 0.02, -0.01])
        assert risk.count == 3

        risk.reset()
        assert risk.count == 0
        assert len(risk.returns_history) == 0


class TestRiskMetricsStandaloneFunctions:
    """Test standalone risk metric functions."""

    def test_compute_sharpe_ratio(self):
        """Test standalone Sharpe ratio function."""
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float)

    def test_compute_sharpe_ratio_torch(self):
        """Test Sharpe ratio with torch tensor."""
        returns = torch.tensor([0.01, 0.02, -0.005, 0.015, 0.01])
        sharpe = compute_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_compute_sortino_ratio(self):
        """Test standalone Sortino ratio function."""
        returns = [0.01, -0.02, 0.015, -0.01, 0.02]
        sortino = compute_sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_compute_max_drawdown_function(self):
        """Test standalone max drawdown function."""
        returns = [0.05, 0.05, -0.10, -0.05, 0.02]
        max_dd = compute_max_drawdown(returns)
        assert isinstance(max_dd, float)
        assert max_dd >= 0
        assert max_dd <= 1.0


# ============================================================================
# Reward Function Tests
# ============================================================================

class TestSimpleReturnReward:
    """Test suite for SimpleReturnReward."""

    def test_initialization(self):
        """Test SimpleReturnReward initialization."""
        reward_fn = SimpleReturnReward(scale_factor=2.0, log_returns=False)
        assert reward_fn.scale_factor == 2.0
        assert reward_fn.log_returns is False

    def test_compute_reward_positive_return(self):
        """Test reward computation with positive return."""
        reward_fn = SimpleReturnReward(scale_factor=1.0)

        state = {}
        action = {}
        next_state = {}
        info = {'portfolio_return': 0.05}

        reward, components = reward_fn.compute_reward(state, action, next_state, info)

        assert reward == pytest.approx(0.05)
        assert 'return' in components

    def test_compute_reward_with_scale_factor(self):
        """Test reward computation with scale factor."""
        reward_fn = SimpleReturnReward(scale_factor=10.0)

        info = {'portfolio_return': 0.05}
        reward, _ = reward_fn.compute_reward({}, {}, {}, info)

        assert reward == pytest.approx(0.5)

    def test_log_returns(self):
        """Test log returns computation."""
        reward_fn = SimpleReturnReward(log_returns=True)

        info = {'portfolio_return': 0.05}
        reward, _ = reward_fn.compute_reward({}, {}, {}, info)

        import math
        expected = math.log(1.05)
        assert reward == pytest.approx(expected, abs=1e-6)


class TestRiskAdjustedReward:
    """Test suite for RiskAdjustedReward."""

    def test_initialization(self):
        """Test RiskAdjustedReward initialization."""
        reward_fn = RiskAdjustedReward(metric='sharpe', window_size=30)
        assert reward_fn.metric == 'sharpe'
        assert reward_fn.risk_metrics.window_size == 30

    def test_compute_reward_sharpe(self):
        """Test Sharpe-based reward."""
        reward_fn = RiskAdjustedReward(metric='sharpe', window_size=10)

        # Feed multiple returns to build history
        for ret in [0.01, 0.02, -0.005, 0.015, 0.01, 0.005, 0.02, 0.01, 0.015, 0.01]:
            info = {'portfolio_return': ret}
            reward, components = reward_fn.compute_reward({}, {}, {}, info)

        assert isinstance(reward, float)
        assert 'risk_adjusted' in components

    def test_compute_reward_sortino(self):
        """Test Sortino-based reward."""
        reward_fn = RiskAdjustedReward(metric='sortino', window_size=10)

        for ret in [0.01, -0.02, 0.015, -0.01, 0.02, 0.005, 0.01, 0.015, 0.01, 0.02]:
            info = {'portfolio_return': ret}
            reward, _ = reward_fn.compute_reward({}, {}, {}, info)

        assert isinstance(reward, float)

    def test_reset(self):
        """Test reset functionality."""
        reward_fn = RiskAdjustedReward()
        info = {'portfolio_return': 0.01}
        reward_fn.compute_reward({}, {}, {}, info)

        reward_fn.reset()
        assert reward_fn.risk_metrics.count == 0


class TestMultiComponentReward:
    """Test suite for MultiComponentReward."""

    def test_initialization(self):
        """Test MultiComponentReward initialization."""
        weights = {'return': 1.0, 'risk': 0.5, 'transaction_cost': 0.1}
        reward_fn = MultiComponentReward(weights=weights)
        assert reward_fn.weights == weights

    def test_compute_reward_all_components(self):
        """Test reward computation with all components."""
        reward_fn = MultiComponentReward()

        info = {
            'portfolio_return': 0.05,
            'portfolio_volatility': 0.02,
            'transaction_costs': 0.001,
            'position_concentration': 0.3,
            'max_drawdown': 0.05,
        }

        reward, components = reward_fn.compute_reward({}, {}, {}, info)

        assert isinstance(reward, float)
        assert 'return' in components
        assert 'volatility' in components
        assert 'transaction_cost' in components
        assert 'diversification' in components

    def test_drawdown_penalty(self):
        """Test drawdown penalty."""
        reward_fn = MultiComponentReward(
            weights={'drawdown': 1.0},
            penalties={'drawdown_threshold': 0.1}
        )

        # No penalty below threshold
        info = {'max_drawdown': 0.05}
        reward, components = reward_fn.compute_reward({}, {}, {}, info)
        assert components['drawdown'] == 0.0

        # Penalty above threshold
        info = {'max_drawdown': 0.20}
        reward, components = reward_fn.compute_reward({}, {}, {}, info)
        assert components['drawdown'] < 0.0


class TestAdaptiveReward:
    """Test suite for AdaptiveReward."""

    def test_initialization(self):
        """Test AdaptiveReward initialization."""
        reward_fn = AdaptiveReward()
        assert reward_fn.current_regime == 'neutral'
        assert reward_fn.base_weights is not None

    def test_regime_detection(self):
        """Test regime detection."""
        reward_fn = AdaptiveReward()

        # High volatility scenario
        info = {'portfolio_volatility': 0.05, 'portfolio_return': 0.01}
        regime = reward_fn.detect_regime(info)
        assert regime == 'high_vol'

        # Bull market scenario
        info = {'portfolio_volatility': 0.01, 'portfolio_return': 0.03}
        regime = reward_fn.detect_regime(info)
        assert regime == 'bull'

    def test_compute_reward_adapts_to_regime(self):
        """Test that reward adapts based on regime."""
        reward_fn = AdaptiveReward()

        # High volatility scenario
        info = {
            'portfolio_volatility': 0.05,
            'portfolio_return': 0.01,
            'transaction_costs': 0.001,
        }

        reward, components = reward_fn.compute_reward({}, {}, {}, info)
        assert isinstance(reward, float)
        assert 'regime' in components


class TestPotentialBasedReward:
    """Test suite for PotentialBasedReward."""

    def test_initialization(self):
        """Test PotentialBasedReward initialization."""
        base_reward = SimpleReturnReward()
        reward_fn = PotentialBasedReward(base_reward=base_reward, gamma=0.99)
        assert reward_fn.gamma == 0.99

    def test_compute_reward_with_shaping(self):
        """Test reward shaping."""
        base_reward = SimpleReturnReward()
        reward_fn = PotentialBasedReward(base_reward=base_reward)

        state = {}
        next_state = {}
        info = {
            'portfolio_return': 0.01,
            'portfolio_value': 1.05
        }

        reward, components = reward_fn.compute_reward(state, {}, next_state, info)

        assert isinstance(reward, float)
        assert 'base_reward' in components
        assert 'shaping' in components


class TestCuriosityReward:
    """Test suite for CuriosityReward."""

    def test_initialization(self):
        """Test CuriosityReward initialization."""
        base_reward = SimpleReturnReward()
        reward_fn = CuriosityReward(base_reward=base_reward, curiosity_weight=0.1)
        assert reward_fn.curiosity_weight == 0.1

    def test_compute_reward_with_curiosity(self):
        """Test curiosity bonus."""
        base_reward = SimpleReturnReward()
        reward_fn = CuriosityReward(base_reward=base_reward)

        state = {'portfolio_value': 1.0}
        next_state = {'portfolio_value': 1.05}
        info = {'portfolio_return': 0.05}

        reward, components = reward_fn.compute_reward(state, {}, next_state, info)

        assert isinstance(reward, float)
        assert 'extrinsic' in components
        assert 'intrinsic' in components


class TestHierarchicalReward:
    """Test suite for HierarchicalReward."""

    def test_initialization(self):
        """Test HierarchicalReward initialization."""
        short_term = SimpleReturnReward()
        long_term = RiskAdjustedReward()
        reward_fn = HierarchicalReward(
            short_term_reward=short_term,
            long_term_reward=long_term
        )
        assert reward_fn.long_term_horizon == 20

    def test_compute_reward_combines_time_scales(self):
        """Test combining short and long term rewards."""
        short_term = SimpleReturnReward()
        long_term = SimpleReturnReward(scale_factor=0.5)
        reward_fn = HierarchicalReward(short_term, long_term, long_term_horizon=5)

        # Feed multiple steps
        for i in range(10):
            info = {'portfolio_return': 0.01}
            reward, components = reward_fn.compute_reward({}, {}, {}, info)

        assert 'short_term' in components
        assert 'long_term' in components


class TestSparseReward:
    """Test suite for SparseReward."""

    def test_initialization(self):
        """Test SparseReward initialization."""
        reward_fn = SparseReward(milestone_reward=10.0)
        assert reward_fn.milestone_reward == 10.0

    def test_no_reward_during_episode(self):
        """Test that no reward given during episode."""
        reward_fn = SparseReward()

        info = {'portfolio_return': 0.01, 'done': False}
        reward, components = reward_fn.compute_reward({}, {}, {}, info)

        assert reward == 0.0

    def test_reward_at_episode_end(self):
        """Test reward at episode end."""
        reward_fn = SparseReward()

        info = {
            'done': True,
            'total_return': 0.15,
            'sharpe_ratio': 1.5
        }

        reward, components = reward_fn.compute_reward({}, {}, {}, info)

        assert reward > 0
        assert 'episode_end' in components


class TestRewardNormalizer:
    """Test suite for RewardNormalizer."""

    def test_initialization(self):
        """Test RewardNormalizer initialization."""
        normalizer = RewardNormalizer(method=NormalizationMethod.STANDARDIZE)
        assert normalizer.method == NormalizationMethod.STANDARDIZE

    def test_standardization(self):
        """Test standardization normalization."""
        normalizer = RewardNormalizer(method=NormalizationMethod.STANDARDIZE)

        # Feed some rewards
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = [normalizer.normalize(r) for r in rewards]

        # Normalized rewards should have different distribution
        assert np.std(normalized) != np.std(rewards)

    def test_clipping(self):
        """Test clipping normalization."""
        normalizer = RewardNormalizer(
            method=NormalizationMethod.CLIP,
            clip_range=(-1.0, 1.0)
        )

        assert normalizer.normalize(2.0) == 1.0
        assert normalizer.normalize(-2.0) == -1.0
        assert normalizer.normalize(0.5) == 0.5


class TestRewardUtilityFunctions:
    """Test utility functions for rewards."""

    def test_combine_rewards(self):
        """Test combining multiple reward functions."""
        reward1 = SimpleReturnReward()
        reward2 = RiskAdjustedReward()

        combined = combine_rewards([reward1, reward2], weights=[0.7, 0.3])

        assert isinstance(combined, MultiComponentReward)

    def test_create_default_reward(self):
        """Test factory function for default rewards."""
        reward_fn = create_default_reward(risk_adjusted=True, adaptive=False)
        assert isinstance(reward_fn, RiskAdjustedReward)

        reward_fn = create_default_reward(adaptive=True)
        assert isinstance(reward_fn, AdaptiveReward)


# ============================================================================
# Market Regime Detection Tests
# ============================================================================

class TestVolatilityRegimeDetector:
    """Test suite for VolatilityRegimeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = VolatilityRegimeDetector(window_size=20)
        assert detector.window_size == 20

    def test_detect_low_volatility(self):
        """Test detection of low volatility regime."""
        detector = VolatilityRegimeDetector(
            absolute_thresholds=(0.10, 0.20, 0.40)
        )

        # Low volatility returns
        returns = [0.001, -0.001, 0.002, -0.002] * 10
        for ret in returns:
            detector.update(ret)

        regime, vol = detector.detect()
        assert regime in [VolatilityRegime.LOW, VolatilityRegime.MEDIUM]

    def test_detect_high_volatility(self):
        """Test detection of high volatility regime."""
        detector = VolatilityRegimeDetector(
            absolute_thresholds=(0.10, 0.20, 0.40)
        )

        # High volatility returns
        returns = [0.05, -0.05, 0.06, -0.06] * 10
        for ret in returns:
            detector.update(ret)

        regime, vol = detector.detect()
        assert regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]


class TestTrendRegimeDetector:
    """Test suite for TrendRegimeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = TrendRegimeDetector(fast_window=10, slow_window=30)
        assert detector.fast_window == 10
        assert detector.slow_window == 30

    def test_detect_bull_trend(self):
        """Test detection of bull trend."""
        detector = TrendRegimeDetector()

        # Positive trend
        returns = [0.02] * 40
        prices = [100 * (1.02 ** i) for i in range(40)]

        for ret, price in zip(returns, prices):
            detector.update(ret, price)

        regime, strength = detector.detect()
        assert regime in [TrendRegime.BULL, TrendRegime.STRONG_BULL]
        assert strength > 0

    def test_detect_bear_trend(self):
        """Test detection of bear trend."""
        detector = TrendRegimeDetector()

        # Negative trend
        returns = [-0.02] * 40
        prices = [100 * (0.98 ** i) for i in range(40)]

        for ret, price in zip(returns, prices):
            detector.update(ret, price)

        regime, strength = detector.detect()
        assert regime in [TrendRegime.BEAR, TrendRegime.STRONG_BEAR]
        assert strength < 0


class TestCorrelationRegimeDetector:
    """Test suite for CorrelationRegimeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = CorrelationRegimeDetector(window_size=30, num_assets=3)
        assert detector.num_assets == 3

    def test_detect_high_correlation(self):
        """Test detection of high correlation."""
        detector = CorrelationRegimeDetector(window_size=20, num_assets=2)

        # Highly correlated returns
        for i in range(30):
            base_return = np.random.normal(0, 0.01)
            returns = [base_return + np.random.normal(0, 0.001) for _ in range(2)]
            detector.update(returns)

        regime, corr = detector.detect()
        assert corr > 0.5  # Should be positively correlated


class TestMarketRegimeDetector:
    """Test suite for MarketRegimeDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MarketRegimeDetector()
        assert detector.volatility_detector is not None
        assert detector.trend_detector is not None

    def test_update_and_detect(self):
        """Test updating and detecting regime."""
        detector = MarketRegimeDetector()

        # Feed data
        for i in range(50):
            return_value = np.random.normal(0.001, 0.01)
            price = 100 * (1 + i * 0.01)
            detector.update(return_value, price)

        regime = detector.detect()

        assert isinstance(regime, MarketRegime)
        assert isinstance(regime.volatility, VolatilityRegime)
        assert isinstance(regime.trend, TrendRegime)
        assert 0 <= regime.confidence <= 1

    def test_get_regime_summary(self):
        """Test regime summary string."""
        detector = MarketRegimeDetector()

        for i in range(30):
            detector.update(0.01, 100 + i)

        detector.detect()
        summary = detector.get_regime_summary()

        assert isinstance(summary, str)
        assert "Volatility" in summary
        assert "Trend" in summary


class TestRegimeUtilities:
    """Test regime utility functions."""

    def test_create_default_regime_detector(self):
        """Test factory function for regime detector."""
        detector = create_default_regime_detector()
        assert isinstance(detector, MarketRegimeDetector)

        detector_with_corr = create_default_regime_detector(
            include_correlation=True,
            num_assets=3
        )
        assert detector_with_corr.correlation_detector is not None

    def test_regime_to_risk_adjustment(self):
        """Test converting regime to risk adjustments."""
        regime = MarketRegime(
            volatility=VolatilityRegime.HIGH,
            trend=TrendRegime.BULL
        )

        adjustments = regime_to_risk_adjustment(regime)

        assert 'position_size_multiplier' in adjustments
        assert 'stop_loss_multiplier' in adjustments
        # High vol should reduce position size
        assert adjustments['position_size_multiplier'] < 1.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestRewardRegimeIntegration:
    """Integration tests for rewards with regime detection."""

    def test_adaptive_reward_with_regime_detector(self):
        """Test adaptive reward using regime detector."""
        # Custom regime detector
        regime_detector = MarketRegimeDetector()

        def custom_regime_fn(info):
            # Simple mock regime detection
            vol = info.get('portfolio_volatility', 0.0)
            return 'high_vol' if vol > 0.03 else 'low_vol'

        reward_fn = AdaptiveReward(regime_detector=custom_regime_fn)

        # High vol scenario
        info = {
            'portfolio_return': 0.01,
            'portfolio_volatility': 0.05,
            'transaction_costs': 0.001,
        }

        reward, components = reward_fn.compute_reward({}, {}, {}, info)
        assert isinstance(reward, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
