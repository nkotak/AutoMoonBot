"""
Market regime detection for adaptive reward functions.

This module provides tools to identify different market conditions:
- Volatility regimes (low, medium, high)
- Trend regimes (bull, bear, sideways)
- Correlation regimes (high, low)
- Volume regimes

Regime detection enables adaptive strategies that adjust trading behavior
based on market conditions. For example:
- High volatility → reduce position sizes, favor risk management
- Bull market → favor momentum, increase exposure
- Bear market → defensive positioning, capital preservation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import math


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Trend regime classifications."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


class CorrelationRegime(Enum):
    """Correlation regime classifications."""
    HIGH_POSITIVE = "high_positive"
    MODERATE_POSITIVE = "moderate_positive"
    UNCORRELATED = "uncorrelated"
    MODERATE_NEGATIVE = "moderate_negative"
    HIGH_NEGATIVE = "high_negative"


class VolumeRegime(Enum):
    """Volume regime classifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketRegime:
    """
    Combined market regime information.

    Attributes:
        volatility: Current volatility regime
        trend: Current trend regime
        correlation: Current correlation regime (if multiple assets)
        volume: Current volume regime
        confidence: Confidence in regime classification (0-1)
        metadata: Additional regime information
    """
    volatility: VolatilityRegime
    trend: TrendRegime
    correlation: Optional[CorrelationRegime] = None
    volume: Optional[VolumeRegime] = None
    confidence: float = 1.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VolatilityRegimeDetector:
    """
    Detect volatility regimes using rolling statistics.

    Classifies market volatility into low/medium/high/extreme based on
    historical percentiles or absolute thresholds.

    Args:
        window_size: Rolling window for volatility calculation
        percentile_thresholds: Percentiles for low/medium/high classification
        absolute_thresholds: Optional absolute volatility thresholds
        annualization_factor: Factor to annualize volatility (252 for daily)
    """

    def __init__(
        self,
        window_size: int = 20,
        percentile_thresholds: Tuple[float, float, float] = (30.0, 70.0, 95.0),
        absolute_thresholds: Optional[Tuple[float, float, float]] = None,
        annualization_factor: int = 252
    ):
        self.window_size = window_size
        self.percentile_thresholds = percentile_thresholds
        self.absolute_thresholds = absolute_thresholds
        self.annualization_factor = annualization_factor

        # Rolling returns buffer
        self.returns_history: List[float] = []

        # Historical volatility for percentile calculation
        self.volatility_history: List[float] = []

    def update(self, return_value: float):
        """
        Update detector with new return.

        Args:
            return_value: Period return
        """
        self.returns_history.append(return_value)
        if len(self.returns_history) > self.window_size * 2:
            self.returns_history.pop(0)

        # Compute volatility if enough data
        if len(self.returns_history) >= self.window_size:
            recent_returns = self.returns_history[-self.window_size:]
            vol = np.std(recent_returns, ddof=1) * np.sqrt(self.annualization_factor)
            self.volatility_history.append(vol)

            # Keep reasonable history for percentiles
            if len(self.volatility_history) > 252:
                self.volatility_history.pop(0)

    def detect(self) -> Tuple[VolatilityRegime, float]:
        """
        Detect current volatility regime.

        Returns:
            regime: Volatility regime
            volatility: Current annualized volatility
        """
        if len(self.returns_history) < self.window_size:
            return VolatilityRegime.MEDIUM, 0.0

        # Compute current volatility
        recent_returns = self.returns_history[-self.window_size:]
        current_vol = np.std(recent_returns, ddof=1) * np.sqrt(self.annualization_factor)

        # Classify using absolute thresholds if provided
        if self.absolute_thresholds is not None:
            low_thresh, high_thresh, extreme_thresh = self.absolute_thresholds

            if current_vol < low_thresh:
                regime = VolatilityRegime.LOW
            elif current_vol < high_thresh:
                regime = VolatilityRegime.MEDIUM
            elif current_vol < extreme_thresh:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.EXTREME

        # Otherwise use historical percentiles
        elif len(self.volatility_history) >= 30:
            percentile = np.percentile(
                self.volatility_history,
                np.linspace(0, 100, len(self.volatility_history))
            )

            p_low, p_high, p_extreme = self.percentile_thresholds
            vol_low = np.percentile(self.volatility_history, p_low)
            vol_high = np.percentile(self.volatility_history, p_high)
            vol_extreme = np.percentile(self.volatility_history, p_extreme)

            if current_vol < vol_low:
                regime = VolatilityRegime.LOW
            elif current_vol < vol_high:
                regime = VolatilityRegime.MEDIUM
            elif current_vol < vol_extreme:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.EXTREME

        else:
            regime = VolatilityRegime.MEDIUM

        return regime, current_vol

    def reset(self):
        """Reset detector state."""
        self.returns_history.clear()
        self.volatility_history.clear()


class TrendRegimeDetector:
    """
    Detect trend regimes using moving averages and momentum.

    Identifies bull/bear/sideways markets using:
    - Moving average crossovers (fast vs slow)
    - Cumulative returns
    - Trend strength

    Args:
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        trend_threshold: Threshold for sideways vs trending
    """

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 30,
        trend_threshold: float = 0.02
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.trend_threshold = trend_threshold

        # Price/return history
        self.returns_history: List[float] = []
        self.prices_history: List[float] = []

    def update(self, return_value: float, price: Optional[float] = None):
        """
        Update detector with new data.

        Args:
            return_value: Period return
            price: Optional price level (for MA crossover)
        """
        self.returns_history.append(return_value)
        if len(self.returns_history) > self.slow_window * 2:
            self.returns_history.pop(0)

        if price is not None:
            self.prices_history.append(price)
            if len(self.prices_history) > self.slow_window * 2:
                self.prices_history.pop(0)

    def detect(self) -> Tuple[TrendRegime, float]:
        """
        Detect current trend regime.

        Returns:
            regime: Trend regime
            trend_strength: Strength of trend (-1 to +1)
        """
        if len(self.returns_history) < self.slow_window:
            return TrendRegime.SIDEWAYS, 0.0

        # Method 1: Cumulative returns
        recent_returns = self.returns_history[-self.slow_window:]
        cumulative_return = np.prod([1 + r for r in recent_returns]) - 1

        # Method 2: Moving average crossover (if prices available)
        ma_signal = 0.0
        if len(self.prices_history) >= self.slow_window:
            fast_ma = np.mean(self.prices_history[-self.fast_window:])
            slow_ma = np.mean(self.prices_history[-self.slow_window:])
            ma_signal = (fast_ma - slow_ma) / slow_ma

        # Method 3: Trend consistency (% of positive returns)
        positive_ratio = np.sum(np.array(recent_returns) > 0) / len(recent_returns)

        # Combine signals
        trend_strength = (
            0.5 * cumulative_return +
            0.3 * ma_signal +
            0.2 * (positive_ratio - 0.5) * 2  # Scale to [-1, 1]
        )

        # Classify trend
        if trend_strength > 2 * self.trend_threshold:
            regime = TrendRegime.STRONG_BULL
        elif trend_strength > self.trend_threshold:
            regime = TrendRegime.BULL
        elif trend_strength > -self.trend_threshold:
            regime = TrendRegime.SIDEWAYS
        elif trend_strength > -2 * self.trend_threshold:
            regime = TrendRegime.BEAR
        else:
            regime = TrendRegime.STRONG_BEAR

        return regime, trend_strength

    def reset(self):
        """Reset detector state."""
        self.returns_history.clear()
        self.prices_history.clear()


class CorrelationRegimeDetector:
    """
    Detect correlation regime for multi-asset portfolios.

    Measures how correlated assets are moving, which impacts diversification
    benefits and risk management.

    Args:
        window_size: Rolling window for correlation calculation
        num_assets: Number of assets to track
    """

    def __init__(
        self,
        window_size: int = 30,
        num_assets: int = 2
    ):
        self.window_size = window_size
        self.num_assets = num_assets

        # Returns matrix: each row is a time step, each column is an asset
        self.returns_matrix: List[List[float]] = []

    def update(self, returns: List[float]):
        """
        Update detector with new returns for all assets.

        Args:
            returns: List of returns for each asset
        """
        assert len(returns) == self.num_assets, \
            f"Expected {self.num_assets} returns, got {len(returns)}"

        self.returns_matrix.append(returns)
        if len(self.returns_matrix) > self.window_size * 2:
            self.returns_matrix.pop(0)

    def detect(self) -> Tuple[CorrelationRegime, float]:
        """
        Detect current correlation regime.

        Returns:
            regime: Correlation regime
            avg_correlation: Average pairwise correlation
        """
        if len(self.returns_matrix) < self.window_size:
            return CorrelationRegime.UNCORRELATED, 0.0

        # Get recent returns matrix
        recent_matrix = np.array(self.returns_matrix[-self.window_size:])

        # Compute correlation matrix
        corr_matrix = np.corrcoef(recent_matrix.T)

        # Average pairwise correlation (exclude diagonal)
        mask = ~np.eye(self.num_assets, dtype=bool)
        avg_corr = np.mean(corr_matrix[mask])

        # Classify correlation regime
        if avg_corr > 0.7:
            regime = CorrelationRegime.HIGH_POSITIVE
        elif avg_corr > 0.3:
            regime = CorrelationRegime.MODERATE_POSITIVE
        elif avg_corr > -0.3:
            regime = CorrelationRegime.UNCORRELATED
        elif avg_corr > -0.7:
            regime = CorrelationRegime.MODERATE_NEGATIVE
        else:
            regime = CorrelationRegime.HIGH_NEGATIVE

        return regime, avg_corr

    def reset(self):
        """Reset detector state."""
        self.returns_matrix.clear()


class VolumeRegimeDetector:
    """
    Detect volume regimes.

    High volume indicates strong conviction in price moves, while low volume
    suggests weak participation.

    Args:
        window_size: Rolling window for volume statistics
        percentile_thresholds: Percentiles for classification
    """

    def __init__(
        self,
        window_size: int = 20,
        percentile_thresholds: Tuple[float, float, float] = (25.0, 75.0, 95.0)
    ):
        self.window_size = window_size
        self.percentile_thresholds = percentile_thresholds

        self.volume_history: List[float] = []

    def update(self, volume: float):
        """
        Update detector with new volume.

        Args:
            volume: Trading volume
        """
        self.volume_history.append(volume)
        if len(self.volume_history) > self.window_size * 2:
            self.volume_history.pop(0)

    def detect(self) -> Tuple[VolumeRegime, float]:
        """
        Detect current volume regime.

        Returns:
            regime: Volume regime
            current_volume: Current volume level
        """
        if len(self.volume_history) < self.window_size:
            return VolumeRegime.NORMAL, 0.0

        current_volume = self.volume_history[-1]

        # Compute percentiles
        p_low, p_high, p_extreme = self.percentile_thresholds
        vol_low = np.percentile(self.volume_history, p_low)
        vol_high = np.percentile(self.volume_history, p_high)
        vol_extreme = np.percentile(self.volume_history, p_extreme)

        # Classify
        if current_volume < vol_low:
            regime = VolumeRegime.LOW
        elif current_volume < vol_high:
            regime = VolumeRegime.NORMAL
        elif current_volume < vol_extreme:
            regime = VolumeRegime.HIGH
        else:
            regime = VolumeRegime.EXTREME

        return regime, current_volume

    def reset(self):
        """Reset detector state."""
        self.volume_history.clear()


class MarketRegimeDetector:
    """
    Combined market regime detector.

    Integrates multiple regime detectors to provide comprehensive market
    condition assessment.

    Args:
        volatility_detector: Volatility regime detector
        trend_detector: Trend regime detector
        correlation_detector: Correlation regime detector (optional)
        volume_detector: Volume regime detector (optional)
    """

    def __init__(
        self,
        volatility_detector: Optional[VolatilityRegimeDetector] = None,
        trend_detector: Optional[TrendRegimeDetector] = None,
        correlation_detector: Optional[CorrelationRegimeDetector] = None,
        volume_detector: Optional[VolumeRegimeDetector] = None
    ):
        self.volatility_detector = volatility_detector or VolatilityRegimeDetector()
        self.trend_detector = trend_detector or TrendRegimeDetector()
        self.correlation_detector = correlation_detector
        self.volume_detector = volume_detector

        self.current_regime: Optional[MarketRegime] = None

    def update(
        self,
        return_value: float,
        price: Optional[float] = None,
        returns_vector: Optional[List[float]] = None,
        volume: Optional[float] = None
    ):
        """
        Update all detectors with new data.

        Args:
            return_value: Period return
            price: Optional price level
            returns_vector: Optional returns for all assets (for correlation)
            volume: Optional volume
        """
        self.volatility_detector.update(return_value)
        self.trend_detector.update(return_value, price)

        if self.correlation_detector is not None and returns_vector is not None:
            self.correlation_detector.update(returns_vector)

        if self.volume_detector is not None and volume is not None:
            self.volume_detector.update(volume)

    def detect(self) -> MarketRegime:
        """
        Detect current market regime.

        Returns:
            Combined market regime
        """
        # Detect individual regimes
        vol_regime, vol_value = self.volatility_detector.detect()
        trend_regime, trend_strength = self.trend_detector.detect()

        corr_regime = None
        corr_value = None
        if self.correlation_detector is not None:
            corr_regime, corr_value = self.correlation_detector.detect()

        volume_regime = None
        volume_value = None
        if self.volume_detector is not None:
            volume_regime, volume_value = self.volume_detector.detect()

        # Compute confidence based on data availability
        confidence = min(
            len(self.volatility_detector.returns_history) / self.volatility_detector.window_size,
            1.0
        )

        # Create combined regime
        regime = MarketRegime(
            volatility=vol_regime,
            trend=trend_regime,
            correlation=corr_regime,
            volume=volume_regime,
            confidence=confidence,
            metadata={
                'volatility_value': vol_value,
                'trend_strength': trend_strength,
                'correlation_value': corr_value,
                'volume_value': volume_value,
            }
        )

        self.current_regime = regime
        return regime

    def get_regime_summary(self) -> str:
        """
        Get human-readable regime summary.

        Returns:
            Regime summary string
        """
        if self.current_regime is None:
            return "No regime detected yet"

        summary = f"Market Regime:\n"
        summary += f"  Volatility: {self.current_regime.volatility.value}\n"
        summary += f"  Trend: {self.current_regime.trend.value}\n"

        if self.current_regime.correlation is not None:
            summary += f"  Correlation: {self.current_regime.correlation.value}\n"

        if self.current_regime.volume is not None:
            summary += f"  Volume: {self.current_regime.volume.value}\n"

        summary += f"  Confidence: {self.current_regime.confidence:.2f}"

        return summary

    def reset(self):
        """Reset all detectors."""
        self.volatility_detector.reset()
        self.trend_detector.reset()
        if self.correlation_detector is not None:
            self.correlation_detector.reset()
        if self.volume_detector is not None:
            self.volume_detector.reset()
        self.current_regime = None


# Utility functions

def create_default_regime_detector(
    include_correlation: bool = False,
    include_volume: bool = False,
    num_assets: int = 2
) -> MarketRegimeDetector:
    """
    Factory function to create default regime detector.

    Args:
        include_correlation: Include correlation detector
        include_volume: Include volume detector
        num_assets: Number of assets for correlation detector

    Returns:
        Configured market regime detector
    """
    volatility_detector = VolatilityRegimeDetector(
        window_size=20,
        percentile_thresholds=(30.0, 70.0, 95.0)
    )

    trend_detector = TrendRegimeDetector(
        fast_window=10,
        slow_window=30,
        trend_threshold=0.02
    )

    correlation_detector = None
    if include_correlation:
        correlation_detector = CorrelationRegimeDetector(
            window_size=30,
            num_assets=num_assets
        )

    volume_detector = None
    if include_volume:
        volume_detector = VolumeRegimeDetector(
            window_size=20,
            percentile_thresholds=(25.0, 75.0, 95.0)
        )

    detector = MarketRegimeDetector(
        volatility_detector=volatility_detector,
        trend_detector=trend_detector,
        correlation_detector=correlation_detector,
        volume_detector=volume_detector
    )

    return detector


def regime_to_risk_adjustment(regime: MarketRegime) -> Dict[str, float]:
    """
    Convert regime to suggested risk adjustments.

    Args:
        regime: Market regime

    Returns:
        Dictionary of risk adjustment factors
    """
    adjustments = {
        'position_size_multiplier': 1.0,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 1.0,
    }

    # Adjust based on volatility
    if regime.volatility == VolatilityRegime.LOW:
        adjustments['position_size_multiplier'] = 1.2
        adjustments['stop_loss_multiplier'] = 0.8
    elif regime.volatility == VolatilityRegime.HIGH:
        adjustments['position_size_multiplier'] = 0.7
        adjustments['stop_loss_multiplier'] = 1.5
    elif regime.volatility == VolatilityRegime.EXTREME:
        adjustments['position_size_multiplier'] = 0.4
        adjustments['stop_loss_multiplier'] = 2.0

    # Adjust based on trend
    if regime.trend in [TrendRegime.STRONG_BULL, TrendRegime.STRONG_BEAR]:
        adjustments['take_profit_multiplier'] = 1.5
    elif regime.trend == TrendRegime.SIDEWAYS:
        adjustments['position_size_multiplier'] *= 0.8

    return adjustments
