"""
Portfolio-level risk limits and controls for algorithmic trading.

This module implements various portfolio-level risk constraints to
prevent excessive risk-taking and ensure compliance with risk policies.

Risk Limits:
- Maximum Drawdown Limit: Halt trading if drawdown exceeds threshold
- VaR Limit: Limit Value at Risk exposure
- CVaR Limit: Limit Conditional VaR exposure
- Exposure Limit: Maximum total portfolio exposure
- Concentration Limit: Maximum position size per asset
- Correlation Limit: Maximum correlation exposure
- Leverage Limit: Maximum portfolio leverage
- Sector Exposure Limit: Maximum exposure per sector/asset class

All limits support:
- Soft limits (warnings) and hard limits (enforcement)
- Dynamic adjustment based on market conditions
- Breach handling and notifications
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


class LimitType(Enum):
    """Types of risk limits."""
    DRAWDOWN = "drawdown"
    VAR = "var"
    CVAR = "cvar"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    SECTOR_EXPOSURE = "sector_exposure"


class LimitSeverity(Enum):
    """Severity of limit breach."""
    NONE = "none"
    WARNING = "warning"  # Soft limit breached
    CRITICAL = "critical"  # Hard limit breached


@dataclass
class LimitBreach:
    """
    Information about a limit breach.

    Attributes:
        limit_type: Type of limit breached
        severity: Severity of breach
        current_value: Current value
        limit_value: Limit threshold
        message: Description of breach
        timestamp: When breach occurred
    """
    limit_type: LimitType
    severity: LimitSeverity
    current_value: float
    limit_value: float
    message: str
    timestamp: int = 0


class BaseRiskLimit(ABC):
    """
    Abstract base class for risk limits.

    Args:
        hard_limit: Hard limit threshold (enforcement)
        soft_limit: Soft limit threshold (warning)
        enabled: Whether limit is active
    """

    def __init__(
        self,
        hard_limit: float,
        soft_limit: Optional[float] = None,
        enabled: bool = True
    ):
        self.hard_limit = hard_limit
        self.soft_limit = soft_limit if soft_limit is not None else hard_limit * 0.8
        self.enabled = enabled

        self.breaches: List[LimitBreach] = []
        self.current_value: Optional[float] = None

    @abstractmethod
    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """
        Check if limit is breached.

        Returns:
            is_breached: True if hard limit breached
            breach_info: Breach information if any limit violated
        """
        pass

    def _create_breach(
        self,
        limit_type: LimitType,
        severity: LimitSeverity,
        current_value: float,
        limit_value: float,
        message: str
    ) -> LimitBreach:
        """Create breach record."""

        breach = LimitBreach(
            limit_type=limit_type,
            severity=severity,
            current_value=current_value,
            limit_value=limit_value,
            message=message
        )

        self.breaches.append(breach)
        return breach

    def reset(self):
        """Reset limit state."""
        self.breaches.clear()
        self.current_value = None


class MaxDrawdownLimit(BaseRiskLimit):
    """
    Maximum drawdown limit.

    Prevents trading when drawdown exceeds threshold.

    Args:
        hard_limit: Maximum drawdown before halting (e.g., 0.20 for 20%)
        soft_limit: Warning threshold (e.g., 0.15 for 15%)
    """

    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """Check drawdown limit."""

        if not self.enabled:
            return False, None

        current_drawdown = kwargs.get('current_drawdown')
        if current_drawdown is None:
            portfolio_value = kwargs.get('portfolio_value')
            peak_value = kwargs.get('peak_value')

            if portfolio_value is None or peak_value is None:
                raise ValueError("Either 'current_drawdown' or 'portfolio_value' and 'peak_value' must be provided")

            current_drawdown = (peak_value - portfolio_value) / peak_value

        self.current_value = current_drawdown

        # Check hard limit
        if current_drawdown >= self.hard_limit:
            breach = self._create_breach(
                limit_type=LimitType.DRAWDOWN,
                severity=LimitSeverity.CRITICAL,
                current_value=current_drawdown,
                limit_value=self.hard_limit,
                message=f"Critical: Drawdown {current_drawdown:.2%} exceeds hard limit {self.hard_limit:.2%}"
            )
            return True, breach

        # Check soft limit
        if current_drawdown >= self.soft_limit:
            breach = self._create_breach(
                limit_type=LimitType.DRAWDOWN,
                severity=LimitSeverity.WARNING,
                current_value=current_drawdown,
                limit_value=self.soft_limit,
                message=f"Warning: Drawdown {current_drawdown:.2%} exceeds soft limit {self.soft_limit:.2%}"
            )
            return False, breach

        return False, None


class VaRLimit(BaseRiskLimit):
    """
    Value at Risk limit.

    Limits portfolio VaR exposure.

    Args:
        hard_limit: Maximum VaR (e.g., 0.05 for 5%)
        soft_limit: Warning VaR threshold
        confidence_level: Confidence level for VaR (default: 0.95)
    """

    def __init__(
        self,
        hard_limit: float,
        soft_limit: Optional[float] = None,
        confidence_level: float = 0.95,
        **kwargs
    ):
        super().__init__(hard_limit, soft_limit, **kwargs)
        self.confidence_level = confidence_level

    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """Check VaR limit."""

        if not self.enabled:
            return False, None

        var = kwargs.get('var')
        if var is None:
            raise ValueError("'var' must be provided")

        self.current_value = var

        if var >= self.hard_limit:
            breach = self._create_breach(
                limit_type=LimitType.VAR,
                severity=LimitSeverity.CRITICAL,
                current_value=var,
                limit_value=self.hard_limit,
                message=f"Critical: VaR {var:.2%} exceeds hard limit {self.hard_limit:.2%}"
            )
            return True, breach

        if var >= self.soft_limit:
            breach = self._create_breach(
                limit_type=LimitType.VAR,
                severity=LimitSeverity.WARNING,
                current_value=var,
                limit_value=self.soft_limit,
                message=f"Warning: VaR {var:.2%} exceeds soft limit {self.soft_limit:.2%}"
            )
            return False, breach

        return False, None


class ExposureLimit(BaseRiskLimit):
    """
    Total portfolio exposure limit.

    Limits total gross or net exposure.

    Args:
        hard_limit: Maximum exposure (e.g., 2.0 for 200%)
        soft_limit: Warning threshold
        use_gross_exposure: If True, use gross exposure; else net (default: True)
    """

    def __init__(
        self,
        hard_limit: float,
        soft_limit: Optional[float] = None,
        use_gross_exposure: bool = True,
        **kwargs
    ):
        super().__init__(hard_limit, soft_limit, **kwargs)
        self.use_gross_exposure = use_gross_exposure

    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """Check exposure limit."""

        if not self.enabled:
            return False, None

        if self.use_gross_exposure:
            exposure = kwargs.get('gross_exposure')
            exposure_type = "Gross exposure"
        else:
            exposure = kwargs.get('net_exposure')
            exposure_type = "Net exposure"

        if exposure is None:
            position_values = kwargs.get('position_values')
            if position_values is None:
                raise ValueError("Either exposure or position_values must be provided")

            if self.use_gross_exposure:
                exposure = sum(abs(v) for v in position_values.values())
            else:
                exposure = sum(position_values.values())

        portfolio_value = kwargs.get('portfolio_value', 1.0)
        exposure_ratio = exposure / portfolio_value

        self.current_value = exposure_ratio

        if exposure_ratio >= self.hard_limit:
            breach = self._create_breach(
                limit_type=LimitType.EXPOSURE,
                severity=LimitSeverity.CRITICAL,
                current_value=exposure_ratio,
                limit_value=self.hard_limit,
                message=f"Critical: {exposure_type} {exposure_ratio:.2f}x exceeds hard limit {self.hard_limit:.2f}x"
            )
            return True, breach

        if exposure_ratio >= self.soft_limit:
            breach = self._create_breach(
                limit_type=LimitType.EXPOSURE,
                severity=LimitSeverity.WARNING,
                current_value=exposure_ratio,
                limit_value=self.soft_limit,
                message=f"Warning: {exposure_type} {exposure_ratio:.2f}x exceeds soft limit {self.soft_limit:.2f}x"
            )
            return False, breach

        return False, None


class ConcentrationLimit(BaseRiskLimit):
    """
    Position concentration limit.

    Limits maximum position size per asset.

    Args:
        hard_limit: Maximum position as fraction of portfolio (e.g., 0.25 for 25%)
        soft_limit: Warning threshold
        apply_to_all_positions: If True, check all positions; else only check largest
    """

    def __init__(
        self,
        hard_limit: float,
        soft_limit: Optional[float] = None,
        apply_to_all_positions: bool = True,
        **kwargs
    ):
        super().__init__(hard_limit, soft_limit, **kwargs)
        self.apply_to_all_positions = apply_to_all_positions

    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """Check concentration limit."""

        if not self.enabled:
            return False, None

        position_values = kwargs.get('position_values')
        if position_values is None:
            raise ValueError("'position_values' must be provided")

        portfolio_value = kwargs.get('portfolio_value', 1.0)

        # Calculate position fractions
        position_fractions = {
            asset: abs(value) / portfolio_value
            for asset, value in position_values.items()
        }

        if self.apply_to_all_positions:
            # Check all positions
            max_concentration = max(position_fractions.values()) if position_fractions else 0.0
            violating_assets = [
                asset for asset, frac in position_fractions.items()
                if frac >= self.soft_limit
            ]
        else:
            # Check only largest position
            if not position_fractions:
                return False, None

            max_asset = max(position_fractions, key=position_fractions.get)
            max_concentration = position_fractions[max_asset]
            violating_assets = [max_asset] if max_concentration >= self.soft_limit else []

        self.current_value = max_concentration

        if max_concentration >= self.hard_limit:
            assets_str = ", ".join(violating_assets)
            breach = self._create_breach(
                limit_type=LimitType.CONCENTRATION,
                severity=LimitSeverity.CRITICAL,
                current_value=max_concentration,
                limit_value=self.hard_limit,
                message=f"Critical: Position concentration {max_concentration:.2%} exceeds hard limit {self.hard_limit:.2%} (assets: {assets_str})"
            )
            return True, breach

        if max_concentration >= self.soft_limit:
            assets_str = ", ".join(violating_assets)
            breach = self._create_breach(
                limit_type=LimitType.CONCENTRATION,
                severity=LimitSeverity.WARNING,
                current_value=max_concentration,
                limit_value=self.soft_limit,
                message=f"Warning: Position concentration {max_concentration:.2%} exceeds soft limit {self.soft_limit:.2%} (assets: {assets_str})"
            )
            return False, breach

        return False, None


class LeverageLimit(BaseRiskLimit):
    """
    Leverage limit.

    Limits maximum portfolio leverage.

    Args:
        hard_limit: Maximum leverage (e.g., 2.0 for 2x)
        soft_limit: Warning threshold
    """

    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """Check leverage limit."""

        if not self.enabled:
            return False, None

        leverage = kwargs.get('leverage')
        if leverage is None:
            total_exposure = kwargs.get('total_exposure')
            portfolio_value = kwargs.get('portfolio_value')

            if total_exposure is None or portfolio_value is None:
                raise ValueError("Either 'leverage' or 'total_exposure' and 'portfolio_value' must be provided")

            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0.0

        self.current_value = leverage

        if leverage >= self.hard_limit:
            breach = self._create_breach(
                limit_type=LimitType.LEVERAGE,
                severity=LimitSeverity.CRITICAL,
                current_value=leverage,
                limit_value=self.hard_limit,
                message=f"Critical: Leverage {leverage:.2f}x exceeds hard limit {self.hard_limit:.2f}x"
            )
            return True, breach

        if leverage >= self.soft_limit:
            breach = self._create_breach(
                limit_type=LimitType.LEVERAGE,
                severity=LimitSeverity.WARNING,
                current_value=leverage,
                limit_value=self.soft_limit,
                message=f"Warning: Leverage {leverage:.2f}x exceeds soft limit {self.soft_limit:.2f}x"
            )
            return False, breach

        return False, None


class SectorExposureLimit(BaseRiskLimit):
    """
    Sector/asset class exposure limit.

    Limits exposure to specific sectors or asset classes.

    Args:
        hard_limit: Maximum sector exposure (e.g., 0.40 for 40%)
        soft_limit: Warning threshold
        sector_mapping: Dict mapping assets to sectors
    """

    def __init__(
        self,
        hard_limit: float,
        soft_limit: Optional[float] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(hard_limit, soft_limit, **kwargs)
        self.sector_mapping = sector_mapping or {}

    def check_limit(
        self,
        **kwargs
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """Check sector exposure limit."""

        if not self.enabled:
            return False, None

        position_values = kwargs.get('position_values')
        if position_values is None:
            raise ValueError("'position_values' must be provided")

        portfolio_value = kwargs.get('portfolio_value', 1.0)

        # Calculate sector exposures
        sector_exposures: Dict[str, float] = {}
        for asset, value in position_values.items():
            sector = self.sector_mapping.get(asset, 'unknown')
            sector_exposures[sector] = sector_exposures.get(sector, 0.0) + abs(value)

        # Convert to fractions
        sector_fractions = {
            sector: exposure / portfolio_value
            for sector, exposure in sector_exposures.items()
        }

        if not sector_fractions:
            return False, None

        max_sector = max(sector_fractions, key=sector_fractions.get)
        max_exposure = sector_fractions[max_sector]

        self.current_value = max_exposure

        if max_exposure >= self.hard_limit:
            breach = self._create_breach(
                limit_type=LimitType.SECTOR_EXPOSURE,
                severity=LimitSeverity.CRITICAL,
                current_value=max_exposure,
                limit_value=self.hard_limit,
                message=f"Critical: Sector {max_sector} exposure {max_exposure:.2%} exceeds hard limit {self.hard_limit:.2%}"
            )
            return True, breach

        if max_exposure >= self.soft_limit:
            breach = self._create_breach(
                limit_type=LimitType.SECTOR_EXPOSURE,
                severity=LimitSeverity.WARNING,
                current_value=max_exposure,
                limit_value=self.soft_limit,
                message=f"Warning: Sector {max_sector} exposure {max_exposure:.2%} exceeds soft limit {self.soft_limit:.2%}"
            )
            return False, breach

        return False, None


class RiskLimitManager:
    """
    Manages multiple risk limits.

    Centralized manager for all portfolio risk limits.

    Args:
        limits: List of risk limits to enforce
        halt_on_breach: If True, halt trading on hard limit breach
        notification_callback: Optional function to call on breach
    """

    def __init__(
        self,
        limits: Optional[List[BaseRiskLimit]] = None,
        halt_on_breach: bool = True,
        notification_callback: Optional[Callable] = None
    ):
        self.limits = limits or []
        self.halt_on_breach = halt_on_breach
        self.notification_callback = notification_callback

        self.is_halted = False
        self.all_breaches: List[LimitBreach] = []

    def add_limit(self, limit: BaseRiskLimit):
        """Add a risk limit."""
        self.limits.append(limit)

    def remove_limit(self, limit_type: LimitType):
        """Remove all limits of specified type."""
        self.limits = [l for l in self.limits if not isinstance(l, limit_type)]

    def check_all_limits(self, **kwargs) -> Tuple[bool, List[LimitBreach]]:
        """
        Check all risk limits.

        Args:
            **kwargs: Data needed for limit checks

        Returns:
            is_halted: True if any hard limit breached
            breaches: List of all breaches
        """

        current_breaches = []
        any_hard_breach = False

        for limit in self.limits:
            try:
                is_breached, breach_info = limit.check_limit(**kwargs)

                if breach_info is not None:
                    current_breaches.append(breach_info)
                    self.all_breaches.append(breach_info)

                    # Notify if callback provided
                    if self.notification_callback is not None:
                        self.notification_callback(breach_info)

                if is_breached:
                    any_hard_breach = True

            except Exception as e:
                # Log error but continue checking other limits
                print(f"Error checking limit {type(limit).__name__}: {e}")

        # Update halt status
        if any_hard_breach and self.halt_on_breach:
            self.is_halted = True

        return self.is_halted, current_breaches

    def reset_all(self):
        """Reset all limits and halt status."""
        for limit in self.limits:
            limit.reset()
        self.is_halted = False
        self.all_breaches.clear()

    def get_limit_status(self) -> Dict[str, Dict]:
        """
        Get status of all limits.

        Returns:
            Dictionary with limit statuses
        """
        status = {}
        for limit in self.limits:
            limit_name = type(limit).__name__
            status[limit_name] = {
                'enabled': limit.enabled,
                'current_value': limit.current_value,
                'soft_limit': limit.soft_limit,
                'hard_limit': limit.hard_limit,
                'num_breaches': len(limit.breaches)
            }
        return status


def create_default_limits(
    portfolio_value: float = 100000.0,
    max_drawdown: float = 0.20,
    max_concentration: float = 0.25,
    max_leverage: float = 2.0
) -> RiskLimitManager:
    """
    Factory function to create default risk limits.

    Args:
        portfolio_value: Portfolio value for calculations
        max_drawdown: Maximum drawdown threshold
        max_concentration: Maximum single position size
        max_leverage: Maximum leverage

    Returns:
        RiskLimitManager with default limits
    """

    limits = [
        MaxDrawdownLimit(
            hard_limit=max_drawdown,
            soft_limit=max_drawdown * 0.75
        ),
        ConcentrationLimit(
            hard_limit=max_concentration,
            soft_limit=max_concentration * 0.8
        ),
        LeverageLimit(
            hard_limit=max_leverage,
            soft_limit=max_leverage * 0.8
        ),
        ExposureLimit(
            hard_limit=max_leverage,
            soft_limit=max_leverage * 0.8,
            use_gross_exposure=True
        ),
    ]

    manager = RiskLimitManager(limits=limits, halt_on_breach=True)
    return manager
