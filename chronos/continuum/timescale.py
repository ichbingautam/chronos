"""
Multi-Timescale Optimization for HOPE.

Implements the insight from Nested Learning that different components
of the optimization (architecture, optimizer, data) operate at different
timescales and should be updated at different frequencies.

Key concepts:
- Fast timescale: Inner loop parameters (updated every step)
- Medium timescale: Outer parameters (updated every K inner steps)
- Slow timescale: Meta-parameters (updated rarely)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from chronos.core.problem import OuterOptimizer
from chronos.utils.logging import get_logger

logger = get_logger("continuum.timescale")


@dataclass
class TimescaleConfig:
    """Configuration for a single timescale."""

    name: str
    update_frequency: int  # Update every N steps
    learning_rate: float
    momentum: float = 0.9
    warmup_steps: int = 0
    cooldown_factor: float = 1.0  # LR multiplier during cooldown


@dataclass
class MultiTimescaleConfig:
    """Configuration for multi-timescale optimization."""

    timescales: List[TimescaleConfig] = field(default_factory=list)

    # Coordination settings
    sync_timescales: bool = True  # Synchronize updates across timescales
    cascade_gradients: bool = True  # Let slow timescales use fast gradients

    # Adaptation
    adaptive_frequency: bool = False  # Adapt update frequency based on progress
    frequency_adaptation_window: int = 100

    @classmethod
    def default_three_level(cls) -> "MultiTimescaleConfig":
        """Create default 3-level timescale configuration."""
        return cls(
            timescales=[
                TimescaleConfig(
                    name="fast",
                    update_frequency=1,
                    learning_rate=0.1,
                    momentum=0.9
                ),
                TimescaleConfig(
                    name="medium",
                    update_frequency=10,
                    learning_rate=0.01,
                    momentum=0.99
                ),
                TimescaleConfig(
                    name="slow",
                    update_frequency=100,
                    learning_rate=0.001,
                    momentum=0.999
                ),
            ]
        )


class TimescaleState:
    """State for a single timescale's optimizer."""

    def __init__(self, config: TimescaleConfig, params: Dict[str, Tensor]):
        self.config = config
        self.step = 0
        self.last_update_step = 0

        # Momentum buffers
        self.momentum_buffer: Dict[str, Tensor] = {
            k: torch.zeros_like(v) for k, v in params.items()
        }

        # Accumulated gradients between updates
        self.accumulated_grad: Dict[str, Tensor] = {
            k: torch.zeros_like(v) for k, v in params.items()
        }
        self.accumulation_count = 0

    def should_update(self, global_step: int) -> bool:
        """Check if this timescale should update at current step."""
        return global_step % self.config.update_frequency == 0

    def get_learning_rate(self, global_step: int) -> float:
        """Get current learning rate with warmup/cooldown."""
        lr = self.config.learning_rate

        # Warmup
        if global_step < self.config.warmup_steps:
            warmup_factor = global_step / max(1, self.config.warmup_steps)
            lr *= warmup_factor

        return lr

    def accumulate_gradient(self, gradient: Dict[str, Tensor]) -> None:
        """Accumulate gradient for delayed update."""
        for name, grad in gradient.items():
            if name in self.accumulated_grad:
                self.accumulated_grad[name] += grad
        self.accumulation_count += 1

    def get_averaged_gradient(self) -> Dict[str, Tensor]:
        """Get gradient averaged over accumulation period."""
        if self.accumulation_count == 0:
            return self.accumulated_grad

        return {
            k: v / self.accumulation_count
            for k, v in self.accumulated_grad.items()
        }

    def reset_accumulator(self) -> None:
        """Reset gradient accumulator after update."""
        for name in self.accumulated_grad:
            self.accumulated_grad[name].zero_()
        self.accumulation_count = 0

    def apply_momentum_update(
        self,
        params: Dict[str, Tensor],
        gradient: Dict[str, Tensor],
        global_step: int
    ) -> Dict[str, Tensor]:
        """Apply momentum-based update."""
        lr = self.get_learning_rate(global_step)
        momentum = self.config.momentum

        updated = {}
        for name, param in params.items():
            if name not in gradient:
                updated[name] = param
                continue

            grad = gradient[name]

            # Update momentum buffer
            self.momentum_buffer[name] = (
                momentum * self.momentum_buffer[name] + grad
            )

            # Apply update
            updated[name] = param - lr * self.momentum_buffer[name]

        self.last_update_step = global_step
        self.step += 1

        return updated


class MultiTimescaleOptimizer(OuterOptimizer):
    """
    Outer optimizer with multi-timescale parameter groups.

    Different parameter groups can be updated at different frequencies,
    reflecting the insight that some hyperparameters should change slowly
    while others can adapt quickly.

    Example:
        - Learning rate: Fast timescale (update frequently)
        - Weight decay: Medium timescale
        - Architecture params: Slow timescale

    Args:
        outer_params: Dictionary of outer parameters with timescale assignments
        config: Multi-timescale configuration
        param_timescale_map: Dict mapping param names to timescale names
    """

    def __init__(
        self,
        outer_params: Dict[str, Tensor],
        config: Optional[MultiTimescaleConfig] = None,
        param_timescale_map: Optional[Dict[str, str]] = None,
        device: torch.device = None
    ):
        # Use fast timescale LR as default
        config = config or MultiTimescaleConfig.default_three_level()
        default_lr = config.timescales[0].learning_rate if config.timescales else 0.01

        super().__init__(outer_params, lr=default_lr, device=device)

        self.config = config

        # Map params to timescales
        self.param_timescale_map = param_timescale_map or {}

        # Default unmapped params to fast timescale
        default_timescale = config.timescales[0].name if config.timescales else "fast"
        for name in outer_params:
            if name not in self.param_timescale_map:
                self.param_timescale_map[name] = default_timescale

        # Initialize per-timescale state
        self._timescale_states: Dict[str, TimescaleState] = {}
        for ts_config in config.timescales:
            # Get params for this timescale
            ts_params = {
                k: v for k, v in outer_params.items()
                if self.param_timescale_map.get(k) == ts_config.name
            }
            if ts_params:
                self._timescale_states[ts_config.name] = TimescaleState(
                    ts_config, ts_params
                )

        self._global_step = 0

        # Stats
        self._update_counts = {ts.name: 0 for ts in config.timescales}

    def compute_hypergradient(
        self,
        trajectories: list,
        inner_problem: Any,
        validation_data: Any = None
    ) -> Dict[str, Tensor]:
        """
        Compute hypergradient (delegated to base implementation).

        This method should be overridden or the optimizer should be
        composed with an actual gradient computation method.
        """
        # Default: return zeros
        return {k: torch.zeros_like(v) for k, v in self.outer_params.items()}

    def step(
        self,
        hypergradient: Dict[str, Tensor],
        force_update: bool = False
    ) -> Dict[str, Tensor]:
        """
        Apply update, respecting timescale update frequencies.

        Args:
            hypergradient: Computed hypergradient
            force_update: If True, update all timescales regardless of frequency

        Returns:
            Updated outer parameters
        """
        self._global_step += 1

        # Accumulate gradients for each timescale
        for ts_name, ts_state in self._timescale_states.items():
            # Get gradients for this timescale's params
            ts_grads = {
                k: hypergradient.get(k, torch.zeros_like(self.outer_params[k]))
                for k in self.outer_params
                if self.param_timescale_map.get(k) == ts_name
            }

            if ts_grads:
                ts_state.accumulate_gradient(ts_grads)

        # Apply updates for timescales that should update
        for ts_name, ts_state in self._timescale_states.items():
            if force_update or ts_state.should_update(self._global_step):
                # Get averaged gradient
                avg_grad = ts_state.get_averaged_gradient()

                # Get current params for this timescale
                ts_params = {
                    k: self.outer_params[k]
                    for k in self.outer_params
                    if self.param_timescale_map.get(k) == ts_name
                }

                # Apply update
                updated = ts_state.apply_momentum_update(
                    ts_params, avg_grad, self._global_step
                )

                # Update outer params
                for k, v in updated.items():
                    self.outer_params[k] = v

                # Reset accumulator
                ts_state.reset_accumulator()
                self._update_counts[ts_name] += 1

                logger.debug(
                    f"Timescale '{ts_name}' updated at step {self._global_step}"
                )

        return self.outer_params

    def get_timescale_info(self) -> Dict[str, Dict]:
        """Get information about each timescale."""
        info = {}
        for ts_name, ts_state in self._timescale_states.items():
            info[ts_name] = {
                "config": ts_state.config,
                "step": ts_state.step,
                "last_update": ts_state.last_update_step,
                "accumulation_count": ts_state.accumulation_count,
                "total_updates": self._update_counts[ts_name],
            }
        return info

    def get_stats(self) -> dict:
        """Get optimizer statistics."""
        return {
            "global_step": self._global_step,
            "update_counts": self._update_counts.copy(),
            "timescale_info": self.get_timescale_info(),
        }


def create_hierarchical_optimizer(
    outer_params: Dict[str, Tensor],
    fast_params: List[str],
    medium_params: List[str],
    slow_params: List[str],
    fast_lr: float = 0.1,
    medium_lr: float = 0.01,
    slow_lr: float = 0.001,
) -> MultiTimescaleOptimizer:
    """
    Convenience function to create a 3-level hierarchical optimizer.

    Args:
        outer_params: All outer parameters
        fast_params: Parameter names for fast timescale
        medium_params: Parameter names for medium timescale
        slow_params: Parameter names for slow timescale
        fast_lr: Fast timescale learning rate
        medium_lr: Medium timescale learning rate
        slow_lr: Slow timescale learning rate

    Returns:
        Configured MultiTimescaleOptimizer
    """
    config = MultiTimescaleConfig(
        timescales=[
            TimescaleConfig("fast", update_frequency=1, learning_rate=fast_lr),
            TimescaleConfig("medium", update_frequency=10, learning_rate=medium_lr),
            TimescaleConfig("slow", update_frequency=100, learning_rate=slow_lr),
        ]
    )

    param_map = {}
    for name in fast_params:
        param_map[name] = "fast"
    for name in medium_params:
        param_map[name] = "medium"
    for name in slow_params:
        param_map[name] = "slow"

    return MultiTimescaleOptimizer(outer_params, config, param_map)
