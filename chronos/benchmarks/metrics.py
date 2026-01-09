"""
Benchmark Metrics for Chronos.

Provides utilities for measuring key performance indicators:
- Throughput (steps/second)
- Network traffic
- Staleness impact on convergence
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chronos.utils.logging import get_logger

logger = get_logger("benchmarks.metrics")


@dataclass
class ThroughputMetrics:
    """Throughput measurement results."""

    outer_steps_per_second: float
    inner_steps_per_second: float
    total_wall_time: float
    total_outer_steps: int
    total_inner_steps: int


def measure_throughput(
    run_fn: Callable,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> ThroughputMetrics:
    """
    Measure optimization throughput.

    Args:
        run_fn: Function that runs one outer step, returns inner step count
        num_iterations: Number of iterations to measure
        warmup_iterations: Warmup iterations (not counted)

    Returns:
        ThroughputMetrics with timing information
    """
    # Warmup
    for _ in range(warmup_iterations):
        run_fn()

    # Measure
    total_inner = 0
    start_time = time.time()

    for _ in range(num_iterations):
        inner_steps = run_fn()
        total_inner += inner_steps if inner_steps else 0

    total_time = time.time() - start_time

    return ThroughputMetrics(
        outer_steps_per_second=num_iterations / total_time,
        inner_steps_per_second=total_inner / total_time,
        total_wall_time=total_time,
        total_outer_steps=num_iterations,
        total_inner_steps=total_inner
    )


@dataclass
class NetworkMetrics:
    """Network traffic measurement results."""

    total_bytes_sent: int
    total_bytes_received: int
    total_messages: int
    avg_message_size: float
    compression_ratio: float
    messages_skipped: int  # Due to significance filter


class NetworkTracker:
    """
    Track network traffic during optimization.

    Wraps send/receive functions to measure bytes transferred.
    """

    def __init__(self):
        self._bytes_sent = 0
        self._bytes_received = 0
        self._messages_sent = 0
        self._messages_received = 0
        self._original_bytes = 0  # Before compression
        self._skipped = 0

    def track_send(self, data: bytes, original_size: int | None = None) -> None:
        """Track bytes sent."""
        self._bytes_sent += len(data)
        self._messages_sent += 1
        self._original_bytes += original_size or len(data)

    def track_receive(self, data: bytes) -> None:
        """Track bytes received."""
        self._bytes_received += len(data)
        self._messages_received += 1

    def track_skip(self) -> None:
        """Track skipped message (significance filter)."""
        self._skipped += 1

    def get_metrics(self) -> NetworkMetrics:
        """Get accumulated metrics."""
        total_messages = self._messages_sent + self._messages_received
        total_bytes = self._bytes_sent + self._bytes_received

        return NetworkMetrics(
            total_bytes_sent=self._bytes_sent,
            total_bytes_received=self._bytes_received,
            total_messages=total_messages,
            avg_message_size=(
                total_bytes / total_messages if total_messages > 0 else 0
            ),
            compression_ratio=(
                self._original_bytes / self._bytes_sent
                if self._bytes_sent > 0 else 1.0
            ),
            messages_skipped=self._skipped
        )

    def reset(self) -> None:
        """Reset all counters."""
        self._bytes_sent = 0
        self._bytes_received = 0
        self._messages_sent = 0
        self._messages_received = 0
        self._original_bytes = 0
        self._skipped = 0


def measure_network_traffic(
    run_fn: Callable[[NetworkTracker], None],
    num_iterations: int = 100
) -> NetworkMetrics:
    """
    Measure network traffic during optimization.

    Args:
        run_fn: Function that runs optimization and uses tracker
        num_iterations: Number of iterations

    Returns:
        NetworkMetrics with traffic information
    """
    tracker = NetworkTracker()

    for _ in range(num_iterations):
        run_fn(tracker)

    return tracker.get_metrics()


@dataclass
class StalenessMetrics:
    """Staleness impact measurement results."""

    avg_staleness: float
    max_staleness: int
    staleness_histogram: dict[int, int]  # staleness -> count

    # Convergence correlation
    loss_by_staleness: dict[int, float]  # avg loss for each staleness level
    commits_rejected: int
    acceptance_rate: float


class StalenessTracker:
    """Track staleness and its impact on optimization."""

    def __init__(self):
        self._staleness_counts: dict[int, int] = {}
        self._staleness_losses: dict[int, list[float]] = {}
        self._rejected = 0
        self._accepted = 0

    def track_commit(
        self,
        staleness: int,
        loss: float,
        accepted: bool
    ) -> None:
        """Track a trajectory commit."""
        if accepted:
            self._accepted += 1
            self._staleness_counts[staleness] = (
                self._staleness_counts.get(staleness, 0) + 1
            )
            if staleness not in self._staleness_losses:
                self._staleness_losses[staleness] = []
            self._staleness_losses[staleness].append(loss)
        else:
            self._rejected += 1

    def get_metrics(self) -> StalenessMetrics:
        """Get staleness metrics."""
        if not self._staleness_counts:
            return StalenessMetrics(
                avg_staleness=0.0,
                max_staleness=0,
                staleness_histogram={},
                loss_by_staleness={},
                commits_rejected=self._rejected,
                acceptance_rate=0.0
            )

        # Compute average staleness
        total_staleness = sum(
            s * c for s, c in self._staleness_counts.items()
        )
        total_commits = sum(self._staleness_counts.values())
        avg_staleness = total_staleness / total_commits if total_commits > 0 else 0

        # Compute average loss by staleness
        loss_by_staleness = {
            s: sum(losses) / len(losses)
            for s, losses in self._staleness_losses.items()
            if losses
        }

        return StalenessMetrics(
            avg_staleness=avg_staleness,
            max_staleness=max(self._staleness_counts.keys()),
            staleness_histogram=self._staleness_counts.copy(),
            loss_by_staleness=loss_by_staleness,
            commits_rejected=self._rejected,
            acceptance_rate=(
                self._accepted / (self._accepted + self._rejected)
                if (self._accepted + self._rejected) > 0 else 0
            )
        )


def measure_staleness_impact(
    run_fn: Callable[[StalenessTracker], None],
    num_iterations: int = 100
) -> StalenessMetrics:
    """
    Measure staleness impact on optimization.

    Args:
        run_fn: Function that runs optimization and uses tracker
        num_iterations: Number of iterations

    Returns:
        StalenessMetrics with staleness analysis
    """
    tracker = StalenessTracker()

    for _ in range(num_iterations):
        run_fn(tracker)

    return tracker.get_metrics()


def compare_configurations(
    base_run_fn: Callable,
    compare_run_fn: Callable,
    num_iterations: int = 100,
    metric: str = "loss"
) -> dict[str, Any]:
    """
    Compare two configurations and compute relative improvement.

    Args:
        base_run_fn: Baseline configuration run function
        compare_run_fn: Comparison configuration run function
        num_iterations: Number of iterations per config
        metric: Metric to compare ("loss", "throughput", "bandwidth")

    Returns:
        Comparison results including improvement percentage
    """
    # Run baseline
    base_results = []
    for _ in range(num_iterations):
        result = base_run_fn()
        base_results.append(result)

    # Run comparison
    compare_results = []
    for _ in range(num_iterations):
        result = compare_run_fn()
        compare_results.append(result)

    # Compute statistics
    base_mean = sum(base_results) / len(base_results)
    compare_mean = sum(compare_results) / len(compare_results)

    improvement = (base_mean - compare_mean) / base_mean * 100 if base_mean != 0 else 0

    return {
        "baseline_mean": base_mean,
        "comparison_mean": compare_mean,
        "improvement_percent": improvement,
        "baseline_results": base_results,
        "comparison_results": compare_results,
    }
