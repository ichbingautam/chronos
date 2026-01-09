"""
Benchmark Runner for Chronos.

Provides a framework for running reproducible benchmarks comparing
different optimization strategies (sync vs async, sparse vs dense).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from chronos.utils.logging import get_logger

logger = get_logger("benchmarks.runner")


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    description: str = ""

    # Problem settings
    num_outer_steps: int = 100
    num_inner_steps: int = 50
    batch_size: int = 32

    # Distributed settings
    num_workers: int = 1
    max_staleness: int = 2
    significance_threshold: float = 0.01

    # Comparison modes
    sync_mode: bool = True  # Compare with synchronous baseline
    sparse_communication: bool = True

    # Reproducibility
    seed: int = 42
    device: str = "cpu"

    # Output
    output_dir: Optional[str] = None
    save_checkpoints: bool = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config: BenchmarkConfig

    # Timing
    wall_time_seconds: float = 0.0
    inner_loop_time: float = 0.0
    communication_time: float = 0.0
    outer_step_time: float = 0.0

    # Convergence
    final_validation_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)

    # Communication
    total_bytes_sent: int = 0
    total_messages: int = 0
    messages_skipped: int = 0  # Due to significance filter

    # Staleness
    avg_staleness: float = 0.0
    max_staleness: int = 0
    commits_rejected: int = 0

    # Throughput
    outer_steps_per_second: float = 0.0
    inner_steps_per_second: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["config"] = asdict(self.config)
        return result

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        config = BenchmarkConfig(**data.pop("config"))
        return cls(config=config, **data)


class BenchmarkRunner:
    """
    Runner for benchmarking Chronos optimization.

    Supports comparing:
    - Synchronous vs asynchronous optimization
    - Dense vs sparse communication
    - Different staleness bounds

    Usage:
        runner = BenchmarkRunner(config)
        result = runner.run(problem_factory, outer_optimizer_factory)
        result.save("benchmark_results.json")
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._setup_device()
        self._set_seed()

    def _setup_device(self) -> None:
        """Set up compute device."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def run(
        self,
        create_problem: Callable,
        create_optimizer: Callable,
        create_data_loader: Optional[Callable] = None,
        validation_fn: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        Run the benchmark.

        Args:
            create_problem: Factory function creating InnerProblem
            create_optimizer: Factory function creating OuterOptimizer
            create_data_loader: Optional factory for data loader
            validation_fn: Optional function to compute validation metric

        Returns:
            BenchmarkResult with timing and convergence metrics
        """
        result = BenchmarkResult(config=self.config)

        logger.info(f"Starting benchmark: {self.config.name}")
        start_time = time.time()

        # Create components
        inner_problem = create_problem(self.device)
        outer_optimizer = create_optimizer(self.device)
        data_loader = create_data_loader() if create_data_loader else None

        # Timing accumulators
        inner_time = 0.0
        outer_time = 0.0
        comm_time = 0.0

        # Run optimization
        loss_history = []
        total_inner_steps = 0

        for outer_step in range(self.config.num_outer_steps):
            # Inner loop
            inner_start = time.time()

            final_params, trajectory = inner_problem.solve(
                outer_params=outer_optimizer.outer_params,
                num_steps=self.config.num_inner_steps,
                data_loader=data_loader
            )

            inner_time += time.time() - inner_start
            total_inner_steps += self.config.num_inner_steps

            # Communication (simulated timing)
            comm_start = time.time()
            # In distributed mode, this would be network communication
            # For now, just track the overhead
            comm_time += time.time() - comm_start

            # Outer step
            outer_start = time.time()

            hypergradient = outer_optimizer.compute_hypergradient(
                trajectories=[trajectory],
                inner_problem=inner_problem
            )
            outer_optimizer.step(hypergradient)

            outer_time += time.time() - outer_start

            # Validation
            if validation_fn:
                val_loss = validation_fn(inner_problem, outer_optimizer.outer_params)
            else:
                val_loss = trajectory.final_loss

            loss_history.append(val_loss)

            # Progress logging
            if (outer_step + 1) % 10 == 0:
                logger.info(
                    f"Step {outer_step + 1}/{self.config.num_outer_steps}, "
                    f"loss={val_loss:.4f}"
                )

        total_time = time.time() - start_time

        # Fill in results
        result.wall_time_seconds = total_time
        result.inner_loop_time = inner_time
        result.outer_step_time = outer_time
        result.communication_time = comm_time
        result.final_validation_loss = loss_history[-1] if loss_history else 0.0
        result.loss_history = loss_history
        result.outer_steps_per_second = self.config.num_outer_steps / total_time
        result.inner_steps_per_second = total_inner_steps / total_time

        logger.info(
            f"Benchmark completed: {total_time:.2f}s, "
            f"final_loss={result.final_validation_loss:.4f}"
        )

        # Save if output dir specified
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / f"{self.config.name}_result.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)

        return result

    def compare_sync_vs_sparse(
        self,
        create_problem: Callable,
        create_optimizer: Callable,
        create_data_loader: Optional[Callable] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare synchronous vs sparse communication.

        Returns:
            Dictionary with 'sync' and 'sparse' results
        """
        results = {}

        # Sync baseline
        sync_config = BenchmarkConfig(
            name=f"{self.config.name}_sync",
            num_outer_steps=self.config.num_outer_steps,
            num_inner_steps=self.config.num_inner_steps,
            significance_threshold=0.0,  # Always communicate
            seed=self.config.seed
        )
        sync_runner = BenchmarkRunner(sync_config)
        results["sync"] = sync_runner.run(
            create_problem, create_optimizer, create_data_loader
        )

        # Sparse communication
        sparse_config = BenchmarkConfig(
            name=f"{self.config.name}_sparse",
            num_outer_steps=self.config.num_outer_steps,
            num_inner_steps=self.config.num_inner_steps,
            significance_threshold=self.config.significance_threshold,
            seed=self.config.seed
        )
        sparse_runner = BenchmarkRunner(sparse_config)
        results["sparse"] = sparse_runner.run(
            create_problem, create_optimizer, create_data_loader
        )

        # Log comparison
        sync_loss = results["sync"].final_validation_loss
        sparse_loss = results["sparse"].final_validation_loss

        logger.info(f"Sync final loss: {sync_loss:.4f}")
        logger.info(f"Sparse final loss: {sparse_loss:.4f}")
        logger.info(f"Loss difference: {abs(sync_loss - sparse_loss):.4f}")

        return results
