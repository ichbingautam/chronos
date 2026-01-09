"""
Worker Agent for Distributed Chronos.

The worker is an autonomous agent that:
1. Connects to the coordinator
2. Checks out outer parameters
3. Runs inner optimization loop
4. Decides when to communicate (significance-triggered)
5. Commits trajectories back to coordinator

Can run in multiple processes/machines for parallel optimization.
"""

import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

import zmq
import torch
from torch import Tensor

from chronos.core.problem import InnerProblem
from chronos.core.state import Trajectory
from chronos.distributed.protocols import (
    MessageType,
    CheckoutRequest,
    CheckoutResponse,
    CommitRequest,
    CommitResponse,
)
from chronos.utils.logging import get_logger

logger = get_logger("worker")


@dataclass
class WorkerConfig:
    """Configuration for a worker agent."""

    coordinator_addr: str = "tcp://localhost:5555"
    worker_id: Optional[str] = None
    inner_steps: int = 100
    heartbeat_interval: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    significance_threshold: float = 0.01  # For sparse communication

    def __post_init__(self):
        if self.worker_id is None:
            self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"


class Worker:
    """
    Autonomous worker for distributed nested optimization.

    Runs an inner optimization loop and communicates results
    back to the coordinator using significance-triggered protocol.

    Args:
        inner_problem: The inner optimization problem to solve
        config: Worker configuration
        data_loader: Iterator providing training data batches
    """

    def __init__(
        self,
        inner_problem: InnerProblem,
        config: Optional[WorkerConfig] = None,
        data_loader: Optional[Iterator] = None
    ):
        self.inner_problem = inner_problem
        self.config = config or WorkerConfig()
        self.data_loader = data_loader

        # ZeroMQ setup
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None

        # State
        self._running = False
        self._current_version: Optional[int] = None
        self._current_outer_params: Optional[Dict[str, Tensor]] = None

        # Stats
        self._stats = {
            "inner_loops_completed": 0,
            "commits_sent": 0,
            "commits_accepted": 0,
            "commits_rejected": 0,
            "start_time": None,
        }

        # Significance filter for sparse communication
        self._accumulated_delta: Optional[Dict[str, Tensor]] = None
        self._last_synced_params: Optional[Dict[str, Tensor]] = None

    @property
    def worker_id(self) -> str:
        return self.config.worker_id

    def connect(self) -> None:
        """Connect to the coordinator."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(self.config.coordinator_addr)

        # Set timeout
        self._socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds

        # Register with coordinator
        self._send_register()

        logger.info(
            f"Worker {self.worker_id} connected to {self.config.coordinator_addr}"
        )

    def disconnect(self) -> None:
        """Disconnect from coordinator."""
        if self._socket:
            try:
                self._send_deregister()
            except:
                pass
            self._socket.close()

        if self._context:
            self._context.term()

        logger.info(f"Worker {self.worker_id} disconnected")

    def run(self, num_iterations: Optional[int] = None) -> None:
        """
        Run the worker loop.

        Args:
            num_iterations: Number of outer iterations to run (None = infinite)
        """
        self._running = True
        self._stats["start_time"] = time.time()
        iteration = 0

        logger.info(f"Worker {self.worker_id} starting main loop")

        while self._running:
            if num_iterations is not None and iteration >= num_iterations:
                break

            try:
                # 1. Checkout outer params
                version, outer_params = self._checkout()
                if outer_params is None:
                    time.sleep(self.config.retry_delay)
                    continue

                self._current_version = version
                self._current_outer_params = outer_params

                # 2. Run inner optimization
                trajectory = self._run_inner_loop(outer_params)

                # 3. Check significance and commit
                if self._should_commit(trajectory):
                    self._commit(trajectory)

                iteration += 1
                self._stats["inner_loops_completed"] += 1

            except KeyboardInterrupt:
                logger.info("Worker interrupted")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(self.config.retry_delay)

        self._running = False

    def stop(self) -> None:
        """Stop the worker loop."""
        self._running = False

    def _checkout(self) -> tuple:
        """
        Checkout outer params from coordinator.

        Returns:
            Tuple of (version, outer_params) or (None, None) on failure
        """
        request = CheckoutRequest(
            worker_id=self.worker_id,
            blocking=True
        )

        for attempt in range(self.config.max_retries):
            try:
                self._socket.send(request.serialize())
                response_data = self._socket.recv()
                response = CheckoutResponse.deserialize(response_data)

                if response.success:
                    logger.debug(
                        f"Checked out version {response.version}"
                    )
                    return response.version, response.outer_params
                else:
                    logger.warning(f"Checkout failed: {response.error}")
                    return None, None

            except zmq.Again:
                logger.warning(f"Checkout timeout (attempt {attempt + 1})")
                time.sleep(self.config.retry_delay)

        return None, None

    def _run_inner_loop(self, outer_params: Dict[str, Tensor]) -> Trajectory:
        """
        Run the inner optimization loop.

        Args:
            outer_params: Fixed outer parameters for this inner solve

        Returns:
            Trajectory recording the optimization history
        """
        # Initialize trajectory
        trajectory = Trajectory(
            version=self._current_version,
            worker_id=self.worker_id,
            outer_params={k: v.detach().clone() for k, v in outer_params.items()}
        )

        # Get initial params
        init_params = self.inner_problem.get_params()

        # Simple SGD for inner loop
        lr = outer_params.get("lr", torch.tensor(0.01))
        if isinstance(lr, Tensor):
            lr = lr.item()

        optimizer = torch.optim.SGD(
            self.inner_problem.model.parameters(),
            lr=lr
        )

        for step in range(self.config.inner_steps):
            optimizer.zero_grad()

            # Get batch
            data = next(self.data_loader) if self.data_loader else None

            # Forward and backward
            loss = self.inner_problem.objective(
                self.inner_problem.get_params(),
                outer_params,
                data
            )
            loss.backward()
            optimizer.step()

            # Record trajectory (sparse - only every N steps or first/last)
            if step == 0 or step == self.config.inner_steps - 1 or step % 10 == 0:
                trajectory.add_step(
                    step=step,
                    params=self.inner_problem.get_params(),
                    loss=loss.item()
                )

        trajectory.finalize(self.inner_problem.get_params())

        logger.debug(
            f"Inner loop completed: {self.config.inner_steps} steps, "
            f"final_loss={trajectory.final_loss:.4f}"
        )

        return trajectory

    def _should_commit(self, trajectory: Trajectory) -> bool:
        """
        Check if trajectory is significant enough to commit.

        Uses significance-triggered protocol: only communicate when
        the parameter update is significant relative to model size.

        Args:
            trajectory: Completed trajectory

        Returns:
            True if should commit, False to skip
        """
        if trajectory.final_params is None:
            return False

        # Compute parameter delta
        delta = trajectory.get_param_delta()
        if not delta:
            return True  # First trajectory, always commit

        # Compute significance: ||Δθ|| / ||θ||
        delta_norm = sum(
            torch.norm(d).item() ** 2 for d in delta.values()
        ) ** 0.5

        param_norm = sum(
            torch.norm(p).item() ** 2
            for p in trajectory.final_params.values()
        ) ** 0.5

        if param_norm < 1e-8:
            return True

        significance = delta_norm / param_norm

        should_commit = significance > self.config.significance_threshold

        if not should_commit:
            logger.debug(
                f"Skipping commit: significance {significance:.4f} < "
                f"threshold {self.config.significance_threshold}"
            )

        return should_commit

    def _commit(self, trajectory: Trajectory) -> bool:
        """
        Commit trajectory to coordinator.

        Args:
            trajectory: Trajectory to commit

        Returns:
            True if accepted, False if rejected
        """
        # Serialize trajectory data
        trajectory_data = {
            "outer_params": trajectory.outer_params,
            "final_params": trajectory.final_params,
            "final_loss": trajectory.final_loss,
            "num_steps": trajectory.num_steps,
            "start_time": trajectory.start_time,
            "end_time": trajectory.end_time,
        }

        request = CommitRequest(
            worker_id=self.worker_id,
            version=trajectory.version,
            trajectory_data=trajectory_data
        )

        for attempt in range(self.config.max_retries):
            try:
                self._socket.send(request.serialize())
                response_data = self._socket.recv()
                response = CommitResponse.deserialize(response_data)

                self._stats["commits_sent"] += 1

                if response.accepted:
                    self._stats["commits_accepted"] += 1
                    logger.debug(
                        f"Commit accepted (weight={response.staleness_weight:.2f})"
                    )

                    if response.new_version:
                        logger.info(
                            f"Outer step completed, new version: {response.new_version}"
                        )
                else:
                    self._stats["commits_rejected"] += 1
                    logger.debug("Commit rejected (too stale)")

                return response.accepted

            except zmq.Again:
                logger.warning(f"Commit timeout (attempt {attempt + 1})")
                time.sleep(self.config.retry_delay)

        return False

    def _send_register(self) -> None:
        """Register with coordinator."""
        message = pickle.dumps({
            "type": MessageType.WORKER_REGISTER,
            "worker_id": self.worker_id
        })
        self._socket.send(message)
        self._socket.recv()  # Wait for ack

    def _send_deregister(self) -> None:
        """Deregister from coordinator."""
        message = pickle.dumps({
            "type": MessageType.WORKER_DEREGISTER,
            "worker_id": self.worker_id
        })
        self._socket.send(message)
        try:
            self._socket.recv(flags=zmq.NOBLOCK)
        except:
            pass

    def get_stats(self) -> dict:
        """Get worker statistics."""
        elapsed = 0
        if self._stats["start_time"]:
            elapsed = time.time() - self._stats["start_time"]

        return {
            **self._stats,
            "elapsed_time": elapsed,
            "current_version": self._current_version,
            "throughput": (
                self._stats["inner_loops_completed"] / elapsed
                if elapsed > 0 else 0
            )
        }


def run_worker(
    inner_problem: InnerProblem,
    coordinator_addr: str = "tcp://localhost:5555",
    num_iterations: Optional[int] = None,
    **config_kwargs
) -> Worker:
    """
    Convenience function to create and run a worker.

    Args:
        inner_problem: Inner optimization problem
        coordinator_addr: Coordinator address
        num_iterations: Number of iterations (None = infinite)
        **config_kwargs: Additional WorkerConfig arguments

    Returns:
        The worker instance (after completion)
    """
    config = WorkerConfig(
        coordinator_addr=coordinator_addr,
        **config_kwargs
    )

    worker = Worker(inner_problem, config)
    worker.connect()

    try:
        worker.run(num_iterations)
    finally:
        worker.disconnect()

    return worker
