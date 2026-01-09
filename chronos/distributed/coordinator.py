"""
Coordinator Service for Distributed Chronos.

The coordinator is a central service that:
1. Manages the MetaState (outer parameters, version, trajectories)
2. Handles versioned checkouts with bounded staleness
3. Aggregates trajectories and triggers outer optimization steps
4. Broadcasts parameter updates to workers

Uses ZeroMQ for reliable request-reply communication.
"""

import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import zmq

from chronos.core.problem import OuterOptimizer
from chronos.core.state import MetaState, Trajectory, Checkpoint
from chronos.core.version import BoundedVersionQueue, VersionTracker
from chronos.distributed.protocols import (
    MessageType,
    CheckoutRequest,
    CheckoutResponse,
    CommitRequest,
    CommitResponse,
    ErrorResponse,
    parse_message,
)
from chronos.utils.logging import get_logger

logger = get_logger("coordinator")


@dataclass
class WorkerInfo:
    """Information about a registered worker."""

    worker_id: str
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    current_version: Optional[int] = None
    status: str = "idle"


class Coordinator:
    """
    Central coordinator for distributed nested optimization.

    Manages:
    - Version-controlled outer parameters
    - Worker checkouts and commits
    - Trajectory aggregation
    - Outer optimization steps

    Args:
        outer_params: Initial outer parameters
        outer_optimizer: Optimizer for updating outer params
        inner_problem: Reference inner problem for hypergradient computation
        port: ZeroMQ port to listen on
        max_in_flight: Maximum concurrent versions (staleness bound)
        min_trajectories: Minimum trajectories before outer step
        heartbeat_timeout: Seconds before considering worker dead
    """

    def __init__(
        self,
        outer_params: Dict[str, Any],
        outer_optimizer: Optional[OuterOptimizer] = None,
        inner_problem: Optional[Any] = None,
        port: int = 5555,
        max_in_flight: int = 3,
        max_staleness: int = 2,
        min_trajectories: int = 1,
        heartbeat_timeout: float = 60.0,
    ):
        self.port = port
        self.min_trajectories = min_trajectories
        self.heartbeat_timeout = heartbeat_timeout
        self.outer_optimizer = outer_optimizer
        self.inner_problem = inner_problem

        # Initialize version tracker
        self.version_tracker = VersionTracker(
            initial_outer_params=outer_params,
            max_in_flight=max_in_flight,
            max_staleness=max_staleness
        )

        # Worker registry
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = threading.RLock()

        # Stats
        self._stats = {
            "checkouts": 0,
            "commits": 0,
            "commits_rejected": 0,
            "outer_steps": 0,
            "start_time": time.time(),
        }

        # ZeroMQ context
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._running = False
        self._server_thread: Optional[threading.Thread] = None

    @property
    def current_version(self) -> int:
        """Current outer parameter version."""
        return self.version_tracker.current_version

    @property
    def outer_params(self) -> Dict[str, Any]:
        """Current outer parameters."""
        return self.version_tracker.outer_params

    @property
    def num_workers(self) -> int:
        """Number of registered workers."""
        with self._lock:
            return len(self._workers)

    def start(self, blocking: bool = False) -> None:
        """
        Start the coordinator service.

        Args:
            blocking: If True, run in current thread; otherwise spawn daemon thread
        """
        if self._running:
            logger.warning("Coordinator already running")
            return

        self._running = True

        if blocking:
            self._run_server()
        else:
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="chronos-coordinator"
            )
            self._server_thread.start()
            logger.info(f"Coordinator started on port {self.port}")

    def stop(self) -> None:
        """Stop the coordinator service."""
        self._running = False

        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)

        logger.info("Coordinator stopped")

    def _run_server(self) -> None:
        """Main server loop."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{self.port}")

        # Set socket timeout for graceful shutdown
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second

        logger.info(f"Coordinator listening on tcp://*:{self.port}")

        while self._running:
            try:
                # Receive request
                message = self._socket.recv()

                # Process and respond
                response = self._handle_message(message)
                self._socket.send(response)

            except zmq.Again:
                # Timeout - check for dead workers
                self._cleanup_dead_workers()
                continue
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                try:
                    error_resp = ErrorResponse(error=str(e))
                    self._socket.send(error_resp.serialize())
                except:
                    pass

    def _handle_message(self, message: bytes) -> bytes:
        """Route message to appropriate handler."""
        try:
            msg_type, payload = parse_message(message)

            if msg_type == MessageType.CHECKOUT_REQUEST:
                return self._handle_checkout(payload)
            elif msg_type == MessageType.COMMIT_REQUEST:
                return self._handle_commit(payload)
            elif msg_type == MessageType.HEARTBEAT:
                return self._handle_heartbeat(payload)
            elif msg_type == MessageType.WORKER_REGISTER:
                return self._handle_register(payload)
            elif msg_type == MessageType.WORKER_DEREGISTER:
                return self._handle_deregister(payload)
            else:
                return ErrorResponse(error=f"Unknown message type: {msg_type}").serialize()

        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return ErrorResponse(error=str(e)).serialize()

    def _handle_checkout(self, payload: dict) -> bytes:
        """Handle checkout request from worker."""
        worker_id = payload["worker_id"]
        blocking = payload.get("blocking", True)

        with self._lock:
            # Register worker if new
            if worker_id not in self._workers:
                self._workers[worker_id] = WorkerInfo(worker_id=worker_id)

            self._workers[worker_id].last_heartbeat = time.time()
            self._workers[worker_id].status = "checking_out"

        # Perform checkout
        result = self.version_tracker.checkout(worker_id, block=blocking)

        if result is None:
            return CheckoutResponse(
                success=False,
                error="No checkout slots available"
            ).serialize()

        version, outer_params = result

        with self._lock:
            self._workers[worker_id].current_version = version
            self._workers[worker_id].status = "computing"
            self._stats["checkouts"] += 1

        logger.debug(f"Worker {worker_id} checked out version {version}")

        return CheckoutResponse(
            success=True,
            version=version,
            outer_params=outer_params
        ).serialize()

    def _handle_commit(self, payload: dict) -> bytes:
        """Handle trajectory commit from worker."""
        worker_id = payload["worker_id"]
        version = payload["version"]
        trajectory_data = payload["trajectory_data"]

        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
                self._workers[worker_id].status = "idle"

        # Reconstruct trajectory from data
        trajectory = self._reconstruct_trajectory(trajectory_data, version, worker_id)

        # Commit to version tracker
        accepted, weight = self.version_tracker.commit(worker_id, trajectory)

        with self._lock:
            if accepted:
                self._stats["commits"] += 1
                logger.debug(
                    f"Worker {worker_id} committed version {version} "
                    f"(weight={weight:.2f})"
                )
            else:
                self._stats["commits_rejected"] += 1
                logger.debug(
                    f"Worker {worker_id} commit rejected - version {version} too stale"
                )

        # Check if we should perform outer step
        new_version = None
        if accepted:
            new_version = self._maybe_outer_step()

        return CommitResponse(
            accepted=accepted,
            staleness_weight=weight,
            new_version=new_version
        ).serialize()

    def _handle_heartbeat(self, payload: dict) -> bytes:
        """Handle heartbeat from worker."""
        worker_id = payload["worker_id"]

        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
                self._workers[worker_id].status = payload.get("status", "idle")

        # Return current version
        return pickle.dumps({
            "type": "heartbeat_ack",
            "current_version": self.current_version
        })

    def _handle_register(self, payload: dict) -> bytes:
        """Handle worker registration."""
        worker_id = payload["worker_id"]

        with self._lock:
            self._workers[worker_id] = WorkerInfo(worker_id=worker_id)

        logger.info(f"Worker {worker_id} registered")

        return pickle.dumps({
            "type": "register_ack",
            "success": True,
            "current_version": self.current_version
        })

    def _handle_deregister(self, payload: dict) -> bytes:
        """Handle worker deregistration."""
        worker_id = payload["worker_id"]

        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                # Release any held checkout
                self.version_tracker.version_queue.release(worker_id)

        logger.info(f"Worker {worker_id} deregistered")

        return pickle.dumps({
            "type": "deregister_ack",
            "success": True
        })

    def _reconstruct_trajectory(
        self,
        data: dict,
        version: int,
        worker_id: str
    ) -> Trajectory:
        """Reconstruct Trajectory object from serialized data."""
        import torch

        trajectory = Trajectory(
            version=version,
            worker_id=worker_id,
            outer_params=data.get("outer_params", {}),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time", time.time())
        )

        # Add final params and loss
        if "final_params" in data:
            trajectory.final_params = data["final_params"]

        if "final_loss" in data:
            # Create minimal trajectory step for loss tracking
            from chronos.core.state import TrajectoryStep
            trajectory.steps.append(TrajectoryStep(
                step=data.get("num_steps", 0) - 1,
                params=data.get("final_params", {}),
                loss=data["final_loss"]
            ))

        return trajectory

    def _maybe_outer_step(self) -> Optional[int]:
        """
        Check if conditions are met for outer optimization step.

        Returns:
            New version if step was taken, None otherwise
        """
        trajectories = self.version_tracker.get_trajectories(clear=False)

        if len(trajectories) < self.min_trajectories:
            return None

        if self.outer_optimizer is None:
            # No optimizer configured - just clear trajectories
            self.version_tracker.get_trajectories(clear=True)
            return None

        # Compute hypergradient and update
        try:
            hypergradient = self.outer_optimizer.compute_hypergradient(
                trajectories=trajectories,
                inner_problem=self.inner_problem
            )

            new_params = self.outer_optimizer.step(hypergradient)
            new_version = self.version_tracker.update_outer_params(new_params)

            # Clear processed trajectories
            self.version_tracker.get_trajectories(clear=True)

            with self._lock:
                self._stats["outer_steps"] += 1

            logger.info(f"Outer step completed, new version: {new_version}")
            return new_version

        except Exception as e:
            logger.error(f"Error in outer step: {e}")
            return None

    def _cleanup_dead_workers(self) -> None:
        """Remove workers that haven't sent heartbeat."""
        now = time.time()
        dead_workers = []

        with self._lock:
            for worker_id, info in self._workers.items():
                if now - info.last_heartbeat > self.heartbeat_timeout:
                    dead_workers.append(worker_id)

            for worker_id in dead_workers:
                del self._workers[worker_id]
                self.version_tracker.version_queue.release(worker_id)
                logger.warning(f"Worker {worker_id} timed out, removed")

    def get_stats(self) -> dict:
        """Get coordinator statistics."""
        with self._lock:
            return {
                **self._stats,
                "uptime": time.time() - self._stats["start_time"],
                "current_version": self.current_version,
                "num_workers": len(self._workers),
                "pending_trajectories": len(
                    self.version_tracker.get_trajectories(clear=False)
                ),
                "version_queue": self.version_tracker.version_queue.get_stats(),
            }

    # --- Synchronous API for local testing ---

    def checkout_sync(self, worker_id: str) -> tuple:
        """Synchronous checkout for local testing."""
        return self.version_tracker.checkout(worker_id)

    def commit_sync(self, worker_id: str, trajectory: Trajectory) -> tuple:
        """Synchronous commit for local testing."""
        return self.version_tracker.commit(worker_id, trajectory)

    def step_outer_sync(self) -> Optional[int]:
        """Synchronous outer step for local testing."""
        return self._maybe_outer_step()
