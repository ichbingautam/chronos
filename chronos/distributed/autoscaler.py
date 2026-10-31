"""
Elastic Worker Auto-Scaling Manager for Chronos Distributed Execution.

Monitors DAG scheduler ready-task backlog and worker utilization in real time,
dynamically provisions worker nodes when latency exceeds SLA thresholds, and
gracefully deprovisions idle nodes after cooldown windows without disrupting in-flight tasks.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from chronos.utils.logging import get_logger

logger = get_logger("autoscaler")


class WorkerNodeState(Enum):
    """Lifecycle state of an elastic worker node."""

    PROVISIONING = auto()
    IDLE = auto()
    BUSY = auto()
    TERMINATING = auto()
    TERMINATED = auto()


@dataclass
class AutoscalerConfig:
    """Configuration for the elastic worker autoscaler."""

    min_workers: int = 1
    max_workers: int = 10
    target_latency_sec: float = 2.0
    scale_up_threshold_tasks: int = 5
    scale_down_cooldown_sec: float = 5.0
    scale_up_step: int = 2
    scale_down_step: int = 1
    worker_capacity: int = 2
    check_interval_sec: float = 0.5

    def __post_init__(self) -> None:
        if self.min_workers < 0:
            raise ValueError("min_workers cannot be negative")
        if self.max_workers < self.min_workers:
            raise ValueError("max_workers cannot be less than min_workers")
        if self.worker_capacity <= 0:
            raise ValueError("worker_capacity must be positive")
        if self.target_latency_sec <= 0:
            raise ValueError("target_latency_sec must be positive")


@dataclass
class AutoscalingMetrics:
    """Snapshot of real-time cluster utilization and backlog metrics."""

    ready_task_count: int
    in_flight_task_count: int
    active_worker_count: int
    idle_worker_count: int
    average_queue_latency_sec: float
    worker_utilization: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Represents a managed worker node within the elastic pool."""

    worker_id: str
    capacity: int = 2
    state: WorkerNodeState = WorkerNodeState.PROVISIONING
    created_at: float = field(default_factory=time.time)
    idle_since: Optional[float] = None
    assigned_tasks: Set[str] = field(default_factory=set)

    @property
    def is_available(self) -> bool:
        """Check if node can accept additional tasks."""
        return (
            self.state in (WorkerNodeState.IDLE, WorkerNodeState.BUSY)
            and len(self.assigned_tasks) < self.capacity
        )

    @property
    def in_flight_count(self) -> int:
        """Return number of tasks currently in flight."""
        return len(self.assigned_tasks)

    def assign_task(self, task_id: str) -> bool:
        """Assign a task to this worker if capacity permits."""
        if not self.is_available:
            return False
        self.assigned_tasks.add(task_id)
        self.state = WorkerNodeState.BUSY
        self.idle_since = None
        return True

    def complete_task(self, task_id: str) -> bool:
        """Complete and remove a task from this worker."""
        if task_id in self.assigned_tasks:
            self.assigned_tasks.remove(task_id)
            if len(self.assigned_tasks) == 0:
                self.idle_since = time.time()
                if self.state != WorkerNodeState.TERMINATING:
                    self.state = WorkerNodeState.IDLE
            return True
        return False


class ElasticWorkerPoolManager:
    """
    Manages elastic worker auto-scaling for Chronos distributed task execution.

    Dynamically provisions and deprovisions worker nodes according to real-time
    DAG queue depth and SLA latency thresholds.
    """

    def __init__(
        self,
        config: Optional[AutoscalerConfig] = None,
        provision_fn: Optional[Callable[[str], Any]] = None,
        terminate_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.config = config or AutoscalerConfig()
        self.provision_fn = provision_fn
        self.terminate_fn = terminate_fn

        self._workers: Dict[str, WorkerNode] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Warm up to min_workers
        with self._lock:
            for _ in range(self.config.min_workers):
                self._provision_worker_internal()

    @property
    def worker_count(self) -> int:
        """Number of active (non-terminated) workers."""
        with self._lock:
            return len([w for w in self._workers.values() if w.state != WorkerNodeState.TERMINATED])

    @property
    def idle_worker_count(self) -> int:
        """Number of currently idle workers."""
        with self._lock:
            return len([w for w in self._workers.values() if w.state == WorkerNodeState.IDLE])

    def _generate_worker_id(self) -> str:
        return f"worker-elastic-{uuid.uuid4().hex[:8]}"

    def _provision_worker_internal(self, worker_id: Optional[str] = None) -> WorkerNode:
        """Internal helper to provision a new worker node."""
        wid = worker_id or self._generate_worker_id()
        node = WorkerNode(
            worker_id=wid,
            capacity=self.config.worker_capacity,
            state=WorkerNodeState.IDLE,
            idle_since=time.time(),
        )
        self._workers[wid] = node
        logger.info(f"Provisioned elastic worker: {wid}")

        if self.provision_fn:
            try:
                self.provision_fn(wid)
            except Exception as e:
                logger.error(f"Error executing custom provision hook for {wid}: {e}")

        return node

    def register_worker(self, worker_id: Optional[str] = None) -> WorkerNode:
        """Register a worker in the pool."""
        with self._lock:
            if self.worker_count >= self.config.max_workers:
                raise RuntimeError(
                    f"Cannot register worker: pool reached max capacity ({self.config.max_workers})"
                )
            return self._provision_worker_internal(worker_id)

    def unregister_worker(self, worker_id: str, force: bool = False) -> bool:
        """
        Unregister and terminate a worker node.

        If force is False, workers with in-flight tasks enter TERMINATING state
        and are only removed when tasks complete.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False

            if not force and worker.in_flight_count > 0:
                worker.state = WorkerNodeState.TERMINATING
                logger.info(
                    f"Worker {worker_id} has {worker.in_flight_count} in-flight tasks, marked TERMINATING"
                )
                return False

            worker.state = WorkerNodeState.TERMINATED
            del self._workers[worker_id]
            logger.info(f"Unregistered and terminated worker: {worker_id}")

            if self.terminate_fn:
                try:
                    self.terminate_fn(worker_id)
                except Exception as e:
                    logger.error(f"Error executing custom terminate hook for {worker_id}: {e}")

            return True

    def get_metrics(
        self, ready_task_count: int = 0, average_queue_latency_sec: float = 0.0
    ) -> AutoscalingMetrics:
        """Compute real-time cluster utilization and backlog metrics."""
        with self._lock:
            active_workers = [
                w
                for w in self._workers.values()
                if w.state in (WorkerNodeState.IDLE, WorkerNodeState.BUSY)
            ]
            idle_workers = [w for w in self._workers.values() if w.state == WorkerNodeState.IDLE]
            total_in_flight = sum(w.in_flight_count for w in active_workers)

            total_capacity = sum(w.capacity for w in active_workers)
            utilization = (total_in_flight / total_capacity) if total_capacity > 0 else 0.0

            return AutoscalingMetrics(
                ready_task_count=ready_task_count,
                in_flight_task_count=total_in_flight,
                active_worker_count=len(active_workers),
                idle_worker_count=len(idle_workers),
                average_queue_latency_sec=average_queue_latency_sec,
                worker_utilization=min(1.0, max(0.0, utilization)),
            )

    def evaluate_and_scale(
        self, ready_task_count: int, average_queue_latency_sec: float
    ) -> Dict[str, Any]:
        """
        Evaluate queue depth and latency against SLA thresholds and perform auto-scaling.

        Returns a dictionary describing the auto-scaling action taken.
        """
        with self._lock:
            metrics = self.get_metrics(ready_task_count, average_queue_latency_sec)
            action = {"scaled_up": 0, "scaled_down": 0, "metrics": metrics}

            # Check scale-up conditions:
            # 1. Ready backlog exceeds task threshold
            # 2. Average queue latency breaches SLA target
            # 3. High worker utilization with non-empty queue
            should_scale_up = (
                ready_task_count >= self.config.scale_up_threshold_tasks
                or average_queue_latency_sec > self.config.target_latency_sec
                or (ready_task_count > 0 and metrics.idle_worker_count == 0)
            )

            if should_scale_up and metrics.active_worker_count < self.config.max_workers:
                available_headroom = self.config.max_workers - metrics.active_worker_count
                to_add = min(self.config.scale_up_step, available_headroom)

                # If queue depth significantly exceeds threshold, scale up more aggressively
                if (
                    ready_task_count > self.config.scale_up_threshold_tasks * 2
                    and available_headroom > to_add
                ):
                    to_add = min(available_headroom, to_add * 2)

                provisioned = []
                for _ in range(to_add):
                    node = self._provision_worker_internal()
                    provisioned.append(node.worker_id)

                action["scaled_up"] = len(provisioned)
                action["provisioned_workers"] = provisioned
                logger.info(
                    f"Auto-scaled UP: added {len(provisioned)} workers (new total: {self.worker_count})"
                )
                return action

            # Check scale-down conditions:
            # 1. Backlog is small or zero
            # 2. Active workers exceed minimum floor
            # 3. Workers have been idle longer than cooldown window
            now = time.time()
            if ready_task_count == 0 and metrics.active_worker_count > self.config.min_workers:
                candidate_workers = [
                    w
                    for w in self._workers.values()
                    if w.state == WorkerNodeState.IDLE
                    and w.idle_since is not None
                    and (now - w.idle_since) >= self.config.scale_down_cooldown_sec
                ]

                # Sort by longest idle first
                candidate_workers.sort(key=lambda w: w.idle_since or 0, reverse=False)

                max_removable = metrics.active_worker_count - self.config.min_workers
                to_remove_count = min(self.config.scale_down_step, max_removable)
                selected_for_removal = candidate_workers[:to_remove_count]

                deprovisioned = []
                for worker in selected_for_removal:
                    if self.unregister_worker(worker.worker_id, force=False):
                        deprovisioned.append(worker.worker_id)

                action["scaled_down"] = len(deprovisioned)
                action["deprovisioned_workers"] = deprovisioned
                if deprovisioned:
                    logger.info(
                        f"Auto-scaled DOWN: removed {len(deprovisioned)} workers (new total: {self.worker_count})"
                    )

            # Cleanup any workers marked TERMINATING that have completed all tasks
            terminating_workers = [
                w
                for w in self._workers.values()
                if w.state == WorkerNodeState.TERMINATING and w.in_flight_count == 0
            ]
            for worker in terminating_workers:
                self.unregister_worker(worker.worker_id, force=True)

            return action

    def acquire_worker_for_task(self, task_id: str) -> Optional[str]:
        """
        Locate and allocate an available worker for an executable task.

        Returns worker_id if assigned, or None if no capacity exists.
        """
        with self._lock:
            # Prefer idle workers first, then busy workers with remaining capacity
            idle_workers = [w for w in self._workers.values() if w.state == WorkerNodeState.IDLE]
            if idle_workers:
                worker = idle_workers[0]
                if worker.assign_task(task_id):
                    return worker.worker_id

            available_workers = [w for w in self._workers.values() if w.is_available]
            if available_workers:
                worker = available_workers[0]
                if worker.assign_task(task_id):
                    return worker.worker_id

            return None

    def release_task(self, worker_id: str, task_id: str) -> bool:
        """Mark task execution complete on the specified worker."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False

            res = worker.complete_task(task_id)
            # If worker was pending termination and has drained in-flight tasks, terminate now
            if worker.state == WorkerNodeState.TERMINATING and worker.in_flight_count == 0:
                self.unregister_worker(worker_id, force=True)

            return res

    def get_worker(self, worker_id: str) -> Optional[WorkerNode]:
        """Retrieve worker node state by identifier."""
        with self._lock:
            return self._workers.get(worker_id)

    def get_all_workers(self) -> List[WorkerNode]:
        """Return a snapshot list of all tracked workers."""
        with self._lock:
            return list(self._workers.values())
