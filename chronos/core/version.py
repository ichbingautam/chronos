"""
Version tracking for bounded staleness in distributed nested optimization.

Implements the versioned, bounded asynchrony protocol that prevents
staleness cascade while allowing parallel inner-loop execution.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import torch
from torch import Tensor

from chronos.core.state import Checkpoint, MetaState, Trajectory


@dataclass
class CheckoutRecord:
    """Records a version checkout by a worker."""

    worker_id: str
    version: int
    checkout_time: float
    outer_params: Dict[str, Tensor]


class BoundedVersionQueue:
    """
    Manages version checkouts with bounded staleness.

    This is the key mechanism for preventing staleness cascade:
    - Workers "checkout" a version of outer params before inner solve
    - At most `max_in_flight` versions can be active simultaneously
    - New checkouts block if too many versions are in flight
    - Stale commits (version too old) are rejected or down-weighted

    Attributes:
        max_in_flight: Maximum number of concurrent versions allowed
        max_staleness: Maximum version difference allowed for commits
    """

    def __init__(
        self,
        max_in_flight: int = 3,
        max_staleness: int = 2,
        timeout: float = 30.0
    ):
        self.max_in_flight = max_in_flight
        self.max_staleness = max_staleness
        self.timeout = timeout

        self._lock = threading.RLock()
        self._checkout_ready = threading.Condition(self._lock)

        # Track active checkouts: version -> set of worker_ids
        self._active_checkouts: Dict[int, Set[str]] = {}

        # Track checkout details: worker_id -> CheckoutRecord
        self._worker_checkouts: Dict[str, CheckoutRecord] = {}

        # Version history for staleness checking
        self._version_history: deque[Checkpoint] = deque(maxlen=max_staleness + 1)

    @property
    def num_in_flight(self) -> int:
        """Number of versions currently in flight."""
        with self._lock:
            return len(self._active_checkouts)

    def can_checkout(self) -> bool:
        """Check if a new checkout is allowed without blocking."""
        with self._lock:
            return self.num_in_flight < self.max_in_flight

    def checkout(
        self,
        worker_id: str,
        meta_state: MetaState,
        block: bool = True
    ) -> Optional[Tuple[int, Dict[str, Tensor]]]:
        """
        Checkout current version of outer params for a worker.

        Args:
            worker_id: Unique identifier for the worker
            meta_state: Current meta state to checkout from
            block: If True, wait for checkout slot; if False, return None if unavailable

        Returns:
            Tuple of (version, outer_params) or None if non-blocking and unavailable
        """
        with self._checkout_ready:
            # Wait for available slot
            start_time = time.time()
            while self.num_in_flight >= self.max_in_flight:
                if not block:
                    return None

                remaining = self.timeout - (time.time() - start_time)
                if remaining <= 0:
                    raise TimeoutError(
                        f"Checkout timeout: {self.num_in_flight} versions in flight"
                    )

                self._checkout_ready.wait(timeout=remaining)

            # Release any previous checkout by this worker
            self._release_worker_checkout(worker_id)

            version = meta_state.version
            outer_params = meta_state.get_outer_params_copy()

            # Record the checkout
            record = CheckoutRecord(
                worker_id=worker_id,
                version=version,
                checkout_time=time.time(),
                outer_params=outer_params
            )

            if version not in self._active_checkouts:
                self._active_checkouts[version] = set()
            self._active_checkouts[version].add(worker_id)
            self._worker_checkouts[worker_id] = record

            return version, outer_params

    def commit(
        self,
        worker_id: str,
        version: int,
        trajectory: Trajectory,
        current_version: int
    ) -> Tuple[bool, float]:
        """
        Commit a completed trajectory from a worker.

        Args:
            worker_id: Worker that completed the trajectory
            version: Version the trajectory was computed against
            trajectory: The completed trajectory
            current_version: Current version of meta state

        Returns:
            Tuple of (accepted, staleness_weight) where:
            - accepted: Whether the commit was accepted
            - staleness_weight: Weight to apply (1.0 = fresh, <1.0 = stale)
        """
        with self._checkout_ready:
            staleness = current_version - version

            # Reject if too stale
            if staleness > self.max_staleness:
                self._release_worker_checkout(worker_id)
                self._checkout_ready.notify_all()
                return False, 0.0

            # Compute staleness weight (exponential decay)
            if staleness == 0:
                weight = 1.0
            else:
                weight = 0.5 ** staleness  # Halve weight for each version of staleness

            # Release the checkout
            self._release_worker_checkout(worker_id)
            self._checkout_ready.notify_all()

            return True, weight

    def release(self, worker_id: str) -> None:
        """Release a worker's checkout without committing."""
        with self._checkout_ready:
            self._release_worker_checkout(worker_id)
            self._checkout_ready.notify_all()

    def _release_worker_checkout(self, worker_id: str) -> None:
        """Internal: release a worker's checkout."""
        if worker_id in self._worker_checkouts:
            record = self._worker_checkouts.pop(worker_id)
            version = record.version

            if version in self._active_checkouts:
                self._active_checkouts[version].discard(worker_id)
                if not self._active_checkouts[version]:
                    del self._active_checkouts[version]

    def record_version(self, checkpoint: Checkpoint) -> None:
        """Record a new version in history."""
        with self._lock:
            self._version_history.append(checkpoint)

    def get_active_workers(self) -> Dict[str, int]:
        """Get mapping of worker_id -> checked out version."""
        with self._lock:
            return {
                worker_id: record.version
                for worker_id, record in self._worker_checkouts.items()
            }

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about version queue state."""
        with self._lock:
            versions_in_flight = list(self._active_checkouts.keys())
            return {
                "num_in_flight": len(versions_in_flight),
                "max_in_flight": self.max_in_flight,
                "versions": versions_in_flight,
                "num_workers": len(self._worker_checkouts),
                "history_size": len(self._version_history)
            }


class VersionTracker:
    """
    High-level version tracking for distributed optimization.

    Combines MetaState with BoundedVersionQueue to provide a
    complete versioned state management solution.
    """

    def __init__(
        self,
        initial_outer_params: Dict[str, Tensor],
        max_in_flight: int = 3,
        max_staleness: int = 2,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize meta state
        outer_params = {
            k: v.to(self.device) if isinstance(v, Tensor) else torch.tensor(v, device=self.device)
            for k, v in initial_outer_params.items()
        }

        self.meta_state = MetaState(outer_params=outer_params, version=0)

        # Initialize version queue
        self.version_queue = BoundedVersionQueue(
            max_in_flight=max_in_flight,
            max_staleness=max_staleness
        )

        # Record initial version
        self.version_queue.record_version(
            Checkpoint.from_meta_state(self.meta_state)
        )

        self._lock = threading.Lock()

    def checkout(
        self,
        worker_id: str,
        block: bool = True
    ) -> Optional[Tuple[int, Dict[str, Tensor]]]:
        """Checkout current outer params for a worker."""
        return self.version_queue.checkout(worker_id, self.meta_state, block)

    def commit(
        self,
        worker_id: str,
        trajectory: Trajectory
    ) -> Tuple[bool, float]:
        """Commit a completed trajectory."""
        with self._lock:
            accepted, weight = self.version_queue.commit(
                worker_id=worker_id,
                version=trajectory.version,
                trajectory=trajectory,
                current_version=self.meta_state.version
            )

            if accepted:
                # Apply staleness weight to trajectory (could modify losses/grads)
                self.meta_state.add_trajectory(trajectory)

            return accepted, weight

    def update_outer_params(self, new_params: Dict[str, Tensor]) -> int:
        """Update outer params and create new version."""
        with self._lock:
            self.meta_state.update_outer_params(new_params)

            # Record new version
            self.version_queue.record_version(
                Checkpoint.from_meta_state(self.meta_state)
            )

            return self.meta_state.version

    def get_trajectories(self, clear: bool = True) -> list[Trajectory]:
        """Get collected trajectories, optionally clearing them."""
        with self._lock:
            if clear:
                return self.meta_state.clear_trajectories()
            return list(self.meta_state.trajectories)

    @property
    def current_version(self) -> int:
        """Current version number."""
        return self.meta_state.version

    @property
    def outer_params(self) -> Dict[str, Tensor]:
        """Current outer parameters."""
        return self.meta_state.get_outer_params_copy()
