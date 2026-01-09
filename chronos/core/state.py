"""
State management for nested optimization.

Defines MetaState, Trajectory, and Checkpoint classes that track the
optimization state, history, and enable versioned updates.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class TrajectoryStep:
    """Single step in an inner optimization trajectory."""

    step: int
    params: Dict[str, Tensor]
    grads: Optional[Dict[str, Tensor]] = None
    loss: float = 0.0

    def to_device(self, device: torch.device) -> "TrajectoryStep":
        """Move tensors to specified device."""
        return TrajectoryStep(
            step=self.step,
            params={k: v.to(device) for k, v in self.params.items()},
            grads={k: v.to(device) for k, v in self.grads.items()} if self.grads else None,
            loss=self.loss
        )

    def detach(self) -> "TrajectoryStep":
        """Detach all tensors from computation graph."""
        return TrajectoryStep(
            step=self.step,
            params={k: v.detach().clone() for k, v in self.params.items()},
            grads={k: v.detach().clone() for k, v in self.grads.items()} if self.grads else None,
            loss=self.loss
        )


@dataclass
class Trajectory:
    """
    Records the history of an inner optimization run.

    This is essential for computing meta-gradients, as we need to
    differentiate through the optimization process.

    Attributes:
        version: The outer parameter version this trajectory was computed against
        worker_id: Identifier of the worker that produced this trajectory
        outer_params: The fixed outer parameters used during this inner solve
        steps: List of TrajectoryStep recording params/grads/loss at each step
        final_params: The final model parameters after inner optimization
        start_time: When the inner solve started
        end_time: When it completed
    """

    version: int
    worker_id: str
    outer_params: Dict[str, Tensor]
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_params: Optional[Dict[str, Tensor]] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_step(
        self,
        step: int,
        params: Dict[str, Tensor],
        grads: Optional[Dict[str, Tensor]] = None,
        loss: float = 0.0,
        detach: bool = True
    ) -> None:
        """
        Record a step in the trajectory.

        Args:
            step: Step number
            params: Model parameters at this step
            grads: Gradients at this step (optional)
            loss: Loss value at this step
            detach: Whether to detach tensors from computation graph
        """
        traj_step = TrajectoryStep(
            step=step,
            params={k: v.detach().clone() if detach else v.clone() for k, v in params.items()},
            grads={k: v.detach().clone() if detach else v.clone() for k, v in grads.items()} if grads else None,
            loss=loss
        )
        self.steps.append(traj_step)

    def finalize(self, final_params: Dict[str, Tensor]) -> None:
        """Mark trajectory as complete with final parameters."""
        self.final_params = {k: v.detach().clone() for k, v in final_params.items()}
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Time taken for this inner solve."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def num_steps(self) -> int:
        """Number of recorded steps."""
        return len(self.steps)

    @property
    def final_loss(self) -> float:
        """Loss at the final step."""
        if not self.steps:
            return float("nan")
        return self.steps[-1].loss

    def get_param_delta(self) -> Dict[str, Tensor]:
        """Compute parameter change from start to end."""
        if len(self.steps) < 2:
            return {}

        first = self.steps[0].params
        last = self.steps[-1].params

        return {k: last[k] - first[k] for k in first.keys()}

    def to_device(self, device: torch.device) -> "Trajectory":
        """Move all tensors to specified device."""
        return Trajectory(
            version=self.version,
            worker_id=self.worker_id,
            outer_params={k: v.to(device) for k, v in self.outer_params.items()},
            steps=[s.to_device(device) for s in self.steps],
            final_params={k: v.to(device) for k, v in self.final_params.items()} if self.final_params else None,
            start_time=self.start_time,
            end_time=self.end_time
        )


@dataclass
class MetaState:
    """
    Complete state of the outer optimization.

    This tracks the current outer parameters, their version, and
    collected trajectories from workers. Used by the coordinator
    to manage distributed optimization.

    Attributes:
        outer_params: Current outer/meta parameters (λ)
        version: Monotonically increasing version counter
        trajectories: Collected inner-loop trajectories awaiting aggregation
        outer_step: Number of outer optimization steps completed
        history: Record of (version, outer_loss) for tracking convergence
    """

    outer_params: Dict[str, Tensor]
    version: int = 0
    trajectories: List[Trajectory] = field(default_factory=list)
    outer_step: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def increment_version(self) -> int:
        """Increment and return new version number."""
        self.version += 1
        return self.version

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a completed trajectory to the collection."""
        self.trajectories.append(trajectory)

    def clear_trajectories(self) -> List[Trajectory]:
        """Clear and return collected trajectories."""
        trajs = self.trajectories
        self.trajectories = []
        return trajs

    def record_outer_step(self, outer_loss: float, hypergradient_norm: float = 0.0) -> None:
        """Record an outer optimization step in history."""
        self.outer_step += 1
        self.history.append({
            "step": self.outer_step,
            "version": self.version,
            "outer_loss": outer_loss,
            "hypergradient_norm": hypergradient_norm,
            "timestamp": time.time(),
            "num_trajectories": len(self.trajectories)
        })

    def get_outer_params_copy(self) -> Dict[str, Tensor]:
        """Get a detached copy of outer parameters."""
        return {k: v.detach().clone() for k, v in self.outer_params.items()}

    def update_outer_params(self, new_params: Dict[str, Tensor]) -> None:
        """Update outer parameters and increment version."""
        self.outer_params = {k: v.clone() for k, v in new_params.items()}
        self.increment_version()


@dataclass
class Checkpoint:
    """
    Immutable snapshot of MetaState for versioning and recovery.

    Used by the version tracker to maintain historical states and
    enable bounded staleness in distributed settings.
    """

    version: int
    outer_params: Dict[str, Tensor]
    outer_step: int
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_meta_state(cls, meta_state: MetaState) -> "Checkpoint":
        """Create checkpoint from current MetaState."""
        return cls(
            version=meta_state.version,
            outer_params={k: v.detach().clone() for k, v in meta_state.outer_params.items()},
            outer_step=meta_state.outer_step,
            timestamp=time.time()
        )

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to CPU for serialization
        data = {
            "version": self.version,
            "outer_params": {k: v.cpu() for k, v in self.outer_params.items()},
            "outer_step": self.outer_step,
            "timestamp": self.timestamp
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path, device: torch.device = None) -> "Checkpoint":
        """Load checkpoint from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        outer_params = data["outer_params"]
        if device:
            outer_params = {k: v.to(device) for k, v in outer_params.items()}

        return cls(
            version=data["version"],
            outer_params=outer_params,
            outer_step=data["outer_step"],
            timestamp=data["timestamp"]
        )
