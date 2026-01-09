"""Tests for version management."""

import pytest
import torch
import threading
import time

from chronos.core.version import BoundedVersionQueue, VersionTracker
from chronos.core.state import MetaState, Trajectory


class TestBoundedVersionQueue:
    """Tests for BoundedVersionQueue."""

    def test_creation(self):
        """Test queue creation."""
        queue = BoundedVersionQueue(max_in_flight=3, max_staleness=2)

        assert queue.max_in_flight == 3
        assert queue.max_staleness == 2

    def test_checkout(self):
        """Test basic checkout."""
        queue = BoundedVersionQueue(max_in_flight=3)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        result = queue.checkout("worker-1", state, block=False)

        assert result is not None
        version, params = result
        assert version == 0
        assert "lr" in params

    def test_checkout_limit(self):
        """Test that checkouts are limited by max_in_flight."""
        queue = BoundedVersionQueue(max_in_flight=2)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        # First two should succeed
        r1 = queue.checkout("worker-1", state, block=False)
        r2 = queue.checkout("worker-2", state, block=False)

        assert r1 is not None
        assert r2 is not None

        # Third should fail (non-blocking)
        r3 = queue.checkout("worker-3", state, block=False)
        assert r3 is None

    def test_commit_accept(self):
        """Test successful commit."""
        queue = BoundedVersionQueue(max_in_flight=3, max_staleness=2)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        queue.checkout("worker-1", state)

        trajectory = Trajectory(version=0, worker_id="worker-1")
        trajectory.finalize({"w": torch.randn(3, 3)})

        accepted, weight = queue.commit("worker-1", 0, trajectory, current_version=0)

        assert accepted is True
        assert weight == 1.0

    def test_commit_staleness_decay(self):
        """Test staleness weight decay."""
        queue = BoundedVersionQueue(max_in_flight=5, max_staleness=3)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)}, version=0)

        queue.checkout("worker-1", state)

        # Simulate version advancing
        trajectory = Trajectory(version=0, worker_id="worker-1")
        trajectory.finalize({"w": torch.randn(3, 3)})

        # Commit with staleness = 1 (current is 1, committed was 0)
        accepted, weight = queue.commit("worker-1", 0, trajectory, current_version=1)

        assert accepted is True
        assert weight < 1.0  # Should be decayed

    def test_commit_too_stale(self):
        """Test rejection of too-stale commits."""
        queue = BoundedVersionQueue(max_in_flight=5, max_staleness=1)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)}, version=0)

        queue.checkout("worker-1", state)

        trajectory = Trajectory(version=0, worker_id="worker-1")
        trajectory.finalize({"w": torch.randn(3, 3)})

        # Staleness = 3, max_staleness = 1 -> reject
        accepted, weight = queue.commit("worker-1", 0, trajectory, current_version=3)

        assert accepted is False

    def test_release(self):
        """Test releasing checkout slot."""
        queue = BoundedVersionQueue(max_in_flight=1)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        queue.checkout("worker-1", state)

        # Should fail - slot taken
        r = queue.checkout("worker-2", state, block=False)
        assert r is None

        # Release
        queue.release("worker-1")

        # Now should succeed
        r = queue.checkout("worker-2", state, block=False)
        assert r is not None

    def test_stats(self):
        """Test statistics tracking."""
        queue = BoundedVersionQueue(max_in_flight=3)
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        queue.checkout("worker-1", state)
        queue.checkout("worker-2", state)

        stats = queue.get_stats()

        assert stats["active_checkouts"] == 2
        assert "worker-1" in stats["workers"]


class TestVersionTracker:
    """Tests for VersionTracker."""

    def test_creation(self):
        """Test tracker creation."""
        params = {"lr": torch.tensor(0.01)}
        tracker = VersionTracker(initial_outer_params=params)

        assert tracker.current_version == 0
        assert "lr" in tracker.outer_params

    def test_checkout_commit_flow(self):
        """Test complete checkout/commit flow."""
        params = {"lr": torch.tensor(0.01)}
        tracker = VersionTracker(initial_outer_params=params)

        # Checkout
        result = tracker.checkout("worker-1")
        assert result is not None
        version, outer_params = result

        # Create trajectory
        trajectory = Trajectory(version=version, worker_id="worker-1")
        trajectory.finalize({"w": torch.randn(3, 3)})

        # Commit
        accepted, weight = tracker.commit("worker-1", trajectory)
        assert accepted is True

    def test_update_outer_params(self):
        """Test updating outer parameters."""
        params = {"lr": torch.tensor(0.01)}
        tracker = VersionTracker(initial_outer_params=params)

        new_params = {"lr": torch.tensor(0.02)}
        new_version = tracker.update_outer_params(new_params)

        assert new_version == 1
        assert tracker.current_version == 1
        assert torch.allclose(tracker.outer_params["lr"], torch.tensor(0.02))

    def test_get_trajectories(self):
        """Test trajectory collection."""
        params = {"lr": torch.tensor(0.01)}
        tracker = VersionTracker(initial_outer_params=params)

        # Checkout and commit
        tracker.checkout("worker-1")
        trajectory = Trajectory(version=0, worker_id="worker-1")
        trajectory.finalize({"w": torch.randn(3, 3)})
        tracker.commit("worker-1", trajectory)

        # Get trajectories
        trajectories = tracker.get_trajectories(clear=False)
        assert len(trajectories) == 1

        # Clear
        trajectories = tracker.get_trajectories(clear=True)
        assert len(trajectories) == 1

        # Should be empty now
        trajectories = tracker.get_trajectories(clear=False)
        assert len(trajectories) == 0
