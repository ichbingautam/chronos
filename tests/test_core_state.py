"""Tests for core state management."""

import time

import pytest
import torch

from chronos.core.state import Checkpoint, MetaState, Trajectory, TrajectoryStep


class TestTrajectoryStep:
    """Tests for TrajectoryStep dataclass."""

    def test_creation(self):
        """Test basic creation."""
        params = {"w": torch.randn(5, 5)}
        step = TrajectoryStep(step=0, params=params, loss=0.5)

        assert step.step == 0
        assert step.loss == 0.5
        assert "w" in step.params

    def test_with_gradients(self):
        """Test creation with gradients."""
        params = {"w": torch.randn(5, 5)}
        grads = {"w": torch.randn(5, 5)}
        step = TrajectoryStep(step=0, params=params, grads=grads, loss=0.5)

        assert step.grads is not None
        assert "w" in step.grads


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_creation(self):
        """Test trajectory creation."""
        trajectory = Trajectory(
            version=1,
            worker_id="worker-1",
            outer_params={"lr": torch.tensor(0.01)}
        )

        assert trajectory.version == 1
        assert trajectory.worker_id == "worker-1"
        assert len(trajectory.steps) == 0

    def test_add_step(self):
        """Test adding steps to trajectory."""
        trajectory = Trajectory(version=1, worker_id="test", outer_params={})

        trajectory.add_step(
            step=0,
            params={"w": torch.randn(3, 3)},
            loss=1.0
        )

        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].step == 0
        assert trajectory.steps[0].loss == 1.0

    def test_finalize(self):
        """Test trajectory finalization."""
        trajectory = Trajectory(version=1, worker_id="test", outer_params={})
        trajectory.add_step(step=0, params={"w": torch.randn(3, 3)}, loss=0.5)

        final_params = {"w": torch.randn(3, 3)}
        trajectory.finalize(final_params)

        assert trajectory.final_params is not None
        assert trajectory.final_loss == 0.5
        assert trajectory.end_time > trajectory.start_time

    def test_num_steps(self):
        """Test step counting."""
        trajectory = Trajectory(version=1, worker_id="test", outer_params={})

        for i in range(5):
            trajectory.add_step(i, {"w": torch.randn(2, 2)}, None, 0.1)

        assert trajectory.num_steps == 5

    def test_get_param_delta(self):
        """Test computing parameter delta."""
        trajectory = Trajectory(version=1, worker_id="test", outer_params={})

        init_w = torch.ones(3, 3)
        trajectory.add_step(0, {"w": init_w.clone()}, None, 1.0)

        # Add second step with updated params
        final_w = init_w + 0.1
        trajectory.add_step(1, {"w": final_w.clone()}, None, 0.5)
        trajectory.finalize({"w": final_w})

        delta = trajectory.get_param_delta()
        assert delta is not None
        assert "w" in delta
        assert torch.allclose(delta["w"], torch.ones(3, 3) * 0.1, atol=1e-5)


class TestMetaState:
    """Tests for MetaState class."""

    def test_creation(self):
        """Test MetaState creation."""
        outer_params = {"lr": torch.tensor(0.01)}
        state = MetaState(outer_params=outer_params)

        assert state.version == 0
        assert state.outer_step == 0
        assert "lr" in state.outer_params

    def test_add_trajectory(self):
        """Test adding trajectory to state."""
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        trajectory = Trajectory(version=0, worker_id="test", outer_params={})
        trajectory.finalize({"w": torch.randn(3, 3)})

        state.add_trajectory(trajectory)

        assert len(state.trajectories) == 1

    def test_increment_version(self):
        """Test version incrementing."""
        state = MetaState(outer_params={"lr": torch.tensor(0.01)})

        initial_version = state.version
        new_version = state.increment_version()

        assert new_version == initial_version + 1
        assert state.version == new_version


class TestCheckpoint:
    """Tests for Checkpoint class."""

    def test_creation(self):
        """Test checkpoint creation."""
        outer_params = {"lr": torch.tensor(0.01)}
        checkpoint = Checkpoint(
            version=1,
            outer_params=outer_params,
            outer_step=10
        )

        assert checkpoint.version == 1
        assert checkpoint.outer_step == 10

    def test_from_meta_state(self):
        """Test creating checkpoint from MetaState."""
        state = MetaState(
            outer_params={"lr": torch.tensor(0.01)},
            version=5
        )
        state.outer_step = 50

        checkpoint = Checkpoint.from_meta_state(state)

        assert checkpoint.version == 5
        assert checkpoint.outer_step == 50

    def test_save_load(self, tmp_path):
        """Test checkpoint serialization."""
        outer_params = {"lr": torch.tensor(0.01), "wd": torch.tensor(0.001)}
        checkpoint = Checkpoint(version=3, outer_params=outer_params, outer_step=30)

        # Save
        save_path = tmp_path / "checkpoint.pt"
        checkpoint.save(save_path)

        assert save_path.exists()

        # Load
        loaded = Checkpoint.load(save_path)

        assert loaded.version == 3
        assert loaded.outer_step == 30
        assert torch.allclose(loaded.outer_params["lr"], outer_params["lr"])
