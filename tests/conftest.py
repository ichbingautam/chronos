"""Test utilities and fixtures for Chronos."""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def device():
    """Get test device (CPU for CI, GPU if available locally)."""
    return torch.device("cpu")


@pytest.fixture
def simple_model(device):
    """Simple 2-layer model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    return model


@pytest.fixture
def outer_params(device):
    """Sample outer parameters for testing."""
    return {
        "lr": torch.tensor(0.01, device=device, requires_grad=True),
        "weight_decay": torch.tensor(0.001, device=device, requires_grad=True),
    }


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    from chronos.core.state import Trajectory, TrajectoryStep

    trajectory = Trajectory(
        version=1,
        worker_id="test-worker",
        outer_params={"lr": torch.tensor(0.01)}
    )

    # Add some steps
    for i in range(5):
        trajectory.add_step(
            step=i,
            params={"weight": torch.randn(10, 20)},
            grads={"weight": torch.randn(10, 20)},
            loss=1.0 / (i + 1)
        )

    trajectory.finalize({"weight": torch.randn(10, 20)})
    return trajectory
