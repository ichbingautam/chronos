"""Tests for continuum module (HOPE features)."""

import pytest
import torch

from chronos.continuum.memory import (
    ContinuumMemory,
    MemoryConfig,
    MemoryEntry,
)
from chronos.continuum.timescale import (
    MultiTimescaleConfig,
    MultiTimescaleOptimizer,
    TimescaleConfig,
    create_hierarchical_optimizer,
)


class TestMemoryEntry:
    """Tests for MemoryEntry."""

    def test_creation(self):
        """Test entry creation."""
        entry = MemoryEntry(
            outer_params={"lr": torch.tensor(0.01)},
            hypergradient={"lr": torch.tensor(0.001)},
            validation_loss=0.5,
            step=10,
            version=1
        )

        assert entry.step == 10
        assert entry.validation_loss == 0.5
        assert entry.weight == 1.0

    def test_get_param_vector(self):
        """Test flattening params to vector."""
        entry = MemoryEntry(
            outer_params={"a": torch.ones(3), "b": torch.ones(2)},
            hypergradient={},
            validation_loss=0.5,
            step=0,
            version=0
        )

        vec = entry.get_param_vector()

        assert vec.shape == (5,)

    def test_compress(self):
        """Test entry compression."""
        entry = MemoryEntry(
            outer_params={"lr": torch.randn(100)},
            hypergradient={"lr": torch.randn(100)},
            validation_loss=0.5,
            step=0,
            version=0
        )

        assert not entry.compressed

        entry.compress()

        assert entry.compressed
        assert entry.outer_params["lr"].dtype == torch.float16


class TestContinuumMemory:
    """Tests for ContinuumMemory."""

    def test_creation(self):
        """Test memory creation."""
        memory = ContinuumMemory()

        assert len(memory._entries) == 0

    def test_store(self):
        """Test storing entries."""
        memory = ContinuumMemory()

        memory.store(
            outer_params={"lr": torch.tensor(0.01)},
            hypergradient={"lr": torch.tensor(0.001)},
            validation_loss=0.5,
            step=0
        )

        assert len(memory._entries) == 1
        assert memory._stats["stores"] == 1

    def test_retrieve_similar(self):
        """Test retrieving similar entries."""
        memory = ContinuumMemory()

        # Store some entries
        for i in range(5):
            memory.store(
                outer_params={"lr": torch.tensor(0.01 * (i + 1))},
                hypergradient={"lr": torch.tensor(0.001)},
                validation_loss=0.5 - 0.1 * i,
                step=i
            )

        # Query
        query = {"lr": torch.tensor(0.02)}
        neighbors = memory.retrieve_similar(query, k=3)

        assert len(neighbors) == 3
        assert all(isinstance(n[0], MemoryEntry) for n in neighbors)

    def test_predict_gradient(self):
        """Test gradient prediction."""
        memory = ContinuumMemory()

        # Store with consistent gradients
        for i in range(3):
            memory.store(
                outer_params={"lr": torch.tensor(0.01)},
                hypergradient={"lr": torch.tensor(0.001)},
                validation_loss=0.5,
                step=i
            )

        predicted = memory.predict_gradient({"lr": torch.tensor(0.01)})

        assert "lr" in predicted
        # Should be close to stored gradients
        assert predicted["lr"].abs().item() < 0.01

    def test_loss_trend(self):
        """Test loss trend computation."""
        memory = ContinuumMemory()

        # Store decreasing losses
        for i in range(10):
            memory.store(
                outer_params={"lr": torch.tensor(0.01)},
                hypergradient={"lr": torch.tensor(0.001)},
                validation_loss=1.0 - 0.1 * i,
                step=i
            )

        mean_loss, slope = memory.get_loss_trend(window=10)

        assert slope < 0  # Decreasing trend


class TestMultiTimescaleOptimizer:
    """Tests for MultiTimescaleOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        params = {
            "fast_param": torch.tensor(0.1, requires_grad=True),
            "slow_param": torch.tensor(0.01, requires_grad=True),
        }

        optimizer = MultiTimescaleOptimizer(params)

        assert optimizer._global_step == 0

    def test_step_frequency(self):
        """Test that different timescales update at different frequencies."""
        params = {
            "fast": torch.tensor(1.0, requires_grad=True),
            "slow": torch.tensor(1.0, requires_grad=True),
        }

        config = MultiTimescaleConfig(
            timescales=[
                TimescaleConfig("fast", update_frequency=1, learning_rate=0.1),
                TimescaleConfig("slow", update_frequency=5, learning_rate=0.01),
            ]
        )

        param_map = {"fast": "fast", "slow": "slow"}
        optimizer = MultiTimescaleOptimizer(params, config, param_map)

        # Run 5 steps
        for _ in range(5):
            grad = {"fast": torch.tensor(1.0), "slow": torch.tensor(1.0)}
            optimizer.step(grad)

        stats = optimizer.get_stats()

        # Fast should update every step, slow every 5
        assert stats["update_counts"]["fast"] == 5
        assert stats["update_counts"]["slow"] == 1

    def test_hierarchical_optimizer(self):
        """Test convenience function."""
        params = {
            "lr": torch.tensor(0.1, requires_grad=True),
            "wd": torch.tensor(0.01, requires_grad=True),
            "arch": torch.tensor(0.001, requires_grad=True),
        }

        optimizer = create_hierarchical_optimizer(
            params,
            fast_params=["lr"],
            medium_params=["wd"],
            slow_params=["arch"]
        )

        assert "lr" in optimizer.param_timescale_map
        assert optimizer.param_timescale_map["lr"] == "fast"
        assert optimizer.param_timescale_map["arch"] == "slow"
