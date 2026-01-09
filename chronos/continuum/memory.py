"""
Continuum Memory Systems (CMS) for HOPE.

Inspired by Google's Nested Learning paradigm, CMS maintains a
memory of optimization trajectories across different timescales,
enabling learning from historical optimization paths.

Key concepts:
- Memory entries store (state, gradient, outcome) tuples
- Entries are weighted by recency and relevance
- Memory informs future optimization decisions
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import math

import torch
from torch import Tensor

from chronos.utils.logging import get_logger

logger = get_logger("continuum.memory")


@dataclass
class MemoryConfig:
    """Configuration for Continuum Memory."""

    # Memory size limits
    max_entries: int = 1000
    max_memory_mb: float = 512.0  # Max memory usage in MB

    # Retention settings
    decay_rate: float = 0.99  # Exponential decay for entry weights
    relevance_threshold: float = 0.01  # Minimum relevance to keep

    # Similarity settings
    similarity_metric: str = "cosine"  # cosine, euclidean
    k_nearest: int = 10  # Number of neighbors for retrieval

    # Compression
    compress_old_entries: bool = True
    compression_age_threshold: int = 100  # Steps before compression


@dataclass
class MemoryEntry:
    """A single entry in continuum memory."""

    # Core data
    outer_params: Dict[str, Tensor]  # λ at this point
    hypergradient: Dict[str, Tensor]  # Computed meta-gradient
    validation_loss: float  # Outcome metric

    # Metadata
    step: int  # Outer optimization step
    version: int  # Version when recorded
    timestamp: float = field(default_factory=time.time)

    # Computed fields
    weight: float = 1.0  # Current importance weight
    compressed: bool = False  # Whether entry is compressed

    def get_param_vector(self) -> Tensor:
        """Flatten outer params to single vector for similarity."""
        return torch.cat([p.flatten() for p in self.outer_params.values()])

    def get_gradient_vector(self) -> Tensor:
        """Flatten hypergradient to single vector."""
        return torch.cat([g.flatten() for g in self.hypergradient.values()])

    def compress(self) -> None:
        """Compress entry to save memory (keep only essential info)."""
        if self.compressed:
            return

        # Quantize to fp16
        self.outer_params = {
            k: v.half() for k, v in self.outer_params.items()
        }
        self.hypergradient = {
            k: v.half() for k, v in self.hypergradient.items()
        }
        self.compressed = True


class ContinuumMemory:
    """
    Continuum Memory System for meta-learning from optimization history.

    Stores and retrieves historical optimization states to:
    1. Warm-start from similar past configurations
    2. Predict promising update directions
    3. Avoid repeating failed optimization paths

    Usage:
        memory = ContinuumMemory(config)

        # After each outer step
        memory.store(outer_params, hypergradient, val_loss, step)

        # When starting optimization from new point
        neighbors = memory.retrieve_similar(current_params, k=5)
        predicted_grad = memory.predict_gradient(current_params)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Storage
        self._entries: deque = deque(maxlen=self.config.max_entries)
        self._current_step = 0

        # Index for fast retrieval (param vectors)
        self._param_index: Optional[Tensor] = None
        self._index_dirty = True

        # Stats
        self._stats = {
            "stores": 0,
            "retrievals": 0,
            "predictions": 0,
            "compressions": 0,
            "evictions": 0,
        }

    def store(
        self,
        outer_params: Dict[str, Tensor],
        hypergradient: Dict[str, Tensor],
        validation_loss: float,
        step: int,
        version: int = 0
    ) -> None:
        """
        Store a new memory entry.

        Args:
            outer_params: Outer parameters at this step
            hypergradient: Computed hypergradient
            validation_loss: Validation loss achieved
            step: Current outer step
            version: Current version
        """
        entry = MemoryEntry(
            outer_params={k: v.detach().clone() for k, v in outer_params.items()},
            hypergradient={k: v.detach().clone() for k, v in hypergradient.items()},
            validation_loss=validation_loss,
            step=step,
            version=version
        )

        self._entries.append(entry)
        self._current_step = step
        self._index_dirty = True
        self._stats["stores"] += 1

        # Decay old entries and compress if needed
        self._decay_weights()
        self._maybe_compress()

        logger.debug(f"Stored memory entry at step {step}, loss={validation_loss:.4f}")

    def retrieve_similar(
        self,
        query_params: Dict[str, Tensor],
        k: Optional[int] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve k most similar historical states.

        Args:
            query_params: Current outer parameters
            k: Number of neighbors (default from config)

        Returns:
            List of (entry, similarity_score) tuples
        """
        if not self._entries:
            return []

        k = k or self.config.k_nearest
        self._stats["retrievals"] += 1

        # Build query vector
        query_vec = torch.cat([p.flatten() for p in query_params.values()])

        # Compute similarities
        similarities = []
        for entry in self._entries:
            entry_vec = entry.get_param_vector().float()  # Convert from fp16 if compressed

            if self.config.similarity_metric == "cosine":
                sim = torch.nn.functional.cosine_similarity(
                    query_vec.unsqueeze(0),
                    entry_vec.unsqueeze(0)
                ).item()
            else:  # euclidean (convert to similarity)
                dist = torch.norm(query_vec - entry_vec).item()
                sim = 1.0 / (1.0 + dist)

            # Weight by entry importance
            weighted_sim = sim * entry.weight
            similarities.append((entry, weighted_sim))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def predict_gradient(
        self,
        query_params: Dict[str, Tensor],
        k: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Predict hypergradient based on similar historical states.

        Uses weighted average of gradients from k-nearest neighbors.

        Args:
            query_params: Current outer parameters
            k: Number of neighbors to use

        Returns:
            Predicted hypergradient dictionary
        """
        neighbors = self.retrieve_similar(query_params, k)

        if not neighbors:
            return {k: torch.zeros_like(v) for k, v in query_params.items()}

        self._stats["predictions"] += 1

        # Weighted average of gradients
        total_weight = sum(sim for _, sim in neighbors)

        if total_weight < 1e-8:
            return {k: torch.zeros_like(v) for k, v in query_params.items()}

        predicted = {}
        for name in query_params.keys():
            weighted_sum = torch.zeros_like(query_params[name])

            for entry, sim in neighbors:
                if name in entry.hypergradient:
                    grad = entry.hypergradient[name].float()  # Decompress if needed
                    weighted_sum += sim * grad

            predicted[name] = weighted_sum / total_weight

        return predicted

    def get_loss_trend(self, window: int = 10) -> Tuple[float, float]:
        """
        Get recent loss trend (mean and slope).

        Returns:
            Tuple of (mean_loss, slope)
        """
        if len(self._entries) < 2:
            return 0.0, 0.0

        recent = list(self._entries)[-window:]
        losses = [e.validation_loss for e in recent]

        mean_loss = sum(losses) / len(losses)

        # Simple linear regression for slope
        n = len(losses)
        x_mean = (n - 1) / 2
        y_mean = mean_loss

        numerator = sum((i - x_mean) * (losses[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 1e-8 else 0.0

        return mean_loss, slope

    def _decay_weights(self) -> None:
        """Apply exponential decay to entry weights."""
        for entry in self._entries:
            age = self._current_step - entry.step
            entry.weight = self.config.decay_rate ** age

    def _maybe_compress(self) -> None:
        """Compress old entries to save memory."""
        if not self.config.compress_old_entries:
            return

        for entry in self._entries:
            age = self._current_step - entry.step
            if age > self.config.compression_age_threshold and not entry.compressed:
                entry.compress()
                self._stats["compressions"] += 1

    def _evict_low_relevance(self) -> None:
        """Remove entries below relevance threshold."""
        initial_count = len(self._entries)

        self._entries = deque(
            [e for e in self._entries if e.weight >= self.config.relevance_threshold],
            maxlen=self.config.max_entries
        )

        evicted = initial_count - len(self._entries)
        if evicted > 0:
            self._stats["evictions"] += evicted
            self._index_dirty = True

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            **self._stats,
            "num_entries": len(self._entries),
            "compressed_entries": sum(1 for e in self._entries if e.compressed),
            "avg_weight": (
                sum(e.weight for e in self._entries) / len(self._entries)
                if self._entries else 0
            ),
        }

    def clear(self) -> None:
        """Clear all memory entries."""
        self._entries.clear()
        self._param_index = None
        self._index_dirty = True
