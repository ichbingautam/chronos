"""
Chronos: Distributed Nested Optimization Framework

A production-ready framework for High-Order Optimization (HOPE) with:
- Versioned bounded asynchrony for staleness handling
- Hybrid state management (sharded coordinator + peer-to-peer)
- Significance-triggered sparse communication
"""

__version__ = "0.1.0"

from chronos.core.problem import InnerProblem, OuterOptimizer
from chronos.core.state import Checkpoint, MetaState, Trajectory

__all__ = [
    "InnerProblem",
    "OuterOptimizer",
    "MetaState",
    "Trajectory",
    "Checkpoint",
]
