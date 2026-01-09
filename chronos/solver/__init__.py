"""Solver implementations for computing meta-gradients."""

from chronos.solver.implicit_diff import ImplicitDifferentiation
from chronos.solver.unrolled import UnrolledDifferentiation

__all__ = [
    "ImplicitDifferentiation",
    "UnrolledDifferentiation",
]
