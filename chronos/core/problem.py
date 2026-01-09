"""
Core abstractions for bilevel optimization problems.

Defines the InnerProblem and OuterOptimizer interfaces that form the foundation
of the nested optimization framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class InnerProblem(ABC):
    """
    Abstract base class for inner optimization problems.

    The inner problem defines the lower-level optimization that is solved
    for a given set of outer parameters (hyperparameters). Examples include:
    - Training a model with fixed learning rate (hyperparameter tuning)
    - Adapting to a specific task (meta-learning/MAML)
    - Training with fixed sample weights (data reweighting)

    Attributes:
        model: The neural network being optimized
        device: Device to run computations on
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @abstractmethod
    def objective(
        self,
        params: Dict[str, Tensor],
        outer_params: Dict[str, Tensor],
        data: Any
    ) -> Tensor:
        """
        Compute the inner objective (loss) for given parameters.

        Args:
            params: Current model parameters (θ)
            outer_params: Outer/meta parameters (λ) - e.g., learning rate, regularization
            data: Batch of training data

        Returns:
            Scalar loss tensor
        """
        pass

    @abstractmethod
    def solve(
        self,
        outer_params: Dict[str, Tensor],
        init_params: Optional[Dict[str, Tensor]] = None,
        num_steps: int = 100,
        data_loader: Any = None
    ) -> Tuple[Dict[str, Tensor], "Trajectory"]:
        """
        Solve the inner optimization problem.

        Args:
            outer_params: Fixed outer parameters for this inner solve
            init_params: Initial model parameters (uses current if None)
            num_steps: Number of inner optimization steps
            data_loader: Iterator providing training batches

        Returns:
            Tuple of (final_params, trajectory) where trajectory contains
            the optimization history needed for meta-gradient computation
        """
        pass

    def get_params(self) -> Dict[str, Tensor]:
        """Get current model parameters as a dictionary."""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def set_params(self, params: Dict[str, Tensor]) -> None:
        """Set model parameters from a dictionary."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(params[name])


class OuterOptimizer(ABC):
    """
    Abstract base class for outer (meta) optimization.

    The outer optimizer updates the hyperparameters/meta-parameters based on
    the trajectories collected from solving inner problems. It computes
    hypergradients (gradients of outer objective w.r.t. outer parameters)
    and applies updates.

    Attributes:
        outer_params: Current outer parameters (λ)
        lr: Learning rate for outer updates
    """

    def __init__(
        self,
        outer_params: Dict[str, Tensor],
        lr: float = 0.01,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.outer_params = {
            k: v.to(self.device) if isinstance(v, Tensor) else torch.tensor(v, device=self.device)
            for k, v in outer_params.items()
        }
        self.lr = lr

        # Enable gradients for outer params
        for param in self.outer_params.values():
            param.requires_grad_(True)

    @abstractmethod
    def compute_hypergradient(
        self,
        trajectories: list["Trajectory"],
        inner_problem: InnerProblem,
        validation_data: Any = None
    ) -> Dict[str, Tensor]:
        """
        Compute the hypergradient (gradient of outer objective w.r.t. outer params).

        This is the key computation in bilevel optimization. Different algorithms
        use different approaches:
        - Implicit differentiation: Uses implicit function theorem
        - Unrolled differentiation: Backprop through inner optimization
        - Neumann approximation: Truncated Neumann series

        Args:
            trajectories: List of optimization trajectories from inner solves
            inner_problem: The inner problem instance
            validation_data: Data for computing outer objective (if different from training)

        Returns:
            Dictionary mapping outer param names to their gradients
        """
        pass

    def step(self, hypergradient: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Update outer parameters using computed hypergradient.

        Args:
            hypergradient: Gradients for each outer parameter

        Returns:
            Updated outer parameters
        """
        with torch.no_grad():
            for name, param in self.outer_params.items():
                if name in hypergradient:
                    param.sub_(self.lr * hypergradient[name])

        return self.get_outer_params()

    def get_outer_params(self) -> Dict[str, Tensor]:
        """Get current outer parameters."""
        return {k: v.clone() for k, v in self.outer_params.items()}


@dataclass
class HyperparameterSpec:
    """Specification for a single hyperparameter."""

    name: str
    initial_value: float
    min_value: float = float("-inf")
    max_value: float = float("inf")
    log_scale: bool = False  # If True, optimize in log space

    def to_tensor(self, device: torch.device = None) -> Tensor:
        """Convert to tensor, optionally in log space."""
        value = self.initial_value
        if self.log_scale:
            value = torch.log(torch.tensor(value))
        else:
            value = torch.tensor(value)

        if device:
            value = value.to(device)
        return value

    def from_tensor(self, tensor: Tensor) -> float:
        """Convert from tensor back to value."""
        value = tensor.item()
        if self.log_scale:
            value = torch.exp(torch.tensor(value)).item()
        return max(self.min_value, min(self.max_value, value))


# Import Trajectory here to avoid circular imports
from chronos.core.state import Trajectory
