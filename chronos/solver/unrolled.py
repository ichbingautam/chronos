"""
Unrolled Differentiation for computing hypergradients.

Backpropagates through the entire inner optimization loop to compute
exact gradients. More accurate than implicit differentiation but
requires more memory (O(K) where K = inner steps).

This is the approach used by MAML and other meta-learning algorithms.
"""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from chronos.core.problem import InnerProblem, OuterOptimizer
from chronos.core.state import Trajectory


class UnrolledDifferentiation(OuterOptimizer):
    """
    Outer optimizer using unrolled differentiation for hypergradients.

    This approach:
    1. Runs K steps of inner optimization while retaining the computation graph
    2. Computes validation loss at the final parameters
    3. Backpropagates through all K steps to get ∂L_val/∂λ

    Pros:
    - Exact gradients (no approximation)
    - Simple to implement

    Cons:
    - Memory scales with K (inner steps)
    - Can be slow for large K

    Args:
        outer_params: Initial outer parameters
        lr: Learning rate for outer updates
        truncation_steps: If set, only backprop through last N steps
        device: Device for computations
    """

    def __init__(
        self,
        outer_params: Dict[str, Tensor],
        lr: float = 0.01,
        truncation_steps: Optional[int] = None,
        device: torch.device = None
    ):
        super().__init__(outer_params, lr, device)
        self.truncation_steps = truncation_steps

    def compute_hypergradient(
        self,
        trajectories: list[Trajectory],
        inner_problem: InnerProblem,
        validation_data: Any = None
    ) -> Dict[str, Tensor]:
        """
        Compute hypergradient by backpropagating through inner loop.

        Note: For this to work, inner optimization must be done with
        create_graph=True so gradients are tracked.

        Args:
            trajectories: Completed inner optimization trajectories
            inner_problem: The inner problem instance
            validation_data: Data for computing outer objective

        Returns:
            Dictionary mapping outer param names to gradients
        """
        if not trajectories:
            return {k: torch.zeros_like(v) for k, v in self.outer_params.items()}

        # Aggregate gradients across trajectories
        aggregated_grads = {k: torch.zeros_like(v) for k, v in self.outer_params.items()}

        for trajectory in trajectories:
            traj_grads = self._compute_trajectory_gradient(
                trajectory, inner_problem, validation_data
            )

            for k in aggregated_grads:
                if k in traj_grads:
                    aggregated_grads[k] += traj_grads[k]

        # Average
        n_traj = len(trajectories)
        return {k: v / n_traj for k, v in aggregated_grads.items()}

    def _compute_trajectory_gradient(
        self,
        trajectory: Trajectory,
        inner_problem: InnerProblem,
        validation_data: Any
    ) -> Dict[str, Tensor]:
        """Compute gradient from a single trajectory."""

        if trajectory.final_params is None:
            raise ValueError("Trajectory must be finalized")

        # Determine which steps to backprop through
        steps = trajectory.steps
        if self.truncation_steps is not None:
            steps = steps[-self.truncation_steps:]

        if not steps:
            return {k: torch.zeros_like(v) for k, v in self.outer_params.items()}

        # The key insight: we need the final params to still be
        # connected to the outer params through the computation graph
        final_params = trajectory.final_params

        # Compute validation loss
        if validation_data is not None:
            val_loss = inner_problem.objective(
                final_params, self.outer_params, validation_data
            )
        else:
            # Use final training loss as proxy
            val_loss = torch.tensor(trajectory.final_loss, device=self.device)
            # This won't have gradients - need actual forward pass
            val_loss = inner_problem.objective(
                final_params, self.outer_params, None
            )

        # Backpropagate to outer params
        grads = {}
        for name, param in self.outer_params.items():
            if param.requires_grad:
                grad = torch.autograd.grad(
                    val_loss, param,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                grads[name] = grad if grad is not None else torch.zeros_like(param)
            else:
                grads[name] = torch.zeros_like(param)

        return grads


class DifferentiableInnerSolver:
    """
    Helper class to run inner optimization with gradient tracking.

    This wraps the inner optimization loop to ensure the computation
    graph is retained for unrolled differentiation.
    """

    def __init__(
        self,
        inner_problem: InnerProblem,
        inner_lr: float = 0.01,
        create_graph: bool = True
    ):
        self.inner_problem = inner_problem
        self.inner_lr = inner_lr
        self.create_graph = create_graph

    def solve(
        self,
        outer_params: Dict[str, Tensor],
        init_params: Optional[Dict[str, Tensor]] = None,
        num_steps: int = 5,
        data_iterator: Any = None
    ) -> Trajectory:
        """
        Run differentiable inner optimization.

        Args:
            outer_params: Fixed outer parameters for this solve
            init_params: Initial model parameters
            num_steps: Number of inner steps
            data_iterator: Iterator yielding training batches

        Returns:
            Trajectory with computation graph intact for backprop
        """
        import uuid

        # Initialize trajectory
        trajectory = Trajectory(
            version=0,  # Will be set by caller
            worker_id=str(uuid.uuid4())[:8],
            outer_params={k: v.clone() for k, v in outer_params.items()}
        )

        # Get or initialize parameters
        if init_params is not None:
            params = {k: v.clone().requires_grad_(True) for k, v in init_params.items()}
        else:
            params = {
                k: v.clone().requires_grad_(True)
                for k, v in self.inner_problem.get_params().items()
            }

        # Run inner optimization
        for step in range(num_steps):
            # Get batch
            data = next(data_iterator) if data_iterator else None

            # Forward pass
            loss = self.inner_problem.objective(params, outer_params, data)

            # Compute gradients
            grads = torch.autograd.grad(
                loss, list(params.values()),
                create_graph=self.create_graph,
                allow_unused=True
            )

            grad_dict = {
                k: g if g is not None else torch.zeros_like(params[k])
                for k, g in zip(params.keys(), grads)
            }

            # Record step (don't detach - keep graph!)
            trajectory.add_step(
                step=step,
                params={k: v.clone() for k, v in params.items()},
                grads={k: v.clone() for k, v in grad_dict.items()},
                loss=loss.item(),
                detach=False  # Keep computation graph!
            )

            # SGD update (stays in graph if create_graph=True)
            # Get learning rate from outer params if available
            lr = outer_params.get("lr", torch.tensor(self.inner_lr))
            if isinstance(lr, Tensor):
                lr_val = lr
            else:
                lr_val = torch.tensor(lr, device=loss.device)

            params = {
                k: v - lr_val * grad_dict[k]
                for k, v in params.items()
            }

        trajectory.finalize({k: v for k, v in params.items()})
        return trajectory
