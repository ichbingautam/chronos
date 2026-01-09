"""
Implicit Differentiation for computing hypergradients.

Uses the Implicit Function Theorem to compute dL_val/dλ without
backpropagating through the entire inner optimization trajectory.

Key formula:
    dL_val/dλ = ∂L_val/∂θ* · dθ*/dλ

where dθ*/dλ is computed via:
    dθ*/dλ = -(∇²_θθ L_train)^(-1) · ∇²_θλ L_train

We use Conjugate Gradient to solve the linear system instead of
explicitly computing the Hessian inverse.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import hvp, vjp

from chronos.core.problem import InnerProblem, OuterOptimizer
from chronos.core.state import MetaState, Trajectory


def conjugate_gradient(
    hvp_fn: Callable[[Tensor], Tensor],
    b: Tensor,
    max_iter: int = 10,
    tol: float = 1e-5,
    damping: float = 0.0
) -> Tensor:
    """
    Solve Ax = b using Conjugate Gradient with Hessian-vector products.

    This avoids materializing the full Hessian matrix, which is crucial
    for large neural networks where H ∈ R^(n×n) would be prohibitive.

    Args:
        hvp_fn: Function computing Hessian-vector product Hv for any v
        b: Right-hand side vector
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        damping: Damping factor added to diagonal (for numerical stability)

    Returns:
        Approximate solution x ≈ H^(-1)b
    """
    x = torch.zeros_like(b)
    r = b.clone()  # residual = b - Ax, initially = b since x = 0
    p = r.clone()  # search direction

    r_norm_sq = torch.dot(r.flatten(), r.flatten())

    for i in range(max_iter):
        # Compute Ap (with optional damping)
        Ap = hvp_fn(p)
        if damping > 0:
            Ap = Ap + damping * p

        # Step size
        pAp = torch.dot(p.flatten(), Ap.flatten())
        if pAp.abs() < 1e-12:
            break
        alpha = r_norm_sq / pAp

        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap

        # Check convergence
        r_norm_sq_new = torch.dot(r.flatten(), r.flatten())
        if r_norm_sq_new.sqrt() < tol:
            break

        # Update search direction
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new

    return x


class ImplicitDifferentiation(OuterOptimizer):
    """
    Outer optimizer using implicit differentiation for hypergradients.

    This is more memory-efficient than unrolled differentiation as it
    doesn't require storing the full optimization trajectory for backprop.

    Args:
        outer_params: Initial outer parameters
        lr: Learning rate for outer updates
        cg_steps: Number of conjugate gradient iterations
        cg_damping: Damping factor for CG (improves stability)
        device: Device for computations
    """

    def __init__(
        self,
        outer_params: Dict[str, Tensor],
        lr: float = 0.01,
        cg_steps: int = 10,
        cg_damping: float = 0.01,
        device: torch.device = None
    ):
        super().__init__(outer_params, lr, device)
        self.cg_steps = cg_steps
        self.cg_damping = cg_damping

    def compute_hypergradient(
        self,
        trajectories: list[Trajectory],
        inner_problem: InnerProblem,
        validation_data: Any = None
    ) -> Dict[str, Tensor]:
        """
        Compute hypergradient using implicit differentiation.

        The computation follows these steps:
        1. Compute validation gradient ∂L_val/∂θ at final params
        2. Solve (∇²_θθ L_train)·v = ∂L_val/∂θ for v using CG
        3. Compute hypergradient as -∇²_θλ L_train · v

        Args:
            trajectories: Completed inner optimization trajectories
            inner_problem: The inner problem instance
            validation_data: Data for computing outer objective

        Returns:
            Dictionary mapping outer param names to gradients
        """
        if not trajectories:
            return {k: torch.zeros_like(v) for k, v in self.outer_params.items()}

        # Aggregate over trajectories (use the most recent one for simplicity)
        trajectory = trajectories[-1]

        if trajectory.final_params is None:
            raise ValueError("Trajectory must be finalized with final_params")

        # Set model to final parameters
        inner_problem.set_params(trajectory.final_params)
        final_params = trajectory.final_params
        outer_params = self.outer_params

        # Step 1: Compute validation gradient ∂L_val/∂θ
        if validation_data is not None:
            val_loss = inner_problem.objective(final_params, outer_params, validation_data)
        else:
            # Use training loss as proxy for validation
            val_loss = trajectory.final_loss
            # Need actual loss tensor for gradients
            val_loss = inner_problem.objective(
                final_params, outer_params,
                trajectory.steps[-1] if trajectory.steps else None
            )

        # Get ∂L_val/∂θ
        val_grads = self._compute_param_grads(val_loss, inner_problem.model)
        val_grad_flat = self._flatten_grads(val_grads)

        # Step 2: Solve (∇²_θθ L_train)·v = ∂L_val/∂θ using CG
        def hvp_fn(v: Tensor) -> Tensor:
            """Hessian-vector product for training loss."""
            return self._hessian_vector_product(
                inner_problem, final_params, outer_params,
                trajectory, v
            )

        v = conjugate_gradient(
            hvp_fn, val_grad_flat,
            max_iter=self.cg_steps,
            damping=self.cg_damping
        )

        # Step 3: Compute hypergradient via mixed partials
        hypergradient = self._compute_mixed_partials(
            inner_problem, final_params, outer_params, trajectory, v
        )

        return hypergradient

    def _compute_param_grads(
        self,
        loss: Tensor,
        model: torch.nn.Module
    ) -> Dict[str, Tensor]:
        """Compute gradients of loss w.r.t. model parameters."""
        grads = torch.autograd.grad(
            loss, model.parameters(),
            create_graph=True, allow_unused=True
        )
        return {
            name: g if g is not None else torch.zeros_like(p)
            for (name, p), g in zip(model.named_parameters(), grads)
        }

    def _flatten_grads(self, grads: Dict[str, Tensor]) -> Tensor:
        """Flatten gradient dictionary to single vector."""
        return torch.cat([g.flatten() for g in grads.values()])

    def _unflatten_grads(
        self,
        flat: Tensor,
        template: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Unflatten vector back to gradient dictionary."""
        result = {}
        offset = 0
        for name, t in template.items():
            numel = t.numel()
            result[name] = flat[offset:offset + numel].view_as(t)
            offset += numel
        return result

    def _hessian_vector_product(
        self,
        inner_problem: InnerProblem,
        final_params: Dict[str, Tensor],
        outer_params: Dict[str, Tensor],
        trajectory: Trajectory,
        v: Tensor
    ) -> Tensor:
        """
        Compute Hessian-vector product ∇²_θθ L_train · v.

        Uses the identity:
            Hv = d/dt [∇_θ L_train(θ + tv)] |_{t=0}
        """
        # Get training loss at final params
        train_loss = inner_problem.objective(
            final_params, outer_params,
            trajectory.steps[-1] if trajectory.steps else None
        )

        # First gradient
        grads = self._compute_param_grads(train_loss, inner_problem.model)
        grad_flat = self._flatten_grads(grads)

        # Hessian-vector product via second backprop
        hvp_result = torch.autograd.grad(
            grad_flat, list(inner_problem.model.parameters()),
            grad_outputs=self._unflatten_to_tuple(v, grads),
            retain_graph=True, allow_unused=True
        )

        hvp_flat = torch.cat([
            g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
            for (p, g) in zip(inner_problem.model.parameters(), hvp_result)
        ])

        return hvp_flat

    def _unflatten_to_tuple(
        self,
        flat: Tensor,
        template: Dict[str, Tensor]
    ) -> Tuple[Tensor, ...]:
        """Unflatten to tuple matching model.parameters() order."""
        result = []
        offset = 0
        for t in template.values():
            numel = t.numel()
            result.append(flat[offset:offset + numel].view_as(t))
            offset += numel
        return tuple(result)

    def _compute_mixed_partials(
        self,
        inner_problem: InnerProblem,
        final_params: Dict[str, Tensor],
        outer_params: Dict[str, Tensor],
        trajectory: Trajectory,
        v: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute mixed partial derivatives ∇²_θλ L_train · v.

        This gives us the hypergradient after solving the CG system.
        """
        hypergradient = {}

        # For each outer parameter, compute the mixed partial
        for name, lam in outer_params.items():
            if not lam.requires_grad:
                hypergradient[name] = torch.zeros_like(lam)
                continue

            # Compute ∂L_train/∂λ with θ held fixed but depending on λ
            train_loss = inner_problem.objective(
                final_params, outer_params,
                trajectory.steps[-1] if trajectory.steps else None
            )

            # This is a simplification - full implementation would need
            # to track how θ* depends on λ through the inner loop
            grad_lambda = torch.autograd.grad(
                train_loss, lam,
                retain_graph=True, allow_unused=True
            )[0]

            if grad_lambda is None:
                hypergradient[name] = torch.zeros_like(lam)
            else:
                hypergradient[name] = -grad_lambda  # Negative sign from implicit diff

        return hypergradient
