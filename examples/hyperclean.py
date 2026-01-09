"""
Data Hyper-Cleaning Example

Demonstrates bilevel optimization for learning optimal sample weights
to clean a dataset with noisy labels.

Outer problem: Learn sample weights w ∈ [0,1]^n that minimize validation loss
Inner problem: Train model on weighted training data

This is a classic bilevel optimization benchmark from:
"Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting" (CVPR 2019)
"""

import argparse
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from chronos.core.problem import InnerProblem, OuterOptimizer
from chronos.core.state import Trajectory
from chronos.solver.unrolled import UnrolledDifferentiation, DifferentiableInnerSolver
from chronos.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class SimpleNN(nn.Module):
    """Simple 2-layer neural network for classification."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class HyperCleaningProblem(InnerProblem):
    """
    Inner problem for data hyper-cleaning.

    The outer parameters are per-sample weights that scale the
    contribution of each training sample to the loss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: Tuple[Tensor, Tensor],
        device: torch.device = None
    ):
        super().__init__(model, device)
        self.train_x, self.train_y = train_data
        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.n_samples = len(self.train_x)

    def objective(
        self,
        params: Dict[str, Tensor],
        outer_params: Dict[str, Tensor],
        data: Any = None
    ) -> Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            params: Model parameters (not used directly - model is set)
            outer_params: Contains 'sample_weights' tensor
            data: Optional batch indices, if None uses all data
        """
        # Get sample weights from outer params
        sample_weights = outer_params.get("sample_weights")
        if sample_weights is None:
            sample_weights = torch.ones(self.n_samples, device=self.device)

        # Normalize weights to sum to n_samples (keeps loss scale consistent)
        sample_weights = F.softmax(sample_weights, dim=0) * self.n_samples

        # Forward pass
        logits = self.model(self.train_x)

        # Per-sample cross entropy
        per_sample_loss = F.cross_entropy(logits, self.train_y, reduction='none')

        # Weighted loss
        weighted_loss = (sample_weights * per_sample_loss).mean()

        return weighted_loss

    def solve(
        self,
        outer_params: Dict[str, Tensor],
        init_params: Optional[Dict[str, Tensor]] = None,
        num_steps: int = 100,
        data_loader: Any = None
    ) -> Tuple[Dict[str, Tensor], Trajectory]:
        """
        Solve inner problem with SGD.
        """
        import uuid

        # Initialize
        if init_params is not None:
            self.set_params(init_params)

        trajectory = Trajectory(
            version=0,
            worker_id=str(uuid.uuid4())[:8],
            outer_params={k: v.detach().clone() for k, v in outer_params.items()}
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        for step in range(num_steps):
            optimizer.zero_grad()

            loss = self.objective(self.get_params(), outer_params, None)
            loss.backward()
            optimizer.step()

            # Record trajectory
            trajectory.add_step(
                step=step,
                params=self.get_params(),
                loss=loss.item()
            )

        trajectory.finalize(self.get_params())
        return self.get_params(), trajectory


def generate_noisy_data(
    n_train: int = 1000,
    n_val: int = 200,
    input_dim: int = 20,
    n_classes: int = 2,
    noise_ratio: float = 0.4,
    seed: int = 42
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
    """
    Generate synthetic classification data with label noise.

    Returns:
        (train_x, train_y), (val_x, val_y), noise_mask
    """
    torch.manual_seed(seed)

    # Generate clean data
    train_x = torch.randn(n_train, input_dim)
    val_x = torch.randn(n_val, input_dim)

    # True labels based on linear decision boundary
    weight = torch.randn(input_dim)
    train_y_clean = (train_x @ weight > 0).long()
    val_y = (val_x @ weight > 0).long()

    # Add noise to training labels
    n_noisy = int(n_train * noise_ratio)
    noise_mask = torch.zeros(n_train, dtype=torch.bool)
    noise_indices = torch.randperm(n_train)[:n_noisy]
    noise_mask[noise_indices] = True

    train_y = train_y_clean.clone()
    train_y[noise_mask] = 1 - train_y[noise_mask]  # Flip labels

    return (train_x, train_y), (val_x, val_y), noise_mask


def evaluate(model: nn.Module, data: Tuple[Tensor, Tensor], device: torch.device) -> float:
    """Evaluate model accuracy on data."""
    model.eval()
    x, y = data[0].to(device), data[1].to(device)
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    model.train()
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Hyper-cleaning example")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--noise-ratio", type=float, default=0.4)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--outer-steps", type=int, default=20)
    parser.add_argument("--outer-lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Generate data
    train_data, val_data, noise_mask = generate_noisy_data(
        n_train=args.n_train,
        noise_ratio=args.noise_ratio,
        seed=args.seed
    )

    logger.info(f"Generated {args.n_train} training samples with {args.noise_ratio:.0%} noise")

    # Initialize model
    model = SimpleNN(input_dim=20, hidden_dim=50, output_dim=2).to(device)

    # Initialize inner problem
    inner_problem = HyperCleaningProblem(model, train_data, device)

    # Initialize sample weights (outer params)
    outer_params = {
        "sample_weights": torch.zeros(args.n_train, device=device, requires_grad=True)
    }

    # Initialize outer optimizer
    outer_optimizer = UnrolledDifferentiation(
        outer_params=outer_params,
        lr=args.outer_lr,
        truncation_steps=10,
        device=device
    )

    # Baseline: train without hyper-cleaning
    logger.info("Training baseline (uniform weights)...")
    model_baseline = SimpleNN(input_dim=20, hidden_dim=50, output_dim=2).to(device)
    baseline_problem = HyperCleaningProblem(model_baseline, train_data, device)
    uniform_weights = {"sample_weights": torch.zeros(args.n_train, device=device)}
    baseline_problem.solve(uniform_weights, num_steps=args.inner_steps * args.outer_steps)
    baseline_acc = evaluate(model_baseline, val_data, device)
    logger.info(f"Baseline validation accuracy: {baseline_acc:.2%}")

    # Bilevel optimization
    logger.info("Starting bilevel optimization...")

    for outer_step in range(args.outer_steps):
        # Reset model for each outer step (or continue from previous)
        model = SimpleNN(input_dim=20, hidden_dim=50, output_dim=2).to(device)
        inner_problem = HyperCleaningProblem(model, train_data, device)

        # Solve inner problem
        final_params, trajectory = inner_problem.solve(
            outer_params=outer_optimizer.outer_params,
            num_steps=args.inner_steps
        )

        # Evaluate on validation
        val_acc = evaluate(model, val_data, device)

        # Compute hypergradient
        hypergradient = outer_optimizer.compute_hypergradient(
            trajectories=[trajectory],
            inner_problem=inner_problem,
            validation_data=val_data
        )

        # Update outer params
        outer_optimizer.step(hypergradient)

        # Log progress
        final_weights = F.softmax(outer_optimizer.outer_params["sample_weights"], dim=0)
        noise_weight = final_weights[noise_mask].sum().item()
        clean_weight = final_weights[~noise_mask].sum().item()

        logger.info(
            f"Outer step {outer_step + 1}/{args.outer_steps}: "
            f"val_acc={val_acc:.2%}, "
            f"noise_weight={noise_weight:.2f}, clean_weight={clean_weight:.2f}"
        )

    # Final evaluation
    final_acc = evaluate(model, val_data, device)
    logger.info(f"\n{'='*50}")
    logger.info(f"Baseline accuracy: {baseline_acc:.2%}")
    logger.info(f"Hyper-cleaning accuracy: {final_acc:.2%}")
    logger.info(f"Improvement: {final_acc - baseline_acc:+.2%}")


if __name__ == "__main__":
    main()
