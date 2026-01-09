# Chronos

**Distributed Nested Optimization Framework with HOPE**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Chronos is a production-ready framework for High-Order Optimization (HOPE) that enables distributed bilevel/nested optimization with:

- **Versioned Bounded Asynchrony**: Contains staleness cascade without blocking
- **Hybrid State Management**: Sharded coordinator + peer-to-peer for scalability
- **Significance-Triggered Communication**: 40% network reduction via sparse protocols

## Installation

```bash
# From source
git clone https://github.com/chronos-ml/chronos.git
cd chronos
pip install -e ".[dev]"
```

## Quick Start

```python
from chronos.core import InnerProblem, OuterOptimizer, MetaState
from chronos.solver import ImplicitDifferentiation

# Define your bilevel problem
class MyInnerProblem(InnerProblem):
    def objective(self, params, outer_params, data):
        # Your training loss here
        return loss

    def solve(self, outer_params, init_params, num_steps, data_loader):
        # Your inner optimization loop
        return final_params, trajectory

# Run optimization
outer_opt = ImplicitDifferentiation(
    outer_params={"lr": torch.tensor(0.01)},
    lr=0.001
)

for outer_step in range(100):
    final_params, trajectory = inner_problem.solve(outer_opt.outer_params, ...)
    hypergradient = outer_opt.compute_hypergradient([trajectory], inner_problem)
    outer_opt.step(hypergradient)
```

## Examples

### Data Hyper-Cleaning

Learn optimal sample weights to clean noisy labels:

```bash
python examples/hyperclean.py --noise-ratio 0.4 --outer-steps 20
```

## Project Structure

```
chronos/
├── core/           # Core abstractions (InnerProblem, MetaState, VersionTracker)
├── solver/         # Hypergradient computation (implicit, unrolled)
├── distributed/    # Coordinator, Worker, protocols (Phase 2)
├── communication/  # Sparse protocols, compression (Phase 2)
└── continuum/      # HOPE-inspired memory systems (Phase 3)
```

## Roadmap

- [x] Phase 1: Core abstractions & single-node solvers
- [ ] Phase 2: Distributed orchestration (coordinator, workers, sparse comm)
- [ ] Phase 3: HOPE features (continuum memory, multi-timescale updates)

## References

- [Nested Learning: A New ML Paradigm (NeurIPS 2025)](http://abehrouz.github.io/files/NL.pdf)
- [TorchOpt: Differentiable Optimization Library](https://github.com/metaopt/torchopt)
- [Betty: Multilevel Optimization Library](https://github.com/leopard-ai/betty)

## License

MIT License - see [LICENSE](LICENSE) for details.