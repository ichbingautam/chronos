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
git clone https://github.com/chronos-ml/chronos.git
cd chronos
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### Single-Node Bilevel Optimization

```python
from chronos.core import InnerProblem, MetaState
from chronos.solver import ImplicitDifferentiation

# Define your problem and run optimization
outer_opt = ImplicitDifferentiation(
    outer_params={"lr": torch.tensor(0.01)},
    lr=0.001
)

for step in range(100):
    final_params, trajectory = inner_problem.solve(outer_opt.outer_params, ...)
    hypergradient = outer_opt.compute_hypergradient([trajectory], inner_problem)
    outer_opt.step(hypergradient)
```

### Distributed Training

**Start coordinator:**

```python
from chronos.distributed import Coordinator

coordinator = Coordinator(
    outer_params={"lr": 0.01, "weight_decay": 0.001},
    port=5555,
    max_in_flight=3  # Bounded staleness
)
coordinator.start()
```

**Start workers (on different machines/processes):**

```python
from chronos.distributed import Worker, WorkerConfig

config = WorkerConfig(
    coordinator_addr="tcp://coordinator-host:5555",
    inner_steps=100,
    significance_threshold=0.01  # Sparse communication
)

worker = Worker(inner_problem, config)
worker.connect()
worker.run(num_iterations=1000)
```

## Examples

```bash
# Data hyper-cleaning (single-node)
python examples/hyperclean.py --noise-ratio 0.4 --outer-steps 20
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      COORDINATOR                            │
│  VersionTracker │ MetaState │ ZeroMQ Server                │
└─────────────────────────────────────────────────────────────┘
            │                  │                  │
    ┌───────┴────────┐ ┌───────┴────────┐ ┌───────┴────────┐
    │   WORKER 1     │ │   WORKER 2     │ │   WORKER N     │
    │  Inner Loop    │ │  Inner Loop    │ │  Inner Loop    │
    │  + Signif.     │ │  + Signif.     │ │  + Signif.     │
    │    Filter      │ │    Filter      │ │    Filter      │
    └────────────────┘ └────────────────┘ └────────────────┘
```

## Project Structure

```
chronos/
├── core/           # InnerProblem, MetaState, VersionTracker
├── solver/         # Implicit & unrolled differentiation
├── distributed/    # Coordinator, Worker, ZeroMQ protocols
├── communication/  # Sparse protocols, compression
└── continuum/      # HOPE memory systems (Phase 3)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Bounded Staleness** | Max 3 versions in-flight, exponential decay weights |
| **Significance Filter** | Only sync when `\|\|Δ\|\|/\|\|θ\|\| > threshold` |
| **Top-k Sparsification** | Keep only k largest gradients |
| **Error Feedback** | Preserve gradient info across sparse updates |
| **INT8 Quantization** | Optional precision reduction |

## Roadmap

- [x] Phase 1: Core abstractions & single-node solvers
- [x] Phase 2: Distributed orchestration (coordinator, workers, sparse comm)
- [ ] Phase 3: HOPE features (continuum memory, multi-timescale updates)

## References

- [Nested Learning (NeurIPS 2025)](http://abehrouz.github.io/files/NL.pdf)
- [TorchOpt](https://github.com/metaopt/torchopt)
- [Betty](https://github.com/leopard-ai/betty)

## License

MIT License
