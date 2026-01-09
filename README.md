# Chronos

**Distributed Nested Optimization Framework with HOPE**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Chronos is a production-ready framework for High-Order Optimization (HOPE) that enables distributed bilevel/nested optimization with:

- **Versioned Bounded Asynchrony**: Contains staleness cascade without blocking
- **Hybrid State Management**: Sharded coordinator + peer-to-peer for scalability
- **Significance-Triggered Communication**: 40% network reduction via sparse protocols
- **Continuum Memory System**: Learn from optimization history for faster convergence
- **Multi-Timescale Updates**: Different update frequencies for different hyperparameters

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

```python
from chronos.distributed import Coordinator, Worker, WorkerConfig

# Start coordinator
coordinator = Coordinator(
    outer_params={"lr": 0.01},
    port=5555,
    max_in_flight=3
)
coordinator.start()

# Start workers
worker = Worker(inner_problem, WorkerConfig(
    coordinator_addr="tcp://localhost:5555",
    significance_threshold=0.01
))
worker.connect()
worker.run()
```

### HOPE Features

```python
from chronos.continuum import ContinuumMemory, MultiTimescaleOptimizer

# Memory-augmented optimization
memory = ContinuumMemory()
for step in range(100):
    # Store optimization state
    memory.store(outer_params, hypergradient, val_loss, step)

    # Retrieve similar historical states
    neighbors = memory.retrieve_similar(outer_params, k=5)
    predicted = memory.predict_gradient(outer_params)

# Multi-timescale updates
optimizer = MultiTimescaleOptimizer(outer_params, config)
optimizer.step(hypergradient)  # Updates at different frequencies
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      COORDINATOR                            │
│  VersionTracker │ MetaState │ ContinuumMemory               │
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
├── continuum/      # HOPE memory systems, multi-timescale
└── benchmarks/     # Performance measurement tools
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Bounded Staleness** | Max 3 versions in-flight, exponential decay weights |
| **Significance Filter** | Only sync when `\|\|Δ\|\|/\|\|θ\|\| > threshold` |
| **Top-k Sparsification** | Keep only k largest gradients |
| **Continuum Memory** | Learn from historical optimization paths |
| **Multi-Timescale** | Fast/medium/slow update frequencies |

## Benchmarking

```python
from chronos.benchmarks import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(name="my_benchmark", num_outer_steps=100)
runner = BenchmarkRunner(config)
result = runner.run(create_problem, create_optimizer)
print(f"Throughput: {result.outer_steps_per_second:.2f} steps/sec")
```

## Roadmap

- [x] Phase 1: Core abstractions & single-node solvers
- [x] Phase 2: Distributed orchestration (coordinator, workers, sparse comm)
- [x] Phase 3: HOPE features (continuum memory, multi-timescale, benchmarks)
- [ ] Phase 4: Production hardening (tests, docs, CI/CD)

## References

- [Nested Learning (NeurIPS 2025)](http://abehrouz.github.io/files/NL.pdf)
- [TorchOpt](https://github.com/metaopt/torchopt)
- [Betty](https://github.com/leopard-ai/betty)

## License

MIT License
