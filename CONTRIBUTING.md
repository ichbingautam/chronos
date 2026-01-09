# Contributing to Chronos

Thank you for your interest in contributing to Chronos! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
# Clone repository
git clone https://github.com/ichbingautam/chronos.git
cd chronos

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code
ruff check chronos/ tests/

# Format code
ruff format chronos/ tests/
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core_state.py -v

# Run with coverage
pytest tests/ --cov=chronos --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with descriptive message
6. Push to your fork
7. Open a Pull Request

## Commit Messages

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Adding tests
- `refactor:` Code refactoring

## Project Structure

```
chronos/
├── core/           # Core abstractions
├── solver/         # Differentiation solvers
├── distributed/    # Coordinator and workers
├── communication/  # Sparse protocols
├── continuum/      # HOPE features
└── benchmarks/     # Performance tools
```

## Questions?

Open an issue or discussion on GitHub.
