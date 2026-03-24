# Contributing to agent-observability

Thank you for your interest in contributing! This document outlines the process.

## Development Setup

```bash
git clone https://github.com/darshjme/agent-observability.git
cd agent-observability
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=agent_observability --cov-report=term-missing
```

All tests must pass before submitting a pull request.

## Code Style

- **Type hints** on every function signature.
- **Google-style docstrings** on all public classes and methods.
- Keep external dependencies at **zero** (stdlib only).
- Follow [PEP 8](https://peps.python.org/pep-0008/).

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Add or update tests for any changed behaviour.
3. Ensure `pytest tests/ -v` passes locally.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a PR with a clear description of the change.

## Reporting Issues

Use [GitHub Issues](https://github.com/darshjme/agent-observability/issues) with a minimal reproducible example.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
