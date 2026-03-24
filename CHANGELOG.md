# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2026-03-24

### Added
- `AgentTracer` with `Trace` and `Span` dataclasses for multi-step agent tracing.
- `AgentLogger` with SHA-256 prompt hashing for privacy-preserving structured logs.
- `MetricsCollector` with percentile aggregation (p95/p99), success rate, and label filtering.
- `TraceExporter` facade supporting fan-out to multiple backends.
- `StdoutExporter` — writes JSONL to any stream (stdout by default).
- `JSONFileExporter` — appends JSONL or overwrites JSON array on disk.
- `ObservabilityContext` — single context manager wiring all components together.
- Full type annotations and Google-style docstrings throughout.
- 40+ unit tests with 100% stdlib — zero external dependencies.
- `pyproject.toml` with `setuptools` build backend.

[0.1.0]: https://github.com/darshjme/agent-observability/releases/tag/v0.1.0
