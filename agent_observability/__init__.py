"""agent-observability: Production-grade LLM agent observability library.

Provides structured span tracing, cost tracking, and log export
for LLM agent runs. Zero external dependencies (stdlib only).

Example:
    >>> from agent_observability import ObservabilityMiddleware, SpanTracer, CostTracker, LogExporter
    >>> middleware = ObservabilityMiddleware(model_id="gpt-4o")
    >>> with middleware.trace_run() as run:
    ...     run.record_tokens(tokens_in=100, tokens_out=50)
"""

from agent_observability.middleware import ObservabilityMiddleware, RunContext
from agent_observability.tracer import SpanTracer, Span
from agent_observability.cost import CostTracker, ModelPricing
from agent_observability.exporter import LogExporter

__all__ = [
    "ObservabilityMiddleware",
    "RunContext",
    "SpanTracer",
    "Span",
    "CostTracker",
    "ModelPricing",
    "LogExporter",
]

__version__ = "0.1.0"
__author__ = "darshjme"
