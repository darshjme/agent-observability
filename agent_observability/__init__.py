"""agent-observability: Structured logging, tracing, and metrics for LLM agent systems.

Quick start::

    from agent_observability import ObservabilityContext

    ctx = ObservabilityContext("my-agent")
    with ctx.trace("run-1"):
        with ctx.span("llm-call") as span:
            # ... call your LLM ...
            ctx.log_call(
                model="gpt-4o",
                prompt="Hello world",
                latency_ms=120.0,
                tokens_in=10,
                tokens_out=40,
                span=span,
            )
    ctx.export_all()
    print(ctx.summary())
"""

from .context import ObservabilityContext
from .exporter import BaseExporter, JSONFileExporter, StdoutExporter, TraceExporter
from .logger import AgentLogger, LLMCallRecord, hash_prompt
from .metrics import AggregatedMetric, MetricPoint, MetricsCollector
from .tracer import AgentTracer, Span, Trace

__all__ = [
    # Context
    "ObservabilityContext",
    # Tracer
    "AgentTracer",
    "Trace",
    "Span",
    # Logger
    "AgentLogger",
    "LLMCallRecord",
    "hash_prompt",
    # Metrics
    "MetricsCollector",
    "MetricPoint",
    "AggregatedMetric",
    # Exporter
    "TraceExporter",
    "BaseExporter",
    "StdoutExporter",
    "JSONFileExporter",
]

__version__ = "0.1.0"
__author__ = "Darshankumar Joshi"
