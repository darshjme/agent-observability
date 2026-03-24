"""ObservabilityContext — context manager wiring tracer, logger, and metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from .exporter import BaseExporter, StdoutExporter, TraceExporter
from .logger import AgentLogger
from .metrics import MetricsCollector
from .tracer import AgentTracer, Span, Trace


class ObservabilityContext:
    """Wires :class:`~.tracer.AgentTracer`, :class:`~.logger.AgentLogger`,
    and :class:`~.metrics.MetricsCollector` into a single cohesive context.

    Use this class as the single entry-point for all observability concerns
    in an agent system.  It provides convenience helpers that record data
    to all three subsystems simultaneously, keeping them in sync without
    boilerplate.

    Args:
        name: Human-readable agent / system name.
        exporters: List of :class:`~.exporter.BaseExporter` backends.  If
            None, a :class:`~.exporter.StdoutExporter` is used.
        emit_logs: Whether the embedded :class:`~.logger.AgentLogger`
            should write JSON lines on each call.
        logger_stream: Optional stream for the logger JSON output.

    Example:
        >>> ctx = ObservabilityContext("my-agent")
        >>> with ctx.trace("run-1") as trace:
        ...     with ctx.span("llm-call") as span:
        ...         ctx.log_call(model="gpt-4o", prompt="Hello", latency_ms=100,
        ...                      tokens_in=5, tokens_out=20, span=span)
        ...     ctx.record_metric("custom_score", 0.95)
        >>> ctx.export_all()
    """

    def __init__(
        self,
        name: str = "agent",
        exporters: Optional[List[BaseExporter]] = None,
        emit_logs: bool = False,
        logger_stream: Optional[Any] = None,
    ) -> None:
        """Initialize ObservabilityContext.

        Args:
            name: Agent / system name.
            exporters: Optional list of trace exporters.
            emit_logs: Whether the logger emits JSON lines to stream.
            logger_stream: Optional output stream for logger JSON.
        """
        self._name = name
        self._tracer = AgentTracer()
        self._logger = AgentLogger(name=name, emit_json=emit_logs, stream=logger_stream)
        self._metrics = MetricsCollector()
        self._exporter = TraceExporter(exporters)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def tracer(self) -> AgentTracer:
        """The underlying :class:`~.tracer.AgentTracer`."""
        return self._tracer

    @property
    def logger(self) -> AgentLogger:
        """The underlying :class:`~.logger.AgentLogger`."""
        return self._logger

    @property
    def metrics(self) -> MetricsCollector:
        """The underlying :class:`~.metrics.MetricsCollector`."""
        return self._metrics

    @property
    def exporter(self) -> TraceExporter:
        """The underlying :class:`~.exporter.TraceExporter`."""
        return self._exporter

    @property
    def name(self) -> str:
        """The agent / system name."""
        return self._name

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager that opens and closes a trace.

        Args:
            name: Human-readable name for the agent run.
            metadata: Optional metadata to attach to the trace.

        Yields:
            The active :class:`~.tracer.Trace`.
        """
        return self._tracer.start_trace(name, metadata)

    def span(
        self,
        name: str,
        trace: Optional[Trace] = None,
        parent: Optional[Span] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager that opens and closes a span.

        Args:
            name: Human-readable name for the operation.
            trace: Target trace; uses the active trace if None.
            parent: Optional parent span.
            metadata: Optional metadata.

        Yields:
            The active :class:`~.tracer.Span`.
        """
        return self._tracer.start_span(name, trace=trace, parent_span=parent, metadata=metadata)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_call(
        self,
        model: str,
        prompt: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        span: Optional[Span] = None,
        trace: Optional[Trace] = None,
    ) -> None:
        """Log a call and simultaneously record latency / token metrics.

        Args:
            model: Model identifier string.
            prompt: Raw prompt text (hashed before storage).
            latency_ms: Round-trip latency in milliseconds.
            tokens_in: Input token count.
            tokens_out: Output token count.
            cost_usd: Estimated cost in US dollars.
            success: Whether the call succeeded.
            error: Error message if the call failed.
            metadata: Arbitrary extra data.
            span: If provided, attach span IDs to the log record.
            trace: If provided, attach trace IDs to the log record.
        """
        trace_id = trace.trace_id if trace else None
        span_id = span.span_id if span else None

        self._logger.log_call(
            model=model,
            prompt=prompt,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            success=success,
            error=error,
            metadata=metadata,
            trace_id=trace_id,
            span_id=span_id,
        )
        labels = {"model": model}
        self._metrics.record(MetricsCollector.LATENCY_MS, latency_ms, labels=labels)
        self._metrics.record(MetricsCollector.TOKENS_IN, float(tokens_in), labels=labels)
        self._metrics.record(MetricsCollector.TOKENS_OUT, float(tokens_out), labels=labels)
        self._metrics.record(MetricsCollector.COST_USD, cost_usd, labels=labels)
        if success:
            self._metrics.record_success(labels=labels)
        else:
            self._metrics.record_error(labels=labels)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an arbitrary metric point.

        Args:
            name: Metric name.
            value: Numeric value.
            labels: Optional key-value labels.
        """
        self._metrics.record(name, value, labels=labels)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_all(self) -> None:
        """Export all recorded traces through all configured exporters."""
        self._exporter.export(self._tracer.traces)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a high-level observability summary.

        Returns:
            Dictionary with trace count, call count, success rate,
            total cost, and aggregated metric summaries.
        """
        return {
            "name": self._name,
            "trace_count": len(self._tracer.traces),
            "call_count": len(self._logger.records),
            "success_rate": self._logger.success_rate(),
            "total_cost_usd": self._logger.total_cost(),
            "metrics": self._metrics.summary(),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all tracer, logger, and metrics data."""
        self._tracer.clear()
        self._logger.clear()
        self._metrics.clear()
