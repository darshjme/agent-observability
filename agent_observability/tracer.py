"""AgentTracer — trace multi-step agent runs with spans."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    """Represents a single traced operation within an agent run.

    Attributes:
        span_id: Unique identifier for this span.
        name: Human-readable name for the operation.
        start_time: Unix timestamp when the span started.
        end_time: Unix timestamp when the span ended (None if still running).
        duration_ms: Duration in milliseconds (None if still running).
        tokens_in: Number of input tokens consumed.
        tokens_out: Number of output tokens generated.
        metadata: Arbitrary key-value metadata attached to this span.
        parent_span_id: ID of the parent span, if any.
        status: Span status: "running", "ok", or "error".
        error: Error message if status is "error".
    """

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_span_id: Optional[str] = None
    status: str = "running"
    error: Optional[str] = None

    def finish(self, tokens_in: int = 0, tokens_out: int = 0, error: Optional[str] = None) -> "Span":
        """Finalize the span, recording end time and duration.

        Args:
            tokens_in: Input tokens consumed during this span.
            tokens_out: Output tokens generated during this span.
            error: Optional error message; sets status to "error" if provided.

        Returns:
            Self, for method chaining.
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        if error:
            self.status = "error"
            self.error = error
        else:
            self.status = "ok"
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the span to a plain dictionary.

        Returns:
            Dictionary representation of this span.
        """
        return {
            "span_id": self.span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "metadata": self.metadata,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class Trace:
    """Represents a complete agent run composed of one or more spans.

    Attributes:
        trace_id: Unique identifier for this trace.
        name: Human-readable name for the agent run.
        spans: Ordered list of spans collected during the run.
        start_time: Unix timestamp when the trace was created.
        end_time: Unix timestamp when the trace was finished.
        metadata: Arbitrary key-value metadata for the entire trace.
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens_in(self) -> int:
        """Total input tokens across all spans."""
        return sum(s.tokens_in for s in self.spans)

    @property
    def total_tokens_out(self) -> int:
        """Total output tokens across all spans."""
        return sum(s.tokens_out for s in self.spans)

    @property
    def duration_ms(self) -> Optional[float]:
        """Total trace duration in milliseconds, or None if not finished."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trace to a plain dictionary.

        Returns:
            Dictionary representation of this trace.
        """
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "spans": [s.to_dict() for s in self.spans],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "metadata": self.metadata,
        }


class AgentTracer:
    """Traces multi-step agent runs by collecting spans.

    Example:
        >>> tracer = AgentTracer()
        >>> with tracer.start_trace("my-agent") as trace:
        ...     with tracer.start_span("llm-call") as span:
        ...         span.finish(tokens_in=100, tokens_out=50)
    """

    def __init__(self) -> None:
        """Initialize the tracer with an empty trace list."""
        self._traces: List[Trace] = []
        self._active_trace: Optional[Trace] = None

    # ------------------------------------------------------------------
    # Trace lifecycle
    # ------------------------------------------------------------------

    def new_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        """Create and register a new trace (not a context manager).

        Args:
            name: Human-readable name for the agent run.
            metadata: Optional metadata to attach to the trace.

        Returns:
            The newly created Trace.
        """
        trace = Trace(name=name, metadata=metadata or {})
        self._traces.append(trace)
        self._active_trace = trace
        return trace

    def finish_trace(self, trace: Trace) -> Trace:
        """Finalize a trace, recording its end time.

        Args:
            trace: The trace to finish.

        Returns:
            The finished trace.
        """
        trace.end_time = time.time()
        if self._active_trace is trace:
            self._active_trace = None
        return trace

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> "_TraceContext":
        """Start a trace as a context manager.

        Args:
            name: Human-readable name for the agent run.
            metadata: Optional metadata to attach to the trace.

        Returns:
            A context manager that yields the trace and auto-finishes it.
        """
        trace = self.new_trace(name, metadata)
        return _TraceContext(self, trace)

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def new_span(
        self,
        name: str,
        trace: Optional[Trace] = None,
        parent_span: Optional[Span] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Create and attach a new span to a trace.

        Args:
            name: Human-readable name for the operation.
            trace: The trace to attach to; uses the active trace if None.
            parent_span: Optional parent span for nested operations.
            metadata: Optional metadata to attach to the span.

        Returns:
            The newly created Span.

        Raises:
            RuntimeError: If no trace is active and none is provided.
        """
        target_trace = trace or self._active_trace
        if target_trace is None:
            raise RuntimeError("No active trace. Call new_trace() or start_trace() first.")
        span = Span(
            name=name,
            parent_span_id=parent_span.span_id if parent_span else None,
            metadata=metadata or {},
        )
        target_trace.spans.append(span)
        return span

    def start_span(
        self,
        name: str,
        trace: Optional[Trace] = None,
        parent_span: Optional[Span] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "_SpanContext":
        """Start a span as a context manager.

        Args:
            name: Human-readable name for the operation.
            trace: The trace to attach to; uses the active trace if None.
            parent_span: Optional parent span for nested operations.
            metadata: Optional metadata to attach to the span.

        Returns:
            A context manager that yields the span.
        """
        span = self.new_span(name, trace, parent_span, metadata)
        return _SpanContext(span)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def traces(self) -> List[Trace]:
        """Return all recorded traces."""
        return list(self._traces)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Look up a trace by its ID.

        Args:
            trace_id: The UUID string of the target trace.

        Returns:
            The matching Trace, or None if not found.
        """
        for t in self._traces:
            if t.trace_id == trace_id:
                return t
        return None

    def clear(self) -> None:
        """Remove all recorded traces and reset active trace."""
        self._traces.clear()
        self._active_trace = None


class _TraceContext:
    """Internal context manager for traces."""

    def __init__(self, tracer: AgentTracer, trace: Trace) -> None:
        self._tracer = tracer
        self._trace = trace

    def __enter__(self) -> Trace:
        return self._trace

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._tracer.finish_trace(self._trace)


class _SpanContext:
    """Internal context manager for spans."""

    def __init__(self, span: Span) -> None:
        self._span = span

    def __enter__(self) -> Span:
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span.status == "running":
            error_msg = str(exc_val) if exc_val else None
            self._span.finish(error=error_msg)
