"""Span tracing module for nested tool call observability.

Provides SpanTracer for creating and managing hierarchical spans
representing tool calls within an agent run.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


class Span:
    """Represents a single traced unit of work (e.g., a tool call).

    Spans can be nested: a parent span may contain child spans.
    Each span tracks timing, metadata, and outcome.

    Attributes:
        span_id: Unique identifier for this span.
        name: Human-readable name (e.g., tool name).
        parent_id: ID of the parent span, or None if root.
        start_time: Unix timestamp when span was started.
        end_time: Unix timestamp when span was ended (None if still open).
        metadata: Arbitrary key-value metadata attached to this span.
        error: Error message if span ended with failure, else None.
        children: List of child Span objects.

    Example:
        >>> span = Span(name="web_search", parent_id=None)
        >>> span.end(metadata={"results": 5})
        >>> d = span.to_dict()
        >>> assert d["name"] == "web_search"
    """

    def __init__(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a new Span.

        Args:
            name: Descriptive name for this span.
            parent_id: ID of the parent span, or None for root spans.
            metadata: Optional initial metadata dict.
        """
        self.span_id: str = str(uuid.uuid4())
        self.name: str = name
        self.parent_id: Optional[str] = parent_id
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = metadata or {}
        self.error: Optional[str] = None
        self.children: List["Span"] = []

    @property
    def duration_ms(self) -> Optional[float]:
        """Return span duration in milliseconds, or None if not yet ended.

        Returns:
            Duration in milliseconds as a float, or None.
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def end(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> "Span":
        """Close this span, recording end time and optional outcome.

        Args:
            metadata: Additional metadata to merge into this span.
            error: Error message if this span represents a failure.

        Returns:
            Self, for method chaining.
        """
        self.end_time = time.time()
        if metadata:
            self.metadata.update(metadata)
        if error:
            self.error = error
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this span and all children to a nested dict.

        Returns:
            Dict with keys: span_id, name, parent_id, start_time,
            end_time, duration_ms, metadata, error, children.
        """
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
            "children": [child.to_dict() for child in self.children],
        }


class SpanTracer:
    """Manages a tree of spans for a single agent run.

    SpanTracer maintains a stack of currently-open spans and
    a flat registry of all spans for efficient lookup.

    Example:
        >>> tracer = SpanTracer()
        >>> root_id = tracer.start_span("run_agent")
        >>> child_id = tracer.start_span("call_tool", parent_id=root_id)
        >>> tracer.end_span(child_id)
        >>> tracer.end_span(root_id)
        >>> spans = tracer.to_dict()
        >>> assert len(spans) == 1  # root with 1 child
    """

    def __init__(self) -> None:
        """Initialize an empty SpanTracer."""
        self._spans: Dict[str, Span] = {}  # span_id -> Span
        self._root_spans: List[Span] = []  # top-level spans (no parent)
        self._open_stack: List[str] = []  # stack of open span_ids

    def start_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create and start a new span.

        If parent_id is None and there are open spans, the current
        top-of-stack span is used as parent automatically.

        Args:
            name: Name of the span.
            parent_id: Explicit parent span ID, or None to auto-parent.
            metadata: Optional initial metadata.

        Returns:
            The new span's span_id string.
        """
        # Auto-parent to top of stack if no explicit parent given
        effective_parent_id = parent_id
        if effective_parent_id is None and self._open_stack:
            effective_parent_id = self._open_stack[-1]

        span = Span(name=name, parent_id=effective_parent_id, metadata=metadata)
        self._spans[span.span_id] = span

        if effective_parent_id is not None:
            parent = self._spans.get(effective_parent_id)
            if parent is not None:
                parent.children.append(span)
        else:
            self._root_spans.append(span)

        self._open_stack.append(span.span_id)
        return span.span_id

    def end_span(
        self,
        span_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Span]:
        """End a span by ID.

        Args:
            span_id: ID of the span to close.
            metadata: Additional metadata to attach on close.
            error: Error message if span ended in failure.

        Returns:
            The closed Span object, or None if span_id not found.
        """
        span = self._spans.get(span_id)
        if span is None:
            return None

        span.end(metadata=metadata, error=error)

        if span_id in self._open_stack:
            self._open_stack.remove(span_id)

        return span

    def get_span(self, span_id: str) -> Optional[Span]:
        """Retrieve a span by ID.

        Args:
            span_id: Span identifier.

        Returns:
            Span object or None.
        """
        return self._spans.get(span_id)

    def current_span_id(self) -> Optional[str]:
        """Return the ID of the currently-open top-of-stack span.

        Returns:
            span_id string or None if stack is empty.
        """
        return self._open_stack[-1] if self._open_stack else None

    def all_spans_flat(self) -> List[Dict[str, Any]]:
        """Return all spans as a flat list of dicts (no nesting).

        Returns:
            List of span dicts with same keys as Span.to_dict()
            but with children as a list of span_id strings only.
        """
        result = []
        for span in self._spans.values():
            d = span.to_dict()
            d["children"] = [c.span_id for c in span.children]
            result.append(d)
        return result

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return all root spans as a nested tree of dicts.

        Returns:
            List of root span dicts, each with nested children.
        """
        return [span.to_dict() for span in self._root_spans]

    def reset(self) -> None:
        """Clear all spans and reset state."""
        self._spans.clear()
        self._root_spans.clear()
        self._open_stack.clear()
