"""Tests for SpanTracer and Span."""

import time
import pytest
from agent_observability.tracer import Span, SpanTracer


class TestSpan:
    def test_span_basic_creation(self):
        span = Span(name="tool_call")
        assert span.name == "tool_call"
        assert span.parent_id is None
        assert span.end_time is None
        assert span.error is None
        assert isinstance(span.span_id, str)
        assert len(span.span_id) == 36  # UUID4 format

    def test_span_end_records_time(self):
        span = Span(name="op")
        time.sleep(0.01)
        span.end()
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # at least 10ms

    def test_span_end_with_metadata(self):
        span = Span(name="search")
        span.end(metadata={"results": 3, "query": "test"})
        assert span.metadata["results"] == 3
        assert span.metadata["query"] == "test"

    def test_span_end_with_error(self):
        span = Span(name="failing_tool")
        span.end(error="Connection timeout")
        assert span.error == "Connection timeout"

    def test_span_duration_none_while_open(self):
        span = Span(name="open")
        assert span.duration_ms is None

    def test_span_to_dict_structure(self):
        span = Span(name="my_tool", metadata={"key": "val"})
        span.end()
        d = span.to_dict()
        assert d["name"] == "my_tool"
        assert d["metadata"]["key"] == "val"
        assert "span_id" in d
        assert "start_time" in d
        assert "end_time" in d
        assert "duration_ms" in d
        assert "children" in d
        assert isinstance(d["children"], list)

    def test_span_initial_metadata(self):
        span = Span(name="s", metadata={"foo": "bar"})
        assert span.metadata["foo"] == "bar"


class TestSpanTracer:
    def test_start_and_end_span(self):
        tracer = SpanTracer()
        sid = tracer.start_span("root")
        assert tracer.current_span_id() == sid
        span = tracer.end_span(sid)
        assert span is not None
        assert span.end_time is not None

    def test_nested_spans(self):
        tracer = SpanTracer()
        root_id = tracer.start_span("root")
        child_id = tracer.start_span("child")
        tracer.end_span(child_id)
        tracer.end_span(root_id)

        tree = tracer.to_dict()
        assert len(tree) == 1
        root_dict = tree[0]
        assert root_dict["name"] == "root"
        assert len(root_dict["children"]) == 1
        assert root_dict["children"][0]["name"] == "child"

    def test_explicit_parent_id(self):
        tracer = SpanTracer()
        root_id = tracer.start_span("root")
        # Start two children explicitly under root
        child1_id = tracer.start_span("child1", parent_id=root_id)
        tracer.end_span(child1_id)
        child2_id = tracer.start_span("child2", parent_id=root_id)
        tracer.end_span(child2_id)
        tracer.end_span(root_id)

        tree = tracer.to_dict()
        assert len(tree[0]["children"]) == 2

    def test_end_span_with_error(self):
        tracer = SpanTracer()
        sid = tracer.start_span("risky")
        span = tracer.end_span(sid, error="boom")
        assert span.error == "boom"

    def test_get_span(self):
        tracer = SpanTracer()
        sid = tracer.start_span("lookup")
        span = tracer.get_span(sid)
        assert span is not None
        assert span.name == "lookup"

    def test_end_nonexistent_span_returns_none(self):
        tracer = SpanTracer()
        result = tracer.end_span("fake-id-000")
        assert result is None

    def test_all_spans_flat(self):
        tracer = SpanTracer()
        root_id = tracer.start_span("root")
        child_id = tracer.start_span("child")
        tracer.end_span(child_id)
        tracer.end_span(root_id)
        flat = tracer.all_spans_flat()
        assert len(flat) == 2
        # Children in flat mode are just IDs
        for s in flat:
            assert isinstance(s["children"], list)

    def test_reset_clears_all(self):
        tracer = SpanTracer()
        tracer.start_span("a")
        tracer.reset()
        assert tracer.to_dict() == []
        assert tracer.current_span_id() is None

    def test_current_span_id_updates_on_stack(self):
        tracer = SpanTracer()
        assert tracer.current_span_id() is None
        s1 = tracer.start_span("s1")
        assert tracer.current_span_id() == s1
        s2 = tracer.start_span("s2")
        assert tracer.current_span_id() == s2
        tracer.end_span(s2)
        assert tracer.current_span_id() == s1
