"""Tests for AgentTracer, Trace, and Span."""

import time

import pytest

from agent_observability.tracer import AgentTracer, Span, Trace


class TestSpan:
    def test_span_defaults(self):
        span = Span(name="test")
        assert span.name == "test"
        assert span.status == "running"
        assert span.end_time is None
        assert span.duration_ms is None
        assert span.tokens_in == 0
        assert span.tokens_out == 0

    def test_span_finish_ok(self):
        span = Span(name="test")
        time.sleep(0.01)
        span.finish(tokens_in=10, tokens_out=5)
        assert span.status == "ok"
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0
        assert span.tokens_in == 10
        assert span.tokens_out == 5

    def test_span_finish_error(self):
        span = Span(name="test")
        span.finish(error="something went wrong")
        assert span.status == "error"
        assert span.error == "something went wrong"

    def test_span_to_dict(self):
        span = Span(name="ops")
        span.finish(tokens_in=3, tokens_out=7)
        d = span.to_dict()
        assert d["name"] == "ops"
        assert d["tokens_in"] == 3
        assert d["tokens_out"] == 7
        assert "span_id" in d
        assert "duration_ms" in d


class TestTrace:
    def test_trace_defaults(self):
        trace = Trace(name="run")
        assert trace.name == "run"
        assert trace.spans == []
        assert trace.end_time is None
        assert trace.duration_ms is None

    def test_trace_token_totals(self):
        trace = Trace(name="run")
        s1 = Span(name="s1")
        s1.finish(tokens_in=10, tokens_out=5)
        s2 = Span(name="s2")
        s2.finish(tokens_in=20, tokens_out=15)
        trace.spans.extend([s1, s2])
        assert trace.total_tokens_in == 30
        assert trace.total_tokens_out == 20

    def test_trace_to_dict(self):
        trace = Trace(name="run")
        s = Span(name="s")
        s.finish()
        trace.spans.append(s)
        trace.end_time = time.time()
        d = trace.to_dict()
        assert d["name"] == "run"
        assert len(d["spans"]) == 1
        assert "trace_id" in d
        assert d["duration_ms"] is not None


class TestAgentTracer:
    def test_new_trace(self):
        tracer = AgentTracer()
        trace = tracer.new_trace("agent-run")
        assert trace.name == "agent-run"
        assert len(tracer.traces) == 1

    def test_finish_trace(self):
        tracer = AgentTracer()
        trace = tracer.new_trace("run")
        tracer.finish_trace(trace)
        assert trace.end_time is not None

    def test_start_trace_context_manager(self):
        tracer = AgentTracer()
        with tracer.start_trace("ctx-run") as trace:
            assert trace.name == "ctx-run"
            assert trace.end_time is None
        assert trace.end_time is not None

    def test_new_span_attached_to_active_trace(self):
        tracer = AgentTracer()
        trace = tracer.new_trace("run")
        span = tracer.new_span("step-1")
        assert span in trace.spans

    def test_new_span_no_active_trace_raises(self):
        tracer = AgentTracer()
        with pytest.raises(RuntimeError):
            tracer.new_span("orphan")

    def test_start_span_context_manager(self):
        tracer = AgentTracer()
        with tracer.start_trace("run"):
            with tracer.start_span("llm-call") as span:
                assert span.status == "running"
            assert span.status == "ok"

    def test_span_parent(self):
        tracer = AgentTracer()
        with tracer.start_trace("run"):
            parent = tracer.new_span("parent")
            child = tracer.new_span("child", parent_span=parent)
            assert child.parent_span_id == parent.span_id

    def test_get_trace_by_id(self):
        tracer = AgentTracer()
        trace = tracer.new_trace("find-me")
        found = tracer.get_trace(trace.trace_id)
        assert found is trace
        assert tracer.get_trace("nonexistent") is None

    def test_clear(self):
        tracer = AgentTracer()
        tracer.new_trace("r1")
        tracer.new_trace("r2")
        tracer.clear()
        assert tracer.traces == []

    def test_span_exception_sets_error(self):
        tracer = AgentTracer()
        with tracer.start_trace("run"):
            with pytest.raises(ValueError):
                with tracer.start_span("bad-op") as span:
                    raise ValueError("oops")
        assert span.status == "error"
        assert "oops" in span.error
