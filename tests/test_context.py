"""Tests for ObservabilityContext."""

import io

import pytest

from agent_observability import ObservabilityContext
from agent_observability.exporter import StdoutExporter


class TestObservabilityContext:
    def _ctx(self, emit_logs: bool = False) -> ObservabilityContext:
        stream = io.StringIO()
        exp_stream = io.StringIO()
        return ObservabilityContext(
            name="test-agent",
            exporters=[StdoutExporter(exp_stream)],
            emit_logs=emit_logs,
            logger_stream=stream,
        )

    def test_name(self):
        ctx = self._ctx()
        assert ctx.name == "test-agent"

    def test_trace_context_manager(self):
        ctx = self._ctx()
        with ctx.trace("run-1") as trace:
            assert trace.name == "run-1"
        assert trace.end_time is not None

    def test_span_context_manager(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            with ctx.span("step") as span:
                assert span.status == "running"
            assert span.status == "ok"

    def test_log_call_stores_record(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            with ctx.span("llm") as span:
                ctx.log_call(
                    model="gpt-4o",
                    prompt="Hello",
                    latency_ms=100.0,
                    tokens_in=5,
                    tokens_out=20,
                    span=span,
                )
        assert len(ctx.logger.records) == 1
        assert ctx.logger.records[0].model == "gpt-4o"

    def test_log_call_populates_metrics(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            ctx.log_call(model="gpt-4o", prompt="Hi", latency_ms=80.0, tokens_in=4, tokens_out=10)
        assert len(ctx.metrics.get("latency_ms")) == 1
        assert len(ctx.metrics.get("tokens_in")) == 1
        assert len(ctx.metrics.get("tokens_out")) == 1

    def test_record_metric(self):
        ctx = self._ctx()
        ctx.record_metric("custom_score", 0.95, labels={"run": "1"})
        points = ctx.metrics.get("custom_score")
        assert len(points) == 1
        assert points[0].value == 0.95

    def test_summary_structure(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            ctx.log_call(model="m", prompt="p", latency_ms=50, success=True)
        s = ctx.summary()
        assert "name" in s
        assert "trace_count" in s
        assert "call_count" in s
        assert "success_rate" in s
        assert "total_cost_usd" in s
        assert "metrics" in s

    def test_summary_counts(self):
        ctx = self._ctx()
        with ctx.trace("r1"):
            ctx.log_call(model="m", prompt="p", latency_ms=10)
        with ctx.trace("r2"):
            ctx.log_call(model="m", prompt="q", latency_ms=20)
        s = ctx.summary()
        assert s["trace_count"] == 2
        assert s["call_count"] == 2

    def test_export_all_runs_without_error(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            with ctx.span("op") as span:
                span.finish(tokens_in=1, tokens_out=2)
        # Should not raise
        ctx.export_all()

    def test_clear_resets_all(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            ctx.log_call(model="m", prompt="p", latency_ms=10)
        ctx.clear()
        assert ctx.tracer.traces == []
        assert ctx.logger.records == []
        assert ctx.metrics.all_points == []

    def test_success_and_error_in_metrics(self):
        ctx = self._ctx()
        with ctx.trace("run"):
            ctx.log_call(model="m", prompt="p", latency_ms=10, success=True)
            ctx.log_call(model="m", prompt="q", latency_ms=20, success=False, error="fail")
        assert ctx.metrics.success_rate() == 0.5

    def test_accessors(self):
        ctx = self._ctx()
        from agent_observability.tracer import AgentTracer
        from agent_observability.logger import AgentLogger
        from agent_observability.metrics import MetricsCollector
        from agent_observability.exporter import TraceExporter
        assert isinstance(ctx.tracer, AgentTracer)
        assert isinstance(ctx.logger, AgentLogger)
        assert isinstance(ctx.metrics, MetricsCollector)
        assert isinstance(ctx.exporter, TraceExporter)
