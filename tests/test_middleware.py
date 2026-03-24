"""Tests for ObservabilityMiddleware and RunContext."""

import pytest
from agent_observability.middleware import ObservabilityMiddleware, RunContext
from agent_observability.cost import CostTracker


class TestRunContext:
    def _make_ctx(self):
        return RunContext(model_id="gpt-4o", cost_tracker=CostTracker())

    def test_initial_state(self):
        ctx = self._make_ctx()
        assert ctx.tokens_in == 0
        assert ctx.tokens_out == 0
        assert ctx.cost_usd == 0.0
        assert ctx.steps == 0
        assert ctx.success is True
        assert ctx.error_msg is None
        assert ctx.end_time is None

    def test_record_tokens_accumulates(self):
        ctx = self._make_ctx()
        ctx.record_tokens(100, 50)
        ctx.record_tokens(200, 100)
        assert ctx.tokens_in == 300
        assert ctx.tokens_out == 150

    def test_record_tokens_computes_cost(self):
        ctx = self._make_ctx()
        ctx.record_tokens(1000, 1000)
        # gpt-4o: $0.005 in + $0.015 out = $0.020
        assert abs(ctx.cost_usd - 0.020) < 1e-9

    def test_add_step(self):
        ctx = self._make_ctx()
        ctx.add_step()
        ctx.add_step(3)
        assert ctx.steps == 4

    def test_fail_marks_run(self):
        ctx = self._make_ctx()
        ctx.fail("something went wrong")
        assert ctx.success is False
        assert ctx.error_msg == "something went wrong"

    def test_set_metadata(self):
        ctx = self._make_ctx()
        ctx.set_metadata("session", "abc123")
        assert ctx.metadata["session"] == "abc123"

    def test_duration_ms_none_while_running(self):
        ctx = self._make_ctx()
        assert ctx.duration_ms is None

    def test_to_dict_keys(self):
        ctx = self._make_ctx()
        import time
        ctx.end_time = time.time()
        d = ctx.to_dict()
        for key in ["run_id", "model_id", "start_time", "end_time", "duration_ms",
                    "tokens_in", "tokens_out", "cost_usd", "steps", "success",
                    "error_msg", "metadata", "spans"]:
            assert key in d

    def test_unknown_model_cost_is_zero(self):
        ctx = RunContext(model_id="unknown-xyz", cost_tracker=CostTracker())
        ctx.record_tokens(1000, 1000)
        # Unknown model: cost stays 0 (no crash)
        assert ctx.cost_usd == 0.0


class TestObservabilityMiddleware:
    def test_trace_run_success(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with mw.trace_run() as run:
            run.record_tokens(100, 50)
            run.add_step()
        assert len(mw.runs) == 1
        r = mw.runs[0]
        assert r.success is True
        assert r.tokens_in == 100
        assert r.steps == 1
        assert r.end_time is not None

    def test_trace_run_exception_marks_failure(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with pytest.raises(RuntimeError):
            with mw.trace_run() as run:
                raise RuntimeError("agent failed")
        assert mw.runs[0].success is False
        assert "agent failed" in mw.runs[0].error_msg

    def test_multiple_runs_accumulate(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        for _ in range(3):
            with mw.trace_run() as run:
                run.record_tokens(100, 50)
        assert len(mw.runs) == 3

    def test_model_override(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with mw.trace_run(model_id="gemini-flash") as run:
            pass
        assert mw.runs[0].model_id == "gemini-flash"

    def test_on_run_complete_callback(self):
        called_with = []
        def cb(ctx):
            called_with.append(ctx.run_id)

        mw = ObservabilityMiddleware(model_id="gpt-4o", on_run_complete=cb)
        with mw.trace_run() as run:
            pass
        assert len(called_with) == 1
        assert called_with[0] == mw.runs[0].run_id

    def test_stats_empty(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        s = mw.stats()
        assert s["total_runs"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_stats_after_runs(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with mw.trace_run() as run:
            run.record_tokens(1000, 500)
            run.add_step(2)
        with pytest.raises(ValueError):
            with mw.trace_run() as run:
                raise ValueError("oops")
        s = mw.stats()
        assert s["total_runs"] == 2
        assert s["successful_runs"] == 1
        assert s["failed_runs"] == 1
        assert s["total_tokens_in"] == 1000
        assert s["total_steps"] == 2

    def test_wrap_decorator(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")

        def agent():
            return "result"

        traced = mw.wrap(agent)
        result = traced()
        assert result == "result"
        assert len(mw.runs) == 1
        assert mw.runs[0].success is True

    def test_total_cost_across_runs(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with mw.trace_run() as run:
            run.record_tokens(1000, 0)  # $0.005
        with mw.trace_run() as run:
            run.record_tokens(1000, 0)  # $0.005
        assert abs(mw.total_cost() - 0.010) < 1e-9

    def test_run_has_tracer(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with mw.trace_run() as run:
            sid = run.tracer.start_span("tool_call")
            run.tracer.end_span(sid)
        spans = mw.runs[0].tracer.to_dict()
        assert len(spans) == 1
        assert spans[0]["name"] == "tool_call"

    def test_metadata_initial(self):
        mw = ObservabilityMiddleware(model_id="gpt-4o")
        with mw.trace_run(metadata={"user": "alice"}) as run:
            pass
        assert mw.runs[0].metadata["user"] == "alice"
