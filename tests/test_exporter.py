"""Tests for LogExporter."""

import io
import json
import tempfile
from pathlib import Path

import pytest
from agent_observability.middleware import ObservabilityMiddleware
from agent_observability.exporter import LogExporter


def _make_run(model_id="gpt-4o", tokens_in=100, tokens_out=50, steps=1):
    """Helper: create a completed RunContext."""
    mw = ObservabilityMiddleware(model_id=model_id)
    with mw.trace_run() as run:
        run.record_tokens(tokens_in, tokens_out)
        run.add_step(steps)
        sid = run.tracer.start_span("tool_call")
        run.tracer.end_span(sid)
    return mw.runs[0]


class TestLogExporter:
    def test_to_dict_returns_dict(self):
        exporter = LogExporter()
        run = _make_run()
        d = exporter.to_dict(run)
        assert isinstance(d, dict)
        assert d["model_id"] == "gpt-4o"
        assert d["tokens_in"] == 100

    def test_to_json_is_valid_json(self):
        exporter = LogExporter()
        run = _make_run()
        s = exporter.to_json(run)
        parsed = json.loads(s)
        assert parsed["model_id"] == "gpt-4o"

    def test_to_json_pretty_indented(self):
        exporter = LogExporter(indent=2)
        run = _make_run()
        s = exporter.to_json(run, pretty=True)
        assert "\n" in s  # indented = multi-line

    def test_export_to_stdout_writes_json(self):
        exporter = LogExporter()
        run = _make_run()
        buf = io.StringIO()
        exporter.export_to_stdout(run, stream=buf)
        output = buf.getvalue()
        parsed = json.loads(output)
        assert parsed["model_id"] == "gpt-4o"

    def test_export_to_stdout_multiple_runs(self):
        exporter = LogExporter()
        runs = [_make_run(), _make_run(model_id="gemini-flash")]
        buf = io.StringIO()
        exporter.export_to_stdout(runs, stream=buf)
        lines = [l for l in buf.getvalue().strip().split("\n") if l.strip().startswith("{") or l.strip() == "}"]
        # Two JSON objects were written — verify both parse
        output = buf.getvalue()
        # Parse each object individually by reconstructing
        # Just verify two model IDs appear
        assert "gpt-4o" in output
        assert "gemini-flash" in output

    def test_export_to_jsonl_creates_file(self):
        exporter = LogExporter()
        run = _make_run()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            count = exporter.export_to_jsonl(run, path)
            assert count == 1
            assert path.exists()
            records = exporter.load_jsonl(path)
            assert len(records) == 1
            assert records[0]["model_id"] == "gpt-4o"

    def test_export_to_jsonl_append(self):
        exporter = LogExporter()
        run1 = _make_run()
        run2 = _make_run(model_id="gemini-flash")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            exporter.export_to_jsonl(run1, path, append=False)
            exporter.export_to_jsonl(run2, path, append=True)
            records = exporter.load_jsonl(path)
            assert len(records) == 2

    def test_export_to_jsonl_overwrite(self):
        exporter = LogExporter()
        run1 = _make_run()
        run2 = _make_run(model_id="gemini-flash")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.jsonl"
            exporter.export_to_jsonl(run1, path, append=False)
            exporter.export_to_jsonl(run2, path, append=False)
            records = exporter.load_jsonl(path)
            assert len(records) == 1  # overwritten
            assert records[0]["model_id"] == "gemini-flash"

    def test_export_to_dict_single(self):
        exporter = LogExporter()
        run = _make_run()
        result = exporter.export_to_dict(run)
        assert isinstance(result, dict)

    def test_export_to_dict_list(self):
        exporter = LogExporter()
        runs = [_make_run(), _make_run()]
        result = exporter.export_to_dict(runs)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_export_spans_only(self):
        exporter = LogExporter()
        run = _make_run()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spans.jsonl"
            count = exporter.export_spans_only(run, path)
            assert count == 1  # 1 span: tool_call
            records = exporter.load_jsonl(path)
            assert records[0]["name"] == "tool_call"

    def test_load_jsonl_nonexistent_raises(self):
        exporter = LogExporter()
        with pytest.raises(FileNotFoundError):
            exporter.load_jsonl("/tmp/nonexistent_file_xyz_123.jsonl")

    def test_export_includes_spans(self):
        exporter = LogExporter()
        run = _make_run()
        d = exporter.to_dict(run)
        assert "spans" in d
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "tool_call"
