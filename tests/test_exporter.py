"""Tests for StdoutExporter, JSONFileExporter, and TraceExporter."""

import io
import json
import tempfile
from pathlib import Path

import pytest

from agent_observability.exporter import JSONFileExporter, StdoutExporter, TraceExporter
from agent_observability.tracer import AgentTracer, Span, Trace


def _make_trace(name: str = "test-trace") -> Trace:
    tracer = AgentTracer()
    trace = tracer.new_trace(name)
    span = tracer.new_span("op")
    span.finish(tokens_in=10, tokens_out=5)
    tracer.finish_trace(trace)
    return trace


class TestStdoutExporter:
    def test_export_one_writes_json(self):
        stream = io.StringIO()
        exp = StdoutExporter(stream=stream)
        trace = _make_trace()
        exp.export_one(trace)
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["name"] == "test-trace"

    def test_export_multiple(self):
        stream = io.StringIO()
        exp = StdoutExporter(stream=stream)
        t1 = _make_trace("run-1")
        t2 = _make_trace("run-2")
        exp.export([t1, t2])
        lines = stream.getvalue().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "run-1"
        assert json.loads(lines[1])["name"] == "run-2"

    def test_pretty_mode(self):
        stream = io.StringIO()
        exp = StdoutExporter(stream=stream, pretty=True)
        trace = _make_trace()
        exp.export_one(trace)
        output = stream.getvalue()
        # Pretty JSON has newlines inside the object
        assert "\n" in output
        parsed = json.loads(output)
        assert "name" in parsed


class TestJSONFileExporter:
    def test_export_append_mode(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        exp = JSONFileExporter(path, append=True)
        t1 = _make_trace("r1")
        t2 = _make_trace("r2")
        exp.export_one(t1)
        exp.export_one(t2)
        records = exp.read_all()
        assert len(records) == 2
        names = {r["name"] for r in records}
        assert names == {"r1", "r2"}

    def test_export_overwrite_mode(self, tmp_path):
        path = tmp_path / "traces.json"
        exp = JSONFileExporter(path, append=False)
        exp.export([_make_trace("first")])
        exp.export([_make_trace("second"), _make_trace("third")])
        records = exp.read_all()
        # Overwrite: only the last export batch remains
        assert len(records) == 2
        assert records[0]["name"] == "second"

    def test_read_all_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        exp = JSONFileExporter(path, append=True)
        assert exp.read_all() == []

    def test_read_all_missing_file(self, tmp_path):
        exp = JSONFileExporter(tmp_path / "missing.jsonl", append=True)
        assert exp.read_all() == []

    def test_path_created_automatically(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "traces.jsonl"
        exp = JSONFileExporter(nested)
        exp.export_one(_make_trace())
        assert nested.exists()


class TestTraceExporter:
    def test_default_exporter_is_stdout(self):
        exp = TraceExporter()
        assert len(exp.exporters) == 1
        assert isinstance(exp.exporters[0], StdoutExporter)

    def test_add_exporter(self):
        exp = TraceExporter([])
        exp.add_exporter(StdoutExporter())
        assert len(exp.exporters) == 1

    def test_fan_out_to_multiple_exporters(self):
        s1 = io.StringIO()
        s2 = io.StringIO()
        exp = TraceExporter([StdoutExporter(s1), StdoutExporter(s2)])
        trace = _make_trace()
        exp.export_one(trace)
        assert json.loads(s1.getvalue())["name"] == "test-trace"
        assert json.loads(s2.getvalue())["name"] == "test-trace"

    def test_export_list(self):
        stream = io.StringIO()
        exp = TraceExporter([StdoutExporter(stream)])
        exp.export([_make_trace("a"), _make_trace("b")])
        lines = stream.getvalue().strip().splitlines()
        assert len(lines) == 2
