"""Tests for AgentLogger and LLMCallRecord."""

import io
import json

import pytest

from agent_observability.logger import AgentLogger, LLMCallRecord, hash_prompt


class TestHashPrompt:
    def test_deterministic(self):
        h1 = hash_prompt("hello world")
        h2 = hash_prompt("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        assert hash_prompt("a") != hash_prompt("b")

    def test_length_16(self):
        h = hash_prompt("test prompt")
        assert len(h) == 16

    def test_hex_chars_only(self):
        h = hash_prompt("test")
        assert all(c in "0123456789abcdef" for c in h)


class TestLLMCallRecord:
    def test_to_dict(self):
        rec = LLMCallRecord(model="gpt-4o", prompt_hash="abc123", latency_ms=100.0)
        d = rec.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["latency_ms"] == 100.0

    def test_to_json(self):
        rec = LLMCallRecord(model="claude-3", prompt_hash="xyz")
        j = rec.to_json()
        parsed = json.loads(j)
        assert parsed["model"] == "claude-3"


class TestAgentLogger:
    def _make_logger(self) -> tuple[AgentLogger, io.StringIO]:
        stream = io.StringIO()
        logger = AgentLogger(name="test-logger", emit_json=True, stream=stream)
        return logger, stream

    def test_log_call_stores_record(self):
        logger, _ = self._make_logger()
        logger.log_call(model="gpt-4o", prompt="Hi", latency_ms=50.0)
        assert len(logger.records) == 1
        assert logger.records[0].model == "gpt-4o"

    def test_log_call_hashes_prompt(self):
        logger, _ = self._make_logger()
        logger.log_call(model="gpt-4o", prompt="secret prompt", latency_ms=10.0)
        assert logger.records[0].prompt_hash == hash_prompt("secret prompt")
        assert "secret prompt" not in logger.records[0].prompt_hash

    def test_log_call_emits_json(self):
        logger, stream = self._make_logger()
        logger.log_call(model="gpt-4o", prompt="Hello", latency_ms=75.0)
        output = stream.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["model"] == "gpt-4o"
        assert parsed["logger"] == "test-logger"

    def test_log_llm_call_stores_record(self):
        logger, _ = self._make_logger()
        logger.log_llm_call(model="claude-3", prompt_hash="abc123", latency_ms=200.0, tokens_in=5, tokens_out=20)
        assert len(logger.records) == 1
        assert logger.records[0].prompt_hash == "abc123"

    def test_total_cost(self):
        logger, _ = self._make_logger()
        logger.log_call(model="m", prompt="p", latency_ms=1, cost_usd=0.01)
        logger.log_call(model="m", prompt="p", latency_ms=1, cost_usd=0.02)
        assert abs(logger.total_cost() - 0.03) < 1e-9

    def test_success_rate_all_success(self):
        logger, _ = self._make_logger()
        logger.log_call(model="m", prompt="p", latency_ms=1, success=True)
        logger.log_call(model="m", prompt="p", latency_ms=1, success=True)
        assert logger.success_rate() == 1.0

    def test_success_rate_mixed(self):
        logger, _ = self._make_logger()
        logger.log_call(model="m", prompt="p", latency_ms=1, success=True)
        logger.log_call(model="m", prompt="p", latency_ms=1, success=False, error="timeout")
        assert logger.success_rate() == 0.5

    def test_success_rate_empty(self):
        logger, _ = self._make_logger()
        assert logger.success_rate() == 1.0

    def test_clear(self):
        logger, _ = self._make_logger()
        logger.log_call(model="m", prompt="p", latency_ms=1)
        logger.clear()
        assert logger.records == []

    def test_trace_span_ids_stored(self):
        logger, _ = self._make_logger()
        logger.log_call(model="m", prompt="p", latency_ms=1, trace_id="t1", span_id="s1")
        rec = logger.records[0]
        assert rec.trace_id == "t1"
        assert rec.span_id == "s1"
