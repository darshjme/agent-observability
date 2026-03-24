"""Tests for MetricsCollector, MetricPoint, and AggregatedMetric."""

import pytest

from agent_observability.metrics import MetricsCollector


class TestMetricsCollector:
    def test_record_and_get(self):
        mc = MetricsCollector()
        mc.record("latency_ms", 100.0)
        points = mc.get("latency_ms")
        assert len(points) == 1
        assert points[0].value == 100.0

    def test_get_label_filter(self):
        mc = MetricsCollector()
        mc.record("latency_ms", 100.0, labels={"model": "gpt-4o"})
        mc.record("latency_ms", 200.0, labels={"model": "claude-3"})
        gpt_points = mc.get("latency_ms", labels={"model": "gpt-4o"})
        assert len(gpt_points) == 1
        assert gpt_points[0].value == 100.0

    def test_aggregate_basic(self):
        mc = MetricsCollector()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            mc.record("score", v)
        agg = mc.aggregate("score")
        assert agg is not None
        assert agg.count == 5
        assert agg.total == 150.0
        assert agg.mean == 30.0
        assert agg.min == 10.0
        assert agg.max == 50.0

    def test_aggregate_returns_none_when_no_data(self):
        mc = MetricsCollector()
        assert mc.aggregate("nonexistent") is None

    def test_aggregate_single_point_no_stdev(self):
        mc = MetricsCollector()
        mc.record("x", 42.0)
        agg = mc.aggregate("x")
        assert agg.stdev is None
        assert agg.mean == 42.0

    def test_aggregate_p95_p99(self):
        mc = MetricsCollector()
        for i in range(1, 101):
            mc.record("val", float(i))
        agg = mc.aggregate("val")
        assert 94.0 <= agg.p95 <= 96.0
        assert 98.0 <= agg.p99 <= 100.0

    def test_success_rate_all_success(self):
        mc = MetricsCollector()
        mc.record_success()
        mc.record_success()
        assert mc.success_rate() == 1.0

    def test_success_rate_mixed(self):
        mc = MetricsCollector()
        mc.record_success()
        mc.record_error()
        assert mc.success_rate() == 0.5

    def test_success_rate_empty(self):
        mc = MetricsCollector()
        assert mc.success_rate() == 1.0

    def test_metric_names(self):
        mc = MetricsCollector()
        mc.record("alpha", 1.0)
        mc.record("beta", 2.0)
        mc.record("alpha", 3.0)
        assert mc.metric_names() == ["alpha", "beta"]

    def test_summary_keys(self):
        mc = MetricsCollector()
        mc.record("latency_ms", 50.0)
        mc.record("tokens_in", 10.0)
        summary = mc.summary()
        assert "latency_ms" in summary
        assert "tokens_in" in summary

    def test_clear(self):
        mc = MetricsCollector()
        mc.record("x", 1.0)
        mc.clear()
        assert mc.all_points == []

    def test_aggregate_to_dict(self):
        mc = MetricsCollector()
        mc.record("x", 1.0)
        mc.record("x", 3.0)
        agg = mc.aggregate("x")
        d = agg.to_dict()
        assert d["name"] == "x"
        assert "mean" in d
        assert "p95" in d
