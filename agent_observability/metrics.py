"""MetricsCollector — collect and aggregate agent metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class MetricPoint:
    """A single data point for a named metric.

    Attributes:
        name: Metric name.
        value: Numeric value.
        timestamp: Unix epoch when this point was recorded.
        labels: Arbitrary key-value labels for filtering/grouping.
    """

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this metric point to a dictionary.

        Returns:
            Dictionary representation of this metric point.
        """
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class AggregatedMetric:
    """Aggregated statistics for a set of metric points.

    Attributes:
        name: Metric name.
        count: Number of data points.
        total: Sum of all values.
        mean: Arithmetic mean.
        median: Median value.
        min: Minimum value.
        max: Maximum value.
        stdev: Standard deviation (None if fewer than 2 points).
        p95: 95th-percentile value.
        p99: 99th-percentile value.
    """

    name: str
    count: int
    total: float
    mean: float
    median: float
    min: float
    max: float
    stdev: Optional[float]
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the aggregated metric to a dictionary.

        Returns:
            Dictionary representation of this aggregated metric.
        """
        return {
            "name": self.name,
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "stdev": self.stdev,
            "p95": self.p95,
            "p99": self.p99,
        }


def _percentile(data: Sequence[float], p: float) -> float:
    """Compute the p-th percentile of a sorted sequence.

    Args:
        data: A sorted sequence of numeric values.
        p: Percentile in the range [0, 100].

    Returns:
        The computed percentile value.
    """
    if not data:
        return 0.0
    idx = (p / 100) * (len(data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(data) - 1)
    fraction = idx - lower
    return data[lower] + fraction * (data[upper] - data[lower])


class MetricsCollector:
    """Collects and aggregates numeric metrics for an agent system.

    Supports recording arbitrary named metrics and provides summary
    statistics including percentiles, mean, median, and standard deviation.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("latency_ms", 120.5, labels={"model": "gpt-4o"})
        >>> collector.record("latency_ms", 85.3)
        >>> summary = collector.aggregate("latency_ms")
        >>> print(summary.mean)
    """

    # Built-in metric name constants
    LATENCY_MS = "latency_ms"
    TOKENS_IN = "tokens_in"
    TOKENS_OUT = "tokens_out"
    COST_USD = "cost_usd"
    SUCCESS = "success"
    ERROR = "error"

    def __init__(self) -> None:
        """Initialize the MetricsCollector with an empty metrics store."""
        self._points: List[MetricPoint] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> MetricPoint:
        """Record a single metric data point.

        Args:
            name: Metric name (e.g. "latency_ms").
            value: Numeric measurement value.
            labels: Optional key-value labels for filtering.
            timestamp: Unix epoch; defaults to current time.

        Returns:
            The newly created :class:`MetricPoint`.
        """
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp or time.time(),
            labels=labels or {},
        )
        self._points.append(point)
        return point

    def record_success(self, labels: Optional[Dict[str, str]] = None) -> MetricPoint:
        """Convenience: record a successful agent call (value=1.0).

        Args:
            labels: Optional key-value labels.

        Returns:
            The newly created :class:`MetricPoint`.
        """
        return self.record(self.SUCCESS, 1.0, labels=labels)

    def record_error(self, labels: Optional[Dict[str, str]] = None) -> MetricPoint:
        """Convenience: record a failed agent call (value=1.0 under "error").

        Args:
            labels: Optional key-value labels.

        Returns:
            The newly created :class:`MetricPoint`.
        """
        return self.record(self.ERROR, 1.0, labels=labels)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """Return all points matching a metric name and optional label filter.

        Args:
            name: Metric name to filter by.
            labels: If provided, only return points whose labels are a
                superset of these key-value pairs.

        Returns:
            Filtered list of :class:`MetricPoint` objects.
        """
        result = [p for p in self._points if p.name == name]
        if labels:
            result = [p for p in result if all(p.labels.get(k) == v for k, v in labels.items())]
        return result

    def aggregate(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[AggregatedMetric]:
        """Compute summary statistics for a metric.

        Args:
            name: Metric name to aggregate.
            labels: Optional label filter (see :meth:`get`).

        Returns:
            An :class:`AggregatedMetric` or None if no points match.
        """
        points = self.get(name, labels)
        if not points:
            return None
        values = sorted(p.value for p in points)
        return AggregatedMetric(
            name=name,
            count=len(values),
            total=sum(values),
            mean=mean(values),
            median=median(values),
            min=values[0],
            max=values[-1],
            stdev=stdev(values) if len(values) >= 2 else None,
            p95=_percentile(values, 95),
            p99=_percentile(values, 99),
        )

    def success_rate(self) -> float:
        """Compute the overall success rate across all recorded calls.

        Considers "success" points as successes and "error" points as
        failures. Returns 1.0 if no relevant points exist.

        Returns:
            Success rate in [0.0, 1.0].
        """
        successes = len(self.get(self.SUCCESS))
        errors = len(self.get(self.ERROR))
        total = successes + errors
        return (successes / total) if total > 0 else 1.0

    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary dict across all recorded metrics.

        Returns:
            Dictionary mapping each metric name to its aggregated stats.
        """
        names = {p.name for p in self._points}
        return {name: (agg.to_dict() if (agg := self.aggregate(name)) else {}) for name in sorted(names)}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def all_points(self) -> List[MetricPoint]:
        """Return a snapshot of all recorded metric points."""
        return list(self._points)

    def metric_names(self) -> List[str]:
        """Return a sorted list of unique metric names recorded so far.

        Returns:
            Sorted list of metric name strings.
        """
        return sorted({p.name for p in self._points})

    def clear(self) -> None:
        """Remove all recorded metric points."""
        self._points.clear()
