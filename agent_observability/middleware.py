"""Observability middleware for wrapping LLM agent runs.

Provides ObservabilityMiddleware for capturing run-level metrics
including timing, token usage, cost, steps, and success/failure state.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from typing import Any, Callable, Dict, Generator, Optional

from agent_observability.cost import CostTracker
from agent_observability.tracer import SpanTracer


class RunContext:
    """Mutable context object for an active agent run.

    Passed to caller code during a traced run to allow incremental
    recording of tokens, steps, and metadata.

    Attributes:
        run_id: Unique identifier for this run.
        model_id: Model being used in this run.
        start_time: Unix timestamp when run began.
        end_time: Unix timestamp when run ended (None while running).
        tokens_in: Accumulated input tokens.
        tokens_out: Accumulated output tokens.
        cost_usd: Accumulated cost in USD.
        steps: Number of agent steps (tool calls, iterations, etc.).
        success: Whether the run completed successfully.
        error_msg: Error message if run failed.
        metadata: Arbitrary extra metadata.
        tracer: SpanTracer instance for this run.
        _cost_tracker: CostTracker used for cost calculation.

    Example:
        >>> ctx = RunContext(model_id="gpt-4o", cost_tracker=CostTracker())
        >>> ctx.record_tokens(100, 50)
        >>> ctx.add_step()
        >>> assert ctx.steps == 1
    """

    def __init__(self, model_id: str, cost_tracker: CostTracker) -> None:
        """Initialize a new RunContext.

        Args:
            model_id: Identifier of the model used in this run.
            cost_tracker: CostTracker to use for cost computation.
        """
        self.run_id: str = str(uuid.uuid4())
        self.model_id: str = model_id
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.cost_usd: float = 0.0
        self.steps: int = 0
        self.success: bool = True
        self.error_msg: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.tracer: SpanTracer = SpanTracer()
        self._cost_tracker: CostTracker = cost_tracker

    def record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage and compute incremental cost.

        Args:
            tokens_in: Number of input tokens to add.
            tokens_out: Number of output tokens to add.
        """
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        try:
            cost = self._cost_tracker.compute_cost(
                self.model_id, tokens_in, tokens_out, record=False
            )
            self.cost_usd += cost
        except ValueError:
            # Unknown model — cost stays at 0
            pass

    def add_step(self, count: int = 1) -> None:
        """Increment the step counter.

        Args:
            count: Number of steps to add (default 1).
        """
        self.steps += count

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata key-value pair.

        Args:
            key: Metadata key.
            value: Metadata value (must be JSON-serializable for export).
        """
        self.metadata[key] = value

    def fail(self, error_msg: str) -> None:
        """Mark this run as failed with an error message.

        Args:
            error_msg: Human-readable error description.
        """
        self.success = False
        self.error_msg = error_msg

    @property
    def duration_ms(self) -> Optional[float]:
        """Return run duration in milliseconds, or None if still running.

        Returns:
            Float milliseconds or None.
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the run context to a dict.

        Returns:
            Dict representation of the run including all metrics,
            metadata, and nested span tree.
        """
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "steps": self.steps,
            "success": self.success,
            "error_msg": self.error_msg,
            "metadata": self.metadata,
            "spans": self.tracer.to_dict(),
        }


class ObservabilityMiddleware:
    """Middleware that wraps agent runs for full observability.

    Captures timing, token usage, cost, step counts, and span trees
    for every run. Supports optional lifecycle callbacks.

    Attributes:
        model_id: Default model ID for runs.
        cost_tracker: CostTracker instance used across runs.
        _on_run_complete: Optional callback invoked on run completion.
        _runs: List of all completed RunContext objects.

    Example:
        >>> middleware = ObservabilityMiddleware(model_id="gpt-4o")
        >>> with middleware.trace_run() as run:
        ...     run.record_tokens(500, 200)
        ...     run.add_step()
        >>> print(middleware.runs[-1].cost_usd)
    """

    def __init__(
        self,
        model_id: str,
        cost_tracker: Optional[CostTracker] = None,
        on_run_complete: Optional[Callable[[RunContext], None]] = None,
    ) -> None:
        """Initialize ObservabilityMiddleware.

        Args:
            model_id: Default model identifier for new runs.
            cost_tracker: Optional shared CostTracker instance.
                Creates a new one if not provided.
            on_run_complete: Optional callback invoked after each run
                completes (success or failure). Receives RunContext.
        """
        self.model_id = model_id
        self.cost_tracker = cost_tracker or CostTracker()
        self._on_run_complete = on_run_complete
        self._runs: list[RunContext] = []

    @property
    def runs(self) -> list[RunContext]:
        """Return all completed run contexts (read-only view).

        Returns:
            List of RunContext objects from all completed runs.
        """
        return list(self._runs)

    @contextlib.contextmanager
    def trace_run(
        self,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[RunContext, None, None]:
        """Context manager to trace a single agent run.

        Automatically records start/end times and handles exceptions
        by marking the run as failed.

        Args:
            model_id: Override the default model_id for this run.
            metadata: Initial metadata to attach to the run.

        Yields:
            RunContext: Mutable context for the active run.

        Example:
            >>> mw = ObservabilityMiddleware(model_id="gpt-4o")
            >>> with mw.trace_run() as run:
            ...     run.record_tokens(100, 50)
            ...     run.add_step()
        """
        effective_model = model_id or self.model_id
        ctx = RunContext(model_id=effective_model, cost_tracker=self.cost_tracker)
        if metadata:
            ctx.metadata.update(metadata)

        try:
            yield ctx
        except Exception as exc:
            ctx.fail(str(exc))
            raise
        finally:
            ctx.end_time = time.time()
            self._runs.append(ctx)
            if self._on_run_complete is not None:
                try:
                    self._on_run_complete(ctx)
                except Exception:
                    pass  # Callbacks must not crash the run

    def wrap(
        self,
        fn: Callable[..., Any],
        model_id: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Decorator/wrapper that traces a callable as an agent run.

        Args:
            fn: Callable to wrap. Will be called with its original
                arguments; RunContext is not passed.
            model_id: Optional model override.

        Returns:
            Wrapped callable with observability tracing.

        Example:
            >>> mw = ObservabilityMiddleware(model_id="gpt-4o")
            >>> def my_agent(): return "done"
            >>> traced = mw.wrap(my_agent)
            >>> result = traced()
            >>> assert mw.runs[-1].success
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.trace_run(model_id=model_id):
                return fn(*args, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper

    def total_cost(self) -> float:
        """Return total cost across all completed runs.

        Returns:
            Sum of cost_usd from all runs in USD.
        """
        return sum(r.cost_usd for r in self._runs)

    def stats(self) -> Dict[str, Any]:
        """Return aggregate stats across all completed runs.

        Returns:
            Dict with keys: total_runs, successful_runs, failed_runs,
            total_tokens_in, total_tokens_out, total_cost_usd,
            total_steps, avg_duration_ms.
        """
        runs = self._runs
        if not runs:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost_usd": 0.0,
                "total_steps": 0,
                "avg_duration_ms": 0.0,
            }

        durations = [r.duration_ms for r in runs if r.duration_ms is not None]
        return {
            "total_runs": len(runs),
            "successful_runs": sum(1 for r in runs if r.success),
            "failed_runs": sum(1 for r in runs if not r.success),
            "total_tokens_in": sum(r.tokens_in for r in runs),
            "total_tokens_out": sum(r.tokens_out for r in runs),
            "total_cost_usd": self.total_cost(),
            "total_steps": sum(r.steps for r in runs),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
        }
