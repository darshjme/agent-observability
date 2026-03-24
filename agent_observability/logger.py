"""AgentLogger — structured JSON logging for LLM calls."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, IO, List, Optional


def _hash_prompt(prompt: str) -> str:
    """Compute a deterministic SHA-256 hex digest of a prompt string.

    Args:
        prompt: The prompt text to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


@dataclass
class LLMCallRecord:
    """Immutable record of a single LLM call.

    Attributes:
        timestamp: Unix epoch when the call was made.
        model: Model identifier (e.g. "gpt-4o").
        prompt_hash: Truncated SHA-256 of the prompt (privacy-preserving).
        latency_ms: Round-trip latency in milliseconds.
        tokens_in: Input (prompt) token count.
        tokens_out: Output (completion) token count.
        cost_usd: Estimated cost in US dollars.
        success: Whether the call returned without error.
        error: Error message if the call failed.
        metadata: Arbitrary extra fields.
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for correlation.
    """

    timestamp: float = field(default_factory=time.time)
    model: str = ""
    prompt_hash: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record to a plain dictionary.

        Returns:
            Dictionary representation of this call record.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the record to a compact JSON string.

        Returns:
            JSON string representation of this call record.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"))


class AgentLogger:
    """Structured JSON logger for LLM calls made by an agent.

    Each call to :meth:`log_call` or :meth:`log_llm_call` appends an
    :class:`LLMCallRecord` to an in-memory list and optionally emits it to
    the configured output stream as a newline-delimited JSON record.

    Args:
        name: Logger name, used as the ``logger`` field in JSON output.
        level: Python logging level (default ``logging.INFO``).
        stream: Output stream for JSON records (default ``sys.stdout``).
        emit_json: When True, write JSON lines to *stream* on each log call.

    Example:
        >>> logger = AgentLogger("my-agent")
        >>> logger.log_call(
        ...     model="gpt-4o",
        ...     prompt="Tell me a joke",
        ...     latency_ms=312.5,
        ...     tokens_in=10,
        ...     tokens_out=40,
        ... )
    """

    def __init__(
        self,
        name: str = "agent",
        level: int = logging.INFO,
        stream: Optional[IO[str]] = None,
        emit_json: bool = True,
    ) -> None:
        """Initialize the AgentLogger.

        Args:
            name: Logger name used in JSON output.
            level: Python logging level threshold.
            stream: Output stream; defaults to sys.stdout.
            emit_json: Whether to write JSON lines to the stream.
        """
        self._name = name
        self._level = level
        self._stream: IO[str] = stream if stream is not None else sys.stdout
        self._emit_json = emit_json
        self._records: List[LLMCallRecord] = []

        # Set up stdlib logger for human-readable side-channel
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_call(
        self,
        model: str,
        prompt: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> LLMCallRecord:
        """Log a single LLM call, hashing the prompt for privacy.

        Args:
            model: Model identifier string.
            prompt: Raw prompt text (will be hashed before storage).
            latency_ms: Round-trip latency in milliseconds.
            tokens_in: Input token count.
            tokens_out: Output token count.
            cost_usd: Estimated cost in US dollars.
            success: Whether the call succeeded.
            error: Error message if the call failed.
            metadata: Arbitrary extra data to include in the record.
            trace_id: Optional trace ID for correlation with tracer.
            span_id: Optional span ID for correlation with tracer.

        Returns:
            The newly created :class:`LLMCallRecord`.
        """
        record = LLMCallRecord(
            model=model,
            prompt_hash=_hash_prompt(prompt),
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            success=success,
            error=error,
            metadata=metadata or {},
            trace_id=trace_id,
            span_id=span_id,
        )
        return self._store_and_emit(record)

    def log_llm_call(
        self,
        model: str,
        prompt_hash: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> LLMCallRecord:
        """Log a call when you already have the prompt hash (no raw prompt).

        Args:
            model: Model identifier string.
            prompt_hash: Pre-computed prompt hash.
            latency_ms: Round-trip latency in milliseconds.
            tokens_in: Input token count.
            tokens_out: Output token count.
            cost_usd: Estimated cost in US dollars.
            success: Whether the call succeeded.
            error: Error message if the call failed.
            metadata: Arbitrary extra data to include in the record.
            trace_id: Optional trace ID for correlation.
            span_id: Optional span ID for correlation.

        Returns:
            The newly created :class:`LLMCallRecord`.
        """
        record = LLMCallRecord(
            model=model,
            prompt_hash=prompt_hash,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            success=success,
            error=error,
            metadata=metadata or {},
            trace_id=trace_id,
            span_id=span_id,
        )
        return self._store_and_emit(record)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[LLMCallRecord]:
        """Return a snapshot of all stored call records."""
        return list(self._records)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    def total_cost(self) -> float:
        """Compute the total estimated cost across all logged calls.

        Returns:
            Sum of ``cost_usd`` for all records.
        """
        return sum(r.cost_usd for r in self._records)

    def success_rate(self) -> float:
        """Compute the fraction of successful calls.

        Returns:
            Success rate in [0.0, 1.0], or 1.0 if no records exist.
        """
        if not self._records:
            return 1.0
        return sum(1 for r in self._records if r.success) / len(self._records)

    def clear(self) -> None:
        """Remove all stored records."""
        self._records.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store_and_emit(self, record: LLMCallRecord) -> LLMCallRecord:
        self._records.append(record)
        if self._emit_json:
            payload = {**record.to_dict(), "logger": self._name}
            self._stream.write(json.dumps(payload) + "\n")
            self._stream.flush()
        self._logger.debug("LLM call logged: model=%s latency=%.1fms", record.model, record.latency_ms)
        return record


def hash_prompt(prompt: str) -> str:
    """Public alias for the prompt-hashing utility.

    Args:
        prompt: The prompt text to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    return _hash_prompt(prompt)
