"""TraceExporter — export traces to JSON file or stdout."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Union

from .tracer import Trace


class BaseExporter(ABC):
    """Abstract base class for trace exporters.

    Subclass this to add support for additional backends such as
    OpenTelemetry, Jaeger, or a custom HTTP collector.
    """

    @abstractmethod
    def export(self, traces: List[Trace]) -> None:
        """Export a list of traces to the target backend.

        Args:
            traces: List of :class:`~agent_observability.tracer.Trace` objects to export.
        """

    @abstractmethod
    def export_one(self, trace: Trace) -> None:
        """Export a single trace to the target backend.

        Args:
            trace: A single :class:`~agent_observability.tracer.Trace` to export.
        """


class StdoutExporter(BaseExporter):
    """Exports traces as newline-delimited JSON to stdout (or any stream).

    Each trace is serialized as a single-line JSON object.  Useful for
    local development, log aggregation pipelines, or CI debugging.

    Args:
        stream: Output stream; defaults to ``sys.stdout``.
        pretty: When True, emit indented JSON (multi-line per trace).

    Example:
        >>> exporter = StdoutExporter()
        >>> exporter.export_one(trace)
    """

    def __init__(self, stream: Optional[IO[str]] = None, pretty: bool = False) -> None:
        """Initialize the StdoutExporter.

        Args:
            stream: Output stream; defaults to sys.stdout.
            pretty: If True, emit indented JSON.
        """
        self._stream: IO[str] = stream if stream is not None else sys.stdout
        self._pretty = pretty

    def export(self, traces: List[Trace]) -> None:
        """Export multiple traces.

        Args:
            traces: List of traces to export.
        """
        for trace in traces:
            self.export_one(trace)

    def export_one(self, trace: Trace) -> None:
        """Export a single trace to the configured stream.

        Args:
            trace: Trace to serialize and write.
        """
        data = trace.to_dict()
        if self._pretty:
            self._stream.write(json.dumps(data, indent=2) + "\n")
        else:
            self._stream.write(json.dumps(data) + "\n")
        self._stream.flush()


class JSONFileExporter(BaseExporter):
    """Exports traces to a JSON file on disk.

    Supports two write modes:

    * ``append=True`` (default): each export appends records to an
      existing file as newline-delimited JSON (JSONL format).
    * ``append=False``: each export **overwrites** the file with a JSON
      array containing all provided traces.

    Args:
        path: Destination file path (created if it doesn't exist).
        append: Whether to append to an existing file (JSONL) or overwrite.
        pretty: When True and append=False, emit indented JSON.

    Example:
        >>> exporter = JSONFileExporter("/tmp/traces.jsonl")
        >>> exporter.export(tracer.traces)
    """

    def __init__(
        self,
        path: Union[str, Path],
        append: bool = True,
        pretty: bool = False,
    ) -> None:
        """Initialize the JSONFileExporter.

        Args:
            path: File path to write traces to.
            append: If True, append JSONL records; otherwise overwrite.
            pretty: If True and not appending, use indented JSON.
        """
        self._path = Path(path)
        self._append = append
        self._pretty = pretty
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The output file path."""
        return self._path

    def export(self, traces: List[Trace]) -> None:
        """Export multiple traces to the configured file.

        Args:
            traces: List of traces to serialize and write.
        """
        if self._append:
            mode = "a"
            with open(self._path, mode, encoding="utf-8") as fh:
                for trace in traces:
                    fh.write(json.dumps(trace.to_dict()) + "\n")
        else:
            data: List[Dict[str, Any]] = [t.to_dict() for t in traces]
            indent = 2 if self._pretty else None
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=indent)
                fh.write("\n")

    def export_one(self, trace: Trace) -> None:
        """Export a single trace to the configured file.

        Args:
            trace: Trace to serialize and write.
        """
        self.export([trace])

    def read_all(self) -> List[Dict[str, Any]]:
        """Read and deserialize all traces from the file.

        Returns:
            List of trace dictionaries.  Returns an empty list if the
            file does not exist.

        Raises:
            ValueError: If the file is neither valid JSONL nor a JSON array.
        """
        if not self._path.exists():
            return []
        content = self._path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        # Try JSONL first
        records: List[Dict[str, Any]] = []
        errors: List[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
        if not errors:
            # Flatten any items that are lists (non-append mode writes a compact JSON array
            # which the JSONL parser reads as a single item containing a list).
            result: List[Dict[str, Any]] = []
            for item in records:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result
        # Fall back to JSON array
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        raise ValueError(f"File {self._path} is not valid JSONL or JSON array. Errors: {errors}")


class TraceExporter:
    """High-level facade that delegates to one or more :class:`BaseExporter` backends.

    Supports fan-out: attach multiple exporters and every trace is sent to
    all of them.  The default configuration writes to stdout.

    Args:
        exporters: Initial list of exporters.  If empty, a
            :class:`StdoutExporter` is added automatically.

    Example:
        >>> exporter = TraceExporter([
        ...     StdoutExporter(),
        ...     JSONFileExporter("/var/log/agent/traces.jsonl"),
        ... ])
        >>> exporter.export_one(trace)
    """

    def __init__(self, exporters: Optional[List[BaseExporter]] = None) -> None:
        """Initialize TraceExporter with the given backends.

        Args:
            exporters: List of BaseExporter instances.  Defaults to a
                single StdoutExporter.
        """
        if exporters is None:
            self._exporters: List[BaseExporter] = [StdoutExporter()]
        else:
            self._exporters = list(exporters)

    def add_exporter(self, exporter: BaseExporter) -> None:
        """Attach an additional exporter backend.

        Args:
            exporter: The exporter to add.
        """
        self._exporters.append(exporter)

    def export(self, traces: List[Trace]) -> None:
        """Export a list of traces to all configured backends.

        Args:
            traces: List of traces to export.
        """
        for exporter in self._exporters:
            exporter.export(traces)

    def export_one(self, trace: Trace) -> None:
        """Export a single trace to all configured backends.

        Args:
            trace: Trace to export.
        """
        for exporter in self._exporters:
            exporter.export_one(trace)

    @property
    def exporters(self) -> List[BaseExporter]:
        """Return the list of attached exporters."""
        return list(self._exporters)
