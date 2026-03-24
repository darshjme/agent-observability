"""Log exporter module for agent observability data.

Supports exporting spans and run data to JSONL files,
stdout (pretty-printed), or Python dicts for custom sinks.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agent_observability.middleware import RunContext


def _default_serializer(obj: Any) -> Any:
    """JSON serialization fallback for non-standard types.

    Args:
        obj: Object that failed standard JSON serialization.

    Returns:
        A JSON-serializable representation.

    Raises:
        TypeError: If the object cannot be serialized.
    """
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class LogExporter:
    """Exports agent run data to various output formats.

    Supports three export destinations:
    - JSONL file (one JSON object per line)
    - stdout (pretty-printed JSON)
    - Python dict (for custom downstream sinks)

    Example:
        >>> exporter = LogExporter()
        >>> mw = ObservabilityMiddleware(model_id="gpt-4o")
        >>> with mw.trace_run() as run:
        ...     run.record_tokens(100, 50)
        >>> exporter.export_to_stdout(mw.runs[-1])
    """

    def __init__(self, indent: int = 2) -> None:
        """Initialize LogExporter.

        Args:
            indent: JSON indentation level for pretty output. Default 2.
        """
        self.indent = indent

    # -------------------------------------------------------------------------
    # Core serialization
    # -------------------------------------------------------------------------

    def to_dict(self, run: RunContext) -> Dict[str, Any]:
        """Convert a RunContext to a plain Python dict.

        Args:
            run: RunContext instance from a completed run.

        Returns:
            Dict representation suitable for custom sinks or further
            processing. Nested spans are included under 'spans' key.
        """
        return run.to_dict()

    def to_json(self, run: RunContext, pretty: bool = False) -> str:
        """Serialize a RunContext to a JSON string.

        Args:
            run: RunContext to serialize.
            pretty: If True, indent with self.indent spaces.

        Returns:
            JSON string.
        """
        indent = self.indent if pretty else None
        return json.dumps(
            run.to_dict(),
            default=_default_serializer,
            indent=indent,
        )

    # -------------------------------------------------------------------------
    # Export targets
    # -------------------------------------------------------------------------

    def export_to_jsonl(
        self,
        runs: Union[RunContext, List[RunContext]],
        path: Union[str, Path],
        append: bool = True,
    ) -> int:
        """Export one or more runs to a JSONL file.

        Each run is serialized as one JSON line. File is created
        if it does not exist.

        Args:
            runs: Single RunContext or list of RunContexts to export.
            path: File path to write to (str or Path).
            append: If True, append to existing file. If False, overwrite.

        Returns:
            Number of records written.
        """
        if isinstance(runs, RunContext):
            runs = [runs]

        mode = "a" if append else "w"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(path, mode, encoding="utf-8") as f:
            for run in runs:
                line = json.dumps(run.to_dict(), default=_default_serializer)
                f.write(line + "\n")
                count += 1
        return count

    def export_to_stdout(
        self,
        runs: Union[RunContext, List[RunContext]],
        stream: Optional[io.TextIOBase] = None,
    ) -> None:
        """Export one or more runs to stdout (or given stream) as pretty JSON.

        Args:
            runs: Single RunContext or list to export.
            stream: Output stream (defaults to sys.stdout).
        """
        if isinstance(runs, RunContext):
            runs = [runs]

        out = stream or sys.stdout
        for run in runs:
            out.write(
                json.dumps(
                    run.to_dict(),
                    default=_default_serializer,
                    indent=self.indent,
                )
            )
            out.write("\n")

    def export_to_dict(
        self,
        runs: Union[RunContext, List[RunContext]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Export runs as Python dict(s) for custom downstream sinks.

        Args:
            runs: Single RunContext or list to export.

        Returns:
            Single dict if input was a single RunContext,
            list of dicts if input was a list.
        """
        if isinstance(runs, RunContext):
            return runs.to_dict()
        return [run.to_dict() for run in runs]

    # -------------------------------------------------------------------------
    # Batch helpers
    # -------------------------------------------------------------------------

    def load_jsonl(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load previously exported JSONL file back into dicts.

        Args:
            path: Path to JSONL file.

        Returns:
            List of run dicts.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        path = Path(path)
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def export_spans_only(
        self,
        run: RunContext,
        path: Union[str, Path],
        append: bool = True,
    ) -> int:
        """Export only the spans from a run to a JSONL file.

        Each span is written as a separate JSON line (flattened, no nesting).

        Args:
            run: RunContext whose spans to export.
            path: Output file path.
            append: Whether to append or overwrite.

        Returns:
            Number of span records written.
        """
        flat_spans = run.tracer.all_spans_flat()
        mode = "a" if append else "w"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(path, mode, encoding="utf-8") as f:
            for span in flat_spans:
                f.write(json.dumps(span, default=_default_serializer) + "\n")
                count += 1
        return count
