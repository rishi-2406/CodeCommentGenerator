"""
Pipeline Logger — Week 7 Core Engine
======================================
Structured logging for all stages of the comment generation pipeline.
Writes both a JSON log (machine-readable) and a human-readable text log.
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StageLog:
    stage: str
    duration_ms: float
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineLog:
    input_file: str
    timestamp: str
    stages: List[StageLog] = field(default_factory=list)
    total_duration_ms: float = 0.0
    comments_generated: int = 0
    output_file: Optional[str] = None


class PipelineLogger:
    """Records timing and summary information for each pipeline stage."""

    def __init__(self, input_file: str):
        self._log = PipelineLog(
            input_file=input_file,
            timestamp=datetime.now().isoformat(),
        )
        self._pipeline_start = time.perf_counter()
        self._stage_start: Optional[float] = None
        self._current_stage: Optional[str] = None

    def begin_stage(self, stage_name: str):
        """Mark the start of a pipeline stage."""
        self._current_stage = stage_name
        self._stage_start = time.perf_counter()

    def end_stage(self, summary: Dict[str, Any] = None, warnings: List[str] = None):
        """Mark the end of a pipeline stage and record its log entry."""
        if self._stage_start is None or self._current_stage is None:
            return
        elapsed_ms = (time.perf_counter() - self._stage_start) * 1000
        self._log.stages.append(StageLog(
            stage=self._current_stage,
            duration_ms=round(elapsed_ms, 2),
            summary=summary or {},
            warnings=warnings or [],
        ))
        self._stage_start = None
        self._current_stage = None

    def set_comments_generated(self, count: int):
        self._log.comments_generated = count

    def set_output_file(self, path: str):
        self._log.output_file = path

    def finalize(self) -> PipelineLog:
        """Finalize the log and compute total duration."""
        self._log.total_duration_ms = round(
            (time.perf_counter() - self._pipeline_start) * 1000, 2
        )
        return self._log

    def save(self, logs_dir: str = "logs") -> Tuple[str, str]:
        """
        Save JSON and text logs to logs_dir.
        Returns (json_path, text_path).
        """
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(logs_dir, f"pipeline_{ts}.json")
        text_path = os.path.join(logs_dir, f"pipeline_{ts}.log")

        log = self.finalize()

        # --- JSON log ---
        import dataclasses

        def _serialize(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [_serialize(i) for i in obj]
            else:
                return obj

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_serialize(log), f, indent=2)

        # --- Human-readable text log ---
        lines = [
            "=" * 60,
            "  CODE COMMENT GENERATION — PIPELINE LOG",
            "=" * 60,
            f"  Input  : {log.input_file}",
            f"  Output : {log.output_file or 'N/A'}",
            f"  Run at : {log.timestamp}",
            f"  Total  : {log.total_duration_ms} ms",
            f"  Comments generated: {log.comments_generated}",
            "-" * 60,
        ]
        for sl in log.stages:
            lines.append(f"  Stage [{sl.stage}]  — {sl.duration_ms} ms")
            for k, v in sl.summary.items():
                lines.append(f"      {k}: {v}")
            for w in sl.warnings:
                lines.append(f"      ⚠ WARNING: {w}")
        lines.append("=" * 60)

        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return json_path, text_path


# Type alias placeholder (Python 3.8 compat)
Tuple_str = tuple
