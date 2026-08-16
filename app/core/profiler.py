"""Pipeline Performance Profiler.

Instruments backend request processing across the 10 core stages:
  1. Receive Upload
  2. Save File
  3. Decode
  4. Resample
  5. Inference
  6. Normalizer
  7. Vocabulary
  8. Menu Context
  9. Parser
  10. Response

Outputs formatted 10-stage performance logs when DEBUG=true.
Highlights any single stage exceeding 300 ms.
"""

import time
from typing import Dict

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PipelineProfiler:
    """Thread-safe request pipeline profiler for end-to-end backend timing."""

    STAGE_KEYS = [
        "Receive Upload",
        "Save File",
        "Decode",
        "Resample",
        "Inference",
        "Normalizer",
        "Vocabulary",
        "Menu Context",
        "Parser",
        "Response",
    ]

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = enabled if enabled is not None else getattr(settings, "debug", True)
        self.stages: Dict[str, float] = {k: 0.0 for k in self.STAGE_KEYS}
        self._start_times: Dict[str, float] = {}
        self.global_start = time.perf_counter()

    def start_stage(self, stage_name: str) -> None:
        """Start timing a specific pipeline stage."""
        if self.enabled:
            self._start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """End timing for a specific pipeline stage and record elapsed ms."""
        if not self.enabled:
            return 0.0
        start = self._start_times.get(stage_name, time.perf_counter())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stages[stage_name] = elapsed_ms
        return elapsed_ms

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Manually record a stage duration in milliseconds."""
        if self.enabled:
            self.stages[stage_name] = duration_ms

    def print_summary(self) -> str:
        """Generate and print the formatted 10-stage timing breakdown report if enabled."""
        if not self.enabled:
            return ""

        total_ms = (time.perf_counter() - self.global_start) * 1000.0

        lines = ["================================"]
        for key in self.STAGE_KEYS:
            val_ms = self.stages.get(key, 0.0)
            highlight = " [EXCEEDS 300ms]" if val_ms > 300.0 else ""
            lines.append(f"{key:<20} {int(round(val_ms)):>5} ms{highlight}")

        lines.append(f"{'TOTAL':<20} {int(round(total_ms)):>5} ms")
        lines.append("================================")

        report = "\n".join(lines)
        print(report)
        logger.info("\n" + report)
        return report
