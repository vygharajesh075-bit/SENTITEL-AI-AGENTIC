
import logging
import math
import statistics
from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any

logger = logging.getLogger("sentinel")


class Observability:
    """Thread-safe in-memory logs and metrics for SENTINEL agent runs."""

    def __init__(
        self,
        max_events: int = 1000,
        slow_run_seconds: float = 2.0,
    ):
        self._lock = Lock()

        try:
            max_events = max(1, int(max_events))
        except (TypeError, ValueError, OverflowError):
            max_events = 1000

        try:
            slow_run_seconds = float(slow_run_seconds)
            if (
                not math.isfinite(slow_run_seconds)
                or slow_run_seconds < 0
            ):
                slow_run_seconds = 2.0
        except (TypeError, ValueError, OverflowError):
            slow_run_seconds = 2.0

        self._events: deque[dict[str, Any]] = deque(
            maxlen=max_events
        )
        self._runs = 0
        self._failures = 0
        self._durations: deque[float] = deque(maxlen=max_events)
        self._actions: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._component_durations: dict[str, deque[float]] = {}
        self._slow_run_seconds = slow_run_seconds

    @staticmethod
    def _valid_duration(value: Any) -> float:
        """Return a finite, non-negative duration or zero."""
        try:
            duration = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0

        if not math.isfinite(duration) or duration < 0:
            return 0.0

        return duration

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Calculate a nearest-rank percentile."""
        if not values:
            return 0.0

        ordered = sorted(values)
        index = max(
            0,
            math.ceil((percentile / 100.0) * len(ordered)) - 1,
        )
        return round(ordered[index], 4)

    @classmethod
    def _latency_summary(cls, values: list[float]) -> dict[str, Any]:
        """Summarize a collection of durations in seconds."""
        if not values:
            return {
                "samples": 0,
                "average_seconds": 0.0,
                "median_seconds": 0.0,
                "p95_seconds": 0.0,
                "max_seconds": 0.0,
            }

        return {
            "samples": len(values),
            "average_seconds": round(
                statistics.fmean(values), 4
            ),
            "median_seconds": round(
                statistics.median(values), 4
            ),
            "p95_seconds": cls._percentile(values, 95),
            "max_seconds": round(max(values), 4),
        }

    def record_event(
        self,
        event: str,
        level: str = "info",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store an event and send it to the application logger."""
        safe_level = str(level).lower()
        if safe_level not in {
            "debug", "info", "warning", "error", "critical"
        }:
            safe_level = "info"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            "level": safe_level,
            "details": details or {},
        }

        with self._lock:
            self._events.append(entry)

        log_method = getattr(logger, safe_level, logger.info)
        log_method("%s | %s", entry["event"], entry["details"])

        return entry

    def record_agent_run(
        self,
        action: str = "no_action",
        duration_seconds: float = 0.0,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an agent run, duration, action, and outcome."""
        duration = self._valid_duration(duration_seconds)
        success = bool(success)
        action = str(action)

        with self._lock:
            self._runs += 1
            self._durations.append(duration)
            self._actions[action] += 1

            if not success:
                self._failures += 1

        event_details = {
            "action": action,
            "duration_seconds": duration,
            "success": success,
            **(details or {}),
        }

        result = self.record_event(
            "agent_run",
            level="info" if success else "error",
            details=event_details,
        )

        if duration >= self._slow_run_seconds:
            self.record_event(
                "slow_agent_run",
                level="warning",
                details={
                    "action": action,
                    "duration_seconds": duration,
                    "threshold_seconds": self._slow_run_seconds,
                },
            )

        return result

    def record_timing(
        self,
        component: str,
        duration_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the duration of an individual component."""
        component = str(component)
        duration = self._valid_duration(duration_seconds)

        with self._lock:
            if component not in self._component_durations:
                self._component_durations[component] = deque(
                    maxlen=self._durations.maxlen
                )

            self._component_durations[component].append(duration)

        return self.record_event(
            "component_timing",
            details={
                "component": component,
                "duration_seconds": duration,
                **(details or {}),
            },
        )

    def record_error(
        self,
        error: Exception | str,
        component: str = "unknown",
    ) -> dict[str, Any]:
        """Record an error and increment its component counter."""
        component = str(component)
        error_type = (
            type(error).__name__
            if isinstance(error, Exception)
            else "Error"
        )

        with self._lock:
            self._errors[component] += 1

        return self.record_event(
            "agent_error",
            level="error",
            details={
                "component": component,
                "error_type": error_type,
                "message": str(error),
            },
        )

    def get_snapshot(self, recent_limit: int = 50) -> dict[str, Any]:
        """Return a thread-safe snapshot of current metrics."""
        try:
            recent_limit = int(recent_limit)
        except (TypeError, ValueError, OverflowError):
            recent_limit = 50

        recent_limit = max(1, min(recent_limit, 1000))

        with self._lock:
            durations = list(self._durations)
            events = list(self._events)[-recent_limit:]
            runs = self._runs
            failures = self._failures
            actions = dict(self._actions)
            errors = dict(self._errors)

            component_durations = {
                name: list(samples)
                for name, samples in self._component_durations.items()
            }

        # Invalid durations are sanitized before insertion, but filtering
        # here also protects metrics from unexpected legacy data.
        durations = [
            value for value in durations
            if math.isfinite(value) and value >= 0
        ]

        average_duration = (
            statistics.fmean(durations) if durations else 0.0
        )

        latency = self._latency_summary(durations)

        component_latency = {
            name: self._latency_summary(
                [
                    value for value in samples
                    if math.isfinite(value) and value >= 0
                ]
            )
            for name, samples in component_durations.items()
        }

        return {
            "status": "degraded" if failures else "ok",
            "agent_runs": runs,
            "agent_failures": failures,
            "failure_rate": (
                round(failures / runs, 4) if runs else 0.0
            ),
            # Keep this legacy field for existing tests and API clients.
            "average_duration_seconds": round(
                average_duration, 4
            ),
            "latency": latency,
            "component_latency": component_latency,
            "actions": actions,
            "errors_by_component": errors,
            "recent_events": events,
        }

    def reset(self) -> None:
        """Clear all counters, samples, and stored events."""
        with self._lock:
            self._events.clear()
            self._runs = 0
            self._failures = 0
            self._durations.clear()
            self._actions.clear()
            self._errors.clear()
            self._component_durations.clear()


observability = Observability()


def record_agent_run(
    action: str = "no_action",
    duration_seconds: float = 0.0,
    success: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return observability.record_agent_run(
        action=action,
        duration_seconds=duration_seconds,
        success=success,
        details=details,
    )


def record_timing(
    component: str,
    duration_seconds: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return observability.record_timing(
        component=component,
        duration_seconds=duration_seconds,
        details=details,
    )


def record_error(
    error: Exception | str,
    component: str = "unknown",
) -> dict[str, Any]:
    return observability.record_error(
        error=error,
        component=component,
    )


def get_observability_snapshot(
    recent_limit: int = 50,
) -> dict[str, Any]:
    return observability.get_snapshot(recent_limit)
