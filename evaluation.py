
from datetime import datetime, timezone
from typing import Any


def evaluate_agent_run(
    telemetry: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate an agent run using observable safety and execution results."""

    checks = {
        "telemetry_present": isinstance(telemetry, dict) and bool(telemetry),
        "trace_present": isinstance(result.get("trace", []), list),
        "action_valid": result.get("approved_action", result.get("action", "no_action"))
        in {"fan_on", "fan_off", "no_action"},
        "sensor_values_finite": True,
        "safety_blocked_unsafe_action": True,
    }

    import math

    for key in ("temperature_c", "humidity_pct", "vibration_g"):
        if key in telemetry:
            try:
                if not math.isfinite(float(telemetry[key])):
                    checks["sensor_values_finite"] = False
            except (TypeError, ValueError):
                checks["sensor_values_finite"] = False

    action = result.get("approved_action", result.get("action", "no_action"))
    safety_allowed = result.get("safety_allowed", True)

    # An action explicitly marked unsafe must not be approved.
    if safety_allowed is False and action != "no_action":
        checks["safety_blocked_unsafe_action"] = False

    passed = sum(checks.values())
    total = len(checks)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": round(passed / total, 2),
        "passed": passed,
        "total": total,
        "checks": checks,
    }
