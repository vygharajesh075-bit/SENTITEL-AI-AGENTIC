

from typing import Any


def _number(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def planning_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Plan a proposed action from sensor readings and diagnosis.
    This node proposes actions only; it does not execute hardware commands.
    """

    telemetry = state.get("telemetry") or {}
    diagnosis = state.get("diagnosis") or {}
    flags = state.get("flags") or []

    if not isinstance(telemetry, dict):
        telemetry = {}

    if not isinstance(diagnosis, dict):
        diagnosis = {}

    if not isinstance(flags, list):
        flags = []

    temperature = _number(telemetry, "temperature_c", float("nan"))
    vibration = _number(telemetry, "vibration_g", float("nan"))
    humidity = _number(telemetry, "humidity_pct", float("nan"))

    confidence = _number(state, "confidence", 0.0)
    sensor_valid = state.get("sensor_valid", True)
    needs_human_review = state.get("needs_human_review", False)
    severity = str(state.get("severity", "low")).lower()

    reasons: list[str] = []

    # Conservative default: do not recommend a state-changing action
    # when sensor data is invalid or a human review is required.
    if not sensor_valid:
        reasons.append("Sensor data is invalid")

    if needs_human_review:
        reasons.append("Human review is required")

    if not all(
        value == value and abs(value) != float("inf")
        for value in (temperature, vibration, humidity)
    ):
        reasons.append("One or more sensor values are missing or non-finite")

    if not 0 <= humidity <= 100:
        reasons.append("Humidity is outside the valid range")

    if confidence < 0.60:
        reasons.append("Diagnosis confidence is below the planning threshold")

    if severity == "high":
        reasons.append("High-severity condition requires conservative handling")

    if reasons:
        action = "no_action"
    elif temperature >= 32 and not telemetry.get("fan_on", False):
        action = "fan_on"
        reasons.append("Temperature is elevated; recommend turning the fan on")
    elif (
        temperature < 28
        and telemetry.get("fan_on", False)
        and vibration < 0.75
    ):
        action = "fan_off"
        reasons.append("Temperature is lower and vibration is below the threshold")
    else:
        action = "no_action"
        reasons.append("No control change is currently required")

    plan = {
        "proposed_action": action,
        "planning_confidence": confidence,
        "planning_reasons": reasons,
        "requires_safety_check": action != "no_action",
    }

    trace = list(state.get("trace") or [])
    trace.append({
        "agent": "planning",
        "action": action,
        "reasons": reasons,
    })

    return {
        **state,
        **plan,
        "trace": trace,
    }


def create_plan(
    telemetry: dict[str, Any],
    diagnosis: dict[str, Any] | None = None,
    confidence: float = 0.0,
    flags: list[str] | None = None,
    severity: str = "low",
    sensor_valid: bool = True,
    needs_human_review: bool = False,
) -> dict[str, Any]:
    """Convenience function for calling the planning node independently."""

    state = {
        "telemetry": telemetry,
        "diagnosis": diagnosis or {},
        "confidence": confidence,
        "flags": flags or [],
        "severity": severity,
        "sensor_valid": sensor_valid,
        "needs_human_review": needs_human_review,
        "trace": [],
    }

    return planning_node(state)
