
import math
from typing import Any


VALID_ACTIONS = {"fan_on", "fan_off", "no_action"}


def validate_action(
    telemetry: dict[str, Any],
    action: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Validate a proposed fan action before it can be executed."""

    reasons = []

    if action not in VALID_ACTIONS:
        reasons.append("Invalid action")

    required = ("temperature_c", "humidity_pct", "vibration_g")

    for key in required:
        try:
            value = float(telemetry[key])
            if not math.isfinite(value):
                reasons.append(f"Invalid sensor value: {key}")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"Missing or invalid sensor value: {key}")

    try:
        temperature = float(telemetry["temperature_c"])
        humidity = float(telemetry["humidity_pct"])
        vibration = float(telemetry["vibration_g"])
        confidence = float(confidence)

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            reasons.append("Invalid confidence")
    except (KeyError, TypeError, ValueError):
        temperature = humidity = vibration = float("nan")
        confidence = 0.0

    if math.isfinite(humidity) and not 0 <= humidity <= 100:
        reasons.append("Humidity outside valid range")

    if action != "no_action" and confidence < 0.60:
        reasons.append("Confidence below minimum threshold")

    if action == "fan_off" and math.isfinite(temperature):
        if temperature >= 32:
            reasons.append("Cannot turn fan off at high temperature")

    if action == "fan_off" and math.isfinite(vibration):
        if vibration >= 0.75:
            reasons.append("Cannot turn fan off during elevated vibration")

    allowed = not reasons

    return {
        "allowed": allowed,
        "approved_action": action if allowed else "no_action",
        "reasons": reasons,
    }
