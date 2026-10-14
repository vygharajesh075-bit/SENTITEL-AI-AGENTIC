def supervisor_decision(state):
    # Safe fallback when the supervisor cannot use Groq.
    flags = state.get("flags", [])
    severity = str(state.get("severity", "low")).lower()
    sensor_valid = state.get("sensor_valid", True)
    confidence = float(state.get("confidence", 0.0) or 0.0)

    if not sensor_valid or severity == "high":
        route = "human_review"
        reason = "Invalid sensor data or high-severity incident requires review."
    elif not flags:
        route = "monitor"
        reason = "No anomalies detected."
    elif confidence < 0.60:
        route = "investigate"
        reason = "Low confidence requires further investigation."
    else:
        route = "plan"
        reason = "Anomaly detected; proceed to action planning."

    return {
        "next_chain": route,
        "route_reason": reason,
        "needs_human_review": route == "human_review",
        "supervisor_provider": "fallback",
    }