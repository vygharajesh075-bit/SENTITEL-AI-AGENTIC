
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from .safety_policy import validate_action


# ============================================================
# 1. COMMON UTILITIES AND DATA MODELS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Telemetry(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    temperature_c: float
    humidity_pct: float
    vibration_g: float
    fan_on: bool
    source: str = "virtual-sensor"


class AgentStep(BaseModel):
    agent: str
    status: Literal["ok", "warning", "blocked", "info", "error"]
    message: str
    evidence: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)


class Incident(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=utc_now)
    severity: Literal["low", "medium", "high"]
    title: str
    diagnosis: str
    evidence: list[str]
    recommended_action: str
    action_taken: str
    verification: str
    status: Literal["monitoring", "resolved", "open"]
    agent_trace: list[AgentStep] = Field(default_factory=list)


class ControlRequest(BaseModel):
    mode: Literal[
        "normal",
        "temperature_fault",
        "vibration_fault",
        "sensor_dropout",
    ]
    enabled: bool = True


class ActionRequest(BaseModel):
    action: Literal["fan_on", "fan_off", "no_action"]
    reason: str = "Dashboard request"


class AgentState(TypedDict, total=False):
    # Core telemetry state
    telemetry: dict[str, Any]
    previous: dict[str, Any] | None
    history: list[dict[str, Any]]

    # Perception and anomaly state
    flags: list[str]
    severity: str
    trace: list[dict[str, Any]]
    sensor_valid: bool

    # Diagnosis and investigation
    diagnosis: str
    diagnosis_evidence: list[str]
    confidence: float
    investigation_evidence: list[str]
    investigation_round: int
    max_investigation_rounds: int

    # Supervisor state
    route_reason: str
    next_chain: str
    needs_human_review: bool
    errors: list[str]

    # Planning and safety
    proposed_action: str
    planner_reason: str
    approved_action: str
    safety_allowed: bool
    execution_mode: str

    # Execution and verification
    action_result: str
    after: dict[str, Any]
    verification: str
    incident: dict[str, Any] | None


# ============================================================
# 2. TRACE AND VALIDATION HELPERS
# ============================================================

def step(
    agent: str,
    status: str,
    message: str,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    """Create a serializable agent-trace entry."""
    return AgentStep(
        agent=agent,
        status=status,
        message=message,
        evidence=evidence or [],
    ).model_dump(mode="json")


def append_trace(
    state: dict[str, Any],
    agent: str,
    status: str,
    message: str,
    evidence: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a new trace list without mutating the input state."""
    return list(state.get("trace", [])) + [
        step(agent, status, message, evidence)
    ]


def as_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float, or return the default."""
    try:
        result = float(value)
        if result != result or abs(result) == float("inf"):
            return default
        return result
    except (TypeError, ValueError, OverflowError):
        return default


# ============================================================
# 3. SENSOR / PERCEPTION CHAIN
# ============================================================

def sensor_node(state: AgentState) -> dict[str, Any]:
    """Validate current telemetry and record sensor evidence."""
    telemetry = dict(state.get("telemetry", {}))

    required = (
        "temperature_c",
        "humidity_pct",
        "vibration_g",
        "fan_on",
    )

    missing = [
        key
        for key in required
        if key not in telemetry or telemetry[key] is None
    ]

    invalid: list[str] = []

    for key in (
        "temperature_c",
        "humidity_pct",
        "vibration_g",
    ):
        if key in telemetry:
            value = as_float(telemetry[key], float("nan"))
            if value != value or abs(value) == float("inf"):
                invalid.append(key)

    # A fan state must be an actual boolean, not an arbitrary string.
    if "fan_on" in telemetry and not isinstance(
        telemetry["fan_on"], bool
    ):
        invalid.append("fan_on")

    evidence = [
        f"Source: {telemetry.get('source', 'unknown')}",
        f"Timestamp: {telemetry.get('timestamp', 'unknown')}",
    ]

    if missing:
        evidence.append("Missing fields: " + ", ".join(missing))

    if invalid:
        evidence.append("Invalid fields: " + ", ".join(invalid))

    valid = not missing and not invalid

    return {
        "telemetry": telemetry,
        "sensor_valid": valid,
        "flags": list(state.get("flags", [])),
        "severity": state.get("severity", "low"),
        "trace": append_trace(
            state,
            "Perception Agent",
            "ok" if valid else "warning",
            (
                "Telemetry passed basic validation."
                if valid
                else "Telemetry is incomplete or contains invalid values."
            ),
            evidence,
        ),
    }


# ============================================================
# 4. ANOMALY DETECTION CHAIN
# ============================================================

def anomaly_node(state: AgentState) -> dict[str, Any]:
    """Apply transparent thresholds to synthetic IoT telemetry."""
    telemetry = state.get("telemetry", {})
    flags: list[str] = []
    evidence: list[str] = []

    if not state.get("sensor_valid", True):
        flags.append("sensor_data_quality")
        evidence.append(
            "Sensor fields are missing or contain invalid values."
        )

    # Do not interpret invalid sensor values as meaningful measurements.
    numeric_valid = True
    for key in ("temperature_c", "humidity_pct", "vibration_g"):
        value = as_float(telemetry.get(key), float("nan"))
        if value != value or abs(value) == float("inf"):
            numeric_valid = False
            break

    if numeric_valid:
        temperature = float(telemetry["temperature_c"])
        humidity = float(telemetry["humidity_pct"])
        vibration = float(telemetry["vibration_g"])

        if temperature >= 40:
            flags.append("critical_temperature")
            evidence.append(
                f"Critical temperature: {temperature:.2f} °C."
            )
        elif temperature >= 32:
            flags.append("high_temperature")
            evidence.append(
                f"Elevated temperature: {temperature:.2f} °C."
            )

        if vibration >= 1.20:
            flags.append("critical_vibration")
            evidence.append(
                f"Critical vibration: {vibration:.3f} g."
            )
        elif vibration >= 0.75:
            flags.append("high_vibration")
            evidence.append(
                f"Elevated vibration: {vibration:.3f} g."
            )

        if not 0 <= humidity <= 100:
            flags.append("humidity_out_of_range")
            evidence.append(
                f"Humidity outside expected range: {humidity:.2f}%."
            )

    severity = "low"

    if any(
        flag in flags
        for flag in (
            "critical_temperature",
            "critical_vibration",
        )
    ):
        severity = "high"
    elif flags:
        severity = "medium"

    return {
        "flags": flags,
        "severity": severity,
        "trace": append_trace(
            state,
            "Anomaly Detection Agent",
            "warning" if flags else "ok",
            (
                f"Detected {len(flags)} anomaly or quality flag(s)."
                if flags
                else "No configured anomaly thresholds were crossed."
            ),
            evidence or ["Configured threshold checks completed."],
        ),
    }


# ============================================================
# 5. HEALTHY / MONITORING CHAIN
# ============================================================

def healthy_node(state: AgentState) -> dict[str, Any]:
    """Handle telemetry that has no detected anomaly flags."""
    return {
        "diagnosis": "No anomaly detected by configured rules.",
        "diagnosis_evidence": [
            "All configured sensor thresholds passed."
        ],
        "confidence": 0.80,
        "proposed_action": "no_action",
        "planner_reason": "Continue normal monitoring.",
        "approved_action": "no_action",
        "safety_allowed": True,
        "action_result": "No action required.",
        "verification": (
            "Monitoring cycle completed; no intervention needed."
        ),
        "trace": append_trace(
            state,
            "Monitoring Agent",
            "ok",
            "No anomaly detected; no intervention proposed.",
        ),
    }


# ============================================================
# 6. INVESTIGATION / DIAGNOSIS CHAIN
# ============================================================

def _fallback_diagnosis(
    state: AgentState,
) -> dict[str, Any]:
    """Provide deterministic diagnosis when no LLM is configured."""
    flags = state.get("flags", [])
    telemetry = state.get("telemetry", {})
    evidence = list(state.get("investigation_evidence", []))

    temperature = as_float(telemetry.get("temperature_c"))
    vibration = as_float(telemetry.get("vibration_g"))
    fan_on = bool(telemetry.get("fan_on", False))

    causes: list[str] = []

    if any("temperature" in flag for flag in flags):
        if temperature >= 32 and not fan_on:
            causes.append(
                "Elevated temperature while the fan is off; "
                "insufficient cooling is a possible cause."
            )
        else:
            causes.append(
                "Elevated temperature; excessive load, insufficient "
                "cooling or a sensor issue remain possible."
            )

    if any("vibration" in flag for flag in flags):
        causes.append(
            "Elevated vibration; imbalance, looseness or sensor "
            "disturbance are possible causes."
        )

    if "sensor_data_quality" in flags:
        causes.append(
            "Telemetry quality problem; validate sensor readings "
            "before taking an automated action."
        )

    if "humidity_out_of_range" in flags:
        causes.append(
            "Humidity is outside the configured range; inspect the "
            "sensor and validate the reading."
        )

    if not causes:
        causes.append(
            "A configured sensor-quality or range check was triggered; "
            "further inspection is recommended."
        )

    if len(flags) > 1:
        confidence = 0.45
    elif flags and "sensor_data_quality" in flags:
        confidence = 0.30
    else:
        confidence = 0.55

    evidence.extend(f"Active flag: {flag}" for flag in flags)

    return {
        "diagnosis": " ".join(causes),
        "evidence": list(dict.fromkeys(evidence))[:10],
        "confidence": confidence,
        "confidence_label": (
            "low" if confidence < 0.60 else "medium"
        ),
    }


def _call_groq_diagnosis(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Call Groq only when an API key is configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        model_name = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )

        llm = ChatGroq(
            model=model_name,
            temperature=0,
            api_key=api_key,
        )

        messages = [
            SystemMessage(
                content=(
                    "You are SENTINEL's IoT diagnosis agent. "
                    "Use only the provided synthetic telemetry and evidence. "
                    "Return a JSON object with keys diagnosis (string), "
                    "evidence (array of strings), and confidence "
                    "(one of low, medium, high). "
                    "Do not invent observations. Separate hypotheses "
                    "from observed evidence. Never issue actuator commands."
                )
            ),
            HumanMessage(content=json.dumps(payload, default=str)),
        ]

        response = llm.invoke(messages)
        content = response.content

        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict)
                else str(item)
                for item in content
            )

        text = str(content).strip()

        if text.startswith("```"):
            text = (
                text.replace("```json", "")
                .replace("```", "")
                .strip()
            )

        result = json.loads(text)
        if not isinstance(result, dict):
            return None

        confidence_map = {
            "low": 0.30,
            "medium": 0.65,
            "high": 0.85,
        }

        label = str(result.get("confidence", "low")).lower()
        result["confidence_label"] = (
            label if label in confidence_map else "low"
        )
        result["confidence"] = confidence_map.get(label, 0.30)

        if not isinstance(result.get("diagnosis"), str):
            return None

        if not isinstance(result.get("evidence", []), list):
            result["evidence"] = []

        return result

    except Exception:
        # Deterministic fallback keeps monitoring available.
        return None


def diagnosis_node(state: AgentState) -> dict[str, Any]:
    """Diagnose anomalies using optional LLM reasoning and rule fallback."""
    telemetry = state.get("telemetry", {})
    previous = state.get("previous")
    flags = list(state.get("flags", []))
    investigation_evidence = list(
        state.get("investigation_evidence", [])
    )

    payload = {
        "current_telemetry": telemetry,
        "previous_telemetry": previous,
        "anomaly_evidence": flags,
        "investigation_evidence": investigation_evidence,
        "severity": state.get("severity", "low"),
        "investigation_round": state.get("investigation_round", 0),
    }

    result = _call_groq_diagnosis(payload)
    if result is None:
        result = _fallback_diagnosis(state)

    diagnosis = str(result.get("diagnosis", "Diagnosis unavailable."))
    confidence = as_float(result.get("confidence"), 0.30)
    confidence = min(max(confidence, 0.0), 1.0)

    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    evidence = list(dict.fromkeys(
        [str(item) for item in evidence]
        + investigation_evidence
        + [f"Active anomaly flag: {flag}" for flag in flags]
    ))[:12]

    status = "warning" if confidence < 0.60 else "ok"

    return {
        "diagnosis": diagnosis,
        "diagnosis_evidence": evidence,
        "confidence": confidence,
        "trace": append_trace(
            state,
            "Diagnosis Agent",
            status,
            f"Diagnosis generated with estimated confidence {confidence:.2f}.",
            evidence,
        ),
    }


# ============================================================
# 7. PLANNING CHAIN
# ============================================================

def planner_node(state: AgentState) -> dict[str, Any]:
    """Propose a conservative action; never execute it here."""
    telemetry = state.get("telemetry", {})
    flags = state.get("flags", [])
    severity = state.get("severity", "low")

    temperature = as_float(telemetry.get("temperature_c"))
    vibration = as_float(telemetry.get("vibration_g"))
    fan_on = bool(telemetry.get("fan_on", False))

    if state.get("needs_human_review", False):
        action = "no_action"
        reason = (
            "Diagnosis remains uncertain; escalate to an operator "
            "instead of acting automatically."
        )
    elif not state.get("sensor_valid", True):
        action = "no_action"
        reason = (
            "Sensor data is incomplete or invalid. Validate the readings "
            "before proposing an intervention."
        )
    elif severity == "high":
        action = "no_action"
        reason = (
            "High-severity event requires operator review before intervention."
        )
    elif temperature >= 32 and not fan_on:
        action = "fan_on"
        reason = (
            "Elevated temperature and fan currently off; propose enabling "
            "the simulated fan for evaluation."
        )
    elif temperature < 28 and fan_on and vibration < 0.75:
        action = "fan_off"
        reason = (
            "Temperature is below the configured cooling threshold and "
            "vibration is normal; propose switching the simulated fan off."
        )
    else:
        action = "no_action"
        reason = (
            "No intervention meets the current conservative rule criteria."
        )

    return {
        "proposed_action": action,
        "planner_reason": reason,
        "trace": append_trace(
            state,
            "Planning Agent",
            "warning" if action == "no_action" and flags else "ok",
            reason,
            [
                f"Severity: {severity}",
                f"Diagnosis confidence: {state.get('confidence', 0):.2f}",
            ],
        ),
    }


# ============================================================
# 8. INDEPENDENT SAFETY CHAIN
# ============================================================

def safety_node(state: AgentState) -> dict[str, Any]:
    """
    Enforce preliminary workflow safeguards and the central safety policy.
    The policy must approve an action before the executor can perform it.
    """
    action = state.get("proposed_action", "no_action")
    telemetry = state.get("telemetry", {})
    flags = state.get("flags", [])
    confidence = as_float(state.get("confidence"), 0.0)

    reasons: list[str] = []

    # Workflow-level checks that must run before policy approval.
    if state.get("needs_human_review", False):
        reasons.append("Supervisor requested human review.")

    if not state.get("sensor_valid", True):
        reasons.append("Telemetry validation failed.")

    if state.get("severity") == "high":
        reasons.append(
            "High-severity events require operator review."
        )

    allowed_actions = {"fan_on", "fan_off", "no_action"}
    if action not in allowed_actions:
        reasons.append("Action is not allowlisted.")

    # Default to denial. Only an explicit policy approval can authorize
    # a state-changing action.
    policy_result: dict[str, Any] = {
        "allowed": False,
        "approved_action": "no_action",
        "reasons": [],
    }

    if not reasons:
        try:
            policy_result = validate_action(
                telemetry=telemetry,
                action=action,
                confidence=confidence,
            )
        except Exception as exc:
            # Fail closed if the safety policy errors or its interface
            # is incompatible with this agent.
            reasons.append(
                f"Safety policy failed; action denied ({type(exc).__name__})."
            )

    policy_allowed = bool(policy_result.get("allowed", False))

    if not policy_allowed and not reasons:
        policy_reasons = policy_result.get("reasons", [])
        if isinstance(policy_reasons, list):
            reasons.extend(str(item) for item in policy_reasons)
        elif policy_reasons:
            reasons.append(str(policy_reasons))
        else:
            reasons.append("Central safety policy denied the action.")

    approved_action = "no_action"

    if not reasons and policy_allowed:
        candidate = policy_result.get("approved_action", "no_action")

        if candidate in allowed_actions:
            approved_action = candidate
        else:
            reasons.append(
                "Safety policy returned an invalid approved action."
            )

    allowed = not reasons and policy_allowed

    # A blocked action must never remain approved.
    if not allowed:
        approved_action = "no_action"

    if allowed:
        message = f"Safety policy approved: {approved_action}."
        status = "ok"
    else:
        message = (
            "Action blocked; no actuator change authorized. "
            + " ".join(reasons)
        )
        status = "blocked"

    return {
        "approved_action": approved_action,
        "safety_allowed": allowed,
        "trace": append_trace(
            state,
            "Safety Controller",
            status,
            message,
            [
                f"Requested action: {action}",
                f"Approved action: {approved_action}",
                f"Confidence: {confidence:.2f}",
                f"Active flags: {flags}",
            ] + reasons,
        ),
    }


# ============================================================
# 9. SIMULATED EXECUTION CHAIN
# ============================================================

def executor_node(state: AgentState) -> dict[str, Any]:
    """Change virtual state only; never control physical hardware."""
    telemetry = dict(state.get("telemetry", {}))
    action = state.get("approved_action", "no_action")

    # Fail closed: no action unless safety explicitly approved it.
    if not state.get("safety_allowed", False):
        action = "no_action"

    if action not in {"fan_on", "fan_off", "no_action"}:
        action = "no_action"

    if action == "fan_on":
        telemetry["fan_on"] = True
        result = "Virtual fan switched on."
    elif action == "fan_off":
        telemetry["fan_on"] = False
        result = "Virtual fan switched off."
    else:
        result = "No simulated actuator change performed."

    return {
        "after": telemetry,
        "action_result": result,
        "trace": append_trace(
            state,
            "Simulated Executor",
            (
                "ok"
                if state.get("safety_allowed", False)
                else "blocked"
            ),
            result,
            [
                f"Action: {action}",
                "Execution target: virtual telemetry state only.",
            ],
        ),
    }


# ============================================================
# 10. VERIFICATION CHAIN
# ============================================================

def verification_node(state: AgentState) -> dict[str, Any]:
    """Verify virtual state changes without claiming physical recovery."""
    before = state.get("telemetry", {})
    after = state.get("after", before)
    action = state.get("approved_action", "no_action")

    if action == "fan_on" and after.get("fan_on") is True:
        verification = (
            "PASS: virtual fan state changed to ON. Physical cooling "
            "effect has not been measured."
        )
        status = "ok"

    elif action == "fan_off" and after.get("fan_on") is False:
        verification = (
            "PASS: virtual fan state changed to OFF. Physical equipment "
            "state has not been measured."
        )
        status = "ok"

    elif action == "no_action":
        verification = (
            "INFO: no actuator change was requested; this is not evidence "
            "that the underlying fault has been resolved."
        )
        status = "info"

    else:
        verification = (
            "WARNING: the expected simulated state change was not observed."
        )
        status = "warning"

    evidence = [
        f"Fan before: {before.get('fan_on')}",
        f"Fan after: {after.get('fan_on')}",
        f"Execution result: {state.get('action_result', 'unknown')}",
    ]

    return {
        "verification": verification,
        "trace": append_trace(
            state,
            "Verification Agent",
            status,
            verification,
            evidence,
        ),
    }


# ============================================================
# 11. COMPATIBILITY HELPER
# ============================================================

def run_agents(
    telemetry: Telemetry,
    previous: Telemetry | None = None,
) -> dict[str, Any]:
    """
    Compatibility runner for legacy callers and smoke tests.

    This runs the nodes in sequence. The full multi-chain graph should
    use its own supervisor and investigation routing.
    """
    state: AgentState = {
        "telemetry": telemetry.model_dump(mode="json"),
        "previous": (
            previous.model_dump(mode="json") if previous else None
        ),
        "flags": [],
        "severity": "low",
        "trace": [],
        "investigation_round": 0,
        "max_investigation_rounds": 1,
        "investigation_evidence": [],
        "confidence": 0.0,
        "needs_human_review": False,
        "sensor_valid": True,
    }

    state.update(sensor_node(state))
    state.update(anomaly_node(state))

    if not state.get("flags"):
        state.update(healthy_node(state))
        return state

    state.update(diagnosis_node(state))
    state.update(planner_node(state))
    state.update(safety_node(state))
    state.update(executor_node(state))
    state.update(verification_node(state))

    return state
