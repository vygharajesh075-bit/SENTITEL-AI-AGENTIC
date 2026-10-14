
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

ALLOWED_ROUTES = {
    "monitor",
    "investigate",
    "plan",
    "human_review",
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT_SECONDS = 8.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _result(
    route: str,
    reason: str,
    provider: str = "fallback",
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    if route not in ALLOWED_ROUTES:
        route = "human_review"

    return {
        "next_chain": route,
        "route_reason": reason,
        "needs_human_review": route == "human_review",
        "supervisor_provider": provider,
        "supervisor_latency_ms": round(latency_ms, 2),
    }




@lru_cache(maxsize=1)
def _get_groq_llm():
    from langchain_groq import ChatGroq

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    options = {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "timeout": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": 0,
    }

    if model == "openai/gpt-oss-20b":
        options["reasoning_effort"] = "low"
        options["reasoning_effort"] = "low"
    else:
        options["reasoning_format"] = "hidden"

    return ChatGroq(**options)



def _parse_groq_json(raw: Any) -> dict[str, Any]:
    """Extract a JSON object from a Groq response."""
    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Groq returned an empty response")

    text = raw.strip()

    # Remove Markdown fences if present.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find a JSON object surrounded by other text.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Groq response did not contain valid JSON")


def _fallback_decision(state: dict[str, Any]) -> dict[str, Any]:
    """Fast deterministic routing; no network request required."""
    flags = state.get("flags") or []
    severity = str(state.get("severity", "low")).strip().lower()
    sensor_valid = state.get("sensor_valid", True)
    confidence = _as_float(state.get("confidence", 1.0), 1.0)

    if sensor_valid is False:
        return _result(
            "human_review",
            "Sensor data is invalid; human review is required.",
        )

    if severity in {"critical", "high"}:
        return _result(
            "human_review",
            "High or critical severity requires human review.",
        )

    if state.get("needs_human_review", False):
        return _result(
            "human_review",
            "The current state requires human review.",
        )

    if not flags:
        return _result(
            "monitor",
            "No active anomaly flags; continue monitoring.",
        )

    if confidence < 0.60:
        return _result(
            "investigate",
            "Confidence is low; investigate before planning.",
        )

    return _result(
        "plan",
        "Anomaly detected; proceed to action planning.",
    )


def _extract_response_text(response: Any) -> str:
    """Extract textual content from common LangChain response formats."""
    content = getattr(response, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text_part = block.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)

        return "\n".join(parts).strip()

    return ""


def _groq_decision(state: dict[str, Any]) -> dict[str, Any] | None:
    """Use Groq for eligible anomaly-routing decisions."""
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY is unavailable; using fallback.")
        return None

    started = perf_counter()

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = _get_groq_llm()

        payload = {
            "flags": state.get("flags") or [],
            "severity": state.get("severity", "unknown"),
            "sensor_valid": state.get("sensor_valid", True),
            "confidence": state.get("confidence", 0.0),
            "telemetry": state.get("telemetry") or {},
            "diagnosis": state.get("diagnosis") or {},
            "needs_human_review": state.get(
                "needs_human_review", False
            ),
        }

        system_prompt = """
You are the routing supervisor for the SENTINEL IoT system.

Return exactly one JSON object, with no Markdown or surrounding prose:
{
  "next_chain": "plan",
  "route_reason": "Anomaly detected"
}

Allowed next_chain values:
monitor, investigate, plan, human_review

Rules:
- Invalid sensor data requires human_review.
- High or critical severity requires human_review.
- If needs_human_review is true, choose human_review.
- If flags is empty, choose monitor.
- If confidence is below 0.60, choose investigate.
- For a confirmed anomaly with adequate confidence, choose plan,
  unless the evidence clearly requires investigation.
- Never claim an actuator action has been executed.
- Do not authorize physical control actions.
- Return a concise JSON object only.
"""

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=json.dumps(payload, default=str)
                ),
            ]
        )

        response_text = _extract_response_text(response)

        if not response_text:
            metadata = getattr(response, "response_metadata", {}) or {}
            usage = getattr(response, "usage_metadata", None)

            logger.warning(
                "Groq returned empty text; finish_reason=%s, model=%s, "
                "usage_metadata_present=%s",
                metadata.get("finish_reason", "unknown"),
                metadata.get("model_name", "unknown"),
                usage is not None,
            )

            additional = getattr(response, "additional_kwargs", {}) or {}
            if additional.get("reasoning_content"):
                logger.warning(
                    "Groq supplied reasoning content but no final answer."
                )

            return None

        data = _parse_groq_json(response_text)

        route = str(data.get("next_chain", "")).strip().lower()
        reason = str(data.get("route_reason", "")).strip()

        if route not in ALLOWED_ROUTES:
            logger.warning("Groq returned an unsupported route.")
            return None

        if not reason:
            reason = "Supervisor selected a route from the current state."

        latency_ms = (perf_counter() - started) * 1000

        return _result(
            route,
            reason,
            provider="groq",
            latency_ms=latency_ms,
        )

    except Exception as exc:
        logger.warning(
            "Groq supervisor failed (%s): %s; using deterministic fallback.",
            type(exc).__name__,
            str(exc)[:300],
        )
        return None


def supervisor_decision(state: Any) -> dict[str, Any]:
    """
    Select a routing decision.

    Obvious cases use the deterministic fast path. Eligible anomalies
    can use Groq, but deterministic safety overrides remain authoritative.
    """
    if not isinstance(state, dict):
        return _result(
            "human_review",
            "Invalid supervisor state; human review is required.",
        )

    fallback = _fallback_decision(state)

    severity = str(state.get("severity", "low")).strip().lower()
    flags = state.get("flags") or []
    confidence = _as_float(state.get("confidence", 1.0), 1.0)

    # Avoid unnecessary API calls for clear-cut cases.
    if (
        state.get("sensor_valid", True) is False
        or severity in {"critical", "high"}
        or state.get("needs_human_review", False)
        or not flags
        or confidence < 0.60
    ):
        return fallback

    groq_result = _groq_decision(state)

    if groq_result is None:
        return fallback

    # Safety overrides always take precedence over model output.
    if state.get("sensor_valid", True) is False:
        return _result(
            "human_review",
            "Sensor data is invalid; human review is required.",
        )

    if severity in {"critical", "high"}:
        return _result(
            "human_review",
            "High or critical severity requires human review.",
        )

    if state.get("needs_human_review", False):
        return _result(
            "human_review",
            "The current state requires human review.",
        )

    # Do not silently dismiss a confirmed anomaly.
    if flags and groq_result["next_chain"] == "monitor":
        return _result(
            "plan",
            "Anomaly flags are present; proceed to action planning.",
            provider="groq",
            latency_ms=groq_result["supervisor_latency_ms"],
        )

    return groq_result
