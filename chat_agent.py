
# backend/app/chat_agent.py

import json
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your backend .env file."
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.2,
)

# main.py registers a function that provides current SENTINEL data.
_data_provider: Callable[[], dict[str, Any]] | None = None


def set_data_provider(provider: Callable[[], dict[str, Any]]) -> None:
    global _data_provider
    _data_provider = provider


def get_sentinel_data() -> dict[str, Any]:
    if _data_provider is None:
        return {
            "status": {"running": False, "message": "Data provider not configured"},
            "readings": [],
            "incidents": [],
            "trace": [],
        }

    try:
        return _data_provider()
    except Exception as exc:
        return {
            "status": {"running": False, "error": str(exc)},
            "readings": [],
            "incidents": [],
            "trace": [],
        }


def to_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


@tool
def inspect_sensor_network() -> str:
    """Inspect recent SENTINEL virtual sensor readings, values, and trends.
    Use this for questions about telemetry, sensors, temperature, humidity,
    fill levels, anomalies, or the latest sensor readings."""
    data = get_sentinel_data()
    readings = data.get("readings", [])

    return to_json({
        "tool": "inspect_sensor_network",
        "reading_count": len(readings),
        "latest_readings": readings[-10:],
        "message": (
            "Showing the latest available readings."
            if readings else
            "No sensor readings are available yet. The simulator may still be starting."
        ),
    })


@tool
def inspect_recent_incidents() -> str:
    """Inspect recent SENTINEL incidents and alerts.
    Use this for questions about faults, alerts, anomalies, or incident history."""
    data = get_sentinel_data()
    incidents = data.get("incidents", [])

    return to_json({
        "tool": "inspect_recent_incidents",
        "incident_count": len(incidents),
        "recent_incidents": incidents[:10],
        "message": (
            "Showing the most recent recorded incidents."
            if incidents else
            "No incidents are currently recorded in the available session data."
        ),
    })


@tool
def inspect_system_status() -> str:
    """Inspect SENTINEL's current application status, simulation mode,
    fan state, connected clients, reading count, and incident count."""
    data = get_sentinel_data()
    status = data.get("status", {})
    readings = data.get("readings", [])
    incidents = data.get("incidents", [])

    return to_json({
        "tool": "inspect_system_status",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reading_count": len(readings),
        "incident_count": len(incidents),
        "latest_reading": readings[-1] if readings else None,
    })


TOOLS = [
    inspect_sensor_network,
    inspect_recent_incidents,
    inspect_system_status,
]

SYSTEM_PROMPT = """
You are SENTINEL AI, the agentic IoT intelligence assistant.

You work with a virtual IoT monitoring system that has sensor telemetry,
incidents, safety checks, an agent workflow, and system status information.

AGENT BEHAVIOUR:
1. Understand what the user is asking.
2. Decide whether you need a tool to answer accurately.
3. Call one or more tools when the question requires current SENTINEL data.
4. Read and reason about the tool results before answering.
5. Give the user a clear, useful answer grounded in those results.

RULES:
- Never invent readings, incidents, timestamps, device states, or tool results.
- If there are no readings, say that no readings are currently available.
- An empty incident list does not prove that no faults exist.
- Do not claim that you controlled hardware or changed simulation settings.
- Do not claim an AI anomaly detector ran unless the supplied data confirms it.
- Explain important warnings in simple language.
- For general questions, answer normally without unnecessary tool calls.
- When appropriate, briefly mention which checks you performed.
- Do not expose API keys, environment variables, or secrets.
"""

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)


def run_chat(
    message: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ]
    })

    messages = result.get("messages", [])
    answer = ""
    trace = []

    for item in messages:
        item_type = getattr(item, "type", "")

        if item_type == "ai":
            content = getattr(item, "content", "")

            if isinstance(content, str) and content.strip():
                answer = content

            for call in getattr(item, "tool_calls", []) or []:
                trace.append({
                    "type": "tool_call",
                    "tool": call.get("name", "unknown"),
                    "input": call.get("args", {}),
                })

        elif item_type == "tool":
            trace.append({
                "type": "tool_result",
                "tool": getattr(item, "name", "unknown"),
                "result": str(getattr(item, "content", ""))[:4000],
            })

    return {
        "answer": answer or "I could not generate an answer. Please try again.",
        "conversation_id": conversation_id,
        "trace": trace,
    }
