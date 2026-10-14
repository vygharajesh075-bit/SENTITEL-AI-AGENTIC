
from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import (
    anomaly_node,
    diagnosis_node,
    executor_node,
    healthy_node,
    safety_node,
    sensor_node,
    step,
    verification_node,
)
from .observability import record_agent_run, record_error, record_timing
from .planning_chain import planning_node
from .supervisor import supervisor_decision

logger = logging.getLogger("sentinel")


class MultiChainState(TypedDict, total=False):
    telemetry: dict[str, Any]
    previous: dict[str, Any] | None

    flags: list[Any]
    severity: str
    trace: list[Any]

    sensor_valid: bool
    diagnosis: str
    confidence: float

    proposed_action: Any
    approved_action: Any
    safety_allowed: bool

    next_chain: str
    route_reason: str
    needs_human_review: bool
    supervisor_provider: str

    action_result: Any
    verification: Any
    incident: Any

    investigation_round: int
    max_investigation_rounds: int

    planning_confidence: float
    planning_reasons: list[str]
    evaluation: Any


def _timed_node(name: str, node: Any):
    """Wrap a graph node with latency and error instrumentation."""

    def wrapped(state: MultiChainState) -> dict[str, Any]:
        started = time.perf_counter()

        try:
            result = node(state)
            duration = time.perf_counter() - started

            record_timing(
                component=f"node.{name}",
                duration_seconds=duration,
            )

            return result

        except Exception as exc:
            duration = time.perf_counter() - started

            record_timing(
                component=f"node.{name}",
                duration_seconds=duration,
                details={"success": False},
            )
            record_error(exc, component=f"node.{name}")

            logger.exception("SENTINEL graph node '%s' failed", name)
            raise

    return wrapped


def supervisor_node(state: MultiChainState) -> dict[str, Any]:
    """Select the next stage using the advanced supervisor."""

    started = time.perf_counter()

    try:
        decision = supervisor_decision(state)
    except Exception as exc:
        record_timing(
            "supervisor.decision",
            time.perf_counter() - started,
            details={"success": False},
        )
        record_error(exc, component="supervisor")
        raise

    duration = time.perf_counter() - started

    record_timing(
        "supervisor.decision",
        duration,
        details={
            "provider": decision.get("supervisor_provider", "fallback"),
            "route": decision.get("next_chain", "investigate"),
        },
    )

    trace = list(state.get("trace", []))

    route = decision.get("next_chain", "investigate")
    reason = decision.get(
        "route_reason",
        "The supervisor selected a route.",
    )
    provider = decision.get("supervisor_provider", "fallback")

    trace.append(
        step(
            "Agentic Supervisor",
            "warning" if route == "human_review" else "ok",
            reason,
            [
                f"Selected route: {route}",
                f"Decision provider: {provider}",
                f"Decision latency: {duration:.4f}s",
            ],
        )
    )

    return {
        **decision,
        "trace": trace,
    }


def route_after_supervisor(state: MultiChainState) -> str:
    """Return a graph node name for the selected route."""

    route = state.get("next_chain", "investigate")

    routes = {
        "monitor": "healthy",
        "investigate": "diagnosis",
        "plan": "planning",
        "human_review": "human_review",
    }

    return routes.get(route, "diagnosis")


def human_review_node(state: MultiChainState) -> dict[str, Any]:
    """Stop automated actions and preserve the current safe device state."""

    reason = state.get(
        "route_reason",
        "Human review is required before proceeding.",
    )

    telemetry = state.get("telemetry") or {}

    # Preserve the existing device state; do not issue a control action.
    after = {
        "fan_on": telemetry.get("fan_on", False),
    }

    trace = list(state.get("trace", []))
    trace.append(
        step(
            "Human Review",
            "warning",
            reason,
            ["No automated control action was executed."],
        )
    )

    return {
        "needs_human_review": True,
        "proposed_action": "no_action",
        "approved_action": "no_action",
        "safety_allowed": False,
        "after": after,
        "action_result": {
            "executed": False,
            "reason": reason,
        },
        "trace": trace,
    }


def route_after_anomaly(state: MultiChainState) -> str:
    """Send anomaly results to the supervisor."""

    return "supervisor"


def build_graph():
    """Construct and compile the instrumented SENTINEL workflow."""

    graph = StateGraph(MultiChainState)

    # Wrap existing nodes to measure their execution time.
    graph.add_node("sensor", _timed_node("sensor", sensor_node))
    graph.add_node("anomaly", _timed_node("anomaly", anomaly_node))
    graph.add_node("supervisor", _timed_node("supervisor", supervisor_node))
    graph.add_node("healthy", _timed_node("healthy", healthy_node))
    graph.add_node("diagnosis", _timed_node("diagnosis", diagnosis_node))
    graph.add_node("planning", _timed_node("planning", planning_node))
    graph.add_node("safety", _timed_node("safety", safety_node))
    graph.add_node("executor", _timed_node("executor", executor_node))
    graph.add_node(
        "verification",
        _timed_node("verification", verification_node),
    )
    graph.add_node(
        "human_review",
        _timed_node("human_review", human_review_node),
    )

    graph.add_edge(START, "sensor")
    graph.add_edge("sensor", "anomaly")

    graph.add_conditional_edges(
        "anomaly",
        route_after_anomaly,
        {"supervisor": "supervisor"},
    )

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "healthy": "healthy",
            "diagnosis": "diagnosis",
            "planning": "planning",
            "human_review": "human_review",
        },
    )

    graph.add_edge("healthy", END)
    graph.add_edge("human_review", END)

    graph.add_edge("diagnosis", "planning")
    graph.add_edge("planning", "safety")
    graph.add_edge("safety", "executor")
    graph.add_edge("executor", "verification")
    graph.add_edge("verification", END)

    return graph.compile()


workflow = build_graph()


def _to_dict(value: Any) -> Any:
    """Convert Pydantic models to dictionaries when supported."""

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()

    return value


def run_agents(
    telemetry: Any,
    previous: Any | None = None,
) -> dict[str, Any]:
    """Run the SENTINEL workflow and record its total execution time."""

    started = time.perf_counter()
    success = False
    action = "no_action"

    try:
        initial_state: MultiChainState = {
            "telemetry": _to_dict(telemetry),
            "previous": _to_dict(previous),
            "flags": [],
            "severity": "low",
            "trace": [],
            "sensor_valid": True,
            "confidence": 0.0,
            "needs_human_review": False,
            "investigation_round": 0,
            "max_investigation_rounds": 2,
            "planning_confidence": 0.0,
            "planning_reasons": [],
        }

        result = workflow.invoke(initial_state)
        success = True

        if isinstance(result, dict):
            action = str(
                result.get("approved_action")
                or result.get("proposed_action")
                or result.get("next_chain")
                or "no_action"
            )

        return result

    except Exception as exc:
        record_error(exc, component="agent_graph")
        raise

    finally:
        duration = time.perf_counter() - started

        record_agent_run(
            action=action,
            duration_seconds=duration,
            success=success,
        )
