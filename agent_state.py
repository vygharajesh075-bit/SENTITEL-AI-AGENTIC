
from typing import Any, TypedDict


class Hypothesis(TypedDict, total=False):
    cause: str
    confidence: float
    supporting_evidence: list[str]
    contradicting_evidence: list[str]


class AgentGraphState(TypedDict, total=False):
    # Original SENTINEL state
    telemetry: dict[str, Any]
    previous: dict[str, Any] | None
    flags: list[str]
    severity: str
    trace: list[dict[str, Any]]
    diagnosis: str
    diagnosis_evidence: list[str]
    proposed_action: str
    planner_reason: str
    approved_action: str
    action_result: str
    after: dict[str, Any]
    verification: str
    incident: dict[str, Any] | None

    # Multi-chain orchestration state
    goal: str
    next_chain: str
    route_reason: str
    hypotheses: list[Hypothesis]
    investigation_round: int
    max_investigation_rounds: int
    tool_calls: int
    max_tool_calls: int
    confidence: float
    needs_human_review: bool
    safety_allowed: bool
    execution_mode: str
    errors: list[str]
