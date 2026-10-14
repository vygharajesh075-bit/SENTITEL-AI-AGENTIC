from datetime import datetime, timezone
from typing import Literal, TypedDict, Any
from pydantic import BaseModel, Field

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
    mode: Literal["normal", "temperature_fault", "vibration_fault", "sensor_dropout"]
    enabled: bool = True

class ActionRequest(BaseModel):
    action: Literal["fan_on", "fan_off", "no_action"]
    reason: str = "Dashboard request"

class AgentState(TypedDict, total=False):
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
