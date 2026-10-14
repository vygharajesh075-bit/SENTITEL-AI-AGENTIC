
import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .models import Telemetry, ControlRequest, ActionRequest
from .simulator import SensorSimulator
from .agent_graph import run_agents
from .agents import safety_node, step
from .store import AuditStore
from .chat_agent import run_chat, set_data_provider
from .evaluation import evaluate_agent_run
from .memory_service import memory_service
from .safety_policy import validate_action
from .planning_chain import planning_node
from .observability import (
    get_observability_snapshot,
    record_timing,
)


load_dotenv()

# --------------------------------------------------
# SENTINEL RUNTIME
# --------------------------------------------------

simulator = SensorSimulator()
store = AuditStore()

readings = deque(maxlen=120)
incidents = deque(maxlen=60)
agent_trace = deque(maxlen=60)
agent_events = deque(maxlen=200)

clients: set[WebSocket] = set()

latest_status = {
    "mode": "normal",
    "running": True,
    "fan_on": False,
    "connected_clients": 0,
    "database": (
        "MongoDB" if store.persistent else "in-memory fallback"
    ),
}

previous_sample: Telemetry | None = None
simulation_task = None

interval = max(
    3,
    int(os.getenv("SIMULATION_INTERVAL_SECONDS", "8")),
)

# Autonomous orchestration state
autonomy_enabled = True
agent_cycles = 0
last_agent_activity = None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def clean(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, list):
        return [clean(item) for item in value]

    if isinstance(value, dict):
        return {
            key: clean(item)
            for key, item in value.items()
        }

    return value


def get_chat_data():
    return {
        "status": latest_status.copy(),
        "readings": [
            reading.model_dump(mode="json")
            for reading in readings
        ],
        "incidents": [
            clean(incident)
            for incident in incidents
        ],
        "trace": [
            clean(item)
            for item in agent_trace
        ],
        "agent_events": [
            clean(item)
            for item in list(agent_events)[:30]
        ],
    }


set_data_provider(get_chat_data)


# --------------------------------------------------
# WEBSOCKET BROADCAST
# --------------------------------------------------

async def broadcast(payload):
    message = json.dumps(payload, default=str)
    dead = []

    for ws in list(clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)

    for ws in dead:
        clients.discard(ws)

    latest_status["connected_clients"] = len(clients)


# --------------------------------------------------
# AUTONOMOUS EVENT LOGGER
# --------------------------------------------------

async def log_agent_event(
    agent,
    event,
    message,
    details=None,
):
    global last_agent_activity

    item = {
        "id": (
            f"{datetime.now(timezone.utc).timestamp()}"
        ),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "agent": agent,
        "event": event,
        "message": message,
        "details": clean(details or {}),
    }

    agent_events.appendleft(item)
    last_agent_activity = item["timestamp"]

    try:
        store.insert("agent_events", item)
    except Exception as exc:
        print(f"[SENTINEL] Event storage warning: {exc}")

    await broadcast({
        "type": "agent_event",
        "data": item,
    })

    return item


# --------------------------------------------------
# AUTONOMOUS SIMULATION LOOP
# --------------------------------------------------

async def simulation_loop():
    global previous_sample
    global agent_cycles

    while True:
        await asyncio.sleep(interval)

        if not autonomy_enabled:
            continue

        cycle_started = time.perf_counter()
        cycle_success = False

        try:
            sample = simulator.next()

            if sample is None:
                await log_agent_event(
                    "Monitoring Agent",
                    "sensor_dropout",
                    "No fresh virtual sensor sample was received.",
                )

                await broadcast({
                    "type": "sensor_status",
                    "data": {
                        "online": False,
                        "message": (
                            "Virtual sensor dropout: "
                            "no fresh sample received."
                        ),
                    },
                })
                continue

            agent_cycles += 1

            await log_agent_event(
                "Monitoring Agent",
                "cycle_started",
                f"Monitoring cycle #{agent_cycles} started.",
                {
                    "cycle": agent_cycles,
                    "mode": simulator.mode,
                },
            )

            # Run the existing multi-agent workflow.
            result = await asyncio.to_thread(
                run_agents,
                sample,
                previous_sample,
            )

            trace = result.get("trace", [])

            for item in trace:
                item_data = clean(item)

                if not isinstance(item_data, dict):
                    item_data = {
                        "result": str(item_data)
                    }

                agent_name = (
                    item_data.get("agent")
                    or item_data.get("name")
                    or item_data.get("node")
                    or "SENTINEL Agent"
                )

                event_status = (
                    item_data.get("status")
                    or item_data.get("event")
                    or "completed"
                )

                event_message = (
                    item_data.get("message")
                    or item_data.get("reason")
                    or item_data.get("action")
                    or f"{agent_name} completed a workflow step."
                )

                await log_agent_event(
                    agent_name,
                    str(event_status),
                    str(event_message),
                    item_data,
                )

            adjusted_data = result.get(
                "after",
                sample.model_dump(mode="json"),
            )

            adjusted = Telemetry.model_validate(
                adjusted_data
            )

            simulator.fan_on = adjusted.fan_on
            previous_sample = adjusted

            readings.append(adjusted)

            for item in reversed(trace):
                agent_trace.appendleft(item)

            store.insert(
                "telemetry",
                adjusted.model_dump(mode="json"),
            )

            store.insert(
                "agent_traces",
                {
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "trace": clean(trace),
                    "mode": simulator.mode,
                    "cycle": agent_cycles,
                },
            )

            incident = result.get("incident")

            if incident:
                incidents.appendleft(incident)
                store.insert("incidents", clean(incident))

                await log_agent_event(
                    "Incident Agent",
                    "incident_detected",
                    "The agent workflow produced an incident.",
                    clean(incident),
                )

            latest_status.update({
                "mode": simulator.mode,
                "running": True,
                "fan_on": simulator.fan_on,
                "autonomy_enabled": autonomy_enabled,
                "agent_cycles": agent_cycles,
            })

            await log_agent_event(
                "Orchestrator",
                "cycle_completed",
                f"Monitoring cycle #{agent_cycles} completed.",
                {
                    "cycle": agent_cycles,
                    "incident_detected": bool(incident),
                    "fan_on": simulator.fan_on,
                },
            )

            await broadcast({
                "type": "telemetry",
                "data": adjusted.model_dump(mode="json"),
                "trace": clean(trace),
                "incident": clean(incident) if incident else None,
                "status": latest_status.copy(),
            })

            cycle_success = True

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            print(f"[SENTINEL] Workflow error: {exc}")

            await log_agent_event(
                "Orchestrator",
                "workflow_error",
                "An autonomous monitoring cycle failed.",
                {
                    "error": str(exc),
                    "cycle": agent_cycles,
                },
            )

            await broadcast({
                "type": "workflow_error",
                "message": str(exc),
            })

        finally:
            record_timing(
                "simulation.cycle",
                time.perf_counter() - cycle_started,
                details={
                    "cycle": agent_cycles,
                    "success": cycle_success,
                },
            )


# --------------------------------------------------
# APPLICATION LIFECYCLE
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global simulation_task

    simulation_task = asyncio.create_task(
        simulation_loop()
    )

    try:
        yield
    finally:
        if simulation_task:
            simulation_task.cancel()

            try:
                await simulation_task
            except asyncio.CancelledError:
                pass

        store.close()


app = FastAPI(
    title="SENTINEL Agentic IoT Intelligence API",
    version="2.1.0",
    description=(
        "Hardware-free virtual IoT simulation with "
        "a multi-agent workflow, autonomous monitoring, "
        "event tracking, and safety controls."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# CHAT AGENT
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )
    conversation_id: str | None = None


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    started = time.perf_counter()
    success = False

    try:
        result = await asyncio.to_thread(
            run_chat,
            request.message,
            request.conversation_id,
        )

        success = True

        return {
            "success": True,
            **result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        print(
            f"[SENTINEL CHAT] Error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The SENTINEL chat agent failed. "
                "Check the backend terminal."
            ),
        )

    finally:
        record_timing(
            "api.chat",
            time.perf_counter() - started,
            details={"success": success},
        )


# --------------------------------------------------
# CORE API ROUTES
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "SENTINEL",
        "status": "online",
        "mode": simulator.mode,
        "llm_configured": bool(
            os.getenv("GROQ_API_KEY")
        ),
        "database": latest_status["database"],
        "autonomy_enabled": autonomy_enabled,
        "agent_cycles": agent_cycles,
        "docs": "/docs",
    }


@app.get("/api/status")
async def get_status():
    return {
        **latest_status,
        "mode": simulator.mode,
        "running": simulation_task is not None,
        "autonomy_enabled": autonomy_enabled,
        "agent_cycles": agent_cycles,
        "last_agent_activity": last_agent_activity,
        "reading_count": len(readings),
        "incident_count": len(incidents),
        "latest": (
            readings[-1].model_dump(mode="json")
            if readings
            else None
        ),
    }


@app.get("/api/readings")
async def get_readings():
    return [
        reading.model_dump(mode="json")
        for reading in readings
    ]


@app.get("/api/incidents")
async def get_incidents():
    return list(incidents)[:60]


@app.get("/api/trace")
async def get_trace():
    return [
        clean(item)
        for item in list(agent_trace)[:60]
    ]


# --------------------------------------------------
# AGENT CONTROL CENTER API
# --------------------------------------------------

@app.get("/api/agent/overview")
async def get_agent_overview():
    return {
        "autonomy_enabled": autonomy_enabled,
        "cycles_completed": agent_cycles,
        "last_activity": last_agent_activity,
        "active_incidents": len(incidents),
        "connected_clients": len(clients),
        "event_count": len(agent_events),
        "recent_events": list(agent_events)[:50],
    }


@app.get("/api/agent/events")
async def get_agent_events():
    return list(agent_events)[:100]


@app.get("/api/agent/metrics")
async def get_agent_metrics():
    """Return SENTINEL agent latency and observability metrics."""
    return get_observability_snapshot()


@app.post("/api/agent/pause")
async def pause_autonomy():
    global autonomy_enabled

    autonomy_enabled = False
    latest_status["autonomy_enabled"] = False

    await log_agent_event(
        "Orchestrator",
        "paused",
        "Autonomous monitoring paused by operator.",
    )

    await broadcast({
        "type": "status",
        "data": latest_status.copy(),
    })

    return {
        "success": True,
        "autonomy_enabled": autonomy_enabled,
        "message": "Autonomous monitoring paused.",
    }


@app.post("/api/agent/resume")
async def resume_autonomy():
    global autonomy_enabled

    autonomy_enabled = True
    latest_status["autonomy_enabled"] = True

    await log_agent_event(
        "Orchestrator",
        "resumed",
        "Autonomous monitoring resumed by operator.",
    )

    await broadcast({
        "type": "status",
        "data": latest_status.copy(),
    })

    return {
        "success": True,
        "autonomy_enabled": autonomy_enabled,
        "message": "Autonomous monitoring resumed.",
    }


# --------------------------------------------------
# SIMULATION CONTROLS
# --------------------------------------------------

@app.post("/api/simulation/mode")
async def set_simulation_mode(
    request: ControlRequest,
):
    started = time.perf_counter()
    success = False

    try:
        simulator.set_mode(
            request.mode if request.enabled else "normal"
        )

        latest_status["mode"] = simulator.mode

        await log_agent_event(
            "Simulation Controller",
            "mode_changed",
            f"Virtual sensor mode set to {simulator.mode}.",
            {
                "mode": simulator.mode,
                "enabled": request.enabled,
            },
        )

        await broadcast({
            "type": "status",
            "data": latest_status.copy(),
        })

        success = True

        return {
            "success": True,
            "mode": simulator.mode,
            "message": (
                f"Virtual sensor mode set to {simulator.mode}."
            ),
        }

    finally:
        record_timing(
            "api.simulation_mode",
            time.perf_counter() - started,
            details={"success": success},
        )


# --------------------------------------------------
# MANUAL CONTROL WITH SAFETY CHECK
# --------------------------------------------------

@app.post("/api/control")
async def manual_control(request: ActionRequest):
    started = time.perf_counter()
    success = False

    try:
        if not readings:
            raise HTTPException(
                status_code=409,
                detail="Wait for the first telemetry sample.",
            )

        current = readings[-1]
        proposed = request.action

        state = {
            "telemetry": current.model_dump(mode="json"),
            "proposed_action": proposed,
        }

        safety_result = safety_node(state)
        approved = safety_result["approved_action"]

        if approved == "fan_on":
            simulator.fan_on = True
        elif approved == "fan_off":
            simulator.fan_on = False

        latest_status["fan_on"] = simulator.fan_on
        safety_trace = safety_result["trace"][-1]

        await log_agent_event(
            "Safety Agent",
            "manual_action_checked",
            f"Safety workflow processed action: {approved}.",
            {
                "requested_action": proposed,
                "approved_action": approved,
                "reason": request.reason,
                "safety": clean(safety_trace),
            },
        )

        await broadcast({
            "type": "manual_action",
            "action": approved,
            "reason": request.reason,
            "safety": clean(safety_trace),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        })

        success = True

        return {
            "success": True,
            "action": approved,
            "reason": request.reason,
            "safety": clean(safety_trace),
        }

    finally:
        record_timing(
            "api.control",
            time.perf_counter() - started,
            details={"success": success},
        )


# --------------------------------------------------
# LIVE WEBSOCKET
# --------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    latest_status["connected_clients"] = len(clients)

    try:
        await ws.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "readings": [
                        reading.model_dump(mode="json")
                        for reading in readings
                    ],
                    "incidents": [
                        clean(item)
                        for item in incidents
                    ],
                    "trace": [
                        clean(item)
                        for item in agent_trace
                    ],
                    "agent_events": list(agent_events),
                    "status": {
                        **latest_status,
                        "autonomy_enabled": autonomy_enabled,
                        "agent_cycles": agent_cycles,
                        "last_agent_activity": last_agent_activity,
                    },
                },
                default=str,
            )
        )

        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass

    finally:
        clients.discard(ws)
        latest_status["connected_clients"] = len(clients)
