"""
serve.py

A small FastAPI service that exposes WasteSegregationEnv (and an optional
trained agent) over HTTP as JSON, so the environment can be driven from a web
or mobile frontend with no Python/Pygame dependency on the client side.

Every endpoint returns the environment's `to_json()` state object, which a
browser or app can render directly (bins, conveyor item, energy, stats).

Endpoints
---------
    POST /session                -> create a new env session, returns {session_id, state}
    GET  /session/{sid}          -> current state
    POST /session/{sid}/step     -> body {"action": int}; step once, returns {state, reward, done}
    POST /session/{sid}/predict  -> step once using the loaded best agent's action
    DELETE /session/{sid}        -> close a session

Run:
    uv run uvicorn serve:app --reload
    # then open http://127.0.0.1:8000/docs for the interactive API explorer

Serving is intentionally optional: FastAPI/uvicorn are declared under the
`[project.optional-dependencies].serve` extra, so the core training/analysis
project has no web dependency. Install with:  uv sync --extra serve
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from environment.custom_env import N_ACTIONS, WasteSegregationEnv

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - only hit without the serve extra
    raise SystemExit(
        "FastAPI is not installed. Install the serving extra with:\n"
        "    uv sync --extra serve\n"
        "then run:  uv run uvicorn serve:app --reload"
    ) from exc


app = FastAPI(
    title="Waste Segregation RL Environment API",
    description="Drive WasteSegregationEnv over HTTP as JSON for a web/mobile frontend.",
    version="1.0.0",
)

# Allow a separate dev front-end (e.g. a local Vite server) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the Three.js web dashboard shipped with the project.
_WEB_ROOT = Path(__file__).resolve().parent / "web"

# In-memory session store. For a production deployment this would be replaced
# by a per-connection actor or a Redis-backed store; kept in-process here to
# stay dependency-light and easy for a marker to run.
_SESSIONS: dict[str, WasteSegregationEnv] = {}
_AGENT = None  # lazily loaded best trained agent, shared across sessions


class StepBody(BaseModel):
    action: int


def _get_env(sid: str) -> WasteSegregationEnv:
    env = _SESSIONS.get(sid)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session '{sid}'.")
    return env


def _load_best_agent():
    """Lazily load the best trained agent via the same selector as main.py."""
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    from main import find_best_model  # local import to avoid a hard dependency
    from training.utils import load_agent

    found = find_best_model()
    if found is None:
        raise HTTPException(
            status_code=409,
            detail="No trained model available. Train an algorithm and run "
                   "`uv run python -m results.analysis` first.",
        )
    algo, model_path = found
    _AGENT = load_agent(algo, model_path)
    return _AGENT


@app.get("/")
def dashboard() -> FileResponse:
    """Serve the Three.js web dashboard."""
    return FileResponse(_WEB_ROOT / "index.html")


@app.post("/session")
def create_session(seed: Optional[int] = None) -> dict:
    sid = uuid.uuid4().hex[:12]
    env = WasteSegregationEnv(seed=seed)
    env.reset(seed=seed)
    _SESSIONS[sid] = env
    return {"session_id": sid, "state": env.to_json()}


@app.get("/session/{sid}")
def get_state(sid: str) -> dict:
    return {"session_id": sid, "state": _get_env(sid).to_json()}


@app.post("/session/{sid}/step")
def step(sid: str, body: StepBody) -> dict:
    env = _get_env(sid)
    if not (0 <= body.action < N_ACTIONS):
        raise HTTPException(status_code=422, detail=f"action must be in [0, {N_ACTIONS - 1}].")
    _, reward, terminated, truncated, _ = env.step(body.action)
    return {
        "session_id": sid,
        "state": env.to_json(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
    }


@app.post("/session/{sid}/predict")
def predict(sid: str) -> dict:
    env = _get_env(sid)
    agent = _load_best_agent()
    obs = env._get_observation()
    action, _ = agent.predict(obs, deterministic=True)
    _, reward, terminated, truncated, _ = env.step(int(action))
    return {
        "session_id": sid,
        "action": int(action),
        "state": env.to_json(),
        "reward": float(reward),
        "done": bool(terminated or truncated),
    }


@app.delete("/session/{sid}")
def close_session(sid: str) -> dict:
    env = _SESSIONS.pop(sid, None)
    if env is not None:
        env.close()
    return {"closed": sid}
