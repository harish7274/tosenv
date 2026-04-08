"""
FastAPI Server — ToS OpenEnv
==============================
Exposes the OpenEnv HTTP interface:

  POST /reset?task=<task_name>&seed=<int>  → TosObservation
  POST /step                               → TosStepResult
  GET  /state                              → TosState
  GET  /health                             → {"status": "ok"}
  GET  /                                   → {"status": "ok", ...}

Also serves the optional OpenEnv web UI at /web
(enabled when ENABLE_WEB_INTERFACE=true).
"""

from __future__ import annotations

import os
import sys

# Ensure the tos_env root is on the path regardless of cwd
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import TosAction, TosObservation, TosStepResult, TosState
from server.tos_environment import TosEnvironment

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ToS Risk Analyzer — OpenEnv",
    description=(
        "An OpenEnv environment for training and evaluating AI agents "
        "on legal document risk assessment tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance (single-session server)
# ---------------------------------------------------------------------------

_env: TosEnvironment = TosEnvironment(task_name="binary_risk")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "name":   "tos-risk-analyzer",
        "version": "1.0.0",
        "tasks":  ["binary_risk", "category_classification", "full_audit"],
        "endpoints": {
            "reset":  "POST /reset?task=<name>&seed=<int>",
            "step":   "POST /step",
            "state":  "GET  /state",
            "health": "GET  /health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=TosObservation)
def reset(
    task: str = Query(
        default=os.getenv("TASK_NAME", "binary_risk"),
        description="Task to run: binary_risk | category_classification | full_audit",
    ),
    seed: int = Query(
        default=42,
        description="Random seed for reproducibility.",
    ),
):
    """Start a new episode. Returns the initial observation."""
    global _env
    try:
        _env = TosEnvironment(task_name=task, seed=seed)
        obs  = _env.reset()
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=TosStepResult)
def step(action: TosAction):
    """Execute an action and return the result."""
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=TosState)
def state():
    """Return current episode state / metadata."""
    try:
        return _env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# OpenEnv validate compatibility shim
# These routes mirror the paths checked by `openenv validate`
# ---------------------------------------------------------------------------

@app.get("/openenv.yaml")
def openenv_yaml():
    """Return openenv.yaml metadata."""
    import pathlib
    yaml_path = pathlib.Path(__file__).parent.parent / "openenv.yaml"
    if yaml_path.exists():
        return JSONResponse(content={"yaml_content": yaml_path.read_text()})
    return JSONResponse(content={"error": "openenv.yaml not found"}, status_code=404)


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
