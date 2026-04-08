import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from server.env import JobApplicationEnv, EpisodeAlreadyDoneError, TaskNotFoundError
from server.models import Action


# ---------------------------------------------------------------------------
# Singleton env + lock
# ---------------------------------------------------------------------------

_env: JobApplicationEnv | None = None
_lock = threading.Lock()


def get_env() -> JobApplicationEnv:
    global _env
    if _env is None:
        _env = JobApplicationEnv()
    return _env


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the singleton on startup
    get_env()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Job Application Environment",
    version="1.0.0",
    description="An OpenEnv-compatible benchmark environment for agent evaluation on recruitment tasks.",
    lifespan=lifespan,
)

def main():
    """Entry point for the openenv multi-mode deployment."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "single_clear_decision"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe used by HF Space ping."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List available task names, descriptions, and difficulties."""
    return {
        "tasks": [
            {
                "name": "single_clear_decision",
                "description": "1 applicant for a junior SWE role with unambiguous fit. Accept/reject + send one email.",
                "difficulty": "easy",
                "max_steps": 4,
                "expected_score": 0.85,
            },
            {
                "name": "batch_with_quota",
                "description": "8 applicants, only 2 slots. Mixed signals — overlapping skills, one overqualified, one missing a cert.",
                "difficulty": "medium",
                "max_steps": 16,
                "expected_score": 0.55,
            },
            {
                "name": "negotiation_and_edge_cases",
                "description": "5 applicants with follow-up threads: counter-offer salary negotiation and a reapplication after rejection.",
                "difficulty": "hard",
                "max_steps": 20,
                "expected_score": 0.25,
            },
        ]
    }


@app.post("/reset")
def reset(body: ResetRequest = None):
    """
    Reset the environment for a given task.
    Empty body `{}` or missing body defaults to task 'single_clear_decision'.
    """
    task_name = (body.task if body else None) or "single_clear_decision"
    with _lock:
        env = get_env()
        try:
            observation = env.reset(task_name)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    return observation.model_dump()


@app.post("/step")
def step(action: Action):
    """Advance the environment by one step with the given action."""
    with _lock:
        env = get_env()
        try:
            result = env.step(action)
        except EpisodeAlreadyDoneError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    return result.model_dump()


@app.get("/state")
def state():
    """Return the current episode state (read-only)."""
    env = get_env()
    return env.state().model_dump()


# ---------------------------------------------------------------------------
# Global exception handler — always return JSON, never HTML
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )

if __name__ == "__main__":
    main()