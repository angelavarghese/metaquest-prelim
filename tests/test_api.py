"""
Smoke tests for the FastAPI server (M05).
Run with: pytest tests/test_api.py -v

These use TestClient (no running server needed).
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

def test_list_tasks_returns_three():
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) == 3
    names = {t["name"] for t in tasks}
    assert "single_clear_decision" in names
    assert "batch_with_quota" in names
    assert "negotiation_and_edge_cases" in names


def test_list_tasks_have_required_fields():
    r = client.get("/tasks")
    for task in r.json()["tasks"]:
        assert "name" in task
        assert "description" in task
        assert "difficulty" in task
        assert "max_steps" in task


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

def test_reset_empty_body_defaults_to_task1():
    r = client.post("/reset", json={})
    assert r.status_code == 200
    obs = r.json()
    assert "applications" in obs
    assert "inbox" in obs
    assert "step" in obs
    assert obs["step"] == 0


def test_reset_explicit_task1():
    r = client.post("/reset", json={"task": "single_clear_decision"})
    assert r.status_code == 200
    obs = r.json()
    assert len(obs["applications"]) == 1


def test_reset_unknown_task_returns_422():
    r = client.post("/reset", json={"task": "nonexistent_task"})
    assert r.status_code == 422


def test_reset_clears_state():
    """Two resets in a row should both return step=0."""
    client.post("/reset", json={"task": "single_clear_decision"})
    r = client.post("/reset", json={"task": "single_clear_decision"})
    assert r.json()["step"] == 0


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

def test_state_returns_episode_state():
    client.post("/reset", json={"task": "single_clear_decision"})
    r = client.get("/state")
    assert r.status_code == 200
    # EpisodeState should have at minimum these fields
    body = r.json()
    assert "step" in body


# ---------------------------------------------------------------------------
# /step — basic happy path
# ---------------------------------------------------------------------------

def test_step_decision_action():
    client.post("/reset", json={"task": "single_clear_decision"})
    # Grab candidate ID from observation
    obs_r = client.post("/reset", json={"task": "single_clear_decision"})
    candidate_id = obs_r.json()["applications"][0]["candidate_id"]

    action = {
        "action_type": "decision",
        "candidate_id": candidate_id,
        "decision": "accept",
        "reason": "Meets all criteria.",
    }
    r = client.post("/step", json=action)
    assert r.status_code == 200
    result = r.json()
    assert "observation" in result
    assert "reward" in result
    assert "done" in result
    assert 0.0 <= result["reward"] <= 1.0


def test_step_compose_email_action():
    obs_r = client.post("/reset", json={"task": "single_clear_decision"})
    candidate_id = obs_r.json()["applications"][0]["candidate_id"]
    candidate_email = obs_r.json()["applications"][0]["email"]

    action = {
        "action_type": "compose_email",
        "recipient": candidate_email,
        "subject": "Your application outcome",
        "body": "Dear Candidate, we are pleased to accept your application for the role.",
        "candidate_id": candidate_id,
    }
    r = client.post("/step", json=action)
    assert r.status_code == 200


def test_step_after_done_returns_409():
    obs_r = client.post("/reset", json={"task": "single_clear_decision"})
    candidate = obs_r.json()["applications"][0]
    cid = candidate["candidate_id"]
    cemail = candidate["email"]

    # Accept
    client.post("/step", json={
        "action_type": "decision", "candidate_id": cid,
        "decision": "accept", "reason": "Good fit."
    })
    # Compose
    client.post("/step", json={
        "action_type": "compose_email", "candidate_id": cid,
        "recipient": cemail, "subject": "Offer", "body": "You are accepted for the role."
    })
    # Send
    client.post("/step", json={"action_type": "send_email", "candidate_id": cid})

    # Now episode should be done — next step must 409
    r = client.post("/step", json={"action_type": "send_email", "candidate_id": cid})
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# Content-type
# ---------------------------------------------------------------------------

def test_reset_requires_json_content_type():
    r = client.post(
        "/reset",
        data="task=single_clear_decision",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    # FastAPI returns 422 for non-JSON bodies when JSON is expected
    assert r.status_code in (415, 422)