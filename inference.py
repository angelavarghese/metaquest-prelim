"""
inference.py — Baseline agent loop for the Job Application Environment.

Environment variables (all consumed via os.environ):
  API_BASE_URL   — OpenAI-compatible base URL  (default: http://localhost:8000/v1)
  MODEL_NAME     — Model to use                (default: gpt-4o-mini)
  HF_TOKEN       — API key, NO default         (required in HF Space / remote runs)
  LOCAL_IMAGE_NAME — If set, used for logging only (optional)

Stdout format (exactly one block per task):
  [START] {"task": ..., "model": ...}
  [STEP]  {"task": ..., "step": ..., "action_type": ..., "reward": ..., "done": ...}
  [END]   {"task": ..., "steps": ..., "final_reward": ..., "score": ...}
"""

import json
import os
import sys
import time
import traceback
from typing import Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")  # intentionally no default

OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

TASKS = ["single_clear_decision", "batch_with_quota", "negotiation_and_edge_cases"]

CONTEXT_WINDOW = 3  # keep last N observations in the prompt

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN or "sk-placeholder",
    base_url=OPENAI_BASE_URL,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI recruitment assistant operating inside a job application review environment.

Your job is to process job applications by taking one action per turn. You must ALWAYS respond with a single valid JSON object matching exactly one of the four action types below. No prose, no markdown, no explanation — raw JSON only.

## Action Types

1. Make a hiring decision:
{"action_type": "decision", "candidate_id": "<id>", "decision": "accept|reject|shortlist", "reason": "<required justification>"}

2. Compose an email to a candidate:
{"action_type": "compose_email", "candidate_id": "<id>", "recipient": "<email address>", "subject": "<subject>", "body": "<full email body>"}

3. Send a composed email (must compose first):
{"action_type": "send_email", "candidate_id": "<id>"}

4. Ask a candidate a clarifying question:
{"action_type": "request_info", "candidate_id": "<id>", "question": "<your question>"}

## Rules
- Always make a decision (accept/reject/shortlist) before composing an email for that candidate.
- After composing an email, call send_email to deliver it.
- Every candidate that receives a decision must also receive a notification email (compose + send).
- In emails: always include the candidate's name, the role name, a clear accept/reject statement, and next steps or personalised feedback. Never leave placeholder text like [NAME] or [ROLE].
- Respect salary band constraints when handling negotiation threads.
- Work efficiently — unnecessary steps are penalised.

## Objective
Maximise your episode score by making correct decisions, sending high-quality emails, and delivering all notifications within the step budget.
"""

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task: str) -> dict:
    r = requests.post(f"{API_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{API_BASE_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def build_user_prompt(obs: dict, step: int, last_reward: float | None, history: list[dict]) -> str:
    lines = []

    if history:
        lines.append("## Recent observations (last {} steps)".format(len(history)))
        for i, past_obs in enumerate(history):
            lines.append(f"### Step {past_obs.get('step', '?')} observation (summary)")
            lines.append(f"- Pending decisions: {past_obs.get('pending_decisions', [])}")
            lines.append(f"- Inbox threads: {len(past_obs.get('inbox', []))}")
            lines.append(f"- Last action result: {past_obs.get('last_action_result')}")

    lines.append("\n## Current observation (step {})".format(step))
    lines.append(f"Task: {obs.get('task_description', '')}")
    lines.append(f"Step: {step}")
    if last_reward is not None:
        lines.append(f"Last step reward: {last_reward:.4f}")
    lines.append(f"Pending decisions: {obs.get('pending_decisions', [])}")

    apps = obs.get("applications", [])
    if apps:
        lines.append(f"\n## Applications ({len(apps)} total)")
        for app in apps:
            lines.append(json.dumps(app))

    inbox = obs.get("inbox", [])
    if inbox:
        lines.append(f"\n## Inbox ({len(inbox)} threads)")
        for thread in inbox:
            lines.append(json.dumps(thread))

    lines.append("\nRespond with a single JSON action object.")
    return "\n".join(lines)


def call_llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> dict:
    """Parse LLM output to a JSON action dict. Falls back to request_info on failure."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Safe fallback — logs a step without mutating meaningful state
        return {
            "action_type": "request_info",
            "candidate_id": "unknown",
            "question": "Could you please clarify your application details?",
        }


# ---------------------------------------------------------------------------
# Stdout helpers
# ---------------------------------------------------------------------------

def emit(tag: str, payload: dict) -> None:
    print(f"[{tag}] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> dict:
    emit("START", {"task": task_name, "model": MODEL_NAME})

    obs = env_reset(task_name)
    step = 0
    last_reward: float | None = None
    cumulative_reward = 0.0
    obs_history: list[dict] = []
    final_reward = 0.0

    try:
        while True:
            # Build prompt
            user_prompt = build_user_prompt(obs, step, last_reward, obs_history[-CONTEXT_WINDOW:])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM
            raw = call_llm(messages)
            action = parse_action(raw)

            # Step environment
            result = env_step(action)

            step_reward = result.get("reward", 0.0)
            done = result.get("done", False)
            cumulative_reward += step_reward
            final_reward = step_reward  # env returns episode score when done, step reward otherwise

            emit("STEP", {
                "task": task_name,
                "step": step,
                "action_type": action.get("action_type", "unknown"),
                "reward": round(step_reward, 4),
                "done": done,
            })

            # Slide observation window
            obs_history.append(obs)
            obs = result.get("observation", obs)
            last_reward = step_reward
            step += 1

            if done:
                # Final score is in the result when done=True
                final_reward = result.get("score", result.get("reward", cumulative_reward))
                break

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        # [END] must always be emitted
        emit("END", {
            "task": task_name,
            "steps": step,
            "final_reward": round(final_reward, 4),
            "score": round(final_reward, 4),
            "error": str(exc),
        })
        return {"task": task_name, "steps": step, "score": final_reward, "error": str(exc)}

    emit("END", {
        "task": task_name,
        "steps": step,
        "final_reward": round(final_reward, 4),
        "score": round(final_reward, 4),
    })
    return {"task": task_name, "steps": step, "score": final_reward}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN and OPENAI_BASE_URL.startswith("https://"):
        print(
            "[WARN] HF_TOKEN is not set. Remote LLM calls will likely fail. "
            "Set HF_TOKEN in your environment.",
            file=sys.stderr,
        )

    results = []
    for task in TASKS:
        result = run_task(task)
        results.append(result)
        time.sleep(1)  # brief pause between tasks

    # Summary to stderr (not stdout, so it doesn't interfere with log parsing)
    print("\n=== Summary ===", file=sys.stderr)
    for r in results:
        score = r.get("score", 0.0)
        err = f"  ERROR: {r['error']}" if "error" in r else ""
        print(f"  {r['task']}: {score:.4f} ({r['steps']} steps){err}", file=sys.stderr)


if __name__ == "__main__":
    main()