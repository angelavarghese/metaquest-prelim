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

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")  # intentionally no default

OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
REQUEST_TIMEOUT_SEC: int = int(os.environ.get("REQUEST_TIMEOUT_SEC", "25"))
LLM_TIMEOUT_SEC: int = int(os.environ.get("LLM_TIMEOUT_SEC", "40"))
# Hard stop below 20 minutes to satisfy submission runtime constraints.
MAX_TOTAL_RUNTIME_SEC: int = int(os.environ.get("MAX_TOTAL_RUNTIME_SEC", "1140"))

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

## Strict workflow per candidate (follow in order)
Step A: decision — accept or reject the candidate.
Step B: compose_email — write their notification email.
Step C: send_email — deliver it.
Only move to the next candidate after completing A→B→C for the current one.

## Decision rules
- Use "accept" or "reject" for every candidate. "shortlist" is only valid when the task description explicitly asks you to build a shortlist — otherwise never use it.
- Each candidate_id must receive exactly ONE decision. Never repeat a decision for the same candidate_id.
- Check "pending_decisions" in the observation — only decide for candidates listed there.
- If "pending_decisions" is empty, skip straight to composing emails for any candidates not yet emailed.

## Email rules
- Always include: candidate's name, the role name, a clear accept/reject statement, next steps (if accepted) or personalised feedback (if rejected).
- Never leave placeholder text like [NAME] or [ROLE] in the email body.
- Respect salary band constraints when handling negotiation threads.

## Efficiency
- Work through candidates one at a time: decide → compose → send, then move to the next.
- Unnecessary or repeated actions waste your step budget and reduce your score.

## Objective
Maximise your episode score by making correct decisions, sending high-quality emails, and delivering all notifications within the step budget.
"""

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task: str) -> dict:
    r = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task": task},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{API_BASE_URL}/step", json=action, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{API_BASE_URL}/state", timeout=REQUEST_TIMEOUT_SEC)
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
    
    if "last_action_result" in obs:
        lines.append(f"Last action result: {obs['last_action_result']}")
        
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
        max_tokens=700,
        temperature=0.2,
        timeout=LLM_TIMEOUT_SEC,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> dict:
    """Parse LLM output to a JSON action dict. Falls back to request_info on failure."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        text = raw[start:end+1]
    else:
        text = raw
        
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

def run_task(task_name: str, deadline_ts: float) -> dict:
    emit("START", {"task": task_name, "model": MODEL_NAME})

    obs = env_reset(task_name)
    step = 0
    last_reward: float | None = None
    cumulative_reward = 0.0
    obs_history: list[dict] = []
    final_reward = 0.0

    try:
        while True:
            if time.time() >= deadline_ts:
                raise TimeoutError(
                    f"Global runtime limit reached ({MAX_TOTAL_RUNTIME_SEC}s)."
                )

            # Build prompt
            user_prompt = build_user_prompt(obs, step, last_reward, obs_history[-CONTEXT_WINDOW:])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM
            raw = call_llm(messages)
            print(f"[DEBUG] raw={raw}", file=sys.stderr)
            action = parse_action(raw)
            print(f"[DEBUG] action={action}", file=sys.stderr)

            # Step environment
            result = env_step(action)

            step_reward = result.get("reward", 0.0)
            done = result.get("done", False)
            cumulative_reward += step_reward
            final_reward = step_reward  # updated to final score below if done

            # When done, get the final score from the breakdown
            if done:
                breakdown = result.get("info", {}).get("final_reward_breakdown", {})
                final_reward = breakdown.get("total", cumulative_reward)

            emit("STEP", {
                "task": task_name,
                "step": step,
                "action_type": action.get("action_type", "unknown"),
                "reward": round(final_reward, 4) if done else round(step_reward, 4),
                "done": done,
            })

            # Slide observation window
            obs_history.append(obs)
            obs = result.get("observation", obs)
            last_reward = step_reward
            step += 1

            if done:
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
    start_ts = time.time()
    deadline_ts = start_ts + MAX_TOTAL_RUNTIME_SEC

    for task in TASKS:
        result = run_task(task, deadline_ts)
        results.append(result)
        if time.time() >= deadline_ts:
            break
        time.sleep(1)  # brief pause between tasks

    # Summary to stderr (not stdout, so it doesn't interfere with log parsing)
    print("\n=== Summary ===", file=sys.stderr)
    for r in results:
        score = r.get("score", 0.0)
        err = f"  ERROR: {r['error']}" if "error" in r else ""
        print(f"  {r['task']}: {score:.4f} ({r['steps']} steps){err}", file=sys.stderr)


if __name__ == "__main__":
    main()