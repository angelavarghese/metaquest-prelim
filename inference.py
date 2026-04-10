"""
inference.py — Baseline agent loop for the Job Application Environment.

Environment variables (all consumed via os.environ):
  API_BASE_URL        — OpenAI-compatible base URL  (default: http://localhost:8000/v1)
  MODEL_NAME          — Model to use                (default: gpt-4o-mini)
  HF_TOKEN            — API key, NO default         (required in HF Space / remote runs)
  LOCAL_IMAGE_NAME    — If set, used for logging only (optional)
  OPENAI_BASE_URL     — Base URL for LLM provider   (default: https://api.openai.com/v1)
  REQUEST_TIMEOUT_SEC — HTTP timeout for env calls  (default: 25)
  LLM_TIMEOUT_SEC     — HTTP timeout for LLM calls  (default: 40)
  MAX_TOTAL_RUNTIME_SEC — Hard stop across all tasks (default: 1140)

Stdout format — EXACTLY one line per event, key=value pairs:
  [START] task=<name> model=<name>
  [STEP]  step=<n> action=<action_type>(...) reward=<f> done=<bool> error=null
  [END]   task=<name> steps=<n> final_reward=<f> score=<f>
"""

import json
import os
import sys
import time
import traceback

import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")          # intentionally no default
LOCAL_IMAGE_NAME: str | None = os.environ.get("LOCAL_IMAGE_NAME")  # logging only

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
{"action_type": "send_email", "candidate_id": "<id>", "thread_id": "<thread_id from compose result>"}

4. Ask a candidate a clarifying question:
{"action_type": "request_info", "candidate_id": "<id>", "question": "<your question>"}

## Strict workflow per candidate (follow in order)
Step A: decision — accept or reject the candidate.
Step B: compose_email — write their notification email.
Step C: send_email — deliver it (use the thread_id returned in the last action result).
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
        for past_obs in history:
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
    text = raw[start:end + 1] if start != -1 and end != -1 else raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "action_type": "request_info",
            "candidate_id": "unknown",
            "question": "Could you please clarify your application details?",
        }


# ---------------------------------------------------------------------------
# Stdout emit — FLAT key=value format required by the evaluator
# ---------------------------------------------------------------------------

def _action_repr(action: dict) -> str:
    """
    Render an action as  action_type(key=val, ...)  e.g.
      decision(candidate_id=C001, decision=accept)
      compose_email(candidate_id=C001, recipient=x@y.com)
    """
    atype = action.get("action_type", "unknown")
    skip = {"action_type"}
    pairs = ", ".join(
        f"{k}={v}" for k, v in action.items() if k not in skip
    )
    return f"{atype}({pairs})"


def emit_start(task: str) -> None:
    print(f"[START] task={task} model={MODEL_NAME}", flush=True)


def emit_step(step: int, action: dict, reward: float, done: bool, error: str | None = None) -> None:
    error_val = "null" if error is None else error
    done_val = str(done).lower()          # "true" / "false"
    action_val = _action_repr(action)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )


def emit_end(task: str, steps: int, final_reward: float, score: float, error: str | None = None) -> None:
    line = f"[END] task={task} steps={steps} final_reward={final_reward:.4f} score={score:.4f}"
    if error is not None:
        line += f" error={error}"
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(task_name: str, deadline_ts: float) -> dict:
    emit_start(task_name)

    obs = env_reset(task_name)
    step = 0
    last_reward: float | None = None
    cumulative_reward = 0.0
    obs_history: list[dict] = []
    final_reward = 0.0
    last_action: dict = {}

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
            last_action = action
            print(f"[DEBUG] action={action}", file=sys.stderr)

            # Step environment
            result = env_step(action)

            step_reward = result.get("reward", 0.0)
            done = result.get("done", False)
            cumulative_reward += step_reward

            if done:
                breakdown = result.get("info", {}).get("final_reward_breakdown", {})
                final_reward = breakdown.get("total", cumulative_reward)
            else:
                final_reward = cumulative_reward

            emit_step(
                step=step,
                action=action,
                reward=final_reward if done else step_reward,
                done=done,
            )

            # Slide observation window
            obs_history.append(obs)
            obs = result.get("observation", obs)
            last_reward = step_reward
            step += 1

            if done:
                break

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit_step(step=step, action=last_action, reward=0.0, done=False, error=str(exc))
        emit_end(task_name, steps=step, final_reward=final_reward, score=final_reward, error=str(exc))
        return {"task": task_name, "steps": step, "score": final_reward, "error": str(exc)}

    emit_end(task_name, steps=step, final_reward=final_reward, score=final_reward)
    return {"task": task_name, "steps": step, "score": final_reward}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN and OPENAI_BASE_URL.startswith("https://"):
        print(
            "[WARN] HF_TOKEN is not set. Remote LLM calls will likely fail. "
            "Set HF_TOKEN in your environment.",
            file=sys.stderr,
        )
    if LOCAL_IMAGE_NAME:
        print(f"[INFO] LOCAL_IMAGE_NAME={LOCAL_IMAGE_NAME}", file=sys.stderr)

    results = []
    start_ts = time.time()
    deadline_ts = start_ts + MAX_TOTAL_RUNTIME_SEC

    for task in TASKS:
        result = run_task(task, deadline_ts)
        results.append(result)
        if time.time() >= deadline_ts:
            break
        time.sleep(1)   # brief pause between tasks

    # Summary to stderr only — never pollute the evaluator's stdout stream
    print("\n=== Summary ===", file=sys.stderr)
    for r in results:
        score = r.get("score", 0.0)
        err = f"  ERROR: {r['error']}" if "error" in r else ""
        print(f"  {r['task']}: {score:.4f} ({r['steps']} steps){err}", file=sys.stderr)


if __name__ == "__main__":
    main()
