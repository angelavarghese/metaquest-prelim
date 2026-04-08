"""
M04 — Environment engine
Stateful episode logic: reset, step, state.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from server.models import (
    Action,
    ApplicationRecord,
    ComposeEmailAction,
    DecisionAction,
    DecisionType,
    EmailMessage,
    EmailThread,
    EpisodeState,
    Observation,
    RequestInfoAction,
    SendEmailAction,
    StepResult,
    TaskConfig,
)
from server.grader import grade_episode, intermediate_step_reward


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class EpisodeAlreadyDoneError(Exception):
    """Raised when step() is called after the episode has ended."""


class TaskNotFoundError(Exception):
    """Raised when the requested task JSON fixture does not exist."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASKS_DIR = Path(__file__).parent / "tasks"


def _load_task(task_name: str) -> TaskConfig:
    path = TASKS_DIR / f"{task_name}.json"
    if not path.exists():
        raise TaskNotFoundError(f"Task fixture not found: {path}")
    raw = json.loads(path.read_text())
    return TaskConfig.model_validate(raw)


def _make_thread_id() -> str:
    return f"thread_{uuid.uuid4().hex[:8]}"


def _simulated_clock(step: int) -> int:
    """Returns a fake Unix timestamp based on step number."""
    return 1_700_000_000 + step * 60


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class JobApplicationEnv:
    """
    Stateful environment for the hiring-agent evaluation.

    Usage:
        env = JobApplicationEnv()
        obs = env.reset("single_clear_decision")
        result = env.step(action)
    """

    def __init__(self) -> None:
        self._task_config: TaskConfig | None = None
        self._episode_state: EpisodeState | None = None

    def reset(self, task_name: str) -> Observation:
        """Load a task and initialise a clean episode. Returns initial observation."""
        self._task_config = _load_task(task_name)
        self._episode_state = EpisodeState(
            task_name=task_name,
            step=0,
            done=False,
            decisions={},
            composed_emails={},
            sent_email_ids=[],
            inbox=[],
            reward_so_far=0.0,
            final_reward=None,
        )
        return self._build_observation(last_action_result=None)

    def step(self, action: Action) -> StepResult:  # type: ignore[type-arg]
        """
        Apply an action and return (observation, reward, done, info).
        Raises EpisodeAlreadyDoneError if called after done.
        """
        if self._episode_state is None or self._task_config is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_state.done:
            raise EpisodeAlreadyDoneError("Episode is already done. Call reset() to start a new one.")

        state = self._episode_state
        task = self._task_config
        state.step += 1

        step_reward = 0.0
        last_action_result: str = ""
        info: dict[str, Any] = {}

        if isinstance(action, DecisionAction):
            last_action_result, step_reward = self._handle_decision(action)
        elif isinstance(action, ComposeEmailAction):
            last_action_result, step_reward = self._handle_compose_email(action)
        elif isinstance(action, SendEmailAction):
            last_action_result, step_reward = self._handle_send_email(action)
        elif isinstance(action, RequestInfoAction):
            last_action_result, step_reward = self._handle_request_info(action)
        else:
            last_action_result = f"Unknown action type: {type(action)}"

        state.reward_so_far += step_reward

        done = self._check_done()
        state.done = done

        if done:
            final_breakdown = grade_episode(state, task)
            state.final_reward = final_breakdown
            info["final_reward_breakdown"] = final_breakdown.model_dump()
            info["episode_summary"] = self._build_summary()

        obs = self._build_observation(last_action_result=last_action_result)

        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            info=info,
        )

    def state(self) -> EpisodeState:
        """Read-only snapshot of the current episode state."""
        if self._episode_state is None:
            raise RuntimeError("Call reset() first.")
        return self._episode_state.model_copy(deep=True)

    def _handle_decision(self, action: DecisionAction) -> tuple[str, float]:
        state = self._episode_state
        task = self._task_config
        cid = action.candidate_id

        valid_ids = {app.candidate_id for app in task.applications}
        if cid not in valid_ids:
            return f"ERROR: Unknown candidate_id '{cid}'", 0.0

        state.decisions[cid] = action.decision

        sim = task.simulated_replies.get(cid, {})
        if (
            sim
            and sim.get("trigger_action") == "decision"
            and sim.get("trigger_decision") == action.decision.value
        ):
            self._inject_simulated_reply(cid, sim)

        return (
            f"Decision recorded: {action.decision.value} for {cid} ({action.reason})",
            0.0,
        )

    def _handle_compose_email(self, action: ComposeEmailAction) -> tuple[str, float]:
        state = self._episode_state
        task = self._task_config
        cid = action.candidate_id

        valid_ids = {app.candidate_id for app in task.applications}
        if cid not in valid_ids:
            return f"ERROR: Unknown candidate_id '{cid}'", 0.0

        thread_id = _make_thread_id()
        email_payload = {
            "candidate_id": cid,
            "recipient": action.recipient,
            "subject": action.subject,
            "body": action.body,
            "thread_id": thread_id,
        }
        state.composed_emails[thread_id] = email_payload

        reward = intermediate_step_reward(state, task, newly_composed_thread_id=thread_id)

        return (
            f"Email composed for {cid} (thread_id={thread_id}). Quality reward: {reward:.2f}",
            reward,
        )

    def _handle_send_email(self, action: SendEmailAction) -> tuple[str, float]:
        state = self._episode_state
        task = self._task_config

        if action.thread_id not in state.composed_emails:
            return f"ERROR: thread_id '{action.thread_id}' not found. Compose email first.", 0.0

        if action.thread_id in state.sent_email_ids:
            return f"WARN: thread_id '{action.thread_id}' already sent.", 0.0

        state.sent_email_ids.append(action.thread_id)
        email_data = state.composed_emails[action.thread_id]
        cid = email_data.get("candidate_id", "")

        sent_msg = EmailMessage(
            from_address="recruiter@company.com",
            to_address=email_data.get("recipient", ""),
            subject=email_data.get("subject", ""),
            body=email_data.get("body", ""),
            timestamp=_simulated_clock(state.step),
        )
        thread = EmailThread(
            thread_id=action.thread_id,
            candidate_id=cid,
            messages=[sent_msg],
        )
        state.inbox.append(thread)

        sim = task.simulated_replies.get(cid, {})
        if (
            sim
            and sim.get("trigger_action") == "send_email"
            and sim.get("trigger_decision") == state.decisions.get(cid, DecisionType.REJECT).value
        ):
            self._inject_simulated_reply(cid, sim)

        return f"Email sent for candidate {cid} (thread_id={action.thread_id})", 0.0

    def _handle_request_info(self, action: RequestInfoAction) -> tuple[str, float]:
        state = self._episode_state
        task = self._task_config
        cid = action.candidate_id

        valid_ids = {app.candidate_id for app in task.applications}
        if cid not in valid_ids:
            return f"ERROR: Unknown candidate_id '{cid}'", 0.0

        thread_id = _make_thread_id()
        msg = EmailMessage(
            from_address="recruiter@company.com",
            to_address=f"{cid}@candidate.sim",
            subject=f"Request for information — {cid}",
            body=action.question,
            timestamp=_simulated_clock(state.step),
        )
        thread = EmailThread(thread_id=thread_id, candidate_id=cid, messages=[msg])
        state.inbox.append(thread)

        return f"Info request sent to {cid}: '{action.question[:60]}...'", 0.0

    def _inject_simulated_reply(self, candidate_id: str, sim: dict[str, Any]) -> None:
        """Append a deterministic candidate reply into the inbox."""
        state = self._episode_state
        reply_msg = EmailMessage(
            from_address=f"{candidate_id}@candidate.sim",
            to_address="recruiter@company.com",
            subject=sim.get("reply_subject", "Re: Your application"),
            body=sim.get("reply_body", ""),
            timestamp=_simulated_clock(state.step + 1),
        )
        thread_id = _make_thread_id()
        thread = EmailThread(
            thread_id=thread_id,
            candidate_id=candidate_id,
            messages=[reply_msg],
        )
        state.inbox.append(thread)

    def _check_done(self) -> bool:
        state = self._episode_state
        task = self._task_config

        if state.step >= task.max_steps:
            return True

        all_decided = set(task.ground_truth_decisions.keys()) == set(state.decisions.keys())

        candidates_needing_email = set(task.required_email_fields.keys())
        sent_candidates: set[str] = set()
        for thread_id in state.sent_email_ids:
            email_data = state.composed_emails.get(thread_id, {})
            cid = email_data.get("candidate_id", "")
            if cid:
                sent_candidates.add(cid)

        all_emails_sent = candidates_needing_email.issubset(sent_candidates)

        return all_decided and all_emails_sent

    def _build_observation(self, last_action_result: str | None) -> Observation:
        state = self._episode_state
        task = self._task_config

        pending = [
            cid
            for cid in task.ground_truth_decisions
            if cid not in state.decisions
        ]

        return Observation(
            applications=task.applications,
            inbox=list(state.inbox),
            pending_decisions=pending,
            step=state.step,
            last_action_result=last_action_result,
            task_description=task.role_description,
        )

    def _build_summary(self) -> dict[str, Any]:
        state = self._episode_state
        task = self._task_config
        breakdown = state.final_reward

        correct = sum(
            1
            for cid, gt in task.ground_truth_decisions.items()
            if state.decisions.get(cid) == gt
        )
        total_candidates = len(task.ground_truth_decisions)

        return {
            "task_name": task.task_name,
            "steps_used": state.step,
            "max_steps": task.max_steps,
            "decisions_correct": correct,
            "decisions_total": total_candidates,
            "final_score": breakdown.total if breakdown else 0.0,
            "breakdown": breakdown.model_dump() if breakdown else {},
        }