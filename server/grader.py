"""
M03 — Grader & reward function
Deterministic per-component scoring. No LLM judge needed.
"""

from __future__ import annotations

import re

from server.models import (
    DecisionType,
    EpisodeState,
    RewardBreakdown,
    TaskConfig,
)


SCORE_EPSILON = 1e-4


# ---------------------------------------------------------------------------
# Email field checkers
# ---------------------------------------------------------------------------

_ACCEPT_PHRASES = [
    "pleased to inform",
    "happy to inform",
    "we would like to move forward",
    "offer you",
    "delighted to offer",
    "successful",
    "selected",
    "we are excited",
    "congratulations",
    "we'd like to extend",
    "we would like to extend",
]

_REJECT_PHRASES = [
    "will not be moving forward",
    "we will not be moving forward",
    "not selected",
    "unsuccessful",
    "decided not to proceed",
    "not be proceeding",
    "not move forward",
    "regret to inform",
    "unfortunately",
    "not be offered",
]

_SHORTLIST_PHRASES = [
    "shortlist",
    "reserve",
    "keep your application on file",
    "consider you for future",
    "hold your application",
]

_NEXT_STEPS_PHRASES = [
    "next step",
    "interview",
    "will be in touch",
    "reach out",
    "schedule",
    "contact you",
    "follow up",
]

_FEEDBACK_PHRASES = [
    "feedback",
    "reason",
    "unfortunately",
    "however",
    "although",
    "while",
    "the role requires",
    "we were looking for",
]

_PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:NAME|ROLE|DATE|POSITION|REASON|FEEDBACK|CONDITION|STRENGTH|GAP)\]",
    re.IGNORECASE,
)


def _contains_any(text: str, phrases: list[str]) -> bool:
    low = text.lower()
    return any(p in low for p in phrases)


def _check_email_quality(
    body: str,
    subject: str,
    candidate_name: str,
    role_description: str,
    required_fields: list[str],
    decision: DecisionType | None,
) -> float:
    """
    Returns a quality score in [0, 1] based on structured field checks.
    Weights:
      +0.30 candidate name present in body
      +0.20 role name / role keyword present
      +0.20 correct accept/reject/shortlist language
      +0.20 next-steps or feedback present
      +0.10 no unresolved placeholders
    """
    score = 0.0
    role_keywords = [w.lower() for w in role_description.split()[:8] if len(w) > 3]

    first_name = candidate_name.split()[0] if candidate_name else ""
    if first_name and first_name.lower() in body.lower():
        score += 0.30

    combined = (body + " " + subject).lower()
    if any(kw in combined for kw in role_keywords):
        score += 0.20

    if decision == DecisionType.ACCEPT:
        if _contains_any(body, _ACCEPT_PHRASES):
            score += 0.20
    elif decision == DecisionType.REJECT:
        if _contains_any(body, _REJECT_PHRASES):
            score += 0.20
    elif decision == DecisionType.SHORTLIST:
        if _contains_any(body, _SHORTLIST_PHRASES):
            score += 0.20
    else:
        if (
            _contains_any(body, _ACCEPT_PHRASES)
            or _contains_any(body, _REJECT_PHRASES)
            or _contains_any(body, _SHORTLIST_PHRASES)
        ):
            score += 0.20

    if decision == DecisionType.ACCEPT or decision == DecisionType.SHORTLIST:
        if _contains_any(body, _NEXT_STEPS_PHRASES):
            score += 0.20
    else:
        if _contains_any(body, _FEEDBACK_PHRASES):
            score += 0.20

    if not _PLACEHOLDER_PATTERN.search(body) and not _PLACEHOLDER_PATTERN.search(subject):
        score += 0.10

    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def grade_episode(episode_state: EpisodeState, task_config: TaskConfig) -> RewardBreakdown:
    """
    Deterministic scoring of a (possibly partial) episode.
    Safe to call at any step — partial credit is given throughout.
    Returns a RewardBreakdown with per-component scores and a weighted total.
    """
    rubric = task_config.scoring_rubric
    candidates = {app.candidate_id: app for app in task_config.applications}
    ground_truth = task_config.ground_truth_decisions

    decision_scores: list[float] = []
    for cid, correct in ground_truth.items():
        agent_decision = episode_state.decisions.get(cid)
        if agent_decision is None:
            decision_scores.append(0.0)
        elif agent_decision == correct:
            decision_scores.append(1.0)
        elif (
            agent_decision == DecisionType.SHORTLIST
            and correct == DecisionType.ACCEPT
        ):
            decision_scores.append(0.5)
        elif (
            agent_decision == DecisionType.SHORTLIST
            and correct == DecisionType.REJECT
        ):
            decision_scores.append(0.25)
        else:
            decision_scores.append(0.0)

    decision_score = (
        sum(decision_scores) / len(decision_scores) if decision_scores else 0.0
    )

    email_quality_scores: list[float] = []
    candidate_email_map: dict[str, dict] = {}
    for thread_id, email_data in episode_state.composed_emails.items():
        cid = email_data.get("candidate_id", "")
        if cid:
            candidate_email_map[cid] = email_data

    for cid in ground_truth:
        required_fields = task_config.required_email_fields.get(cid, [])
        if not required_fields:
            continue

        email_data = candidate_email_map.get(cid)
        if not email_data:
            email_quality_scores.append(0.0)
            continue

        candidate = candidates.get(cid)
        decision = episode_state.decisions.get(cid)

        q = _check_email_quality(
            body=email_data.get("body", ""),
            subject=email_data.get("subject", ""),
            candidate_name=candidate.name if candidate else "",
            role_description=task_config.role_description,
            required_fields=required_fields,
            decision=decision,
        )
        email_quality_scores.append(q)

    email_quality_score = (
        sum(email_quality_scores) / len(email_quality_scores)
        if email_quality_scores
        else 0.0
    )

    candidates_needing_email = set(task_config.required_email_fields.keys())

    sent_candidates: set[str] = set()
    for thread_id in episode_state.sent_email_ids:
        email_data = episode_state.composed_emails.get(thread_id)
        if email_data:
            cid = email_data.get("candidate_id", "")
            if cid:
                sent_candidates.add(cid)

    if candidates_needing_email:
        delivery_score = len(sent_candidates & candidates_needing_email) / len(
            candidates_needing_email
        )
    else:
        delivery_score = 1.0

    steps_fraction = episode_state.step / task_config.max_steps
    threshold = rubric.efficiency_threshold
    if steps_fraction > threshold:
        raw_penalty = (steps_fraction - threshold) / (1.0 - threshold)
        efficiency_penalty = round(min(1.0, raw_penalty), 4)
    else:
        efficiency_penalty = 0.0

    total = (
        rubric.decision_weight * decision_score
        + rubric.email_quality_weight * email_quality_score
        + rubric.delivery_weight * delivery_score
        - rubric.efficiency_weight * efficiency_penalty
    )
    total = round(max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, total)), 4)

    return RewardBreakdown(
        decision_score=round(decision_score, 4),
        email_quality_score=round(email_quality_score, 4),
        delivery_score=round(delivery_score, 4),
        efficiency_penalty=efficiency_penalty,
        total=total,
    )


def intermediate_step_reward(
    episode_state: EpisodeState,
    task_config: TaskConfig,
    newly_composed_thread_id: str | None = None,
) -> float:
    """
    Small positive reward (+0.05) when a valid compose_email passes field checks.
    Called inside env.step() after a compose_email action.
    """
    if newly_composed_thread_id is None:
        return 0.0

    rubric = task_config.scoring_rubric
    email_data = episode_state.composed_emails.get(newly_composed_thread_id)
    if not email_data:
        return 0.0

    cid = email_data.get("candidate_id", "")
    candidates = {app.candidate_id: app for app in task_config.applications}
    candidate = candidates.get(cid)
    decision = episode_state.decisions.get(cid)
    required_fields = task_config.required_email_fields.get(cid, [])

    q = _check_email_quality(
        body=email_data.get("body", ""),
        subject=email_data.get("subject", ""),
        candidate_name=candidate.name if candidate else "",
        role_description=task_config.role_description,
        required_fields=required_fields,
        decision=decision,
    )

    if q >= 0.5:
        return rubric.intermediate_compose_reward
    return 0.0