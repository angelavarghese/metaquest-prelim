"""
Tests for grader.py — M03
Run with: pytest tests/test_grader.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from server.models import DecisionType, EpisodeState, TaskConfig
from server.grader import grade_episode, _check_email_quality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def load_task(name: str) -> TaskConfig:
    import json
    path = Path(__file__).parent.parent / "server" / "tasks" / f"{name}.json"
    return TaskConfig.model_validate(json.loads(path.read_text()))


TASK1 = load_task("single_clear_decision")
TASK2 = load_task("batch_with_quota")
TASK3 = load_task("negotiation_and_edge_cases")


def _perfect_state(task: TaskConfig) -> EpisodeState:
    """Build an episode state where every decision is correct and emails are composed+sent."""
    state = EpisodeState(task_name=task.task_name, step=1)
    candidates = {app.candidate_id: app for app in task.applications}

    for cid, decision in task.ground_truth_decisions.items():
        state.decisions[cid] = decision

    for cid, req_fields in task.required_email_fields.items():
        candidate = candidates[cid]
        decision = task.ground_truth_decisions[cid]
        thread_id = f"thread_{cid}"

        # Build a passing email body
        if decision == DecisionType.ACCEPT:
            body = (
                f"Dear {candidate.name},\n\n"
                f"We are pleased to inform you that we would like to move forward with your application "
                f"for the Senior Product Manager role. As a next step, we will schedule an interview. "
                f"Please let us know if you have any questions."
            )
            subject = f"Your Application — {task.role_description[:30]}"
        elif decision == DecisionType.SHORTLIST:
            body = (
                f"Dear {candidate.name},\n\n"
                f"We would like to shortlist your application for future consideration. "
                f"We will keep your application on file and reach out if a suitable position opens."
            )
            subject = f"Your Application — {task.role_description[:30]}"
        else:
            body = (
                f"Dear {candidate.name},\n\n"
                f"We regret to inform you that we will not be moving forward with your application. "
                f"Unfortunately, after careful consideration, the role requires specific certifications "
                f"that were not present in your profile. We wish you the best in your search."
            )
            subject = f"Your Application — {task.role_description[:30]}"

        state.composed_emails[thread_id] = {
            "candidate_id": cid,
            "subject": subject,
            "body": body,
            "recipient": candidate.email,
            "thread_id": thread_id,
        }
        state.sent_email_ids.append(thread_id)

    state.step = 2  # efficient
    return state


def _random_state(task: TaskConfig) -> EpisodeState:
    """Build a state where all decisions are wrong and no emails sent."""
    state = EpisodeState(task_name=task.task_name, step=task.max_steps)
    for cid, decision in task.ground_truth_decisions.items():
        # Flip the decision
        if decision == DecisionType.ACCEPT:
            state.decisions[cid] = DecisionType.REJECT
        else:
            state.decisions[cid] = DecisionType.ACCEPT
    return state


# ---------------------------------------------------------------------------
# Core scoring tests
# ---------------------------------------------------------------------------

class TestPerfectAgent:
    def test_task1_perfect_score(self):
        state = _perfect_state(TASK1)
        rb = grade_episode(state, TASK1)
        assert rb.total >= 0.85, f"Expected >= 0.85, got {rb.total}"

    def test_task2_perfect_decision_score(self):
        state = _perfect_state(TASK2)
        rb = grade_episode(state, TASK2)
        assert rb.decision_score == pytest.approx(1.0, abs=0.01)

    def test_task3_perfect_decision_score(self):
        state = _perfect_state(TASK3)
        rb = grade_episode(state, TASK3)
        assert rb.decision_score == pytest.approx(1.0, abs=0.01)


class TestRandomAgent:
    def test_task1_low_score(self):
        state = _random_state(TASK1)
        rb = grade_episode(state, TASK1)
        assert rb.total < 0.15, f"Expected < 0.15, got {rb.total}"

    def test_task2_low_score(self):
        state = _random_state(TASK2)
        rb = grade_episode(state, TASK2)
        assert rb.total < 0.15, f"Expected < 0.15, got {rb.total}"


class TestScoreClamping:
    def test_total_always_between_0_and_1(self):
        for task in [TASK1, TASK2, TASK3]:
            for state in [_perfect_state(task), _random_state(task)]:
                rb = grade_episode(state, task)
                assert 0.0 <= rb.total <= 1.0
                assert 0.0 <= rb.decision_score <= 1.0
                assert 0.0 <= rb.email_quality_score <= 1.0
                assert 0.0 <= rb.delivery_score <= 1.0
                assert 0.0 <= rb.efficiency_penalty <= 1.0

    def test_total_never_negative(self):
        """Even a catastrophically bad agent should not go below 0."""
        state = _random_state(TASK2)
        state.step = TASK2.max_steps  # max efficiency penalty
        rb = grade_episode(state, TASK2)
        assert rb.total >= 0.0


class TestDeterminism:
    def test_same_inputs_same_output(self):
        state = _perfect_state(TASK2)
        rb1 = grade_episode(state, TASK2)
        rb2 = grade_episode(state, TASK2)
        assert rb1.model_dump() == rb2.model_dump()

    def test_random_state_deterministic(self):
        state = _random_state(TASK3)
        rb1 = grade_episode(state, TASK3)
        rb2 = grade_episode(state, TASK3)
        assert rb1.model_dump() == rb2.model_dump()


class TestPartialCredit:
    def test_shortlist_when_should_accept_gets_partial(self):
        """Shortlisting a candidate who should be accepted gives 0.5 credit."""
        state = EpisodeState(task_name="single_clear_decision", step=1)
        state.decisions["C001"] = DecisionType.SHORTLIST  # correct is accept
        rb = grade_episode(state, TASK1)
        assert rb.decision_score == pytest.approx(0.5, abs=0.01)

    def test_wrong_direction_gets_zero(self):
        state = EpisodeState(task_name="single_clear_decision", step=1)
        state.decisions["C001"] = DecisionType.REJECT  # should be accept
        rb = grade_episode(state, TASK1)
        assert rb.decision_score == pytest.approx(0.0, abs=0.01)

    def test_no_decision_gets_zero(self):
        state = EpisodeState(task_name="single_clear_decision", step=1)
        # No decisions made
        rb = grade_episode(state, TASK1)
        assert rb.decision_score == pytest.approx(0.0, abs=0.01)


class TestEfficiencyPenalty:
    def test_no_penalty_under_threshold(self):
        state = _perfect_state(TASK1)
        state.step = 2  # well under max_steps=4, threshold at 0.8 * 4 = 3.2
        rb = grade_episode(state, TASK1)
        assert rb.efficiency_penalty == pytest.approx(0.0, abs=0.01)

    def test_penalty_when_over_threshold(self):
        state = _perfect_state(TASK1)
        state.step = TASK1.max_steps  # at max
        rb = grade_episode(state, TASK1)
        assert rb.efficiency_penalty > 0.0

    def test_penalty_does_not_make_total_negative(self):
        state = EpisodeState(task_name="single_clear_decision", step=TASK1.max_steps)
        rb = grade_episode(state, TASK1)
        assert rb.total >= 0.0


class TestDeliveryScore:
    def test_full_delivery_when_all_sent(self):
        state = _perfect_state(TASK1)
        rb = grade_episode(state, TASK1)
        assert rb.delivery_score == pytest.approx(1.0, abs=0.01)

    def test_zero_delivery_when_nothing_sent(self):
        state = EpisodeState(task_name="single_clear_decision", step=1)
        state.decisions["C001"] = DecisionType.ACCEPT
        # No emails composed or sent
        rb = grade_episode(state, TASK1)
        assert rb.delivery_score == pytest.approx(0.0, abs=0.01)

    def test_partial_delivery(self):
        """Send email for only half of candidates that need one."""
        state = _perfect_state(TASK2)
        # Remove half the sent emails
        all_sent = list(state.sent_email_ids)
        state.sent_email_ids = all_sent[: len(all_sent) // 2]
        rb = grade_episode(state, TASK2)
        assert 0.0 < rb.delivery_score < 1.0


class TestEmailQuality:
    def test_perfect_accept_email(self):
        score = _check_email_quality(
            body="Dear Priya,\n\nWe are pleased to inform you that we would like to move forward with your application for the Junior Software Engineer role. As a next step, we will schedule a technical interview within two business days.",
            subject="Your Application for Junior Software Engineer at Acme Corp",
            candidate_name="Priya Sharma",
            role_description="Junior Software Engineer at Acme Corp",
            required_fields=["name", "role", "accept_language", "next_steps"],
            decision=DecisionType.ACCEPT,
        )
        assert score >= 0.85

    def test_unresolved_placeholder_penalised(self):
        score = _check_email_quality(
            body="Dear [NAME], we are pleased to offer you the [ROLE] position.",
            subject="Your Application",
            candidate_name="Test User",
            role_description="Junior Software Engineer",
            required_fields=["name", "role", "accept_language", "next_steps"],
            decision=DecisionType.ACCEPT,
        )
        assert score < 0.90  # loses 0.10 for placeholder, may lose more

    def test_wrong_language_direction(self):
        """Email says 'rejected' but decision is accept — should not get language credit."""
        score = _check_email_quality(
            body="Dear Priya, we regret to inform you that you were not selected.",
            subject="Application Update",
            candidate_name="Priya Sharma",
            role_description="Junior Software Engineer",
            required_fields=["name", "role", "accept_language", "next_steps"],
            decision=DecisionType.ACCEPT,
        )
        # Gets credit for name (0.30), maybe role (0.20), no penalty from placeholder (0.10)
        # But should NOT get accept_language credit (0.20) or next_steps credit (0.20)
        assert score <= 0.60


# ---------------------------------------------------------------------------
# No external API calls (regression guard)
# ---------------------------------------------------------------------------

class TestNoExternalCalls:
    def test_grade_episode_makes_no_network_calls(self, monkeypatch):
        import urllib.request

        original_urlopen = urllib.request.urlopen

        def fail_if_called(*args, **kwargs):
            raise AssertionError("grader.py made an external network call!")

        monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)

        state = _perfect_state(TASK1)
        grade_episode(state, TASK1)  # should not raise