"""
Tests for env.py — M04
Run with: pytest tests/test_env.py -v
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from server.models import (
    ComposeEmailAction,
    DecisionAction,
    DecisionType,
    RequestInfoAction,
    SendEmailAction,
)
from server.env import EpisodeAlreadyDoneError, JobApplicationEnv, TaskNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env() -> JobApplicationEnv:
    return JobApplicationEnv()


def accept_action(cid: str, reason: str = "Meets all criteria") -> DecisionAction:
    return DecisionAction(candidate_id=cid, decision=DecisionType.ACCEPT, reason=reason)


def reject_action(cid: str, reason: str = "Does not meet requirements") -> DecisionAction:
    return DecisionAction(candidate_id=cid, decision=DecisionType.REJECT, reason=reason)


def compose_action(cid: str, recipient: str, subject: str, body: str) -> ComposeEmailAction:
    return ComposeEmailAction(
        candidate_id=cid,
        recipient=recipient,
        subject=subject,
        body=body,
    )


def send_action(cid: str, thread_id: str) -> SendEmailAction:
    return SendEmailAction(candidate_id=cid, thread_id=thread_id)


def extract_thread_id(message: str) -> str:
    match = re.search(r"thread_id=(thread_[A-Za-z0-9]+)", message)
    assert match is not None, f"Could not find thread_id in message: {message!r}"
    return match.group(1)


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self):
        env = make_env()
        obs = env.reset("single_clear_decision")
        assert obs.step == 0
        assert len(obs.applications) == 1
        assert obs.pending_decisions == ["C001"]

    def test_reset_clean_state_regardless_of_prior(self):
        env = make_env()
        env.reset("single_clear_decision")
        # Partially advance
        env.step(accept_action("C001"))
        # Now reset again — state must be clean
        obs = env.reset("single_clear_decision")
        assert obs.step == 0
        assert obs.pending_decisions == ["C001"]
        assert obs.inbox == []

    def test_reset_unknown_task_raises(self):
        env = make_env()
        with pytest.raises(TaskNotFoundError):
            env.reset("does_not_exist")

    def test_reset_different_tasks_sequentially(self):
        env = make_env()
        obs1 = env.reset("single_clear_decision")
        assert len(obs1.applications) == 1
        obs2 = env.reset("batch_with_quota")
        assert len(obs2.applications) == 8

    def test_task_description_in_observation(self):
        env = make_env()
        obs = env.reset("single_clear_decision")
        assert len(obs.task_description) > 0


# ---------------------------------------------------------------------------
# step() — basic mechanics
# ---------------------------------------------------------------------------

class TestStepBasics:
    def test_step_increments_counter(self):
        env = make_env()
        env.reset("single_clear_decision")
        result = env.step(accept_action("C001"))
        assert result.observation.step == 1

    def test_step_removes_from_pending(self):
        env = make_env()
        env.reset("single_clear_decision")
        result = env.step(accept_action("C001"))
        assert "C001" not in result.observation.pending_decisions

    def test_step_before_reset_raises(self):
        env = make_env()
        with pytest.raises(RuntimeError):
            env.step(accept_action("C001"))

    def test_step_unknown_candidate_returns_error(self):
        env = make_env()
        env.reset("single_clear_decision")
        result = env.step(accept_action("ZZZZ"))
        assert "ERROR" in result.observation.last_action_result

    def test_compose_email_produces_thread_id(self):
        env = make_env()
        env.reset("single_clear_decision")
        env.step(accept_action("C001"))
        result = env.step(
            compose_action(
                "C001",
                "priya.sharma@email.com",
                "Your Application for Junior Software Engineer",
                "Dear Priya, we are pleased to inform you that we would like to move forward. As a next step, we will schedule an interview.",
            )
        )
        assert "thread_" in result.observation.last_action_result

    def test_send_email_requires_existing_thread(self):
        env = make_env()
        env.reset("single_clear_decision")
        result = env.step(send_action("C001", "thread_nonexistent"))
        assert "ERROR" in result.observation.last_action_result

    def test_request_info_action(self):
        env = make_env()
        env.reset("single_clear_decision")
        action = RequestInfoAction(
            candidate_id="C001",
            question="Can you clarify your experience with distributed systems?",
        )
        result = env.step(action)
        assert result.observation.last_action_result is not None
        assert len(result.observation.inbox) == 1


# ---------------------------------------------------------------------------
# Done conditions
# ---------------------------------------------------------------------------

class TestDoneConditions:
    def _complete_task1(self, env: JobApplicationEnv):
        """Fully complete task1 in minimal steps."""
        env.reset("single_clear_decision")
        env.step(accept_action("C001"))
        r = env.step(
            compose_action(
                "C001",
                "priya.sharma@email.com",
                "Junior Software Engineer Role at Acme Corp",
                "Dear Priya, we are pleased to inform you that we would like to move forward with your application. As a next step, we will schedule a technical interview within two business days.",
            )
        )
        # Extract thread_id from last_action_result
        msg = r.observation.last_action_result or ""
        thread_id = extract_thread_id(msg)
        result = env.step(send_action("C001", thread_id))
        return result

    def test_done_when_all_decisions_and_emails_sent(self):
        env = make_env()
        result = self._complete_task1(env)
        assert result.done is True

    def test_not_done_until_emails_sent(self):
        env = make_env()
        env.reset("single_clear_decision")
        env.step(accept_action("C001"))
        r = env.step(
            compose_action(
                "C001",
                "priya.sharma@email.com",
                "Junior SWE",
                "Dear Priya, we are pleased to inform you that you are accepted. Next step: interview.",
            )
        )
        # Email composed but NOT sent
        assert r.done is False

    def test_done_at_max_steps(self):
        env = make_env()
        obs = env.reset("single_clear_decision")
        max_steps = 4
        result = None
        for _ in range(max_steps):
            result = env.step(
                RequestInfoAction(candidate_id="C001", question="Tell me more about yourself.")
            )
        assert result is not None
        assert result.done is True

    def test_step_after_done_raises(self):
        env = make_env()
        self._complete_task1(env)
        with pytest.raises(EpisodeAlreadyDoneError):
            env.step(accept_action("C001"))

    def test_final_reward_present_when_done(self):
        env = make_env()
        result = self._complete_task1(env)
        assert result.done is True
        assert "final_reward_breakdown" in result.info


# ---------------------------------------------------------------------------
# state() tests
# ---------------------------------------------------------------------------

class TestStateReadOnly:
    def test_state_returns_copy(self):
        env = make_env()
        env.reset("single_clear_decision")
        s1 = env.state()
        s1.decisions["FAKE"] = DecisionType.ACCEPT  # mutate the copy
        s2 = env.state()
        assert "FAKE" not in s2.decisions  # original unaffected

    def test_state_before_reset_raises(self):
        env = make_env()
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_reflects_decisions(self):
        env = make_env()
        env.reset("single_clear_decision")
        env.step(accept_action("C001"))
        s = env.state()
        assert s.decisions.get("C001") == DecisionType.ACCEPT


# ---------------------------------------------------------------------------
# Simulated replies (task 3)
# ---------------------------------------------------------------------------

class TestSimulatedReplies:
    def test_counter_offer_injected_on_accept(self):
        """C001 in task3 sends counter-offer when accepted."""
        env = make_env()
        env.reset("negotiation_and_edge_cases")
        env.step(accept_action("C001"))
        s = env.state()
        # Inbox should have a reply from C001
        reply_threads = [t for t in s.inbox if t.candidate_id == "C001"]
        assert len(reply_threads) > 0
        assert "192" in reply_threads[0].messages[0].body  # counter at $192k

    def test_reapplication_reply_injected_on_reject_send(self):
        """C002 in task3 sends reapplication request after rejection email is sent."""
        env = make_env()
        env.reset("negotiation_and_edge_cases")

        # Reject C002
        env.step(reject_action("C002", "Stronger candidate selected"))
        r = env.step(
            compose_action(
                "C002",
                "ben.okafor@email.com",
                "Your Application for Senior Product Manager",
                "Dear Ben, unfortunately we will not be moving forward with your application. The role requires a longer track record. We wish you the best.",
            )
        )
        msg = r.observation.last_action_result or ""
        thread_id = extract_thread_id(msg)
        env.step(send_action("C002", thread_id))

        s = env.state()
        c2_threads = [t for t in s.inbox if t.candidate_id == "C002"]
        # Should have at least the sent email + the reply
        reply_bodies = [m.body for t in c2_threads for m in t.messages]
        assert any("feedback" in b.lower() or "reappl" in b.lower() for b in reply_bodies)


# ---------------------------------------------------------------------------
# Multi-task walkthrough
# ---------------------------------------------------------------------------

class TestManualWalkthrough:
    def _walk_task1(self):
        env = make_env()
        env.reset("single_clear_decision")
        env.step(accept_action("C001"))
        r = env.step(
            compose_action(
                "C001",
                "priya.sharma@email.com",
                "Junior SWE Role",
                "Dear Priya, we are pleased to inform you that we would like to move forward. As a next step we will schedule an interview.",
            )
        )
        msg = r.observation.last_action_result or ""
        thread_id = extract_thread_id(msg)
        result = env.step(send_action("C001", thread_id))
        assert result.done
        return result

    def _walk_task2(self):
        env = make_env()
        env.reset("batch_with_quota")
        decisions = {
            "C001": "accept", "C002": "reject", "C003": "reject",
            "C004": "accept", "C005": "reject", "C006": "shortlist",
            "C007": "shortlist", "C008": "reject",
        }
        apps = {
            "C001": "marcus.rivera@email.com",
            "C002": "aisha.okonkwo@email.com",
            "C003": "sam.chen@email.com",
            "C004": "jordan.lee@email.com",
            "C005": "fatima.alhassan@email.com",
            "C006": "tomasz.w@email.com",
            "C007": "preethi.nair@email.com",
            "C008": "devon.harper@email.com",
        }
        names = {
            "C001": "Marcus", "C002": "Aisha", "C003": "Sam",
            "C004": "Jordan", "C005": "Fatima", "C006": "Tomasz",
            "C007": "Preethi", "C008": "Devon",
        }
        thread_ids = {}
        for cid, dec in decisions.items():
            env.step(DecisionAction(candidate_id=cid, decision=DecisionType(dec), reason="Evaluated"))

        for cid, email in apps.items():
            dec = decisions[cid]
            name = names[cid]
            if dec == "accept":
                body = f"Dear {name}, we are pleased to inform you that we would like to move forward. Next step: interview scheduled."
            elif dec == "shortlist":
                body = f"Dear {name}, we would like to shortlist your application for future consideration and keep it on file."
            else:
                body = f"Dear {name}, unfortunately we will not be moving forward. The role requires specific certifications that were not present. We wish you the best."
            r = env.step(compose_action(cid, email, "Your Application — Nexus Tech", body))
            msg = r.observation.last_action_result or ""
            tid = extract_thread_id(msg)
            thread_ids[cid] = tid

        result = None
        for cid, tid in thread_ids.items():
            result = env.step(send_action(cid, tid))

        assert result is not None
        assert result.done
        return result

    def test_task1_walkthrough(self):
        result = self._walk_task1()
        assert result.done
        assert result.info.get("final_reward_breakdown", {}).get("total", 0) > 0

    def test_task2_walkthrough(self):
        result = self._walk_task2()
        assert result.done

    def test_task3_walkthrough_no_exception(self):
        """Task 3 must complete without unhandled exceptions."""
        env = make_env()
        env.reset("negotiation_and_edge_cases")
        decisions = {
            "C001": "accept", "C002": "reject", "C003": "reject",
            "C004": "reject", "C005": "reject",
        }
        emails = {
            "C001": ("leila.amir@email.com", "Leila"),
            "C002": ("ben.okafor@email.com", "Ben"),
            "C003": ("ingrid.solberg@email.com", "Ingrid"),
            "C004": ("rafael.moreno@email.com", "Rafael"),
            "C005": ("yuki.tanaka@email.com", "Yuki"),
        }

        for cid, dec in decisions.items():
            env.step(DecisionAction(candidate_id=cid, decision=DecisionType(dec), reason="Evaluated"))

        thread_ids = {}
        for cid, (email, name) in emails.items():
            dec = decisions[cid]
            if dec == "accept":
                body = f"Dear {name}, we are pleased to inform you that we would like to move forward for the Senior Product Manager role. Next step: we will schedule a call."
            else:
                body = f"Dear {name}, unfortunately we will not be moving forward with your application. The role requires extensive B2B SaaS leadership. We wish you the best."
            r = env.step(compose_action(cid, email, "Senior PM Role — Orbital Systems", body))
            msg = r.observation.last_action_result or ""
            tid = extract_thread_id(msg)
            thread_ids[cid] = tid

        result = None
        for cid, tid in thread_ids.items():
            result = env.step(send_action(cid, tid))

        assert result is not None
        assert result.done