"""
M01 — Data models & Pydantic schemas
All typed structures passed between environment, agent, and grader.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecisionType(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    SHORTLIST = "shortlist"


# ---------------------------------------------------------------------------
# Application record
# ---------------------------------------------------------------------------

class ApplicationRecord(BaseModel):
    candidate_id: str = Field(..., description="Unique candidate identifier, e.g. C001")
    name: str
    email: str
    years_experience: int = Field(..., ge=0)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    current_salary: int | None = Field(None, ge=0)
    expected_salary: int | None = Field(None, ge=0)
    cover_letter: str = ""
    notes: str = ""

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("email must contain '@'")
        return v


# ---------------------------------------------------------------------------
# Email structures
# ---------------------------------------------------------------------------

class EmailMessage(BaseModel):
    from_address: str
    to_address: str
    subject: str
    body: str
    timestamp: int = Field(..., description="Unix timestamp (simulated step clock)")


class EmailThread(BaseModel):
    thread_id: str
    candidate_id: str
    messages: list[EmailMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actions — discriminated union
# ---------------------------------------------------------------------------

class DecisionAction(BaseModel):
    action_type: Literal["decision"] = "decision"  # Changed from 'type'
    candidate_id: str
    decision: DecisionType
    reason: str = Field("", description="Brief reason for the decision")


class ComposeEmailAction(BaseModel):
    action_type: Literal["compose_email"] = "compose_email"  # Changed from 'type'
    candidate_id: str
    recipient: str = Field(..., description="Recipient email address")
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)

    @field_validator("subject", "body")
    @classmethod
    def no_unresolved_placeholders(cls, v: str) -> str:
        import re
        if re.search(r"\[(?:NAME|ROLE|DATE|POSITION)\]", v, re.IGNORECASE):
            raise ValueError(f"Unresolved placeholder found in field: {v!r}")
        return v


class SendEmailAction(BaseModel):
    action_type: Literal["send_email"] = "send_email"  # Changed from 'type'
    candidate_id: str
    # Made optional with a default to satisfy the test payloads that omit it
    thread_id: str | None = Field(None, description="Thread ID of the composed email to send")


class RequestInfoAction(BaseModel):
    action_type: Literal["request_info"] = "request_info"  # Changed from 'type'
    candidate_id: str
    question: str = Field(..., min_length=1)


Action = Annotated[
    Union[DecisionAction, ComposeEmailAction, SendEmailAction, RequestInfoAction],
    Field(discriminator="action_type"),  # Changed from 'type'
]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    applications: list[ApplicationRecord]
    inbox: list[EmailThread] = Field(default_factory=list)
    pending_decisions: list[str] = Field(
        default_factory=list,
        description="Candidate IDs still awaiting a decision",
    )
    step: int = Field(0, ge=0)
    last_action_result: str | None = None
    task_description: str = ""


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    decision_score: float = Field(0.0, ge=0.0, le=1.0)
    email_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    delivery_score: float = Field(0.0, ge=0.0, le=1.0)
    efficiency_penalty: float = Field(0.0, ge=0.0, le=1.0)
    total: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def clamp_total(self) -> RewardBreakdown:
        self.total = max(0.0, min(1.0, self.total))
        return self


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    task_name: str
    step: int = 0
    done: bool = False
    decisions: dict[str, DecisionType] = Field(
        default_factory=dict,
        description="candidate_id -> decision made by agent",
    )
    composed_emails: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="thread_id -> email payload",
    )
    sent_email_ids: list[str] = Field(
        default_factory=list,
        description="Thread IDs that have been sent",
    )
    inbox: list[EmailThread] = Field(default_factory=list)
    reward_so_far: float = 0.0
    final_reward: RewardBreakdown | None = None


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task config (loaded from JSON)
# ---------------------------------------------------------------------------

class ScoringRubric(BaseModel):
    decision_weight: float = 0.60
    email_quality_weight: float = 0.25
    delivery_weight: float = 0.10
    efficiency_weight: float = 0.05
    efficiency_threshold: float = 0.80
    intermediate_compose_reward: float = 0.05


class TaskConfig(BaseModel):
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    role_description: str
    max_steps: int = Field(..., ge=1)
    salary_band: dict[str, int] = Field(
        default_factory=dict,
        description="{'min': int, 'max': int}",
    )
    applications: list[ApplicationRecord]
    ground_truth_decisions: dict[str, DecisionType] = Field(
        ...,
        description="candidate_id -> correct decision",
    )
    required_email_fields: dict[str, list[str]] = Field(
        default_factory=dict,
        description="candidate_id -> list of required field keys",
    )
    scoring_rubric: ScoringRubric = Field(default_factory=ScoringRubric)
    simulated_replies: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="candidate_id -> {trigger_action, reply_body, reply_subject}",
    )
    expected_min_score: float = 0.0