# Job Application Environment

An [OpenEnv](https://github.com/openenv/openenv)-compatible benchmark for evaluating
AI agents on real-world recruitment workflows. Agents read job applications, make
accept/reject decisions, compose personalised notification emails, send them, and
handle follow-up threads (salary negotiation, reapplication requests). Scoring is
fully deterministic — no LLM judge required.

## Environment description

Recruiters and hiring managers process hundreds of applications, draft notifications,
and navigate edge cases (counter-offers, appeals) every day. This environment models
that workflow at varying complexity levels, making it a practical benchmark for agents
that need to reason, communicate, and stay within policy constraints across multi-step
episodes.

## Observation space

| Field | Type | Description |
|---|---|---|
| `applications` | `list[ApplicationRecord]` | All candidates visible to the agent |
| `inbox` | `list[EmailThread]` | Email threads including simulated candidate replies |
| `pending_decisions` | `list[str]` | Candidate IDs still awaiting a decision |
| `step` | `int` | Current step number |
| `last_action_result` | `str \| null` | Result message from the previous action |
| `task_description` | `str` | Full role description and constraints |

Each `ApplicationRecord` contains: `candidate_id`, `name`, `email`,
`years_experience`, `skills`, `certifications`, `current_salary`,
`expected_salary`, `cover_letter`.

## Action space

Four action types, selected via the `action_type` discriminator field:

```json
{"action_type": "decision",      "candidate_id": "C001", "decision": "accept|reject|shortlist", "reason": "..."}
{"action_type": "compose_email", "candidate_id": "C001", "recipient": "x@y.com", "subject": "...", "body": "..."}
{"action_type": "send_email",    "candidate_id": "C001", "thread_id": "<from compose result>"}
{"action_type": "request_info",  "candidate_id": "C001", "question": "..."}
```

Workflow per candidate: `decision` → `compose_email` → `send_email`.

## Tasks

| Task | Difficulty | Candidates | Max steps | Expected score |
|---|---|---|---|---|
| `single_clear_decision` | Easy | 1 | 4 | ~0.85 |
| `batch_with_quota` | Medium | 8 (2 seats) | 24 | ~0.55 |
| `negotiation_and_edge_cases` | Hard | 5 + follow-ups | 20 | ~0.25 |

**Easy:** One applicant who clearly meets all criteria. Accept, compose, send.

**Medium:** Eight applicants for two seats. Mixed signals — one overqualified,
one missing a required cert, two strong backups to shortlist. Agent must rank
and provide personalised rejection reasons to all six rejected candidates.

**Hard:** Five applicants for one senior PM seat. After accepting the best
candidate, they counter-offer above the salary band (which is firm). A rejected
candidate sends a reapplication requesting structured feedback. Agent must handle
both follow-up threads within the step budget.

## Reward function

| Component | Weight | Description |
|---|---|---|
| Decision accuracy | 0.60 | Correct accept/reject vs ground truth. Partial credit: shortlist when accept = 0.5 |
| Email quality | 0.25 | Structured field checks: name, role, correct language, next-steps/feedback, no placeholders |
| Delivery | 0.10 | Fraction of required notifications actually sent |
| Efficiency penalty | 0.05 | Penalty for using >80% of step budget |

Intermediate rewards (+0.05) are given per valid `compose_email` to provide
signal throughout the episode rather than only at the end.

To reproduce:
```bash
export API_BASE_URL=https://Navyasri12355-job-application-env.hf.space
export OPENAI_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=<your-token>
python inference.py 2>/dev/null
```

## Setup

### Local (without Docker)
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
# in a second terminal:
export API_BASE_URL=http://localhost:8000
python inference.py
```

### Docker
```bash
docker build . -t job-app-env
docker run -p 7860:7860 job-app-env
```

### Validate before submitting
```bash
pip install openenv-core
./scripts/validate-submission.sh https://Navyasri12355-job-application-env.hf.space .
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{"task": "single_clear_decision"}` |
| `POST` | `/step` | Submit an action. Body: serialised action object |
| `GET` | `/state` | Read current episode state |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/health` | Liveness probe |