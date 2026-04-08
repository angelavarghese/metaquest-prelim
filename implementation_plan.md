# Job Application Environment Project Plan

## Phase 1 — Core Environment

### M01: Data models & Pydantic schemas
Define all typed structures the environment will pass around — observation, action, reward, application record, task config.

*   `models.py` — `ApplicationRecord`, `Observation`, `Action`, `RewardBreakdown`, `EpisodeState`, `TaskConfig` as Pydantic v2 models with full field validation.
*   `Action` has a discriminated union: `decision` (accept/reject/shortlist), `compose_email` (subject, body, recipient), `send_email` (trigger send), `request_info` (ask candidate a clarifying question). Each action type validated independently.
*   `Observation` contains: `applications: list[ApplicationRecord]`, `inbox: list[EmailThread]`, `pending_decisions: list[str]`, `step: int`, `last_action_result: str | None`, `task_description: str`.
*   `RewardBreakdown` exposes per-component scores so the agent and grader can both inspect them: `decision_score`, `email_quality_score`, `delivery_score`, `efficiency_penalty`, `total`.

**Done when:**
*   `python -c "from models import Observation, Action, TaskConfig"` runs without error. 
*   All fields have type annotations and validators. No forward references unresolved.
*   Files created/updated: 
    *   `server/models.py`

---

### M02: Task configs & ground-truth fixtures
Three fully-specified task definitions with synthetic application data, correct decisions, and reference email templates.

*   **Task 1 — easy (`single_clear_decision`):** 1 applicant for a junior SWE role. Candidate meets all criteria unambiguously (4+ years exp, required skills present). Ground truth: accept. One email to send. Max steps: 4. Expected agent score ≥ 0.85.
*   **Task 2 — medium (`batch_with_quota`):** 8 applicants, only 2 slots. Mixed signals — overlapping skill sets, one overqualified, one missing a required cert. Agent must rank, reject 6 with personalised reasons, accept 2 with next-steps. Max steps: 16. Expected agent score ~0.55.
*   **Task 3 — hard (`negotiation_and_edge_cases`):** 5 applicants. One accepted candidate replies with a counter-offer salary negotiation. One rejection triggers a reapplication asking for feedback. Agent must handle follow-up threads, compose nuanced replies, stay within policy constraints (salary band provided). Max steps: 20. Expected score ~0.25.
*   Each task is stored as `tasks/task_{name}.json` with: `applications[]`, `ground_truth_decisions{}`, `required_email_fields{}`, `salary_band`, `role_description`, `max_steps`, `scoring_rubric{}`.

**Done when:**
*   All three JSON fixtures load without schema errors. 
*   Ground truth decisions are stored per candidate ID. 
*   Reference emails contain all required fields. 
*   A human reviewer has sanity-checked that task 3 is genuinely harder than task 1.
*   Files created/updated:
    *   `server/tasks/single_clear_decision.json`
    *   `server/tasks/batch_with_quota.json`
    *   `server/tasks/negotiation_and_edge_cases.json`

**Depends on:** M01

---

### M03: Grader & reward function
Deterministic scoring logic — per-component, partial credit, no LLM judge needed.

*   `grader.py` — `grade_episode(episode_state, task_config) -> RewardBreakdown`. Fully deterministic. Called at each step to emit a partial reward and at episode end for final score.
*   **Decision score (0.6 weight):** compare each agent decision against `ground_truth_decisions`. Correct accept/reject = 1.0 per candidate. Partial credit: shortlist when ground truth is accept = 0.5. Wrong direction = 0.0. Score averaged across all candidates in the task.
*   **Email quality score (0.25 weight):** structured field checks only — no LLM judge. Required fields from task config checked: candidate name present (+0.3), role name present (+0.2), correct accept/reject language (+0.2), next-steps or feedback present (+0.2), no placeholder text like `[NAME]` remaining (+0.1). Checked via regex + substring search.
*   **Delivery score (0.10 weight):** was `send_email` action called for every candidate that needed a notification? 1.0 if all sent, proportional otherwise.
*   **Efficiency penalty (0.05 weight, subtracted):** `steps_used / max_steps` above a 0.8 threshold. Penalises agents that loop or stall. Never makes total score negative.
*   **Intermediate step rewards:** +0.05 for each valid `compose_email` that passes field checks (gives signal mid-episode, not just at end).

**Done when:**
*   Unit tests pass: perfect agent scores 1.0, random agent scores < 0.15, all scores clamped to [0, 1], grader is deterministic (same inputs → same output every run). 
*   No external API calls inside grader.
*   Files created/updated:
    *   `server/grader.py`
    *   `tests/test_grader.py`

**Depends on:** M01, M02

---

### M04: Environment engine
Stateful episode logic — reset, step, state. Clean transitions, episode boundaries, error handling.

*   `env.py` — `JobApplicationEnv` class with `reset(task_name) -> Observation`, `step(action) -> StepResult`, `state() -> EpisodeState`. Pure Python, no async needed here (FastAPI layer handles that).
*   `reset()`: loads task config from JSON, initialises episode state (all applications pending, inbox empty, step=0), returns initial observation with all application records visible to the agent.
*   `step(action)`: validates action type, mutates episode state (mark decision, append email to inbox, trigger simulated candidate reply for task 3 edge cases), calls grader for step reward, checks done conditions (all candidates decided + all emails sent, or max_steps reached), returns `StepResult(observation, reward, done, info)`.
*   **Simulated candidate replies for task 3:** deterministic reply triggers keyed on candidate ID + action type. E.g., candidate C004 always sends a salary counter when accepted. Stored in task fixture so it is reproducible.
*   **Done conditions:** all required decisions made AND all required emails sent, OR `step >= max_steps`. Episode score computed at done.

**Done when:**
*   Manual walkthrough of all three tasks completes without exception. 
*   `reset()` always returns a clean state regardless of prior episode state. 
*   `step()` after done raises `EpisodeAlreadyDoneError`. 
*   `state()` is read-only and never mutates env.
*   Files created/updated:
    *   `server/env.py`
    *   `tests/test_env.py`

**Depends on:** M01, M02, M03


## Phase 2 — API server & spec compliance

### M05: FastAPI server & OpenEnv spec
HTTP endpoints, `openenv.yaml`, and spec compliance so `openenv validate` passes.

*   `main.py` — FastAPI app with routes: 
    *   `POST /reset` (body: `{"task": "single_clear_decision"}`)
    *   `POST /step` (body: serialised `Action`)
    *   `GET /state`
    *   `GET /tasks` (list available task names + descriptions)
    *   `GET /health` (returns 200, used by HF Space ping).
*   All endpoints return JSON with correct HTTP status codes. `/reset` and `/step` accept `Content-Type: application/json`. `/reset` with empty body `{}` defaults to task 1 — this is what the submission validator pings.
*   `openenv.yaml`: name, version, description, tasks list with name/description/difficulty per task, observation_space schema, action_space schema, `reward_range: [0.0, 1.0]`, max_steps per task, tags: `["recruitment", "email", "decision-making", "real-world"]`.
*   **Env singleton pattern:** one `JobApplicationEnv` instance per process (not per request). Thread-safe resets using a lock — needed for the HF Space single-container deployment.

**Done when:**
*   `openenv validate` exits 0. 
*   `curl -X POST http://localhost:8000/reset -d '{}'` returns HTTP 200 with a valid Observation JSON. 
*   `GET /tasks` returns all three tasks.
*   Files created/updated:
    *   `server/main.py`
    *   `openenv.yaml`

**Depends on:** M04


## Phase 3 — Inference script

### M06: `inference.py` — baseline agent loop
OpenAI-client agent that runs all three tasks, emits exact START/STEP/END stdout format, reproduces baseline scores.

*   Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` (as api_key), optionally `LOCAL_IMAGE_NAME` from environment. Defaults set for `API_BASE_URL` and `MODEL_NAME` only — never for `HF_TOKEN`.
*   **System prompt:** explains available action types with JSON schema examples, the task objective, and explicit instruction to always output valid JSON matching one of the four action types. No markdown, no prose — raw JSON action only.
*   **User prompt at each step:** serialised Observation as JSON + step number + last reward. Keeps last 3 observations in context window to avoid token overflow on task 3.
*   Runs all three tasks in sequence within one execution. Emits one `[START]` block per task, one `[STEP]` per step, one `[END]` per task. Total wall-clock time must stay under 20 min on 2 vCPU / 8 GB — achievable since max steps = 4+16+20 = 40 total LLM calls.
*   JSON parse failures fall back to a default `request_info` action (safe no-op that still logs a `[STEP]`). Episode always terminates cleanly — `[END]` always emitted even on exception, matching the sample script pattern.

**Done when:**
*   Script runs end-to-end without error. 
*   `stdout` contains exactly one `[START]`, N `[STEP]`s, and one `[END]` per task. 
*   All reward values in `[END]` are in `[0, 1]`. 
*   Re-running produces scores within ±0.05 of the documented baseline (minor variation from LLM temperature is acceptable).
*   Files created/updated:
    *   `inference.py`

**Depends on:** M05


## Phase 4 — Deployment

### M07: Dockerfile & HF Space deployment
Containerised server, clean build, Space live and responding to `/reset` within 30s.

*   **Dockerfile** in repo root: `FROM python:3.11-slim`, install `requirements.txt`, copy `server/` and `openenv.yaml`, expose port 7860 (HF Space default), `CMD["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]`. No root user. Image must build in under 10 min.
*   **requirements.txt**: pinned versions — `fastapi==0.111.*`, `uvicorn[standard]==0.30.*`, `pydantic==2.7.*`, `openai==1.30.*`, `openenv-core` (latest). No heavy ML libraries — keeps image under 500 MB.
*   **HF Space config** in `README.md` YAML front matter: `sdk: docker`, `app_port: 7860`, `tags:[openenv, recruitment, agent-benchmark]`. Space set to public.
*   **Verify:** `docker build . -t job-app-env && docker run -p 7860:7860 job-app-env` starts cleanly. `curl -X POST http://localhost:7860/reset -d '{}'` returns 200.

**Done when:**
*   `docker build` exits 0 with no errors. 
*   HF Space URL responds to `POST /reset` with HTTP 200. 
*   `./validate-submission.sh <HF_URL>` passes all 3 checks (Step 1: HF live, Step 2: docker build, Step 3: openenv validate).
*   Files created/updated:
    *   `Dockerfile`
    *   `requirements.txt`
    *   `.dockerignore`

**Depends on:** M05, M06


## Phase 5 — Documentation & final checks

### M08: README, baseline scores & submission
Complete documentation, recorded baseline scores, and final validator run before submitting.

*   **README.md sections:** environment description and motivation (why this task matters for agent evaluation), observation space (field-by-field), action space (all four action types with JSON examples), task descriptions with expected difficulty and scoring breakdown, setup instructions (docker + local), baseline scores table.
*   **Baseline scores table:** run `inference.py` against a known model (e.g., Qwen2.5-72B via HF router) three times, record mean ± std. Document expected scores: task 1 ~0.82, task 2 ~0.51, task 3 ~0.23. Include the exact command used to reproduce.
*   **Final pre-submission checklist:** all five items in the competition UI checked — `inference.py` read & followed, env vars present, defaults correct (not `HF_TOKEN`), all LLM calls via OpenAI client, stdout format exact.
*   Run `./validate-submission.sh <HF_URL> .` one final time, confirm "All 3/3 checks passed". Submit GitHub repo URL + HF Space URL.

**Done when:**
*   Validator script exits 0. 
*   README contains all required sections. 
*   Baseline scores are documented with reproduction command. 
*   All five pre-submission checklist items can be honestly checked.
*   Files created/updated:
    *   `README.md`

**Depends on:** M06, M07
