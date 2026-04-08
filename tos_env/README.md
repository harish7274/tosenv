---
title: ToS Risk Analyzer
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - legal-ai
  - reinforcement-learning
  - document-analysis
  - NLP
short_description: OpenEnv environment for Terms of Service risk analysis
---

# ToS Risk Analyzer — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://pypi.org/project/openenv-core/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An **OpenEnv**-compliant reinforcement learning environment that trains and evaluates AI agents on **real-world legal document risk assessment** — the task of reading Terms of Service agreements and identifying clauses that are harmful or unfair to users.

---

## Why This Environment?

Millions of people accept Terms of Service agreements every day without reading them. Studies show that fully reading one year's worth of encountered privacy policies alone would take 76 working days. AI agents that can accurately detect risky clauses at scale would provide immediate practical value for consumers, regulators, and compliance teams.

This environment challenges agents to:
1. Distinguish risky from safe clauses (binary classification)
2. Categorise *why* a clause is risky (multi-class with partial credit)
3. Audit an entire ToS document end-to-end (F1-scored structured extraction)

---

## Observation Space

```python
TosObservation(
    document_text: str,      # The clause or full document to analyse
    task_name: str,          # "binary_risk" | "category_classification" | "full_audit"
    task_description: str,   # Instructions for the agent
    step_number: int,        # Current step in the episode
    max_steps: int,          # Episode step limit
    metadata: dict,          # clause_id, document_id, difficulty, hints
    last_action_error: str|None  # Error from previous action (null if none)
)
```

## Action Space

```python
TosAction(
    # Task 1 — Binary Risk
    verdict: str | None,        # "risky" or "safe"

    # Task 2 — Category Classification
    category: str | None,       # Privacy | Liability | Termination | Payments | Changes | Other
    reasoning: str | None,      # Explanation (earns bonus score)

    # Task 3 — Full Audit
    findings: list | None,      # [{"clause_text": "...", "category": "...", "risk_score": 7}]
)
```

---

## Tasks

### Task 1 — Binary Risk Classification 🟢 Easy

**Objective:** Given a single ToS clause, determine if it is `"risky"` or `"safe"` for users.

| | |
|---|---|
| **Max Steps** | 1 |
| **Score Range** | 0.0 or 1.0 (binary) |
| **Expected Baseline** | ~0.65 (random = 0.50) |

**Example:**
- Clause: *"We may share your personal information with third-party advertising partners without prior notice."*
- Correct action: `{"verdict": "risky"}` → Score: `1.0`

---

### Task 2 — Category Classification 🟡 Medium

**Objective:** Given a risky clause, classify it into one of five categories: `Privacy`, `Liability`, `Termination`, `Payments`, or `Changes`. Provide a reasoning explanation for bonus points.

| | |
|---|---|
| **Max Steps** | 1 |
| **Score Range** | 0.0 – 1.0 (partial credit) |
| **Scoring** | Exact match: 0.85 + reasoning bonus up to 0.15 |
| **Partial Credit** | Adjacent category: 0.40 |
| **Expected Baseline** | ~0.55 |

**Scoring breakdown:**
```
score = category_score + reasoning_bonus
  category_score:  0.85 (exact), 0.40 (adjacent), 0.00 (wrong)
  reasoning_bonus: up to 0.15 based on keyword coverage
```

---

### Task 3 — Full Document Audit 🔴 Hard

**Objective:** Given a full ToS document (10+ clauses), identify **all** risky clauses with their categories and risk scores (1–10). Up to 3 refinement steps.

| | |
|---|---|
| **Max Steps** | 3 (agent can refine findings) |
| **Score Range** | 0.0 – 1.0 |
| **Scoring** | F1 (60%) + Category accuracy (25%) + Risk score accuracy (15%) |
| **Per-Step Reward** | Improvement over previous best score |
| **Expected Baseline** | ~0.35 |

**Scoring formula:**
```
total = 0.60 * F1 + 0.25 * category_accuracy + 0.15 * risk_score_accuracy

F1 = 2 * precision * recall / (precision + recall)
  precision = correctly_found / total_found
  recall    = correctly_found / total_ground_truth

category_accuracy = avg category correctness of matched clauses
risk_score_accuracy = avg score: exact=1.0, ±1→0.8, ±2→0.5, ±3→0.2, ±4→0
```

---

## Reward Design

### Partial Progress Signals

The reward function provides **meaningful signal across the full trajectory**:

- **Task 1:** Binary signal, but correct answers are immediately informative for policy gradients.
- **Task 2:** Continuous score 0–1 with category adjacency and reasoning quality bonus. An agent that correctly identifies the risk *domain* (even if not exact) gets partial credit.
- **Task 3:** Multi-component F1-based reward with **per-step improvement bonuses**. Step 1 gives the base score; steps 2–3 reward incremental improvements, discouraging trivial repeated submissions.

### Penalty Avoidance

- Submitting no findings → `0.0` reward (not negative, but maximal opportunity cost)
- Invalid action formats → `last_action_error` field set in next observation for self-correction

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
# 1. Clone / navigate to the environment
cd tos_env

# 2. Install dependencies
pip install -r server/requirements.txt

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 4. Verify it's running
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Docker

```bash
# Build
docker build -t tos-risk-analyzer .

# Run
docker run -p 8000:8000 tos-risk-analyzer

# Test
curl http://localhost:8000/health
```

---

## Running the Inference Script

```bash
# Set environment variables
export HF_TOKEN="your_hf_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:8000"   # or your HF Space URL

# Run inference against all 3 tasks
python inference.py
```

Expected output format:
```
[START] task=binary_risk env=tos-risk-analyzer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"verdict":"risky"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=category_classification env=tos-risk-analyzer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category":"Privacy","reasoning":"..."} reward=0.92 done=true error=null
[END] success=true steps=1 score=0.92 rewards=0.92

[START] task=full_audit env=tos-risk-analyzer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"findings":[...]} reward=0.48 done=false error=null
[STEP] step=2 action={"findings":[...]} reward=0.12 done=false error=null
[STEP] step=3 action={"findings":[...]} reward=0.05 done=true error=null
[END] success=false steps=3 score=0.65 rewards=0.48,0.12,0.05
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset?task=<name>&seed=<int>` | Start new episode |
| `POST` | `/step` | Submit action body as JSON |
| `GET`  | `/state` | Get episode metadata |
| `GET`  | `/health` | Health check |
| `GET`  | `/` | API info |

### Example: Full Episode

```python
import requests

BASE = "http://localhost:8000"

# Reset for Task 1
obs = requests.post(f"{BASE}/reset", params={"task": "binary_risk", "seed": 42}).json()
print(obs["document_text"])  # clause text

# Step with action
result = requests.post(f"{BASE}/step", json={"verdict": "risky"}).json()
print(result["reward"])  # 1.0 or 0.0
print(result["done"])    # True

# State
state = requests.get(f"{BASE}/state").json()
print(state["cumulative_reward"])
```

---

## Running Tests

```bash
cd tos_env
pip install pytest
pytest tests/ -v
```

---

## Baseline Scores

Scores achieved with `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Inference:

| Task | Difficulty | Score |
|------|-----------|-------|
| `binary_risk` | Easy | ~0.75 |
| `category_classification` | Medium | ~0.62 |
| `full_audit` | Hard | ~0.38 |
| **Average** | | **~0.58** |

*Note: Scores vary by seed. The above are approximate averages over 5 runs.*

---

## Project Structure

```
tos_env/
├── __init__.py              # Public API exports
├── models.py                # TosObservation, TosAction, TosReward, TosState
├── client.py                # TosEnvClient (Python SDK)
├── graders.py               # Deterministic graders for all 3 tasks
├── inference.py             # Baseline inference script (OpenAI client)
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # HF Spaces-compatible container
├── pyproject.toml           # Package config
├── .dockerignore
├── data/
│   ├── __init__.py
│   └── corpus.py            # 40 labelled clauses + 3 full documents
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI application (reset/step/state endpoints)
│   ├── tos_environment.py   # Core environment logic
│   └── requirements.txt     # Server dependencies
└── tests/
    ├── __init__.py
    └── test_environment.py  # Unit tests for graders + environment
```

---

## Hugging Face Space Deployment

```bash
# Push to HF Space (after installing openenv-core CLI)
pip install openenv-core[cli]
openenv push --repo-id your-username/tos-risk-analyzer
```

Or manually:
1. Create a new HF Space with SDK = "docker"
2. Upload all files from `tos_env/`
3. Set Space hardware to CPU Basic (2 vCPU, 16 GB)
4. The app will auto-start on port 7860

---

## License

MIT License — see [LICENSE](LICENSE) for details.
