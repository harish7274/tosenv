"""
inference.py — ToS Risk Analyzer OpenEnv Baseline
===================================================
A baseline inference script that runs an LLM agent against all 3 tasks
and emits structured stdout logs in the mandatory OpenEnv format.

Mandatory env vars:
  HF_TOKEN       — Hugging Face / OpenAI-compatible API key
  API_BASE_URL   — Base URL for the LLM inference endpoint
                   (default: https://router.huggingface.co/v1)
  MODEL_NAME     — Model identifier
                   (default: Qwen/Qwen2.5-72B-Instruct)

Optional:
  ENV_BASE_URL   — ToS env server URL (default: http://localhost:8000)
  MAX_STEPS      — Max steps per episode (default: 3)

Stdout format (MANDATORY — do not alter field names or ordering):
  [START] task=<task_name> env=tos-risk-analyzer model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "3"))
BENCHMARK    = "tos-risk-analyzer"

TASKS = ["binary_risk", "category_classification", "full_audit"]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN or "hf_placeholder",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# HTTP helpers (raw requests — no openenv-core dependency required)
# ---------------------------------------------------------------------------

import requests

_SESSION = requests.Session()


def _reset(task: str, seed: int = 42) -> Dict[str, Any]:
    resp = _SESSION.post(f"{ENV_BASE_URL}/reset", params={"task": task, "seed": seed}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _step(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = _SESSION.post(f"{ENV_BASE_URL}/step", json=action_payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _state() -> Dict[str, Any]:
    resp = _SESSION.get(f"{ENV_BASE_URL}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM agent helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a legal AI expert specialising in Terms of Service analysis.
You will be given a clause or document and must analyse it for risks.
Always respond with a valid JSON object and no other text.
"""


def _build_task1_prompt(obs: Dict) -> str:
    return f"""Analyse the following Terms of Service clause and determine if it is risky or safe for users.

CLAUSE:
{obs['document_text']}

Respond with ONLY this JSON (no other text):
{{"verdict": "risky"}}
or
{{"verdict": "safe"}}"""


def _build_task2_prompt(obs: Dict) -> str:
    return f"""Classify the following risky Terms of Service clause into one of these categories:
Privacy, Liability, Termination, Payments, Changes, Other

CLAUSE:
{obs['document_text']}

Respond with ONLY this JSON (no other text):
{{"category": "<category>", "reasoning": "<1-2 sentences explaining why>"}}"""


def _build_task3_prompt(obs: Dict, step: int) -> str:
    hint = ""
    if step > 1:
        hint = "\nThis is a refinement step. Improve your previous findings."
    return f"""You are auditing the following Terms of Service document for risky clauses.{hint}

DOCUMENT:
{obs['document_text'][:6000]}

Identify ALL risky clauses. For each, provide:
- clause_text: exact quote from the document
- category: one of Privacy/Liability/Termination/Payments/Changes/Other
- risk_score: integer 1-10

Respond with ONLY this JSON (no other text):
{{"findings": [{{"clause_text": "...", "category": "...", "risk_score": 8}}]}}"""


def _call_llm(prompt: str) -> str:
    """Call the LLM and return raw content string."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


def _parse_json_safe(text: str) -> Optional[Dict]:
    """Try to parse JSON from LLM output, with some cleanup."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code block
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try extracting bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _build_action(task_name: str, parsed: Optional[Dict], step: int) -> Dict[str, Any]:
    """Convert parsed LLM output to action payload."""
    if parsed is None:
        # Fallback actions
        if task_name == "binary_risk":
            return {"verdict": "risky"}
        elif task_name == "category_classification":
            return {"category": "Privacy", "reasoning": "Unable to parse LLM output."}
        else:
            return {"findings": []}

    if task_name == "binary_risk":
        return {"verdict": parsed.get("verdict", "risky")}
    elif task_name == "category_classification":
        return {
            "category":  parsed.get("category", "Other"),
            "reasoning": parsed.get("reasoning", ""),
        }
    else:
        findings = parsed.get("findings", [])
        # Ensure each finding has required fields
        cleaned = []
        for f in findings:
            if isinstance(f, dict) and "clause_text" in f:
                cleaned.append({
                    "clause_text": str(f.get("clause_text", "")),
                    "category":    str(f.get("category", "Other")),
                    "risk_score":  int(f.get("risk_score", 5)),
                })
        return {"findings": cleaned}


# ---------------------------------------------------------------------------
# Logging helpers (MANDATORY format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: Dict, reward: float, done: bool, error: Optional[str]):
    action_str = json.dumps(action, separators=(",", ":"))
    done_str   = "true" if done else "false"
    err_str    = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------

def run_episode(task_name: str, seed: int = 42) -> float:
    """
    Run one episode for the given task.
    Returns the final score.
    """
    log_start(task=task_name, model=MODEL_NAME)

    all_rewards: List[float] = []
    done       = False
    step_num   = 0
    final_score = 0.0
    last_error: Optional[str] = None

    try:
        # Reset
        obs = _reset(task=task_name, seed=seed)
        max_ep_steps = obs.get("max_steps", MAX_STEPS)

        while not done and step_num < max_ep_steps:
            step_num += 1

            # Build prompt based on task
            if task_name == "binary_risk":
                prompt = _build_task1_prompt(obs)
            elif task_name == "category_classification":
                prompt = _build_task2_prompt(obs)
            else:
                prompt = _build_task3_prompt(obs, step_num)

            # Call LLM
            raw_response = _call_llm(prompt)
            parsed       = _parse_json_safe(raw_response)
            action       = _build_action(task_name, parsed, step_num)

            # Step the environment
            try:
                result = _step(action)
                reward     = float(result.get("reward", 0.0))
                done       = bool(result.get("done", True))
                obs        = result.get("observation", obs)
                info       = result.get("info", {})
                last_error = obs.get("last_action_error") if isinstance(obs, dict) else None

                # Track best score for reporting
                step_score = info.get("score", info.get("step_score", reward))
                final_score = max(final_score, float(step_score))

            except Exception as e:
                reward     = 0.0
                done       = True
                last_error = str(e)

            all_rewards.append(reward)
            log_step(
                step=step_num,
                action=action,
                reward=reward,
                done=done,
                error=last_error,
            )

            if not done:
                time.sleep(0.5)  # Rate-limit courtesy

    except Exception as e:
        last_error = str(e)
        # Emit at least one step log on exception
        if step_num == 0:
            step_num = 1
            all_rewards = [0.0]
            log_step(step=1, action={}, reward=0.0, done=True, error=last_error)

    success = final_score >= 0.7
    log_end(
        success=success,
        steps=step_num,
        score=final_score,
        rewards=all_rewards,
    )

    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check server is up
    try:
        resp = _SESSION.get(f"{ENV_BASE_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"ERROR: Environment server not healthy at {ENV_BASE_URL}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to environment server at {ENV_BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    for i, task in enumerate(TASKS):
        seed = 42 + i  # Different seed per task for variety
        run_episode(task_name=task, seed=seed)
        time.sleep(1)


if __name__ == "__main__":
    main()
