"""
ToS Environment — Core Server-Side Logic
=========================================
Implements the OpenEnv Environment base class interface:
  reset()  → TosObservation
  step()   → TosStepResult
  state()  → TosState

Supports three tasks selected via the TASK_NAME environment variable
(or the `task` query param on the /reset endpoint):
  - binary_risk            (Task 1, Easy)
  - category_classification (Task 2, Medium)
  - full_audit             (Task 3, Hard)
"""

from __future__ import annotations

import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional

# Ensure parent package is importable when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TosAction, TosObservation, TosReward, TosState, TosStepResult
from data.corpus import (
    CLAUSES,
    CLAUSE_BY_ID,
    DOCUMENTS,
    DOCUMENT_BY_ID,
    RISKY_CLAUSES,
    SAFE_CLAUSES,
)
from graders import (
    grade_task1_binary_risk,
    grade_task2_category,
    grade_task3_full_audit,
)

# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_NAMES = ("binary_risk", "category_classification", "full_audit")

TASK_DESCRIPTIONS = {
    "binary_risk": (
        "You are analysing a Terms of Service clause. "
        "Determine whether the clause is 'risky' or 'safe' for users. "
        "Set action.verdict to 'risky' or 'safe'."
    ),
    "category_classification": (
        "You are analysing a risky Terms of Service clause. "
        "Classify it into exactly one of these categories: "
        "Privacy, Liability, Termination, Payments, Changes, Other. "
        "Set action.category to your answer and action.reasoning to your explanation."
    ),
    "full_audit": (
        "You are auditing a complete Terms of Service document. "
        "Identify ALL risky clauses. For each, provide: "
        "clause_text (exact quote), category (Privacy/Liability/Termination/Payments/Changes/Other), "
        "and risk_score (integer 1-10). "
        "Set action.findings to a list of these objects."
    ),
}

MAX_STEPS = {
    "binary_risk":            1,
    "category_classification": 1,
    "full_audit":             3,  # Up to 3 refinement steps on audit
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class TosEnvironment:
    """
    Terms of Service Risk Analyzer — OpenEnv Environment.

    Implements the gymnasium-style step/reset/state interface.
    Thread-safe for single-session usage (one episode at a time).
    """

    def __init__(self, task_name: str = "binary_risk", seed: Optional[int] = None):
        if task_name not in TASK_NAMES:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {TASK_NAMES}"
            )
        self.task_name = task_name
        self._rng = random.Random(seed)

        # Episode state — populated by reset()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0

        # Task-specific episode data
        self._current_clause: Optional[Dict[str, Any]] = None   # Tasks 1 & 2
        self._current_document: Optional[Dict[str, Any]] = None  # Task 3

        # Track best audit score across steps (Task 3)
        self._best_audit_score: float = 0.0
        self._best_audit_breakdown: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(self, task_name: Optional[str] = None) -> TosObservation:
        """
        Start a new episode.  Returns the initial observation.
        """
        if task_name and task_name in TASK_NAMES:
            self.task_name = task_name

        self._episode_id       = str(uuid.uuid4())
        self._step_count       = 0
        self._done             = False
        self._cumulative_reward = 0.0
        self._best_audit_score  = 0.0
        self._best_audit_breakdown = {}

        if self.task_name == "binary_risk":
            # Mix risky and safe roughly 50 / 50
            pool = self._rng.choice([RISKY_CLAUSES, SAFE_CLAUSES])
            self._current_clause = self._rng.choice(pool)
            doc_text = self._current_clause["text"]
            meta: Dict[str, Any] = {
                "clause_id":  self._current_clause["id"],
                "difficulty": "easy",
            }

        elif self.task_name == "category_classification":
            self._current_clause = self._rng.choice(RISKY_CLAUSES)
            doc_text = self._current_clause["text"]
            meta = {
                "clause_id":  self._current_clause["id"],
                "difficulty": "medium",
                "valid_categories": [
                    "Privacy", "Liability", "Termination",
                    "Payments", "Changes", "Other"
                ],
            }

        else:  # full_audit
            self._current_document = self._rng.choice(DOCUMENTS)
            doc_text = self._current_document["text"]
            meta = {
                "document_id": self._current_document["id"],
                "title":       self._current_document["title"],
                "difficulty":  "hard",
                "hint": (
                    "The document contains multiple paragraphs separated by blank lines. "
                    "Each paragraph may be a clause."
                ),
            }

        return TosObservation(
            document_text=doc_text,
            task_name=self.task_name,
            task_description=TASK_DESCRIPTIONS[self.task_name],
            step_number=0,
            max_steps=MAX_STEPS[self.task_name],
            metadata=meta,
            last_action_error=None,
        )

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, action: TosAction) -> TosStepResult:
        """
        Execute the agent's action and return (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        self._step_count += 1
        error_msg: Optional[str] = None

        # ── Task 1 — Binary risk ──────────────────────────────────────────
        if self.task_name == "binary_risk":
            assert self._current_clause is not None
            score, feedback = grade_task1_binary_risk(
                verdict=action.verdict,
                ground_truth_is_risky=self._current_clause["is_risky"],
            )
            reward = score
            done   = True
            info: Dict[str, Any] = {
                "grader": "binary_risk",
                "score":   score,
                "feedback": feedback,
                "ground_truth": {
                    "is_risky": self._current_clause["is_risky"],
                    "category": self._current_clause["category"],
                },
            }

        # ── Task 2 — Category classification ─────────────────────────────
        elif self.task_name == "category_classification":
            assert self._current_clause is not None
            score, breakdown, feedback = grade_task2_category(
                category=action.category,
                reasoning=action.reasoning,
                ground_truth_category=self._current_clause["category"],
            )
            reward = score
            done   = True
            info = {
                "grader":    "category_classification",
                "score":     score,
                "breakdown": breakdown,
                "feedback":  feedback,
                "ground_truth": {
                    "category":   self._current_clause["category"],
                    "risk_score": self._current_clause["risk_score"],
                },
            }

        # ── Task 3 — Full audit ───────────────────────────────────────────
        else:
            assert self._current_document is not None
            findings = action.findings or []
            gt_clauses = self._current_document["ground_truth_risky_clauses"]

            score, breakdown, feedback = grade_task3_full_audit(
                agent_findings=findings,
                gt_clauses=gt_clauses,
                full_corpus=CLAUSE_BY_ID,
            )

            # Partial reward: improvements over previous best score
            improvement = max(0.0, score - self._best_audit_score)
            step_reward  = score if self._step_count == 1 else improvement

            # Track best
            if score > self._best_audit_score:
                self._best_audit_score     = score
                self._best_audit_breakdown = breakdown

            reward = round(step_reward, 4)
            # Episode ends either at max steps or if agent achieves perfect score
            done = (self._step_count >= MAX_STEPS[self.task_name]) or (score >= 1.0)

            info = {
                "grader":       "full_audit",
                "step_score":   score,
                "step_reward":  reward,
                "best_score":   self._best_audit_score,
                "breakdown":    breakdown,
                "feedback":     feedback,
                "ground_truth_count": len(gt_clauses),
            }

        self._cumulative_reward += reward

        # If done, also penalise for steps wasted on error
        if error_msg:
            reward = max(0.0, reward - 0.05)

        self._done = done

        # Build next observation (shows updated step, same doc)
        if self.task_name in ("binary_risk", "category_classification"):
            doc_text = self._current_clause["text"]  # type: ignore
            meta     = {"clause_id": self._current_clause["id"]}  # type: ignore
        else:
            doc_text = self._current_document["text"]  # type: ignore
            meta     = {
                "document_id": self._current_document["id"],  # type: ignore
                "title":       self._current_document["title"],  # type: ignore
            }

        obs = TosObservation(
            document_text=doc_text,
            task_name=self.task_name,
            task_description=TASK_DESCRIPTIONS[self.task_name],
            step_number=self._step_count,
            max_steps=MAX_STEPS[self.task_name],
            metadata=meta,
            last_action_error=error_msg,
        )

        return TosStepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    # -----------------------------------------------------------------------
    # state()
    # -----------------------------------------------------------------------

    def state(self) -> TosState:
        """Return current episode metadata."""
        doc_id = None
        if self._current_document:
            doc_id = self._current_document.get("id")
        elif self._current_clause:
            doc_id = self._current_clause.get("id")

        return TosState(
            episode_id=self._episode_id,
            task_name=self.task_name,
            step_count=self._step_count,
            max_steps=MAX_STEPS[self.task_name],
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            document_id=doc_id,
        )
