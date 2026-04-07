"""
ToS Risk Analyzer — OpenEnv Models
====================================
Typed Pydantic data structures for:
  - TosObservation  : what the agent sees
  - TosAction       : what the agent can do
  - TosReward       : reward signal returned after each step
  - TosState        : episode metadata
  - TosStepResult   : full step() return bundle
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — what the agent sees
# ---------------------------------------------------------------------------

class TosObservation(BaseModel):
    """Observation returned by reset() and step()."""

    # The primary content the agent must reason about
    document_text: str = Field(
        description="The Terms of Service text (full document or a single clause, depending on task)."
    )
    task_name: str = Field(
        description="Name of the current task: 'binary_risk', 'category_classification', or 'full_audit'."
    )
    task_description: str = Field(
        description="Human-readable description of what the agent must do."
    )
    step_number: int = Field(
        default=0,
        description="Current step index within the episode."
    )
    max_steps: int = Field(
        default=1,
        description="Maximum steps allowed in this episode."
    )
    # Optional hints / metadata passed along with the observation
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary information: clause_id, document_id, difficulty, etc."
    )
    # Previous action error (null on first step)
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message from the previous action, or null if none."
    )


# ---------------------------------------------------------------------------
# Actions — what the agent can do
# ---------------------------------------------------------------------------

class TosAction(BaseModel):
    """Action submitted by the agent via step()."""

    # Task 1 — Binary risk verdict
    verdict: Optional[str] = Field(
        default=None,
        description="Task 1 — 'risky' or 'safe'. Required for 'binary_risk' task."
    )

    # Task 2 — Category classification
    category: Optional[str] = Field(
        default=None,
        description=(
            "Task 2 — One of: 'Privacy', 'Liability', 'Termination', 'Payments', 'Changes', 'Other'. "
            "Required for 'category_classification' task."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's textual explanation (used for partial credit in Tasks 2 & 3)."
    )

    # Task 3 — Full document audit
    findings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Task 3 — List of identified risky clauses. Each item must have: "
            "{'clause_text': str, 'category': str, 'risk_score': int (1-10)}."
        )
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class TosReward(BaseModel):
    """Reward signal returned alongside each observation."""

    value: float = Field(
        description="Scalar reward in [0.0, 1.0]."
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Sub-component scores that make up the total reward (for debugging)."
    )
    feedback: str = Field(
        default="",
        description="Human-readable feedback explaining the reward."
    )


# ---------------------------------------------------------------------------
# State — episode metadata
# ---------------------------------------------------------------------------

class TosState(BaseModel):
    """Episode state / metadata, returned by state()."""

    episode_id: str = Field(description="Unique identifier for the current episode.")
    task_name: str = Field(description="Active task name.")
    step_count: int = Field(description="Steps taken so far.")
    max_steps: int = Field(description="Maximum steps allowed.")
    done: bool = Field(description="Whether the episode has ended.")
    cumulative_reward: float = Field(description="Total reward accumulated so far.")
    document_id: Optional[str] = Field(
        default=None,
        description="Identifier of the document being analysed."
    )


# ---------------------------------------------------------------------------
# StepResult — full bundle returned by step()
# ---------------------------------------------------------------------------

class TosStepResult(BaseModel):
    """Complete result returned by step()."""

    observation: TosObservation
    reward: float = Field(description="Scalar reward value in [0.0, 1.0].")
    done: bool = Field(description="True when the episode is complete.")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary info dict (reward breakdown, grader details, etc.)."
    )
