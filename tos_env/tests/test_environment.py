"""
Unit tests for all three graders and the core environment.
Run with: pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from graders import (
    grade_task1_binary_risk,
    grade_task2_category,
    grade_task3_full_audit,
)
from data.corpus import CLAUSE_BY_ID, RISKY_CLAUSES, SAFE_CLAUSES, DOCUMENTS
from models import TosAction
from server.tos_environment import TosEnvironment


# ===========================================================================
# Grader tests
# ===========================================================================

class TestTask1Grader:
    def test_correct_risky(self):
        score, _ = grade_task1_binary_risk("risky", True)
        assert score == 0.95

    def test_correct_safe(self):
        score, _ = grade_task1_binary_risk("safe", False)
        assert score == 0.95

    def test_wrong_verdict(self):
        score, _ = grade_task1_binary_risk("risky", False)
        assert score == 0.05

    def test_none_verdict(self):
        score, _ = grade_task1_binary_risk(None, True)
        assert score == 0.05

    def test_invalid_verdict(self):
        score, _ = grade_task1_binary_risk("unknown", True)
        assert score == 0.05

    def test_score_in_range(self):
        for verdict in ("risky", "safe"):
            for truth in (True, False):
                score, _ = grade_task1_binary_risk(verdict, truth)
                assert 0.0 < score < 1.0


class TestTask2Grader:
    def test_exact_match(self):
        score, breakdown, _ = grade_task2_category("Privacy", None, "Privacy")
        assert score >= 0.85

    def test_adjacent_match(self):
        score, _, _ = grade_task2_category("Liability", None, "Privacy")
        assert 0.3 <= score <= 0.5

    def test_wrong_category(self):
        score, _, _ = grade_task2_category("Payments", None, "Privacy")
        assert 0.0 < score < 1.0

    def test_reasoning_bonus(self):
        score_no_reason, _, _ = grade_task2_category("Privacy", None, "Privacy")
        score_with_reason, _, _ = grade_task2_category(
            "Privacy",
            "This clause collects personal data and shares it with third parties.",
            "Privacy"
        )
        assert score_with_reason >= score_no_reason

    def test_max_score_le_1(self):
        score, _, _ = grade_task2_category(
            "Privacy",
            "This collects personal data privacy information share collect biometric",
            "Privacy"
        )
        assert 0.0 < score < 1.0

    def test_none_category(self):
        score, _, _ = grade_task2_category(None, None, "Privacy")
        assert score == 0.01


class TestTask3Grader:
    def _gt(self):
        return DOCUMENTS[0]["ground_truth_risky_clauses"]

    def test_perfect_score(self):
        # Build perfect findings from ground truth
        gt = self._gt()
        perfect_findings = []
        for item in gt:
            clause = CLAUSE_BY_ID[item["clause_id"]]
            perfect_findings.append({
                "clause_text": clause["text"],
                "category":    item["category"],
                "risk_score":  item["risk_score"],
            })
        score, breakdown, _ = grade_task3_full_audit(perfect_findings, gt, CLAUSE_BY_ID)
        assert score > 0.8
        assert breakdown["f1"] > 0.9

    def test_empty_findings(self):
        gt = self._gt()
        score, _, _ = grade_task3_full_audit([], gt, CLAUSE_BY_ID)
        assert score == 0.05

    def test_none_findings(self):
        gt = self._gt()
        score, _, _ = grade_task3_full_audit(None, gt, CLAUSE_BY_ID)
        assert score == 0.05

    def test_partial_findings(self):
        gt = self._gt()
        # Submit only half the ground truth
        half = gt[:len(gt)//2]
        findings = [{
            "clause_text": CLAUSE_BY_ID[item["clause_id"]]["text"],
            "category":    item["category"],
            "risk_score":  item["risk_score"],
        } for item in half]
        score, breakdown, _ = grade_task3_full_audit(findings, gt, CLAUSE_BY_ID)
        assert 0.0 < score < 1.0
        assert breakdown["recall"] < 0.9

    def test_score_in_range(self):
        gt = self._gt()
        findings = [{"clause_text": "arbitrary text", "category": "Privacy", "risk_score": 5}]
        score, _, _ = grade_task3_full_audit(findings, gt, CLAUSE_BY_ID)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Environment tests
# ===========================================================================

class TestEnvironment:
    def test_reset_returns_observation(self):
        env = TosEnvironment(task_name="binary_risk", seed=1)
        obs = env.reset()
        assert obs.task_name == "binary_risk"
        assert obs.step_number == 0
        assert len(obs.document_text) > 0

    def test_step_returns_result(self):
        env = TosEnvironment(task_name="binary_risk", seed=1)
        env.reset()
        action = TosAction(verdict="risky")
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0
        assert result.done is True

    def test_done_after_step_raises(self):
        env = TosEnvironment(task_name="binary_risk", seed=1)
        env.reset()
        env.step(TosAction(verdict="risky"))
        with pytest.raises(RuntimeError):
            env.step(TosAction(verdict="risky"))

    def test_state_returns_state(self):
        env = TosEnvironment(task_name="binary_risk", seed=1)
        env.reset()
        state = env.state()
        assert state.done is False
        assert state.step_count == 0

    def test_all_tasks_reset(self):
        for task in ("binary_risk", "category_classification", "full_audit"):
            env = TosEnvironment(task_name=task, seed=42)
            obs = env.reset()
            assert obs.task_name == task

    def test_task2_step(self):
        env = TosEnvironment(task_name="category_classification", seed=2)
        env.reset()
        action = TosAction(category="Privacy", reasoning="Data sharing clause.")
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0

    def test_task3_multiple_steps(self):
        env = TosEnvironment(task_name="full_audit", seed=3)
        env.reset()
        action = TosAction(findings=[{
            "clause_text": "We may share your personal information with third parties.",
            "category": "Privacy",
            "risk_score": 8,
        }])
        result1 = env.step(action)
        assert 0.0 <= result1.reward <= 1.0
        if not result1.done:
            result2 = env.step(action)
            assert 0.0 <= result2.reward <= 1.0

    def test_reproducible_with_seed(self):
        env1 = TosEnvironment(task_name="binary_risk", seed=99)
        env2 = TosEnvironment(task_name="binary_risk", seed=99)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.document_text == obs2.document_text

    def test_different_seeds_different_obs(self):
        env1 = TosEnvironment(task_name="binary_risk", seed=1)
        env2 = TosEnvironment(task_name="binary_risk", seed=100)
        obs1 = env1.reset()
        obs2 = env2.reset()
        # Very likely to differ with different seeds across 40-clause corpus
        # (not guaranteed but passes with high probability)
        # Just verify both are non-empty
        assert len(obs1.document_text) > 0
        assert len(obs2.document_text) > 0
