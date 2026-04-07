"""
Graders — Programmatic Scoring (0.0 – 1.0)
============================================
Three graders, one per task.  All are deterministic given the same
ground-truth corpus and the same agent action.

  grade_task1_binary_risk(action, truth) → float
  grade_task2_category(action, truth)    → float
  grade_task3_full_audit(action, truth)  → float, breakdown dict
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ===========================================================================
# Task 1 — Binary Risk Classification (EASY)
# ===========================================================================

def grade_task1_binary_risk(
    verdict: str | None,
    ground_truth_is_risky: bool,
) -> Tuple[float, str]:
    """
    Score Task 1: binary risky / safe verdict.

    Scoring:
      - Correct verdict               → 1.0
      - Wrong verdict                 → 0.0
      - Missing / invalid verdict     → 0.0

    Returns (score, feedback_string).
    """
    if not verdict:
        return 0.0, "No verdict provided. Expected 'risky' or 'safe'."

    normalised = verdict.strip().lower()
    if normalised not in ("risky", "safe"):
        return 0.0, f"Invalid verdict '{verdict}'. Must be 'risky' or 'safe'."

    agent_says_risky = (normalised == "risky")
    correct = (agent_says_risky == ground_truth_is_risky)

    if correct:
        label = "risky" if ground_truth_is_risky else "safe"
        return 1.0, f"Correct! The clause is {label}."
    else:
        expected = "risky" if ground_truth_is_risky else "safe"
        return 0.0, f"Incorrect. The clause is {expected}, but you said '{normalised}'."


# ===========================================================================
# Task 2 — Category Classification (MEDIUM)
# ===========================================================================

# Category adjacency for partial credit
_ADJACENT: Dict[str, set] = {
    "Privacy":     {"Privacy", "Liability", "Changes"},
    "Liability":   {"Liability", "Privacy", "Termination", "Changes"},
    "Termination": {"Termination", "Liability", "Payments"},
    "Payments":    {"Payments", "Termination", "Changes"},
    "Changes":     {"Changes", "Privacy", "Payments", "Liability"},
    "Other":       {"Other"},
}

# Keywords that must appear in reasoning for bonus reasoning credit
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Privacy":     ["data", "personal", "information", "share", "collect", "privacy", "biometric"],
    "Liability":   ["liable", "liability", "damage", "indemnif", "waiver", "warranty"],
    "Termination": ["terminat", "suspend", "cancel", "account", "end"],
    "Payments":    ["fee", "charge", "payment", "refund", "billing", "price"],
    "Changes":     ["modif", "change", "update", "alter", "amend"],
    "Other":       [],
}

VALID_CATEGORIES = {"Privacy", "Liability", "Termination", "Payments", "Changes", "Other"}


def _normalise_category(cat: str) -> str:
    """Map common variations to canonical category names."""
    mapping = {
        "data": "Privacy",
        "security": "Privacy",
        "ip": "Liability",
        "intellectual property": "Liability",
        "damage": "Liability",
        "cancel": "Termination",
        "account": "Termination",
        "billing": "Payments",
        "subscription": "Payments",
        "update": "Changes",
        "amendment": "Changes",
    }
    lowered = cat.strip().lower()
    # Try direct canonical match first
    for canon in VALID_CATEGORIES:
        if lowered == canon.lower():
            return canon
    # Try substring mapping
    for key, value in mapping.items():
        if key in lowered:
            return value
    return "Other"


def _reasoning_keyword_score(reasoning: str | None, category: str) -> float:
    """Return 0.0–0.15 bonus based on keyword coverage in reasoning."""
    if not reasoning:
        return 0.0
    keywords = _CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        return 0.0
    lowered = reasoning.lower()
    hits = sum(1 for kw in keywords if kw in lowered)
    return min(0.15, round(hits / len(keywords) * 0.15, 4))


def grade_task2_category(
    category: str | None,
    reasoning: str | None,
    ground_truth_category: str,
) -> Tuple[float, Dict[str, float], str]:
    """
    Score Task 2: category classification with reasoning bonus.

    Scoring:
      - Exact category match          → 0.85
      - Adjacent category match       → 0.40
      - Wrong category                → 0.0
      - Reasoning keyword bonus       → up to +0.15
      Total max                       → 1.0

    Returns (score, breakdown, feedback_string).
    """
    if not category:
        return 0.0, {"category": 0.0, "reasoning": 0.0}, "No category provided."

    normalised = _normalise_category(category)
    breakdown: Dict[str, float] = {}

    adjacent = _ADJACENT.get(ground_truth_category, {ground_truth_category})

    if normalised == ground_truth_category:
        cat_score = 0.85
        cat_feedback = f"Correct category: '{ground_truth_category}'."
    elif normalised in adjacent:
        cat_score = 0.40
        cat_feedback = (
            f"Close — '{normalised}' is adjacent to '{ground_truth_category}'. "
            f"Correct answer: '{ground_truth_category}'."
        )
    else:
        cat_score = 0.0
        cat_feedback = (
            f"Wrong category. '{normalised}' is not related to '{ground_truth_category}'."
        )

    breakdown["category"] = cat_score
    reasoning_bonus = _reasoning_keyword_score(reasoning, ground_truth_category)
    breakdown["reasoning"] = reasoning_bonus

    total = min(1.0, cat_score + reasoning_bonus)
    feedback = (
        f"{cat_feedback} "
        f"Reasoning bonus: {reasoning_bonus:.2f}. "
        f"Total score: {total:.2f}."
    )
    return round(total, 4), breakdown, feedback


# ===========================================================================
# Task 3 — Full Document Audit (HARD)   F1 + quality scoring
# ===========================================================================

def _text_overlap(agent_text: str, gt_text: str, threshold: float = 0.30) -> bool:
    """
    Return True if the agent's clause text overlaps sufficiently
    with the ground-truth clause text.
    Uses simple bag-of-words Jaccard similarity.
    """
    def tokenise(t: str) -> set:
        return set(re.findall(r"\w+", t.lower()))

    agent_tokens = tokenise(agent_text)
    gt_tokens    = tokenise(gt_text)
    if not agent_tokens or not gt_tokens:
        return False
    intersection = agent_tokens & gt_tokens
    union        = agent_tokens | gt_tokens
    return len(intersection) / len(union) >= threshold


def _match_findings_to_ground_truth(
    agent_findings: List[Dict[str, Any]],
    gt_clauses: List[Dict[str, Any]],  # list of {clause_id, category, risk_score}
    full_corpus: Dict[str, Dict],      # id → clause dict (from corpus.py)
) -> Tuple[int, int, int]:
    """
    Match agent findings to ground-truth risky clauses via text overlap.

    Returns (true_positives, false_positives, false_negatives).
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for finding in agent_findings:
        agent_text = finding.get("clause_text", "")
        found_match = False
        for gt in gt_clauses:
            if gt["clause_id"] in matched_gt:
                continue
            gt_text = full_corpus.get(gt["clause_id"], {}).get("text", "")
            if _text_overlap(agent_text, gt_text):
                tp += 1
                matched_gt.add(gt["clause_id"])
                found_match = True
                break
        if not found_match:
            fp += 1

    fn = len(gt_clauses) - len(matched_gt)
    return tp, fp, fn


def _category_accuracy(
    agent_findings: List[Dict[str, Any]],
    gt_clauses: List[Dict[str, Any]],
    full_corpus: Dict[str, Dict],
) -> float:
    """For correctly identified clauses, score category accuracy (0–1)."""
    scores = []
    matched_gt = set()

    for finding in agent_findings:
        agent_text = finding.get("clause_text", "")
        agent_cat  = finding.get("category", "Other")
        for gt in gt_clauses:
            if gt["clause_id"] in matched_gt:
                continue
            gt_text = full_corpus.get(gt["clause_id"], {}).get("text", "")
            if _text_overlap(agent_text, gt_text):
                matched_gt.add(gt["clause_id"])
                # Exact match
                if _normalise_category(agent_cat) == gt["category"]:
                    scores.append(1.0)
                # Adjacent
                elif _normalise_category(agent_cat) in _ADJACENT.get(gt["category"], set()):
                    scores.append(0.5)
                else:
                    scores.append(0.0)
                break

    return sum(scores) / len(scores) if scores else 0.0


def _risk_score_accuracy(
    agent_findings: List[Dict[str, Any]],
    gt_clauses: List[Dict[str, Any]],
    full_corpus: Dict[str, Dict],
) -> float:
    """For correctly identified clauses, score risk_score accuracy (0–1)."""
    scores = []
    matched_gt = set()

    for finding in agent_findings:
        agent_text       = finding.get("clause_text", "")
        agent_risk_score = finding.get("risk_score", 5)
        try:
            agent_risk_score = int(agent_risk_score)
        except (ValueError, TypeError):
            agent_risk_score = 5

        for gt in gt_clauses:
            if gt["clause_id"] in matched_gt:
                continue
            gt_text = full_corpus.get(gt["clause_id"], {}).get("text", "")
            if _text_overlap(agent_text, gt_text):
                matched_gt.add(gt["clause_id"])
                diff = abs(agent_risk_score - gt["risk_score"])
                # Perfect: 0 diff→1.0, off by 1→0.8, off by 2→0.5, off by 3→0.2, ≥4→0
                score_map = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2}
                scores.append(score_map.get(diff, 0.0))
                break

    return sum(scores) / len(scores) if scores else 0.0


def grade_task3_full_audit(
    agent_findings: List[Dict[str, Any]] | None,
    gt_clauses: List[Dict[str, Any]],
    full_corpus: Dict[str, Dict],
    w_f1: float = 0.60,
    w_cat: float = 0.25,
    w_risk: float = 0.15,
) -> Tuple[float, Dict[str, float], str]:
    """
    Score Task 3: full document audit.

    Scoring breakdown (weights sum to 1.0):
      - F1 score on clause detection           (60%)
      - Category accuracy (on matched clauses) (25%)
      - Risk score accuracy (on matched)       (15%)

    Returns (total_score, breakdown, feedback).
    """
    if not agent_findings:
        return 0.0, {"f1": 0.0, "category_accuracy": 0.0, "risk_score_accuracy": 0.0}, \
               "No findings provided."

    tp, fp, fn = _match_findings_to_ground_truth(agent_findings, gt_clauses, full_corpus)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cat_acc   = _category_accuracy(agent_findings, gt_clauses, full_corpus)
    risk_acc  = _risk_score_accuracy(agent_findings, gt_clauses, full_corpus)

    total = round(w_f1 * f1 + w_cat * cat_acc + w_risk * risk_acc, 4)

    breakdown = {
        "f1":                 round(f1, 4),
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "true_positives":     tp,
        "false_positives":    fp,
        "false_negatives":    fn,
        "category_accuracy":  round(cat_acc, 4),
        "risk_score_accuracy": round(risk_acc, 4),
    }

    feedback = (
        f"F1={f1:.2f} (P={precision:.2f}, R={recall:.2f}), "
        f"TP={tp}, FP={fp}, FN={fn}. "
        f"Category accuracy={cat_acc:.2f}, Risk score accuracy={risk_acc:.2f}. "
        f"Total score={total:.2f}."
    )

    return total, breakdown, feedback
