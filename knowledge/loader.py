"""Loader utilities for the quadratic-equation knowledge tree and hint bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_TREE_PATH = BASE_DIR / "knowledge_tree.json"
HINT_BANK_PATH = BASE_DIR / "hint_bank.json"

VALID_DIFFICULTIES = {"low", "medium", "high"}
VALID_HINT_LEVELS = {"level_1", "level_2", "level_3"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required knowledge file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_knowledge_tree() -> dict[str, Any]:
    return _load_json(KNOWLEDGE_TREE_PATH)


def _load_hint_bank() -> dict[str, Any]:
    return _load_json(HINT_BANK_PATH)


def get_knowledge_points() -> list[dict[str, Any]]:
    """Return all knowledge points from the quadratic-equation knowledge tree."""
    knowledge_tree = _load_knowledge_tree()
    knowledge_points = knowledge_tree.get("knowledge_points")

    if not isinstance(knowledge_points, list):
        raise ValueError("knowledge_tree.json must contain a knowledge_points list.")

    return knowledge_points


def _find_knowledge_point(kp_id: str) -> dict[str, Any]:
    for knowledge_point in get_knowledge_points():
        if knowledge_point.get("kp_id") == kp_id:
            return knowledge_point

    raise ValueError(f"Unknown knowledge point id: {kp_id}")


def get_questions(kp_id: str, difficulty: str) -> list[dict[str, Any]]:
    """Return all questions for a knowledge point and difficulty level."""
    difficulty = difficulty.lower()

    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. Expected one of: {sorted(VALID_DIFFICULTIES)}"
        )

    knowledge_point = _find_knowledge_point(kp_id)
    difficulty_levels = knowledge_point.get("difficulty_levels", {})
    questions = difficulty_levels.get(difficulty)

    if not isinstance(questions, list):
        raise ValueError(f"No question list found for {kp_id} at difficulty '{difficulty}'.")

    return questions


def get_question(
    kp_id: str,
    difficulty: str,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Return a specific question, or the first question for the requested group."""
    questions = get_questions(kp_id, difficulty)

    if not questions:
        raise ValueError(f"No questions available for {kp_id} at difficulty '{difficulty}'.")

    if question_id is None:
        return questions[0]

    for question in questions:
        if question.get("question_id") == question_id:
            return question

    raise ValueError(
        f"Question id '{question_id}' not found for {kp_id} at difficulty '{difficulty}'."
    )


def get_hint(question_id: str, hint_level: str) -> str:
    """Return the text hint for a question and hint level."""
    if hint_level not in VALID_HINT_LEVELS:
        raise ValueError(
            f"Invalid hint level '{hint_level}'. Expected one of: {sorted(VALID_HINT_LEVELS)}"
        )

    hint_bank = _load_hint_bank()
    hints_by_question = hint_bank.get("hints", {})
    question_hints = hints_by_question.get(question_id)

    if question_hints is None:
        raise ValueError(f"No hints found for question id: {question_id}")

    hint = question_hints.get(hint_level)
    if hint is None:
        raise ValueError(f"No {hint_level} hint found for question id: {question_id}")

    return hint


if __name__ == "__main__":
    knowledge_points = get_knowledge_points()
    sample_question = get_question("KP1", "low")
    sample_hint = get_hint(sample_question["question_id"], "level_1")

    print(f"Loaded {len(knowledge_points)} knowledge points.")
    print(f"Sample question: {sample_question['question']}")
    print(f"Sample level_1 hint: {sample_hint}")
