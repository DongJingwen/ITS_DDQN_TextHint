"""Build fixed-size numerical state vectors for the tutoring environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from config.config import MAX_ATTEMPTS, MAX_RESPONSE_TIME, STATE_DIM
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.config import MAX_ATTEMPTS, MAX_RESPONSE_TIME, STATE_DIM


DIFFICULTY_ENCODING = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

MAX_DIFFICULTY_INDEX = 2
MAX_KNOWLEDGE_POINT_INDEX = 8


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def encode_difficulty(difficulty: str | int | float) -> float:
    """Encode question difficulty as a normalized value in [0, 1]."""
    if isinstance(difficulty, str):
        difficulty_key = difficulty.lower()
        if difficulty_key not in DIFFICULTY_ENCODING:
            raise ValueError("difficulty must be one of: low, medium, high")
        difficulty_index = DIFFICULTY_ENCODING[difficulty_key]
    else:
        difficulty_index = int(difficulty)
        if difficulty_index not in DIFFICULTY_ENCODING.values():
            raise ValueError("numeric difficulty must be 0, 1, or 2")

    return difficulty_index / MAX_DIFFICULTY_INDEX


def normalize_knowledge_point_index(knowledge_point_index: str | int) -> float:
    """Normalize KP1-KP8 or integer indices to [0, 1]."""
    if isinstance(knowledge_point_index, str):
        kp_text = knowledge_point_index.upper().replace("KP", "")
        if not kp_text.isdigit():
            raise ValueError("knowledge_point_index must be like 'KP1' or an integer 1-8")
        kp_index = int(kp_text)
    else:
        kp_index = int(knowledge_point_index)

    if kp_index < 1 or kp_index > MAX_KNOWLEDGE_POINT_INDEX:
        raise ValueError("knowledge_point_index must be between 1 and 8")

    return (kp_index - 1) / (MAX_KNOWLEDGE_POINT_INDEX - 1)


def build_state(
    mastery_level: float,
    question_difficulty: str | int | float,
    recent_accuracy: float,
    response_time: float,
    attempts: int | float,
    knowledge_point_index: str | int,
) -> np.ndarray:
    """Return the six-dimensional state vector as float32."""
    state = np.array(
        [
            _clip(float(mastery_level)),
            encode_difficulty(question_difficulty),
            _clip(float(recent_accuracy)),
            _clip(float(response_time) / MAX_RESPONSE_TIME),
            _clip(float(attempts) / MAX_ATTEMPTS),
            normalize_knowledge_point_index(knowledge_point_index),
        ],
        dtype=np.float32,
    )

    if state.shape[0] != STATE_DIM:
        raise ValueError(f"State dimension mismatch: expected {STATE_DIM}, got {state.shape[0]}")

    return state


if __name__ == "__main__":
    example_state = build_state(
        mastery_level=0.62,
        question_difficulty="medium",
        recent_accuracy=0.67,
        response_time=48.0,
        attempts=2,
        knowledge_point_index="KP4",
    )

    print(example_state)
    print(example_state.dtype)
