"""Probabilistic student simulator for the text-hint ITS prototype."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


DIFFICULTY_VALUES = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
}

HINT_EFFECTS = {
    0: 0.30,
    1: 0.20,
    2: 0.10,
}


@dataclass
class StudentSimulator:
    """Simulate student correctness, efficiency, attempts, and mastery growth."""

    student_id: str = "sim_student"
    ability: float = 0.5
    learning_rate: float = 0.04
    forgetting_rate: float = 0.01
    base_response_time: float = 45.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.ability = self._clip(self.ability)
        self.mastery = self.ability
        self.interaction_count = 0

    @staticmethod
    def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _difficulty_value(self, difficulty: str | float | int) -> float:
        if isinstance(difficulty, str):
            normalized = difficulty.lower()
            if normalized not in DIFFICULTY_VALUES:
                raise ValueError(
                    f"Invalid difficulty '{difficulty}'. Expected low, medium, or high."
                )
            return DIFFICULTY_VALUES[normalized]

        difficulty_value = float(difficulty)
        if difficulty_value not in DIFFICULTY_VALUES.values():
            return self._clip(difficulty_value)
        return difficulty_value

    def _hint_effect(self, action: int) -> float:
        if action not in HINT_EFFECTS:
            raise ValueError("Invalid action. Expected 0, 1, or 2 for hint level.")
        return HINT_EFFECTS[action]

    def probability_correct(self, difficulty: str | float | int, action: int) -> float:
        """Estimate correctness probability from ability, difficulty, and hint effect."""
        difficulty_value = self._difficulty_value(difficulty)
        hint_effect = self._hint_effect(action)
        logit = 4.0 * (self.ability - difficulty_value + hint_effect)
        return self._sigmoid(logit)

    def simulate_response_time(self, difficulty: str | float | int, action: int) -> float:
        """Generate a positive response time in seconds."""
        difficulty_value = self._difficulty_value(difficulty)
        hint_effect = self._hint_effect(action)
        support_speedup = hint_effect * 0.6
        expected_time = self.base_response_time * (
            1.0 + difficulty_value - self.ability - support_speedup
        )
        noise = self.rng.gauss(0.0, 5.0)
        return max(5.0, expected_time + noise)

    def simulate_attempts(
        self,
        is_correct: bool,
        difficulty: str | float | int,
        action: int,
    ) -> int:
        """Generate the number of attempts used for the interaction."""
        difficulty_value = self._difficulty_value(difficulty)
        hint_effect = self._hint_effect(action)
        struggle_score = difficulty_value - self.ability - hint_effect

        if is_correct and struggle_score <= -0.15:
            return 1
        if is_correct:
            return 1 if self.rng.random() < 0.65 else 2
        if struggle_score > 0.25:
            return 3
        return 2 if self.rng.random() < 0.6 else 3

    def update_mastery(
        self,
        is_correct: bool,
        difficulty: str | float | int,
        action: int,
        attempts: int,
    ) -> float:
        """Update ability/mastery after one learning interaction."""
        difficulty_value = self._difficulty_value(difficulty)
        hint_effect = self._hint_effect(action)

        challenge_bonus = max(0.0, difficulty_value - self.ability)
        hint_dependence_penalty = hint_effect * 0.4
        attempt_penalty = max(0, attempts - 1) * 0.01

        if is_correct:
            delta = self.learning_rate * (0.5 + challenge_bonus - hint_dependence_penalty)
            delta -= attempt_penalty
        else:
            delta = -self.forgetting_rate * (0.5 + difficulty_value)

        self.ability = self._clip(self.ability + delta)
        self.mastery = self.ability
        return self.mastery

    def simulate_interaction(
        self,
        difficulty: str | float | int,
        action: int,
    ) -> dict[str, Any]:
        """Simulate one tutoring interaction after receiving a hint action."""
        probability = self.probability_correct(difficulty, action)
        is_correct = self.rng.random() < probability
        response_time = self.simulate_response_time(difficulty, action)
        attempts = self.simulate_attempts(is_correct, difficulty, action)
        previous_mastery = self.mastery
        updated_mastery = self.update_mastery(is_correct, difficulty, action, attempts)

        self.interaction_count += 1

        return {
            "student_id": self.student_id,
            "correct": is_correct,
            "probability_correct": probability,
            "response_time": response_time,
            "attempts": attempts,
            "previous_mastery": previous_mastery,
            "mastery": updated_mastery,
            "ability": self.ability,
            "difficulty_value": self._difficulty_value(difficulty),
            "hint_effect": self._hint_effect(action),
            "action": action,
            "interaction_count": self.interaction_count,
        }

    def reset(self, ability: float | None = None) -> None:
        """Reset the simulator, optionally with a new starting ability."""
        if ability is not None:
            self.ability = self._clip(ability)
        self.mastery = self.ability
        self.interaction_count = 0


if __name__ == "__main__":
    student = StudentSimulator(student_id="demo_student", ability=0.45, rng=random.Random(7))

    for action in (0, 1, 2):
        result = student.simulate_interaction(difficulty="medium", action=action)
        print(
            {
                "action": result["action"],
                "correct": result["correct"],
                "probability_correct": round(result["probability_correct"], 3),
                "response_time": round(result["response_time"], 2),
                "attempts": result["attempts"],
                "mastery": round(result["mastery"], 3),
            }
        )
