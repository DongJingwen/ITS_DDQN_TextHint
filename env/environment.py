"""Tutoring environment for text-hint recommendation experiments."""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.config import ACTION_DIM, MAX_RESPONSE_TIME, MAX_STEPS_PER_EPISODE, REWARD_WEIGHTS
from env.state_builder import build_state
from knowledge import loader as KnowledgeLoader
from student.simulator import StudentSimulator


ACTION_TO_HINT_LEVEL = {
    0: "level_1",
    1: "level_2",
    2: "level_3",
}

DIFFICULTY_ORDER = ("low", "medium", "high")


class TutoringEnvironment:
    """A small RL environment for simulated quadratic-equation tutoring."""

    def __init__(
        self,
        student: StudentSimulator | None = None,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        initial_ability: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.initial_ability = initial_ability
        self.student = student or StudentSimulator(
            student_id="sim_student",
            ability=initial_ability,
            rng=self.rng,
        )
        self.max_steps = max_steps
        self.knowledge_points = KnowledgeLoader.get_knowledge_points()
        self.question_sequence = self._build_question_sequence()
        self.recent_correctness: deque[float] = deque(maxlen=3)
        self.current_step = 0
        self.current_question_index = 0
        self.current_question: dict[str, Any] | None = None
        self.current_kp_id = "KP1"
        self.current_difficulty = "low"
        self.last_response_time = 0.0
        self.last_attempts = 0

    def _build_question_sequence(self) -> list[dict[str, Any]]:
        sequence = []
        for knowledge_point in self.knowledge_points:
            kp_id = knowledge_point["kp_id"]
            for difficulty in DIFFICULTY_ORDER:
                for question in KnowledgeLoader.get_questions(kp_id, difficulty):
                    sequence.append(
                        {
                            "kp_id": kp_id,
                            "kp_name": knowledge_point["name"],
                            "difficulty": difficulty,
                            "question": question,
                        }
                    )

        if not sequence:
            raise ValueError("Knowledge tree does not contain any questions.")

        return sequence

    def _set_current_question(self) -> None:
        question_item = self.question_sequence[
            self.current_question_index % len(self.question_sequence)
        ]
        self.current_kp_id = question_item["kp_id"]
        self.current_difficulty = question_item["difficulty"]
        self.current_question = question_item["question"]

    def _recent_accuracy(self) -> float:
        if not self.recent_correctness:
            return 0.0
        return sum(self.recent_correctness) / len(self.recent_correctness)

    def _build_current_state(self) -> np.ndarray:
        return build_state(
            mastery_level=self.student.mastery,
            question_difficulty=self.current_difficulty,
            recent_accuracy=self._recent_accuracy(),
            response_time=self.last_response_time,
            attempts=self.last_attempts,
            knowledge_point_index=self.current_kp_id,
        )

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        self.student.reset(ability=self.initial_ability)
        self.recent_correctness.clear()
        self.current_step = 0
        self.current_question_index = 0
        self.last_response_time = 0.0
        self.last_attempts = 0
        self._set_current_question()
        return self._build_current_state()

    def _calculate_reward(
        self,
        action: int,
        interaction: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        correct_component = (
            REWARD_WEIGHTS["correct"]
            if interaction["correct"]
            else REWARD_WEIGHTS["incorrect_penalty"]
        )

        mastery_gain = interaction["mastery"] - interaction["previous_mastery"]
        improvement_component = REWARD_WEIGHTS["improvement"] if mastery_gain > 0 else 0.0

        normalized_response_time = min(
            interaction["response_time"] / MAX_RESPONSE_TIME,
            1.0,
        )
        response_time_component = REWARD_WEIGHTS["efficient_response_time"] * (
            1.0 - normalized_response_time
        )

        extra_attempts = max(0, interaction["attempts"] - 1)
        attempts_component = REWARD_WEIGHTS["extra_attempt_penalty"] * extra_attempts

        hint_usage_component = REWARD_WEIGHTS["hint_usage_penalty"] * (3 - action) / 3

        unnecessary_strong_hint_component = 0.0
        if action == 0 and interaction["previous_mastery"] >= 0.75:
            unnecessary_strong_hint_component = REWARD_WEIGHTS[
                "unnecessary_strong_hint_penalty"
            ]

        components = {
            "correctness": correct_component,
            "improvement": improvement_component,
            "response_time": response_time_component,
            "attempts": attempts_component,
            "hint_usage": hint_usage_component,
            "unnecessary_strong_hint": unnecessary_strong_hint_component,
        }
        return sum(components.values()), components

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply a hint-level action and return next_state, reward, done, info."""
        if action < 0 or action >= ACTION_DIM:
            raise ValueError(f"Invalid action {action}. Expected 0, 1, or 2.")
        if self.current_question is None:
            raise RuntimeError("Environment must be reset before calling step().")

        hint_level = ACTION_TO_HINT_LEVEL[action]
        question_id = self.current_question["question_id"]
        hint_text = KnowledgeLoader.get_hint(question_id, hint_level)

        interaction = self.student.simulate_interaction(
            difficulty=self.current_difficulty,
            action=action,
        )
        reward, reward_components = self._calculate_reward(action, interaction)

        self.recent_correctness.append(1.0 if interaction["correct"] else 0.0)
        self.last_response_time = interaction["response_time"]
        self.last_attempts = interaction["attempts"]

        info = {
            "kp_id": self.current_kp_id,
            "difficulty": self.current_difficulty,
            "question_id": question_id,
            "question": self.current_question["question"],
            "answer": self.current_question["answer"],
            "action": action,
            "hint_level": hint_level,
            "hint_text": hint_text,
            "correct": interaction["correct"],
            "probability_correct": interaction["probability_correct"],
            "response_time": interaction["response_time"],
            "attempts": interaction["attempts"],
            "previous_mastery": interaction["previous_mastery"],
            "mastery": interaction["mastery"],
            "recent_accuracy": self._recent_accuracy(),
            "reward_components": reward_components,
        }

        self.current_step += 1
        self.current_question_index += 1
        done = self.current_step >= self.max_steps

        if not done:
            self._set_current_question()

        next_state = self._build_current_state()
        return next_state, reward, done, info


if __name__ == "__main__":
    environment = TutoringEnvironment(max_steps=5, initial_ability=0.45, seed=11)
    state = environment.reset()
    done = False
    total_reward = 0.0

    print(f"Initial state: {state}")

    while not done:
        action = environment.rng.randrange(ACTION_DIM)
        next_state, reward, done, info = environment.step(action)
        total_reward += reward
        print(
            {
                "step": environment.current_step,
                "question_id": info["question_id"],
                "action": action,
                "correct": info["correct"],
                "attempts": info["attempts"],
                "reward": round(reward, 3),
                "mastery": round(info["mastery"], 3),
            }
        )
        state = next_state

    print(f"Final state: {state}")
    print(f"Total reward: {round(total_reward, 3)}")
