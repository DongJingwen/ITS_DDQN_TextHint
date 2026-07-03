"""Simple real-user experimental session interface.

This module is intentionally not a web UI. It provides a small Python interface
for command-line or notebook-based real-student evaluation workflows.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from env.environment import TutoringEnvironment


class RealUserSession:
    """Manage one student's pretest, learning session, and posttest records."""

    def __init__(
        self,
        student_id: str,
        agent: Any,
        environment: TutoringEnvironment,
        db_manager: Any | None = None,
        student_name: str | None = None,
    ) -> None:
        self.student_id = student_id
        self.student_name = student_name
        self.agent = agent
        self.environment = environment
        self.db_manager = db_manager

        self.pretest_score: float | None = None
        self.posttest_score: float | None = None
        self.learning_gain: float | None = None
        self.estimated_level: float | None = None
        self.initial_difficulty: str | None = None
        self.interactions: list[dict[str, Any]] = []

    @staticmethod
    def _normalize_score(score: float) -> float:
        score = float(score)
        if score > 1.0:
            score = score / 100.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _difficulty_from_score(score: float) -> str:
        if score < 0.5:
            return "low"
        if score < 0.8:
            return "medium"
        return "high"

    def run_pretest(self, pretest_score: float) -> tuple[float, str]:
        """Estimate initial mastery and difficulty from a pretest score."""
        normalized_score = self._normalize_score(pretest_score)
        self.pretest_score = normalized_score
        self.estimated_level = normalized_score
        self.initial_difficulty = self._difficulty_from_score(normalized_score)

        if self.db_manager is not None:
            self.db_manager.add_student(
                student_id=self.student_id,
                name=self.student_name,
                group_type="real",
                initial_level=self.estimated_level,
            )
            self.db_manager.save_pretest_result(
                student_id=self.student_id,
                score=self.pretest_score,
                estimated_level=self.estimated_level,
                initial_difficulty=self.initial_difficulty,
            )

        return self.estimated_level, self.initial_difficulty

    def _reset_environment_for_real_user(self, max_steps: int) -> Any:
        if self.estimated_level is not None and hasattr(self.environment, "initial_ability"):
            self.environment.initial_ability = self.estimated_level

        if hasattr(self.environment, "max_steps"):
            self.environment.max_steps = max_steps

        return self.environment.reset()

    def start_learning_session(self, max_steps: int = 20) -> dict[str, Any]:
        """Run a learning session using the existing environment interface."""
        state = self._reset_environment_for_real_user(max_steps=max_steps)
        done = False
        step = 0
        session_interactions: list[dict[str, Any]] = []

        while not done and step < max_steps:
            epsilon_before = len(getattr(self.agent, "epsilon_history", []))
            action = self.agent.select_action(state)
            epsilon = None
            epsilon_history = getattr(self.agent, "epsilon_history", [])
            if len(epsilon_history) > epsilon_before:
                epsilon = epsilon_history[-1]

            next_state, reward, done, info = self.environment.step(action)
            step += 1

            interaction = {
                "student_id": self.student_id,
                "episode": 1,
                "step": step,
                "kp_id": info["kp_id"],
                "question_id": info["question_id"],
                "difficulty": info["difficulty"],
                "state": state,
                "action": action,
                "hint_level": info["hint_level"],
                "reward": reward,
                "correct": info["correct"],
                "response_time": info["response_time"],
                "attempts": info["attempts"],
                "mastery_level": info["mastery"],
                "epsilon": epsilon,
            }
            session_interactions.append(interaction)
            self.interactions.append(interaction)

            if self.db_manager is not None:
                self.db_manager.log_interaction(**interaction)
                self.db_manager.log_hint_usage(
                    student_id=self.student_id,
                    kp_id=info["kp_id"],
                    question_id=info["question_id"],
                    action=action,
                    hint_level=info["hint_level"],
                    used=True,
                )

            state = next_state

        return self._build_summary(session_interactions)

    def run_posttest(self, posttest_score: float) -> float:
        """Save posttest score and return learning gain."""
        normalized_score = self._normalize_score(posttest_score)
        self.posttest_score = normalized_score
        baseline = self.pretest_score if self.pretest_score is not None else 0.0
        self.learning_gain = normalized_score - baseline

        if self.db_manager is not None:
            self.db_manager.save_posttest_result(
                student_id=self.student_id,
                score=self.posttest_score,
                learning_gain=self.learning_gain,
            )

        return self.learning_gain

    def _build_summary(self, interactions: list[dict[str, Any]]) -> dict[str, Any]:
        total_reward = sum(item["reward"] for item in interactions)
        response_times = [item["response_time"] for item in interactions]
        attempts = [item["attempts"] for item in interactions]
        correct_values = [1.0 if item["correct"] else 0.0 for item in interactions]
        hint_counts = Counter(item["hint_level"] for item in interactions)

        interaction_count = len(interactions)
        return {
            "student_id": self.student_id,
            "pretest_score": self.pretest_score,
            "posttest_score": self.posttest_score,
            "learning_gain": self.learning_gain,
            "total_reward": total_reward,
            "average_response_time": (
                sum(response_times) / interaction_count if interaction_count else 0.0
            ),
            "average_attempts": sum(attempts) / interaction_count if interaction_count else 0.0,
            "hint_usage_distribution": dict(hint_counts),
            "correct_rate": sum(correct_values) / interaction_count if interaction_count else 0.0,
            "steps": interaction_count,
        }

    def get_summary(self) -> dict[str, Any]:
        """Return summary statistics for all learning interactions so far."""
        return self._build_summary(self.interactions)


if __name__ == "__main__":
    from agent.double_dqn import DoubleDQNAgent
    from database.db_manager import DatabaseManager

    db_path = PROJECT_ROOT / "outputs" / "logs" / "real_user_session_demo.db"
    db = DatabaseManager(db_path)
    db.initialize_database()

    demo_agent = DoubleDQNAgent(device="cpu")
    demo_environment = TutoringEnvironment(max_steps=3, seed=21)
    session = RealUserSession(
        student_id="DEMO001",
        student_name="Demo Student",
        agent=demo_agent,
        environment=demo_environment,
        db_manager=db,
    )

    print(session.run_pretest(0.62))
    print(session.start_learning_session(max_steps=3))
    print(session.run_posttest(0.76))
    print(session.get_summary())
    print(db.get_interactions("DEMO001"))
    db.close()
