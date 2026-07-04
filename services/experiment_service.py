"""Experiment service helpers for the desktop GUI.

This service is the bridge between GUI pages and backend experiment utilities.
It keeps database logging, pre-test scoring, and learning-page helper methods
outside the widgets themselves.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent.double_dqn import DoubleDQNAgent
from database.db_manager import DatabaseManager
from env.environment import ACTION_TO_HINT_LEVEL, TutoringEnvironment
from env.state_builder import build_state
from knowledge import loader as KnowledgeLoader


@dataclass(frozen=True)
class PretestQuestion:
    question_id: str
    prompt: str
    accepted_answers: tuple[str, ...]


@dataclass(frozen=True)
class PosttestQuestion:
    question_id: str
    prompt: str
    accepted_answers: tuple[str, ...]


@dataclass(frozen=True)
class LearningQuestion:
    kp_id: str
    question_id: str
    difficulty: str
    prompt: str
    answer: str


class ExperimentService:
    """Provide lightweight experiment utilities for GUI pages."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._pretest_questions = [
            PretestQuestion(
                question_id="pre_q1",
                prompt="Which equation is quadratic: x + 3 = 7 or x^2 - 4x + 3 = 0?",
                accepted_answers=("x^2 - 4x + 3 = 0", "x^2-4x+3=0"),
            ),
            PretestQuestion(
                question_id="pre_q2",
                prompt="Write x^2 = 5x - 6 in standard form.",
                accepted_answers=("x^2 - 5x + 6 = 0", "x^2-5x+6=0"),
            ),
            PretestQuestion(
                question_id="pre_q3",
                prompt="Solve x^2 - 5x + 6 = 0.",
                accepted_answers=("x = 2 or x = 3", "x=2 or x=3", "2,3", "3,2"),
            ),
            PretestQuestion(
                question_id="pre_q4",
                prompt="Find the discriminant of x^2 - 6x + 9 = 0.",
                accepted_answers=("0",),
            ),
            PretestQuestion(
                question_id="pre_q5",
                prompt="Use the quadratic formula to solve x^2 - 4x + 3 = 0.",
                accepted_answers=("x = 1 or x = 3", "x=1 or x=3", "1,3", "3,1"),
            ),
        ]
        self._posttest_questions = [
            PosttestQuestion(
                question_id="post_q1",
                prompt="Identify the coefficients a, b, and c in 2x^2 - 7x + 3 = 0.",
                accepted_answers=("a=2,b=-7,c=3", "a = 2, b = -7, c = 3"),
            ),
            PosttestQuestion(
                question_id="post_q2",
                prompt="Solve x^2 - 9 = 0.",
                accepted_answers=("x = 3 or x = -3", "x=3 or x=-3", "3,-3", "-3,3"),
            ),
            PosttestQuestion(
                question_id="post_q3",
                prompt="Solve x^2 - 7x + 12 = 0 by factoring.",
                accepted_answers=("x = 3 or x = 4", "x=3 or x=4", "3,4", "4,3"),
            ),
            PosttestQuestion(
                question_id="post_q4",
                prompt="Find the discriminant of x^2 - 4x + 5 = 0.",
                accepted_answers=("-4",),
            ),
            PosttestQuestion(
                question_id="post_q5",
                prompt="How many real roots does x^2 - 4x + 5 = 0 have?",
                accepted_answers=("0", "none", "no real roots", "zero"),
            ),
        ]
        self.learning_environment = TutoringEnvironment(max_steps=50, seed=42)
        self.learning_questions = self._load_learning_questions()
        self.current_question_index = 0
        self.current_attempts = 0
        self.question_started_at = time.monotonic()
        self.last_hint_action: int | None = None
        self.last_hint_level: str | None = None
        self.last_hint_text: str | None = None
        self.learning_step = 0
        self.agent = DoubleDQNAgent(device="cpu")
        self._try_load_latest_model()
        self.db_path = Path(db_path) if db_path else PROJECT_ROOT / "outputs" / "logs" / "real_user_experiment.db"
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.initialize_database()

    def register_student(self, student_id: str, student_name: str | None = None) -> None:
        self.db_manager.add_student(
            student_id=student_id,
            name=student_name,
            group_type="real",
        )

    def save_pretest_result(
        self,
        student_id: str,
        score: float,
        estimated_level: float,
        initial_difficulty: str,
    ) -> None:
        self.db_manager.save_pretest_result(
            student_id=student_id,
            score=score,
            estimated_level=estimated_level,
            initial_difficulty=initial_difficulty,
        )

    def save_posttest_result(
        self,
        student_id: str,
        posttest_score: float,
        learning_gain: float,
    ) -> None:
        self.db_manager.save_posttest_result(
            student_id=student_id,
            score=posttest_score,
            learning_gain=learning_gain,
        )

    @staticmethod
    def summarize_progress(
        learning_records: list[dict[str, Any]] | None,
        estimated_mastery_level: float | None = None,
    ) -> dict[str, Any]:
        records = learning_records or []
        records_by_question: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            question_id = str(record.get("question_id", "unknown"))
            records_by_question.setdefault(question_id, []).append(record)

        questions_attempted = len(records_by_question)
        correct_questions = sum(
            1
            for question_records in records_by_question.values()
            if any(bool(record.get("correct")) for record in question_records)
        )
        correct_rate = (
            correct_questions / questions_attempted
            if questions_attempted
            else 0.0
        )

        attempts_per_question = [
            max(int(record.get("attempts", 0)) for record in question_records)
            for question_records in records_by_question.values()
        ]
        average_attempts = (
            sum(attempts_per_question) / len(attempts_per_question)
            if attempts_per_question
            else 0.0
        )
        hints_used = sum(1 for record in records if record.get("hint_generated"))
        response_times = [
            float(record.get("response_time", 0.0))
            for record in records
            if record.get("response_time") is not None
        ]
        average_response_time = (
            sum(response_times) / len(response_times)
            if response_times
            else 0.0
        )

        current_mastery = None
        if estimated_mastery_level is not None:
            current_mastery = max(
                0.0,
                min(
                    1.0,
                    estimated_mastery_level
                    + (0.03 * correct_questions)
                    - (0.01 * max(0, len(records) - correct_questions)),
                ),
            )

        return {
            "questions_attempted": questions_attempted,
            "correct_rate": correct_rate,
            "average_attempts": average_attempts,
            "average_response_time": average_response_time,
            "hints_used": hints_used,
            "current_mastery_level": current_mastery,
        }

    @staticmethod
    def build_result_summary(
        student_id: str,
        pretest_score: float | None,
        posttest_score: float | None,
        learning_gain: float | None,
        learning_records: list[dict[str, Any]] | None,
        estimated_mastery_level: float | None = None,
    ) -> dict[str, Any]:
        progress = ExperimentService.summarize_progress(
            learning_records=learning_records,
            estimated_mastery_level=estimated_mastery_level,
        )
        return {
            "student_id": student_id,
            "pretest_score": pretest_score,
            "posttest_score": posttest_score,
            "learning_gain": learning_gain,
            "total_hints_used": progress["hints_used"],
            "average_attempts": progress["average_attempts"],
            "average_response_time": progress["average_response_time"],
            "correct_rate": progress["correct_rate"],
        }

    @staticmethod
    def export_result_summary(
        summary: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> Path:
        path = Path(output_path) if output_path else PROJECT_ROOT / "outputs" / "logs" / "gui_result_summaries.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "student_id",
            "pretest_score",
            "posttest_score",
            "learning_gain",
            "total_hints_used",
            "average_attempts",
            "average_response_time",
            "correct_rate",
        ]
        write_header = not path.exists()

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: summary.get(key) for key in fieldnames})

        return path

    def get_pretest_questions(self) -> list[PretestQuestion]:
        return list(self._pretest_questions)

    def get_posttest_questions(self) -> list[PosttestQuestion]:
        return list(self._posttest_questions)

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return "".join(answer.lower().split())

    def score_pretest(self, answers_by_question_id: dict[str, str]) -> tuple[float, int, int]:
        questions = self.get_pretest_questions()
        return self._score_questions(questions, answers_by_question_id)

    def score_posttest(self, answers_by_question_id: dict[str, str]) -> tuple[float, int, int]:
        questions = self.get_posttest_questions()
        return self._score_questions(questions, answers_by_question_id)

    def _score_questions(
        self,
        questions: list[PretestQuestion] | list[PosttestQuestion],
        answers_by_question_id: dict[str, str],
    ) -> tuple[float, int, int]:
        correct_count = 0

        for question in questions:
            student_answer = self._normalize_answer(
                answers_by_question_id.get(question.question_id, "")
            )
            accepted_answers = {
                self._normalize_answer(answer)
                for answer in question.accepted_answers
            }
            if student_answer in accepted_answers:
                correct_count += 1

        total_questions = len(questions)
        score = correct_count / total_questions if total_questions else 0.0
        return score, correct_count, total_questions

    @staticmethod
    def estimate_initial_level(pretest_score: float) -> float:
        return max(0.0, min(1.0, float(pretest_score)))

    @staticmethod
    def assign_initial_difficulty(pretest_score: float) -> str:
        if pretest_score < 0.5:
            return "low"
        if pretest_score < 0.8:
            return "medium"
        return "high"

    def _try_load_latest_model(self) -> None:
        model_dir = PROJECT_ROOT / "outputs" / "models"
        model_files = sorted(model_dir.glob("double_dqn_*.pt"))
        if model_files:
            self.agent.load(model_files[-1])

    def _load_learning_questions(self) -> list[LearningQuestion]:
        questions = []
        for item in self.learning_environment.question_sequence:
            question = item["question"]
            questions.append(
                LearningQuestion(
                    kp_id=item["kp_id"],
                    question_id=question["question_id"],
                    difficulty=item["difficulty"],
                    prompt=question["question"],
                    answer=question["answer"],
                )
            )
        return questions

    def start_learning_session(self, initial_difficulty: str | None = None) -> LearningQuestion:
        self.current_question_index = self._initial_question_index(initial_difficulty)
        self.current_attempts = 0
        self.learning_step = 0
        self.question_started_at = time.monotonic()
        self.last_hint_action = None
        self.last_hint_level = None
        self.last_hint_text = None
        return self.get_current_learning_question()

    def _initial_question_index(self, initial_difficulty: str | None) -> int:
        if initial_difficulty is None:
            return 0
        for index, question in enumerate(self.learning_questions):
            if question.difficulty == initial_difficulty:
                return index
        return 0

    def get_current_learning_question(self) -> LearningQuestion:
        if not self.learning_questions:
            raise ValueError("No learning questions are available.")
        return self.learning_questions[self.current_question_index % len(self.learning_questions)]

    def move_to_next_learning_question(self) -> LearningQuestion:
        self.current_question_index = (self.current_question_index + 1) % len(
            self.learning_questions
        )
        self.current_attempts = 0
        self.question_started_at = time.monotonic()
        self.last_hint_action = None
        self.last_hint_level = None
        self.last_hint_text = None
        return self.get_current_learning_question()

    def check_learning_answer(
        self,
        student_answer: str,
        mastery_level: float,
        recent_accuracy: float,
    ) -> dict[str, Any]:
        question = self.get_current_learning_question()
        self.current_attempts += 1
        self.learning_step += 1
        response_time = time.monotonic() - self.question_started_at
        is_correct = self._answers_match(student_answer, question.answer)
        state = self._build_learning_state(
            mastery_level=mastery_level,
            recent_accuracy=recent_accuracy,
            response_time=response_time,
            attempts=self.current_attempts,
            question=question,
        )

        if is_correct:
            reward = self._estimate_logged_reward(
                correct=True,
                correct_after_hint=self.last_hint_level is not None,
                attempts=self.current_attempts,
                response_time=response_time,
                hint_action=self.last_hint_action,
                mastery_level=mastery_level,
            )
            return {
                "correct": True,
                "message": "Correct. You can move to the next question.",
                "question": question,
                "attempts": self.current_attempts,
                "response_time": response_time,
                "hint_action": self.last_hint_action,
                "hint_level": self.last_hint_level,
                "hint_text": self.last_hint_text,
                "correct_after_hint": self.last_hint_level is not None,
                "hint_generated": False,
                "state": state,
                "epsilon": None,
                "reward": reward,
                "step": self.learning_step,
            }

        hint_result = self.recommend_hint(
            mastery_level=mastery_level,
            recent_accuracy=recent_accuracy,
            response_time=response_time,
            attempts=self.current_attempts,
            question=question,
            state=state,
        )
        reward = self._estimate_logged_reward(
            correct=False,
            correct_after_hint=False,
            attempts=self.current_attempts,
            response_time=response_time,
            hint_action=hint_result["action"],
            mastery_level=mastery_level,
        )
        return {
            "correct": False,
            "message": "Incorrect. Review the recommended hint and try again.",
            "question": question,
            "attempts": self.current_attempts,
            "response_time": response_time,
            "hint_action": hint_result["action"],
            "hint_level": hint_result["hint_level"],
            "hint_text": hint_result["hint_text"],
            "correct_after_hint": False,
            "hint_generated": True,
            "state": state,
            "epsilon": hint_result["epsilon"],
            "reward": reward,
            "step": self.learning_step,
        }

    def recommend_hint(
        self,
        mastery_level: float,
        recent_accuracy: float,
        response_time: float,
        attempts: int,
        question: LearningQuestion | None = None,
        state: Any | None = None,
    ) -> dict[str, Any]:
        question = question or self.get_current_learning_question()
        state = state if state is not None else self._build_learning_state(
            mastery_level=mastery_level,
            recent_accuracy=recent_accuracy,
            response_time=response_time,
            attempts=attempts,
            question=question,
        )
        action = self.agent.select_action(state)
        hint_level = ACTION_TO_HINT_LEVEL[action]
        hint_text = KnowledgeLoader.get_hint(question.question_id, hint_level)

        self.last_hint_action = action
        self.last_hint_level = hint_level
        self.last_hint_text = hint_text

        return {
            "action": action,
            "hint_level": hint_level,
            "hint_text": hint_text,
            "epsilon": self.agent.epsilon_history[-1] if self.agent.epsilon_history else None,
        }

    def log_learning_interaction(
        self,
        student_id: str,
        result: dict[str, Any],
        mastery_level: float,
        episode: int = 1,
    ) -> None:
        question = result["question"]
        action = result["hint_action"] if result["hint_action"] is not None else -1
        hint_level = result["hint_level"] if result["hint_level"] is not None else "none"

        self.db_manager.log_interaction(
            student_id=student_id,
            episode=episode,
            step=int(result.get("step", self.learning_step)),
            kp_id=question.kp_id,
            question_id=question.question_id,
            difficulty=question.difficulty,
            state=result.get("state", []),
            action=action,
            hint_level=hint_level,
            reward=float(result.get("reward", 0.0)),
            correct=bool(result["correct"]),
            response_time=float(result["response_time"]),
            attempts=int(result["attempts"]),
            mastery_level=float(mastery_level),
            epsilon=result.get("epsilon"),
        )

        if result.get("hint_generated") and result["hint_action"] is not None:
            self.db_manager.log_hint_usage(
                student_id=student_id,
                kp_id=question.kp_id,
                question_id=question.question_id,
                action=int(result["hint_action"]),
                hint_level=str(result["hint_level"]),
                used=True,
            )

    def close_database(self) -> None:
        self.db_manager.close()

    def _build_learning_state(
        self,
        mastery_level: float,
        recent_accuracy: float,
        response_time: float,
        attempts: int,
        question: LearningQuestion,
    ) -> Any:
        return build_state(
            mastery_level=mastery_level,
            question_difficulty=question.difficulty,
            recent_accuracy=recent_accuracy,
            response_time=response_time,
            attempts=attempts,
            knowledge_point_index=question.kp_id,
        )

    @staticmethod
    def _estimate_logged_reward(
        correct: bool,
        correct_after_hint: bool,
        attempts: int,
        response_time: float,
        hint_action: int | None,
        mastery_level: float,
    ) -> float:
        reward = 1.0 if correct else -0.5
        if correct_after_hint:
            reward += 0.3
        reward -= 0.1 * max(0, attempts - 1)
        reward += 0.2 if response_time <= 30.0 else -0.2
        if hint_action is not None:
            reward -= 0.05
        if hint_action == 0 and mastery_level >= 0.8:
            reward -= 0.1
        return reward

    def _answers_match(self, student_answer: str, correct_answer: str) -> bool:
        return self._normalize_learning_answer(student_answer) == self._normalize_learning_answer(
            correct_answer
        )

    @staticmethod
    def _normalize_learning_answer(answer: str) -> str:
        text = answer.lower().strip()
        text = text.replace(" ", "")
        text = text.replace("或", "or")
        text = text.replace("，", ",")
        return text
