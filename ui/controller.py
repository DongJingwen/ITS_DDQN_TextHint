"""GUI controller for page-level state.

Business logic will be connected in later GUI steps. For now, this object stores
only the current student identity entered on the login page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from services.experiment_service import (
    ExperimentService,
    LearningQuestion,
    PosttestQuestion,
    PretestQuestion,
)
from services.session_manager import SessionManager, StudentSession


@dataclass
class GUIController:
    session_manager: SessionManager
    experiment_service: ExperimentService
    student_id: str | None = None
    student_name: str | None = None

    def __init__(
        self,
        session_manager: SessionManager | None = None,
        experiment_service: ExperimentService | None = None,
    ) -> None:
        self.session_manager = session_manager or SessionManager()
        self.experiment_service = experiment_service or ExperimentService()
        self.student_id = None
        self.student_name = None

    def login_student(
        self,
        student_id: str,
        student_name: str | None = None,
    ) -> StudentSession:
        session = self.session_manager.start_student_session(student_id, student_name)
        self.student_id = session.student_id
        self.student_name = session.student_name
        self.experiment_service.register_student(
            student_id=session.student_id,
            student_name=session.student_name,
        )
        return session

    def get_pretest_questions(self) -> list[PretestQuestion]:
        return self.experiment_service.get_pretest_questions()

    def get_posttest_questions(self) -> list[PosttestQuestion]:
        return self.experiment_service.get_posttest_questions()

    def submit_pretest(self, answers_by_question_id: dict[str, str]) -> dict[str, object]:
        score, correct_count, total_questions = self.experiment_service.score_pretest(
            answers_by_question_id
        )
        estimated_level = self.experiment_service.estimate_initial_level(score)
        initial_difficulty = self.experiment_service.assign_initial_difficulty(score)
        self.session_manager.save_pretest_result(
            pretest_score=score,
            estimated_level=estimated_level,
            initial_difficulty=initial_difficulty,
        )
        session = self.session_manager.current_student
        if session is not None:
            self.experiment_service.save_pretest_result(
                student_id=session.student_id,
                score=score,
                estimated_level=estimated_level,
                initial_difficulty=initial_difficulty,
            )

        return {
            "score": score,
            "correct_count": correct_count,
            "total_questions": total_questions,
            "estimated_level": estimated_level,
            "initial_difficulty": initial_difficulty,
        }

    def start_learning_session(self) -> LearningQuestion:
        session = self.session_manager.current_student
        initial_difficulty = session.initial_difficulty if session else None
        return self.experiment_service.start_learning_session(initial_difficulty)

    def get_current_learning_question(self) -> LearningQuestion:
        return self.experiment_service.get_current_learning_question()

    def submit_learning_answer(self, answer: str) -> dict[str, object]:
        session = self.session_manager.current_student
        if session is None:
            raise ValueError("No active student session.")

        mastery_level = session.estimated_level if session.estimated_level is not None else 0.5
        recent_accuracy = self.session_manager.get_recent_accuracy()
        result = self.experiment_service.check_learning_answer(
            student_answer=answer,
            mastery_level=mastery_level,
            recent_accuracy=recent_accuracy,
        )

        question = result["question"]
        self.session_manager.save_learning_record(
            {
                "student_id": session.student_id,
                "kp_id": question.kp_id,
                "question_id": question.question_id,
                "difficulty": question.difficulty,
                "answer": answer,
                "correct": result["correct"],
                "attempts": result["attempts"],
                "response_time": result["response_time"],
                "hint_action": result["hint_action"],
                "hint_level": result["hint_level"],
                "hint_generated": result.get("hint_generated", False),
                "correct_after_hint": result["correct_after_hint"],
                "reward": result.get("reward", 0.0),
            }
        )
        self.experiment_service.log_learning_interaction(
            student_id=session.student_id,
            result=result,
            mastery_level=mastery_level,
        )
        return result

    def move_to_next_learning_question(self) -> LearningQuestion:
        return self.experiment_service.move_to_next_learning_question()

    def get_progress_summary(self) -> dict[str, object]:
        session = self.session_manager.current_student
        if session is None:
            return {
                "questions_attempted": 0,
                "correct_rate": 0.0,
            "average_attempts": 0.0,
            "average_response_time": 0.0,
            "hints_used": 0,
            "current_mastery_level": None,
        }

        return self.experiment_service.summarize_progress(
            learning_records=session.learning_records,
            estimated_mastery_level=session.estimated_level,
        )

    def submit_posttest(self, posttest_score: float) -> dict[str, float]:
        session = self.session_manager.current_student
        if session is None:
            raise ValueError("No active student session.")

        pretest_score = session.pretest_score if session.pretest_score is not None else 0.0
        learning_gain = posttest_score - pretest_score
        setattr(session, "posttest_score", posttest_score)
        setattr(session, "learning_gain", learning_gain)
        self.experiment_service.save_posttest_result(
            student_id=session.student_id,
            posttest_score=posttest_score,
            learning_gain=learning_gain,
        )
        return {
            "posttest_score": posttest_score,
            "learning_gain": learning_gain,
        }

    def submit_posttest_answers(
        self,
        answers_by_question_id: dict[str, str],
    ) -> dict[str, object]:
        score, correct_count, total_questions = self.experiment_service.score_posttest(
            answers_by_question_id
        )
        result = self.submit_posttest(score)
        return {
            "score": score,
            "correct_count": correct_count,
            "total_questions": total_questions,
            "learning_gain": result["learning_gain"],
        }

    def get_result_summary(self) -> dict[str, object]:
        session = self.session_manager.current_student
        if session is None:
            return {
                "student_id": "-",
                "pretest_score": None,
                "posttest_score": None,
                "learning_gain": None,
                "total_hints_used": 0,
                "average_attempts": 0.0,
                "average_response_time": 0.0,
                "correct_rate": 0.0,
            }

        return self.experiment_service.build_result_summary(
            student_id=session.student_id,
            pretest_score=session.pretest_score,
            posttest_score=getattr(session, "posttest_score", None),
            learning_gain=getattr(session, "learning_gain", None),
            learning_records=session.learning_records,
            estimated_mastery_level=session.estimated_level,
        )

    def export_result_summary(self) -> Path:
        summary = self.get_result_summary()
        return self.experiment_service.export_result_summary(summary)

    def clear_current_session(self) -> None:
        self.session_manager.current_student = None
        self.student_id = None
        self.student_name = None
