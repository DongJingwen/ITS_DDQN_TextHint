"""Session state manager for the desktop GUI extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StudentSession:
    student_id: str
    student_name: str | None = None
    pretest_score: float | None = None
    estimated_level: float | None = None
    initial_difficulty: str | None = None
    learning_records: list[dict[str, Any]] | None = None


class SessionManager:
    """Store lightweight GUI session state without touching backend logic."""

    def __init__(self) -> None:
        self.current_student: StudentSession | None = None

    def start_student_session(
        self,
        student_id: str,
        student_name: str | None = None,
    ) -> StudentSession:
        clean_student_id = student_id.strip()
        clean_student_name = student_name.strip() if student_name else None

        if not clean_student_id:
            raise ValueError("Student ID is required.")

        self.current_student = StudentSession(
            student_id=clean_student_id,
            student_name=clean_student_name,
            learning_records=[],
        )
        return self.current_student

    def has_active_student(self) -> bool:
        return self.current_student is not None

    def save_pretest_result(
        self,
        pretest_score: float,
        estimated_level: float,
        initial_difficulty: str,
    ) -> StudentSession:
        if self.current_student is None:
            raise ValueError("No active student session.")

        self.current_student.pretest_score = pretest_score
        self.current_student.estimated_level = estimated_level
        self.current_student.initial_difficulty = initial_difficulty
        return self.current_student

    def save_learning_record(self, record: dict[str, Any]) -> StudentSession:
        if self.current_student is None:
            raise ValueError("No active student session.")
        if self.current_student.learning_records is None:
            self.current_student.learning_records = []

        self.current_student.learning_records.append(record)
        return self.current_student

    def get_recent_accuracy(self, window_size: int = 3) -> float:
        if self.current_student is None or not self.current_student.learning_records:
            return 0.0

        recent_records = self.current_student.learning_records[-window_size:]
        correct_values = [1.0 if record.get("correct") else 0.0 for record in recent_records]
        return sum(correct_values) / len(correct_values)
