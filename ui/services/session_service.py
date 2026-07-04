"""Thin service layer between the desktop UI and existing backend modules."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from database.db_manager import DatabaseManager


DEFAULT_DB_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_user_experiment.db"


@dataclass
class StudentProfile:
    student_id: str
    name: str | None = None


class UISessionService:
    """Keep UI state separate from RL/environment implementation details."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.initialize_database()
        self.student_profile: StudentProfile | None = None

    def login_student(self, student_id: str, name: str | None = None) -> StudentProfile:
        student_id = student_id.strip()
        name = name.strip() if name else None
        if not student_id:
            raise ValueError("Student ID is required.")

        self.db_manager.add_student(
            student_id=student_id,
            name=name,
            group_type="real",
            initial_level=None,
        )
        self.student_profile = StudentProfile(student_id=student_id, name=name)
        return self.student_profile

    def close(self) -> None:
        self.db_manager.close()

