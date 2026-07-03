"""Optional SQLite manager for later real-user experiment logging."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SCHEMA_PATH = BASE_DIR / "schema.sql"


class DatabaseManager:
    """Small SQLite wrapper for real-user ITS experiment data."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    @staticmethod
    def _state_to_json(state: Any) -> str:
        if isinstance(state, str):
            return state
        if hasattr(state, "tolist"):
            state = state.tolist()
        return json.dumps(state)

    def initialize_database(self) -> None:
        """Create all tables from schema.sql."""
        with SCHEMA_PATH.open("r", encoding="utf-8") as file:
            schema_sql = file.read()
        self.connection.executescript(schema_sql)
        self.connection.commit()

    def add_student(
        self,
        student_id: str,
        name: str | None = None,
        group_type: str = "real",
        initial_level: float | None = None,
    ) -> None:
        """Insert or update a student profile."""
        current_level = initial_level
        existing_student = self.get_student(student_id)

        if existing_student is None:
            self.connection.execute(
                """
                INSERT INTO students (
                    student_id, name, group_type, created_at, initial_level, current_level
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    name,
                    group_type,
                    self._timestamp(),
                    initial_level,
                    current_level,
                ),
            )
        else:
            self.connection.execute(
                """
                UPDATE students
                SET name = ?, group_type = ?, initial_level = ?, current_level = ?
                WHERE student_id = ?
                """,
                (name, group_type, initial_level, current_level, student_id),
            )
        self.connection.commit()

    def save_pretest_result(
        self,
        student_id: str,
        score: float,
        estimated_level: float,
        initial_difficulty: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO pretest_results (
                student_id, score, estimated_level, initial_difficulty, timestamp
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, score, estimated_level, initial_difficulty, self._timestamp()),
        )
        self.connection.execute(
            """
            UPDATE students
            SET initial_level = ?, current_level = ?
            WHERE student_id = ?
            """,
            (estimated_level, estimated_level, student_id),
        )
        self.connection.commit()

    def save_posttest_result(
        self,
        student_id: str,
        score: float,
        learning_gain: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO posttest_results (
                student_id, score, learning_gain, timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (student_id, score, learning_gain, self._timestamp()),
        )
        self.connection.commit()

    def log_interaction(
        self,
        student_id: str,
        episode: int,
        step: int,
        kp_id: str,
        question_id: str,
        difficulty: str,
        state: Any,
        action: int,
        hint_level: str,
        reward: float,
        correct: bool | int,
        response_time: float,
        attempts: int,
        mastery_level: float,
        epsilon: float | None = None,
    ) -> None:
        """Save one student-question interaction."""
        self.connection.execute(
            """
            INSERT INTO interaction_logs (
                student_id, episode, step, kp_id, question_id, difficulty, state,
                action, hint_level, reward, correct, response_time, attempts,
                mastery_level, epsilon, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                episode,
                step,
                kp_id,
                question_id,
                difficulty,
                self._state_to_json(state),
                action,
                hint_level,
                reward,
                int(bool(correct)),
                response_time,
                attempts,
                mastery_level,
                epsilon,
                self._timestamp(),
            ),
        )
        self.connection.execute(
            """
            UPDATE students
            SET current_level = ?
            WHERE student_id = ?
            """,
            (mastery_level, student_id),
        )
        self.connection.commit()

    def log_hint_usage(
        self,
        student_id: str,
        kp_id: str,
        question_id: str,
        action: int,
        hint_level: str,
        used: bool | int = True,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO hint_usage_logs (
                student_id, kp_id, question_id, action, hint_level, used, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                kp_id,
                question_id,
                action,
                hint_level,
                int(bool(used)),
                self._timestamp(),
            ),
        )
        self.connection.commit()

    def get_student(self, student_id: str) -> dict[str, Any] | None:
        cursor = self.connection.execute(
            "SELECT * FROM students WHERE student_id = ?",
            (student_id,),
        )
        return self._row_to_dict(cursor.fetchone())

    def get_interactions(self, student_id: str | None = None) -> list[dict[str, Any]]:
        if student_id is None:
            cursor = self.connection.execute(
                "SELECT * FROM interaction_logs ORDER BY id"
            )
        else:
            cursor = self.connection.execute(
                "SELECT * FROM interaction_logs WHERE student_id = ? ORDER BY id",
                (student_id,),
            )
        return self._rows_to_dicts(cursor.fetchall())

    def close(self) -> None:
        self.connection.close()


if __name__ == "__main__":
    test_db_path = PROJECT_ROOT / "outputs" / "logs" / "test_experiment.db"
    db = DatabaseManager(test_db_path)
    db.initialize_database()
    db.add_student("S001", name="Test Student", group_type="real", initial_level=0.4)
    db.save_pretest_result(
        student_id="S001",
        score=65.0,
        estimated_level=0.55,
        initial_difficulty="medium",
    )
    db.log_interaction(
        student_id="S001",
        episode=1,
        step=1,
        kp_id="KP1",
        question_id="KP1_L_Q1",
        difficulty="low",
        state=[0.55, 0.0, 0.0, 0.2, 0.33, 0.0],
        action=1,
        hint_level="level_2",
        reward=1.2,
        correct=True,
        response_time=24.5,
        attempts=1,
        mastery_level=0.58,
        epsilon=0.31,
    )
    db.log_hint_usage(
        student_id="S001",
        kp_id="KP1",
        question_id="KP1_L_Q1",
        action=1,
        hint_level="level_2",
        used=True,
    )

    print(db.get_student("S001"))
    print(db.get_interactions("S001"))
    db.close()
