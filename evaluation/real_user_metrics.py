"""Compute real-user evaluation metrics from optional SQLite logs."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_user_experiment.db"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_user_metrics.csv"

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on local environment
    pd = None


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _fetch_dicts(db_path: str | Path, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _safe_mean(values: list[float | int | None]) -> float | None:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    return mean(clean_values)


def _latest_by_student(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        student_id = row.get("student_id")
        if student_id is None:
            continue
        if student_id not in latest or int(row.get("id", 0)) > int(latest[student_id].get("id", 0)):
            latest[student_id] = row
    return latest


def _table_exists(db_path: str | Path, table_name: str) -> bool:
    rows = _fetch_dicts(
        db_path,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return bool(rows)


def compute_learning_gain(db_path: str | Path) -> dict[str, Any]:
    """Return aggregate and per-student pre/post learning gain metrics."""
    if not _table_exists(db_path, "pretest_results") or not _table_exists(db_path, "posttest_results"):
        return {
            "mean_pretest_score": None,
            "mean_posttest_score": None,
            "mean_learning_gain": None,
            "per_student_learning_gain": [],
        }

    pretests = _latest_by_student(
        _fetch_dicts(db_path, "SELECT * FROM pretest_results ORDER BY id")
    )
    posttests = _latest_by_student(
        _fetch_dicts(db_path, "SELECT * FROM posttest_results ORDER BY id")
    )

    summaries = []
    for student_id in sorted(set(pretests) | set(posttests)):
        pretest_score = pretests.get(student_id, {}).get("score")
        posttest_score = posttests.get(student_id, {}).get("score")
        stored_gain = posttests.get(student_id, {}).get("learning_gain")
        learning_gain = stored_gain
        if learning_gain is None and pretest_score is not None and posttest_score is not None:
            learning_gain = float(posttest_score) - float(pretest_score)

        summaries.append(
            {
                "student_id": student_id,
                "pretest_score": pretest_score,
                "posttest_score": posttest_score,
                "learning_gain": learning_gain,
            }
        )

    return {
        "mean_pretest_score": _safe_mean([row["pretest_score"] for row in summaries]),
        "mean_posttest_score": _safe_mean([row["posttest_score"] for row in summaries]),
        "mean_learning_gain": _safe_mean([row["learning_gain"] for row in summaries]),
        "per_student_learning_gain": summaries,
    }


def compute_efficiency_metrics(db_path: str | Path) -> dict[str, Any]:
    """Return response-time, attempt, hint-count, and time-to-mastery metrics."""
    if not _table_exists(db_path, "interaction_logs"):
        return {
            "average_response_time": None,
            "average_attempts": None,
            "average_number_of_hints_used": None,
            "time_to_mastery": None,
        }

    interactions = _fetch_dicts(db_path, "SELECT * FROM interaction_logs ORDER BY id")
    hint_rows = (
        _fetch_dicts(db_path, "SELECT * FROM hint_usage_logs ORDER BY id")
        if _table_exists(db_path, "hint_usage_logs")
        else []
    )

    students = sorted({row["student_id"] for row in interactions if row.get("student_id")})
    hint_counts_by_student = []
    for student_id in students:
        hint_counts_by_student.append(
            sum(1 for row in hint_rows if row.get("student_id") == student_id and row.get("used") == 1)
        )

    time_to_mastery_values = []
    for student_id in students:
        student_rows = [row for row in interactions if row.get("student_id") == student_id]
        for row in student_rows:
            mastery_level = row.get("mastery_level")
            if mastery_level is not None and float(mastery_level) >= 0.8:
                time_to_mastery_values.append(row.get("step"))
                break

    return {
        "average_response_time": _safe_mean([row.get("response_time") for row in interactions]),
        "average_attempts": _safe_mean([row.get("attempts") for row in interactions]),
        "average_number_of_hints_used": _safe_mean(hint_counts_by_student),
        "time_to_mastery": _safe_mean(time_to_mastery_values),
    }


def compute_hint_effectiveness(db_path: str | Path) -> dict[str, Any]:
    """Return hint-level correctness, reward, and usage metrics."""
    if not _table_exists(db_path, "interaction_logs"):
        return {
            "correct_after_hint_rate": None,
            "hint_usage_distribution": {},
            "average_reward_by_hint_level": {},
            "correctness_rate_by_hint_level": {},
        }

    interactions = _fetch_dicts(db_path, "SELECT * FROM interaction_logs ORDER BY id")
    if not interactions:
        return {
            "correct_after_hint_rate": None,
            "hint_usage_distribution": {},
            "average_reward_by_hint_level": {},
            "correctness_rate_by_hint_level": {},
        }

    hint_levels = ["level_1", "level_2", "level_3"]
    usage_distribution = {
        level: sum(1 for row in interactions if row.get("hint_level") == level)
        for level in hint_levels
    }
    average_reward_by_hint_level = {}
    correctness_rate_by_hint_level = {}

    for level in hint_levels:
        level_rows = [row for row in interactions if row.get("hint_level") == level]
        average_reward_by_hint_level[level] = _safe_mean([row.get("reward") for row in level_rows])
        correctness_rate_by_hint_level[level] = _safe_mean([row.get("correct") for row in level_rows])

    return {
        "correct_after_hint_rate": _safe_mean([row.get("correct") for row in interactions]),
        "hint_usage_distribution": usage_distribution,
        "average_reward_by_hint_level": average_reward_by_hint_level,
        "correctness_rate_by_hint_level": correctness_rate_by_hint_level,
    }


def compute_student_level_summary(db_path: str | Path) -> list[dict[str, Any]]:
    """Return per-student current level and interaction metrics."""
    if not _table_exists(db_path, "students"):
        return []

    students = _fetch_dicts(db_path, "SELECT * FROM students ORDER BY student_id")
    interactions = (
        _fetch_dicts(db_path, "SELECT * FROM interaction_logs ORDER BY id")
        if _table_exists(db_path, "interaction_logs")
        else []
    )

    summaries = []
    for student in students:
        student_id = student["student_id"]
        student_rows = [row for row in interactions if row.get("student_id") == student_id]
        summaries.append(
            {
                "student_id": student_id,
                "current_level": student.get("current_level"),
                "correct_rate": _safe_mean([row.get("correct") for row in student_rows]),
                "average_reward": _safe_mean([row.get("reward") for row in student_rows]),
                "average_response_time": _safe_mean([row.get("response_time") for row in student_rows]),
                "average_attempts": _safe_mean([row.get("attempts") for row in student_rows]),
            }
        )

    return summaries


def save_real_user_metrics(
    db_path: str | Path,
    output_csv_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Compute and save one CSV combining student-level and aggregate metrics."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    learning_gain = compute_learning_gain(db_path)
    efficiency = compute_efficiency_metrics(db_path)
    hint_effectiveness = compute_hint_effectiveness(db_path)
    student_summaries = compute_student_level_summary(db_path)

    fieldnames = [
        "student_id",
        "current_level",
        "pretest_score",
        "posttest_score",
        "learning_gain",
        "correct_rate",
        "average_reward",
        "average_response_time",
        "average_attempts",
        "mean_pretest_score",
        "mean_posttest_score",
        "mean_learning_gain",
        "overall_average_response_time",
        "overall_average_attempts",
        "average_number_of_hints_used",
        "time_to_mastery",
        "correct_after_hint_rate",
        "level_1_hint_count",
        "level_2_hint_count",
        "level_3_hint_count",
        "level_1_average_reward",
        "level_2_average_reward",
        "level_3_average_reward",
        "level_1_correctness_rate",
        "level_2_correctness_rate",
        "level_3_correctness_rate",
    ]

    gain_by_student = {
        row["student_id"]: row for row in learning_gain["per_student_learning_gain"]
    }

    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        if not student_summaries:
            writer.writerow(_aggregate_metrics_row(learning_gain, efficiency, hint_effectiveness))
        else:
            for student_summary in student_summaries:
                student_id = student_summary["student_id"]
                row = _aggregate_metrics_row(learning_gain, efficiency, hint_effectiveness)
                row.update(student_summary)
                row.update(gain_by_student.get(student_id, {}))
                writer.writerow(row)

    return output_csv_path


def _aggregate_metrics_row(
    learning_gain: dict[str, Any],
    efficiency: dict[str, Any],
    hint_effectiveness: dict[str, Any],
) -> dict[str, Any]:
    usage = hint_effectiveness.get("hint_usage_distribution", {})
    reward_by_level = hint_effectiveness.get("average_reward_by_hint_level", {})
    correct_by_level = hint_effectiveness.get("correctness_rate_by_hint_level", {})

    return {
        "mean_pretest_score": learning_gain.get("mean_pretest_score"),
        "mean_posttest_score": learning_gain.get("mean_posttest_score"),
        "mean_learning_gain": learning_gain.get("mean_learning_gain"),
        "overall_average_response_time": efficiency.get("average_response_time"),
        "overall_average_attempts": efficiency.get("average_attempts"),
        "average_number_of_hints_used": efficiency.get("average_number_of_hints_used"),
        "time_to_mastery": efficiency.get("time_to_mastery"),
        "correct_after_hint_rate": hint_effectiveness.get("correct_after_hint_rate"),
        "level_1_hint_count": usage.get("level_1", 0),
        "level_2_hint_count": usage.get("level_2", 0),
        "level_3_hint_count": usage.get("level_3", 0),
        "level_1_average_reward": reward_by_level.get("level_1"),
        "level_2_average_reward": reward_by_level.get("level_2"),
        "level_3_average_reward": reward_by_level.get("level_3"),
        "level_1_correctness_rate": correct_by_level.get("level_1"),
        "level_2_correctness_rate": correct_by_level.get("level_2"),
        "level_3_correctness_rate": correct_by_level.get("level_3"),
    }


def _default_db_path() -> Path:
    demo_db = PROJECT_ROOT / "outputs" / "logs" / "real_user_session_demo.db"
    if demo_db.exists():
        return demo_db
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH
    db_files = sorted((PROJECT_ROOT / "outputs" / "logs").glob("*.db"))
    if not db_files:
        raise FileNotFoundError("No SQLite database found under outputs/logs.")
    return db_files[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute real-user metrics from SQLite logs.")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_path = args.db_path or _default_db_path()
    output_path = save_real_user_metrics(db_path, args.output)

    print(f"Database: {db_path}")
    print(f"Learning gain: {compute_learning_gain(db_path)}")
    print(f"Efficiency: {compute_efficiency_metrics(db_path)}")
    print(f"Hint effectiveness: {compute_hint_effectiveness(db_path)}")
    print(f"Saved real-user metrics: {output_path}")
