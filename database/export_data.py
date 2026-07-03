"""Export optional SQLite experiment data to CSV files."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "outputs" / "logs" / "exports"
DEFAULT_DB_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_user_experiment.db"

EXPORT_TABLES = [
    "students",
    "pretest_results",
    "posttest_results",
    "interaction_logs",
    "hint_usage_logs",
]


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _write_rows_to_csv(rows: list[sqlite3.Row], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        if not rows:
            writer = csv.writer(file)
            writer.writerow([])
            return

        fieldnames = rows[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def export_table_to_csv(
    db_path: str | Path,
    table_name: str,
    output_csv_path: str | Path,
) -> Path:
    """Export one SQLite table to a CSV file."""
    if table_name not in EXPORT_TABLES:
        raise ValueError(f"Unsupported table '{table_name}'. Expected one of {EXPORT_TABLES}.")

    output_csv_path = Path(output_csv_path)
    with _connect(db_path) as connection:
        rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()

    _write_rows_to_csv(rows, output_csv_path)
    return output_csv_path


def export_all_tables(db_path: str | Path, output_dir: str | Path = DEFAULT_EXPORT_DIR) -> list[Path]:
    """Export all experiment tables to CSV files."""
    output_dir = Path(output_dir)
    exported_paths = []

    for table_name in EXPORT_TABLES:
        output_csv_path = output_dir / f"{table_name}.csv"
        exported_paths.append(export_table_to_csv(db_path, table_name, output_csv_path))

    return exported_paths


def export_student_summary(
    db_path: str | Path,
    output_csv_path: str | Path = DEFAULT_EXPORT_DIR / "student_summary.csv",
) -> Path:
    """Export per-student summary metrics to CSV."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            s.student_id AS student_id,
            pre.score AS pretest_score,
            post.score AS posttest_score,
            post.learning_gain AS learning_gain,
            COUNT(i.id) AS total_interactions,
            AVG(i.reward) AS average_reward,
            AVG(i.correct) AS correct_rate,
            AVG(i.response_time) AS average_response_time,
            AVG(i.attempts) AS average_attempts,
            SUM(CASE WHEN i.hint_level = 'level_1' THEN 1 ELSE 0 END) AS level_1_hint_count,
            SUM(CASE WHEN i.hint_level = 'level_2' THEN 1 ELSE 0 END) AS level_2_hint_count,
            SUM(CASE WHEN i.hint_level = 'level_3' THEN 1 ELSE 0 END) AS level_3_hint_count
        FROM students s
        LEFT JOIN (
            SELECT p1.*
            FROM pretest_results p1
            INNER JOIN (
                SELECT student_id, MAX(id) AS max_id
                FROM pretest_results
                GROUP BY student_id
            ) latest_pre
                ON p1.student_id = latest_pre.student_id
                AND p1.id = latest_pre.max_id
        ) pre ON s.student_id = pre.student_id
        LEFT JOIN (
            SELECT p2.*
            FROM posttest_results p2
            INNER JOIN (
                SELECT student_id, MAX(id) AS max_id
                FROM posttest_results
                GROUP BY student_id
            ) latest_post
                ON p2.student_id = latest_post.student_id
                AND p2.id = latest_post.max_id
        ) post ON s.student_id = post.student_id
        LEFT JOIN interaction_logs i
            ON s.student_id = i.student_id
        GROUP BY
            s.student_id,
            pre.score,
            post.score,
            post.learning_gain
        ORDER BY s.student_id
    """

    with _connect(db_path) as connection:
        rows = connection.execute(query).fetchall()

    fieldnames = [
        "student_id",
        "pretest_score",
        "posttest_score",
        "learning_gain",
        "total_interactions",
        "average_reward",
        "correct_rate",
        "average_response_time",
        "average_attempts",
        "level_1_hint_count",
        "level_2_hint_count",
        "level_3_hint_count",
    ]

    with output_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_clean_summary_row(dict(row)))

    return output_csv_path


def _clean_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    for count_field in (
        "total_interactions",
        "level_1_hint_count",
        "level_2_hint_count",
        "level_3_hint_count",
    ):
        row[count_field] = int(row[count_field] or 0)

    return row


def _default_db_path() -> Path:
    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH

    db_files = sorted((PROJECT_ROOT / "outputs" / "logs").glob("*.db"))
    if not db_files:
        raise FileNotFoundError(
            "No SQLite database found under outputs/logs. Run main.py init-db or a real-user demo first."
        )
    return db_files[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SQLite experiment data to CSV.")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_path = args.db_path or _default_db_path()

    exported_table_paths = export_all_tables(db_path, args.output_dir)
    summary_path = export_student_summary(db_path, args.output_dir / "student_summary.csv")

    print(f"Database: {db_path}")
    for path in exported_table_paths:
        print(f"Exported table: {path}")
    print(f"Exported student summary: {summary_path}")
