"""Collect key outputs for a PhD supervisor meeting package."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PACKAGE_DIR = OUTPUTS_DIR / "meeting_package"


KEY_FILES = [
    PROJECT_ROOT / "README_PROJECT_DESIGN.md",
    OUTPUTS_DIR / "logs" / "final_result_summary.csv",
    OUTPUTS_DIR / "logs" / "final_result_summary.md",
    OUTPUTS_DIR / "figures" / "reward_curve.png",
    OUTPUTS_DIR / "figures" / "loss_curve.png",
    OUTPUTS_DIR / "figures" / "dynamic_epsilon_curve.png",
    OUTPUTS_DIR / "figures" / "average_attempts_curve.png",
    OUTPUTS_DIR / "figures" / "average_response_time_curve.png",
    OUTPUTS_DIR / "figures" / "hint_action_distribution.png",
]

KEY_DIRECTORIES = [
    OUTPUTS_DIR / "figures" / "baseline",
    OUTPUTS_DIR / "figures" / "ablation",
]


def _copy_file(source: Path, destination_dir: Path) -> Path | None:
    if not source.exists():
        print(f"Skipping missing file: {source}")
        return None

    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def _copy_directory_files(source_dir: Path, destination_dir: Path) -> list[Path]:
    copied_files = []
    if not source_dir.exists():
        print(f"Skipping missing directory: {source_dir}")
        return copied_files

    target_subdir = destination_dir / source_dir.name
    target_subdir.mkdir(parents=True, exist_ok=True)

    for source in sorted(source_dir.glob("*")):
        if source.is_file():
            destination = target_subdir / source.name
            shutil.copy2(source, destination)
            copied_files.append(destination)

    return copied_files


def prepare_meeting_package() -> list[Path]:
    """Create outputs/meeting_package and copy available meeting artifacts."""
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    copied_files = []

    for source in KEY_FILES:
        copied = _copy_file(source, PACKAGE_DIR)
        if copied is not None:
            copied_files.append(copied)

    for source_dir in KEY_DIRECTORIES:
        copied_files.extend(_copy_directory_files(source_dir, PACKAGE_DIR))

    return copied_files


if __name__ == "__main__":
    copied = prepare_meeting_package()

    print(f"Meeting package directory: {PACKAGE_DIR}")
    if not copied:
        print("No files were copied.")
    else:
        print("Copied files:")
        for path in copied:
            print(f"- {path}")
