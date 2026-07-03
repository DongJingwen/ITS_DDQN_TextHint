"""Generate final experimental result summaries from existing logs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
SUMMARY_CSV_PATH = LOG_DIR / "final_result_summary.csv"
SUMMARY_MD_PATH = LOG_DIR / "final_result_summary.md"

NOT_AVAILABLE = "Not available"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _to_float(value)
    if number is None:
        return NOT_AVAILABLE if value is None else str(value)
    return f"{number:.{digits}f}"


def _latest_training_log() -> Path | None:
    fixed_path = LOG_DIR / "training_log.csv"
    if fixed_path.exists():
        return fixed_path

    timestamped_logs = sorted(LOG_DIR.glob("training_log_*.csv"))
    if not timestamped_logs:
        return None
    return timestamped_logs[-1]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_training() -> dict[str, Any]:
    training_log = _latest_training_log()
    rows = _read_csv(training_log) if training_log else []
    if not rows:
        return {
            "source": str(training_log) if training_log else NOT_AVAILABLE,
            "final_average_reward": NOT_AVAILABLE,
            "best_average_reward": NOT_AVAILABLE,
            "final_loss": NOT_AVAILABLE,
            "average_epsilon": NOT_AVAILABLE,
            "most_selected_hint_level": NOT_AVAILABLE,
            "average_response_time": NOT_AVAILABLE,
            "average_attempts": NOT_AVAILABLE,
            "average_hint_usage": NOT_AVAILABLE,
        }

    final_row = rows[-1]
    rewards = [_to_float(row.get("total_reward")) for row in rows]
    epsilons = [_to_float(row.get("average_epsilon")) for row in rows]
    response_times = [_to_float(row.get("average_response_time")) for row in rows]
    attempts = [_to_float(row.get("average_attempts")) for row in rows]

    action_counts = {
        "level_1_strong": sum(_to_float(row.get("action_0_count")) or 0 for row in rows),
        "level_2_medium": sum(_to_float(row.get("action_1_count")) or 0 for row in rows),
        "level_3_weak": sum(_to_float(row.get("action_2_count")) or 0 for row in rows),
    }
    most_selected = max(action_counts, key=action_counts.get)
    total_hints = sum(action_counts.values())

    clean_rewards = [value for value in rewards if value is not None]
    return {
        "source": str(training_log),
        "final_average_reward": final_row.get("total_reward", NOT_AVAILABLE),
        "best_average_reward": max(clean_rewards) if clean_rewards else NOT_AVAILABLE,
        "final_loss": final_row.get("average_loss", NOT_AVAILABLE),
        "average_epsilon": _mean([value for value in epsilons if value is not None]),
        "most_selected_hint_level": most_selected,
        "average_response_time": _mean([value for value in response_times if value is not None]),
        "average_attempts": _mean([value for value in attempts if value is not None]),
        "average_hint_usage": total_hints / len(rows) if rows else NOT_AVAILABLE,
    }


def summarize_baselines() -> dict[str, Any]:
    rows = _read_csv(LOG_DIR / "baseline_comparison.csv")
    if not rows:
        return {
            "best_performing_method": NOT_AVAILABLE,
            "reward_improvement_over_random": NOT_AVAILABLE,
            "reward_improvement_over_rule_based": NOT_AVAILABLE,
            "attempts_reduction": NOT_AVAILABLE,
        }

    by_policy = {row.get("policy"): row for row in rows}
    best_row = max(rows, key=lambda row: _to_float(row.get("average_reward")) or float("-inf"))
    ddqn = by_policy.get("double_dqn", {})
    random_row = by_policy.get("random", {})
    rule_row = by_policy.get("rule_based", {})

    ddqn_reward = _to_float(ddqn.get("average_reward"))
    random_reward = _to_float(random_row.get("average_reward"))
    rule_reward = _to_float(rule_row.get("average_reward"))
    ddqn_attempts = _to_float(ddqn.get("average_attempts"))
    best_baseline_attempts = min(
        [
            value
            for value in (
                _to_float(random_row.get("average_attempts")),
                _to_float(rule_row.get("average_attempts")),
            )
            if value is not None
        ],
        default=None,
    )

    attempts_reduction = None
    if ddqn_attempts is not None and best_baseline_attempts is not None:
        attempts_reduction = best_baseline_attempts - ddqn_attempts

    return {
        "best_performing_method": best_row.get("policy", NOT_AVAILABLE),
        "reward_improvement_over_random": (
            ddqn_reward - random_reward
            if ddqn_reward is not None and random_reward is not None
            else NOT_AVAILABLE
        ),
        "reward_improvement_over_rule_based": (
            ddqn_reward - rule_reward
            if ddqn_reward is not None and rule_reward is not None
            else NOT_AVAILABLE
        ),
        "attempts_reduction": attempts_reduction if attempts_reduction is not None else NOT_AVAILABLE,
    }


def summarize_ablation() -> dict[str, Any]:
    ablation_dir = LOG_DIR / "ablation"
    dynamic_rows = _read_csv(ablation_dir / "dynamic_entropy.csv")
    fixed_rows = _read_csv(ablation_dir / "fixed.csv")
    linear_rows = _read_csv(ablation_dir / "linear_decay.csv")

    if not dynamic_rows or not fixed_rows or not linear_rows:
        return {
            "dynamic_vs_fixed_reward_difference": NOT_AVAILABLE,
            "dynamic_vs_linear_reward_difference": NOT_AVAILABLE,
            "dynamic_vs_fixed_attempt_difference": NOT_AVAILABLE,
            "dynamic_vs_linear_attempt_difference": NOT_AVAILABLE,
        }

    dynamic_reward = _mean([_to_float(row.get("total_reward")) for row in dynamic_rows if _to_float(row.get("total_reward")) is not None])
    fixed_reward = _mean([_to_float(row.get("total_reward")) for row in fixed_rows if _to_float(row.get("total_reward")) is not None])
    linear_reward = _mean([_to_float(row.get("total_reward")) for row in linear_rows if _to_float(row.get("total_reward")) is not None])
    dynamic_attempts = _mean([_to_float(row.get("average_attempts")) for row in dynamic_rows if _to_float(row.get("average_attempts")) is not None])
    fixed_attempts = _mean([_to_float(row.get("average_attempts")) for row in fixed_rows if _to_float(row.get("average_attempts")) is not None])
    linear_attempts = _mean([_to_float(row.get("average_attempts")) for row in linear_rows if _to_float(row.get("average_attempts")) is not None])

    return {
        "dynamic_vs_fixed_reward_difference": (
            dynamic_reward - fixed_reward
            if dynamic_reward is not None and fixed_reward is not None
            else NOT_AVAILABLE
        ),
        "dynamic_vs_linear_reward_difference": (
            dynamic_reward - linear_reward
            if dynamic_reward is not None and linear_reward is not None
            else NOT_AVAILABLE
        ),
        "dynamic_vs_fixed_attempt_difference": (
            dynamic_attempts - fixed_attempts
            if dynamic_attempts is not None and fixed_attempts is not None
            else NOT_AVAILABLE
        ),
        "dynamic_vs_linear_attempt_difference": (
            dynamic_attempts - linear_attempts
            if dynamic_attempts is not None and linear_attempts is not None
            else NOT_AVAILABLE
        ),
    }


def summarize_real_users() -> dict[str, Any]:
    rows = _read_csv(LOG_DIR / "real_user_metrics.csv")
    if not rows:
        return {
            "mean_pretest": NOT_AVAILABLE,
            "mean_posttest": NOT_AVAILABLE,
            "mean_learning_gain": NOT_AVAILABLE,
            "correct_after_hint_rate": NOT_AVAILABLE,
        }

    first_row = rows[0]
    return {
        "mean_pretest": first_row.get("mean_pretest_score") or NOT_AVAILABLE,
        "mean_posttest": first_row.get("mean_posttest_score") or NOT_AVAILABLE,
        "mean_learning_gain": first_row.get("mean_learning_gain") or NOT_AVAILABLE,
        "correct_after_hint_rate": first_row.get("correct_after_hint_rate") or NOT_AVAILABLE,
    }


def build_summary() -> dict[str, dict[str, Any]]:
    return {
        "training_performance": summarize_training(),
        "efficiency": summarize_training(),
        "baseline_comparison": summarize_baselines(),
        "ablation": summarize_ablation(),
        "real_users": summarize_real_users(),
    }


def save_summary_csv(summary: dict[str, dict[str, Any]], output_path: Path = SUMMARY_CSV_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        for section, metrics in summary.items():
            for metric, value in metrics.items():
                writer.writerow({"section": section, "metric": metric, "value": _fmt(value)})
    return output_path


def save_summary_markdown(summary: dict[str, dict[str, Any]], output_path: Path = SUMMARY_MD_PATH) -> Path:
    lines = [
        "# Final Experimental Result Summary",
        "",
        "## Training Performance",
        f"- Final average reward: {_fmt(summary['training_performance']['final_average_reward'])}",
        f"- Best average reward: {_fmt(summary['training_performance']['best_average_reward'])}",
        f"- Final loss: {_fmt(summary['training_performance']['final_loss'])}",
        f"- Average epsilon: {_fmt(summary['training_performance']['average_epsilon'])}",
        f"- Most selected hint level: {summary['training_performance']['most_selected_hint_level']}",
        "",
        "## Efficiency",
        f"- Average response time: {_fmt(summary['efficiency']['average_response_time'])}",
        f"- Average attempts: {_fmt(summary['efficiency']['average_attempts'])}",
        f"- Average hint usage per episode: {_fmt(summary['efficiency']['average_hint_usage'])}",
        "",
        "## Baseline Comparison",
        f"- Best performing method: {summary['baseline_comparison']['best_performing_method']}",
        f"- Reward improvement over random: {_fmt(summary['baseline_comparison']['reward_improvement_over_random'])}",
        f"- Reward improvement over rule-based: {_fmt(summary['baseline_comparison']['reward_improvement_over_rule_based'])}",
        f"- Attempts reduction: {_fmt(summary['baseline_comparison']['attempts_reduction'])}",
        "",
        "## Ablation",
        f"- Dynamic epsilon vs fixed epsilon reward difference: {_fmt(summary['ablation']['dynamic_vs_fixed_reward_difference'])}",
        f"- Dynamic epsilon vs linear decay reward difference: {_fmt(summary['ablation']['dynamic_vs_linear_reward_difference'])}",
        f"- Dynamic epsilon vs fixed epsilon attempts difference: {_fmt(summary['ablation']['dynamic_vs_fixed_attempt_difference'])}",
        f"- Dynamic epsilon vs linear decay attempts difference: {_fmt(summary['ablation']['dynamic_vs_linear_attempt_difference'])}",
        "",
        "## Real Users",
        f"- Mean pre-test: {_fmt(summary['real_users']['mean_pretest'])}",
        f"- Mean post-test: {_fmt(summary['real_users']['mean_posttest'])}",
        f"- Mean learning gain: {_fmt(summary['real_users']['mean_learning_gain'])}",
        f"- Correct-after-hint rate: {_fmt(summary['real_users']['correct_after_hint_rate'])}",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    summary = build_summary()
    csv_path = save_summary_csv(summary)
    md_path = save_summary_markdown(summary)
    print(f"Saved final result summary CSV: {csv_path}")
    print(f"Saved final result summary Markdown: {md_path}")


if __name__ == "__main__":
    main()
