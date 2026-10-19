"""Minimal frozen-policy sensitivity analysis for experiment 20260908T012149Z_c44d903b."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch


FROZEN_ROOT = Path(
    "/Users/jingwendong/Documents/UPM_Phd/backup/ITS_DDQN_TextHint"
)
EXPERIMENT_ROOT = (
    FROZEN_ROOT
    / "outputs/journal_experiment/final/20260908T012149Z_c44d903b"
)
OUTPUT_ROOT = Path(__file__).resolve().parent
TRAINING_SEEDS = tuple(range(42, 52))
EXPECTED_EVALUATION_SEEDS = tuple(range(10000, 10100))
REDUCED_HINT_EFFECTS = {0: 0.24, 1: 0.16, 2: 0.08}

sys.path.insert(0, str(FROZEN_ROOT))

from experiments.journal_hint_policy_experiment import (  # noqa: E402
    ExperimentConfig,
    evaluate_policy,
    make_agent,
)
from evaluation.policies import GreedyQPolicy  # noqa: E402
from student import simulator as simulator_module  # noqa: E402


CONDITIONS = (
    "baseline",
    "hint_effects_20pct_lower",
    "hint_intensity_penalty_20pct_higher",
    "strong_high_mastery_penalty_minus_0_12",
    "high_mastery_threshold_0_80",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate_configuration() -> tuple[ExperimentConfig, dict]:
    reference = json.loads(
        (EXPERIMENT_ROOT / "ddqn_seed_42/manifest.json").read_text(encoding="utf-8")
    )
    configuration = reference["configuration"]
    if tuple(configuration["evaluation_seeds"]) != EXPECTED_EVALUATION_SEEDS:
        raise RuntimeError("Frozen evaluation seeds do not match 10000--10099.")
    if configuration["steps_per_episode"] != 50:
        raise RuntimeError("Frozen episode horizon is not 50 steps.")
    if reference["environment_configuration"]["initial_ability"] != 0.5:
        raise RuntimeError("Frozen initial ability is not 0.5.")

    for seed in TRAINING_SEEDS:
        run_dir = EXPERIMENT_ROOT / f"ddqn_seed_{seed}"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["configuration"] != configuration:
            raise RuntimeError(f"Configuration mismatch for DDQN seed {seed}.")
        checkpoint = run_dir / "checkpoint.pt"
        if sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for DDQN seed {seed}.")

    config = ExperimentConfig(
        training_seeds=tuple(configuration["training_seeds"]),
        evaluation_seeds=tuple(configuration["evaluation_seeds"]),
        training_episodes=configuration["training_episodes"],
        steps_per_episode=configuration["steps_per_episode"],
        learning_rate=configuration["learning_rate"],
        gamma=configuration["gamma"],
        replay_capacity=configuration["replay_capacity"],
        batch_size=configuration["batch_size"],
        target_update_interval=configuration["target_update_interval"],
        exploration_mode=configuration["exploration_mode"],
        epsilon_min=configuration["epsilon_min"],
        epsilon_max=configuration["epsilon_max"],
        device=configuration["device"],
    )
    return config, reference


def baseline_and_rescored_episodes(seed: int) -> list[dict]:
    run_dir = EXPERIMENT_ROOT / f"ddqn_seed_{seed}"
    steps = read_csv(run_dir / "evaluation_steps.csv")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in steps:
        grouped[int(row["evaluation_seed"])].append(row)

    results = []
    for evaluation_seed in EXPECTED_EVALUATION_SEEDS:
        episode_steps = grouped[evaluation_seed]
        if len(episode_steps) != 50:
            raise RuntimeError(
                f"Seed {seed}, evaluation seed {evaluation_seed} does not have 50 steps."
            )
        rewards = {condition: 0.0 for condition in CONDITIONS if condition != "hint_effects_20pct_lower"}
        action_counts = [0, 0, 0]
        correct = 0
        for row in episode_steps:
            action = int(row["action"])
            previous_mastery = float(row["previous_mastery"])
            original_reward = float(row["reward"])
            action_counts[action] += 1
            correct += row["correct"] == "True"
            rewards["baseline"] += original_reward
            rewards["hint_intensity_penalty_20pct_higher"] += (
                original_reward + (-0.12 - -0.10) * (3 - action) / 3
            )
            rewards["strong_high_mastery_penalty_minus_0_12"] += (
                original_reward
                + (-0.12 - -0.10)
                * (action == 0 and previous_mastery >= 0.75)
            )
            rewards["high_mastery_threshold_0_80"] += (
                original_reward
                + 0.10 * (action == 0 and 0.75 <= previous_mastery < 0.80)
            )
        for condition, reward in rewards.items():
            results.append(
                {
                    "training_seed": seed,
                    "evaluation_seed": evaluation_seed,
                    "condition": condition,
                    "episode_reward": reward,
                    "correct_rate": correct / 50,
                    "strong_proportion": action_counts[0] / 50,
                    "partial_proportion": action_counts[1] / 50,
                    "minimal_proportion": action_counts[2] / 50,
                }
            )
    return results


def reduced_hint_effect_episodes(seed: int, config: ExperimentConfig) -> list[dict]:
    agent = make_agent("ddqn", seed, config)
    checkpoint = EXPERIMENT_ROOT / f"ddqn_seed_{seed}/checkpoint.pt"
    state_dict = torch.load(checkpoint, map_location=config.device, weights_only=True)
    agent.online_network.load_state_dict(state_dict)
    agent.update_target_network()
    policy = GreedyQPolicy(agent)
    results = []
    for episode, evaluation_seed in enumerate(config.evaluation_seeds, start=1):
        summary, _ = evaluate_policy(policy, evaluation_seed, config, episode=episode)
        steps = summary["steps"]
        results.append(
            {
                "training_seed": seed,
                "evaluation_seed": evaluation_seed,
                "condition": "hint_effects_20pct_lower",
                "episode_reward": summary["total_reward"],
                "correct_rate": summary["correct_rate"],
                "strong_proportion": summary["action_0_count"] / steps,
                "partial_proportion": summary["action_1_count"] / steps,
                "minimal_proportion": summary["action_2_count"] / steps,
            }
        )
    return results


def summarize(episode_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in episode_rows:
        grouped[(row["condition"], int(row["training_seed"]))].append(row)

    run_rows = []
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            rows = grouped[(condition, seed)]
            if len(rows) != 100:
                raise RuntimeError(f"Condition {condition}, seed {seed} does not have 100 episodes.")
            run_rows.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "mean_episode_reward": statistics.mean(r["episode_reward"] for r in rows),
                    "correctness": statistics.mean(r["correct_rate"] for r in rows),
                    "strong_proportion": statistics.mean(r["strong_proportion"] for r in rows),
                    "partial_proportion": statistics.mean(r["partial_proportion"] for r in rows),
                    "minimal_proportion": statistics.mean(r["minimal_proportion"] for r in rows),
                }
            )

    baseline = {
        int(row["training_seed"]): row
        for row in run_rows
        if row["condition"] == "baseline"
    }
    aggregate_rows = []
    for condition in CONDITIONS:
        rows = [r for r in run_rows if r["condition"] == condition]
        rewards = [r["mean_episode_reward"] for r in rows]
        paired_reward_changes = [
            r["mean_episode_reward"] - baseline[int(r["training_seed"])]["mean_episode_reward"]
            for r in rows
        ]
        aggregate_rows.append(
            {
                "condition": condition,
                "mean_episode_reward": statistics.mean(rewards),
                "between_run_sd_episode_reward": statistics.stdev(rewards),
                "baseline_relative_reward_change": statistics.mean(paired_reward_changes),
                "baseline_relative_reward_change_percent": (
                    100 * statistics.mean(paired_reward_changes)
                    / statistics.mean(r["mean_episode_reward"] for r in run_rows if r["condition"] == "baseline")
                ),
                "correctness": statistics.mean(r["correctness"] for r in rows),
                "strong_proportion": statistics.mean(r["strong_proportion"] for r in rows),
                "partial_proportion": statistics.mean(r["partial_proportion"] for r in rows),
                "minimal_proportion": statistics.mean(r["minimal_proportion"] for r in rows),
            }
        )
    return run_rows, aggregate_rows


def main() -> None:
    config, reference = load_and_validate_configuration()
    episode_rows = []
    for seed in TRAINING_SEEDS:
        episode_rows.extend(baseline_and_rescored_episodes(seed))

    original_hint_effects = dict(simulator_module.HINT_EFFECTS)
    try:
        simulator_module.HINT_EFFECTS.clear()
        simulator_module.HINT_EFFECTS.update(REDUCED_HINT_EFFECTS)
        for seed in TRAINING_SEEDS:
            episode_rows.extend(reduced_hint_effect_episodes(seed, config))
    finally:
        simulator_module.HINT_EFFECTS.clear()
        simulator_module.HINT_EFFECTS.update(original_hint_effects)

    run_rows, aggregate_rows = summarize(episode_rows)
    write_csv(OUTPUT_ROOT / "sensitivity_episode_results.csv", episode_rows)
    write_csv(OUTPUT_ROOT / "sensitivity_run_summary.csv", run_rows)
    write_csv(OUTPUT_ROOT / "sensitivity_aggregate_summary.csv", aggregate_rows)
    metadata = {
        "experiment_id": reference["experiment_id"],
        "policy": "frozen DDQN; no retraining",
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(EXPECTED_EVALUATION_SEEDS),
        "episodes_per_checkpoint": 100,
        "steps_per_episode": 50,
        "conditions": {
            "baseline": "Frozen transitions and original rewards.",
            "hint_effects_20pct_lower": "Simulator rerun with {0: 0.24, 1: 0.16, 2: 0.08}.",
            "hint_intensity_penalty_20pct_higher": "Exact frozen-transition rescoring with coefficient -0.12.",
            "strong_high_mastery_penalty_minus_0_12": "Exact frozen-transition rescoring; threshold remains previous_mastery >= 0.75.",
            "high_mastery_threshold_0_80": "Exact frozen-transition rescoring; -0.10 applies when previous_mastery >= 0.80.",
        },
        "assumptions": [],
        "failures": [],
    }
    (OUTPUT_ROOT / "sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
