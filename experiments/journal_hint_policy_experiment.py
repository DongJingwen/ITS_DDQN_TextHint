"""Primary five-policy comparison. Execution requires an explicit --run flag."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
from uuid import uuid4

import numpy as np
import torch

from agent.double_dqn import DoubleDQNAgent
from agent.dqn import DQNAgent
from agent.replay_buffer import ReplayBuffer
from config import config as defaults
from env.environment import TutoringEnvironment
from evaluation.policies import (FixedHintPolicy, GreedyQPolicy, HintPolicy,
                                 RandomHintPolicy, RuleBasedHintPolicy)
from student.simulator import DIFFICULTY_VALUES, HINT_EFFECTS
from training.run_episode import run_episode
from utils.reproducibility import set_global_seed

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random", "fixed", "rule_based", "dqn", "ddqn")


@dataclass(frozen=True)
class ExperimentConfig:
    training_seeds: tuple[int, ...] = (42,)
    evaluation_seeds: tuple[int, ...] = tuple(range(10000, 10020))
    training_episodes: int = 500
    steps_per_episode: int = 50
    learning_rate: float = defaults.LEARNING_RATE
    gamma: float = defaults.GAMMA
    replay_capacity: int = defaults.BUFFER_SIZE
    batch_size: int = defaults.BATCH_SIZE
    target_update_interval: int = defaults.TARGET_UPDATE_FREQUENCY
    exploration_mode: str = "linear_decay"
    epsilon_min: float = defaults.EPSILON_MIN
    epsilon_max: float = defaults.EPSILON_MAX
    device: str = "cpu"

    def __post_init__(self):
        if self.steps_per_episode != 50:
            raise ValueError("Primary experiment preserves the 50-step horizon.")
        if self.exploration_mode != "linear_decay":
            raise ValueError("Primary DQN and DDQN both require linear_decay exploration.")
        for seeds in (self.training_seeds, self.evaluation_seeds):
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError("Seed sets must be nonempty and contain no duplicates.")
            if any(not 0 <= seed < 2**32 for seed in seeds):
                raise ValueError("Seeds must be integers in [0, 2**32).")
        if set(self.training_seeds) & set(self.evaluation_seeds):
            raise ValueError("Training and evaluation seed sets must be disjoint.")
        if self.training_episodes < 1 or not 1 <= self.batch_size <= self.replay_capacity:
            raise ValueError("Invalid training budget or replay/batch sizes.")
        if self.target_update_interval < 1 or self.learning_rate <= 0:
            raise ValueError("Learning rate and target update interval must be positive.")
        if not 0 <= self.epsilon_min <= self.epsilon_max <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid epsilon bounds or gamma.")


def stream_seed(seed: int, purpose: str) -> int:
    """Stable independent RNG streams; does not depend on Python hash randomization."""
    return int.from_bytes(hashlib.sha256(f"{purpose}:{seed}".encode()).digest()[:8], "big")


def make_environment(seed: int, config: ExperimentConfig) -> TutoringEnvironment:
    """Single environment factory for every algorithm and baseline."""
    return TutoringEnvironment(max_steps=config.steps_per_episode,
                               initial_ability=0.5, seed=seed)


def make_agent(policy: str, seed: int, config: ExperimentConfig) -> DoubleDQNAgent:
    if policy not in ("dqn", "ddqn"):
        raise ValueError("Only DQN/DDQN have trainable agents.")
    set_global_seed(seed)
    cls = DQNAgent if policy == "dqn" else DoubleDQNAgent
    return cls(state_dim=6, action_dim=3, learning_rate=config.learning_rate,
               gamma=config.gamma, epsilon_mode=config.exploration_mode,
               epsilon_min=config.epsilon_min, epsilon_max=config.epsilon_max,
               target_update_frequency=config.target_update_interval,
               device=config.device, rng=random.Random(stream_seed(seed, "exploration")))


def evaluate_policy(policy: HintPolicy, evaluation_seed: int,
                    config: ExperimentConfig, episode: int = 1) -> tuple[dict, list[dict]]:
    """Evaluate one scenario. No replay buffer, optimizer, or update API is used."""
    environment = make_environment(evaluation_seed, config)
    state = environment.reset()
    initial_mastery = float(state[0])
    records = []
    done = False
    while not done:
        before = {
            "episode": episode,
            "simulated_student_id": environment.student.student_id,
            "recent_accuracy": environment._recent_accuracy(),
            "previous_response_time": environment.last_response_time,
            "previous_attempts": environment.last_attempts,
        }
        q_values = [None, None, None]
        if isinstance(policy, GreedyQPolicy):
            with torch.no_grad():
                tensor = torch.as_tensor(state, dtype=torch.float32,
                                         device=policy.agent.device).unsqueeze(0)
                q_values = policy.agent.online_network(tensor).squeeze(0).cpu().tolist()
        action = policy.select_action(state)
        next_state, reward, done, info = environment.step(action)
        records.append({
            **before,
            "evaluation_seed": evaluation_seed, "step": len(records) + 1,
            "state": json.dumps(state.tolist()), "next_state": json.dumps(next_state.tolist()),
            "action": action, "reward": reward, "done": done,
            **{key: info[key] for key in ("question_id", "kp_id", "difficulty", "hint_level",
                                         "correct", "probability_correct", "response_time",
                                         "attempts", "previous_mastery", "mastery")},
            "knowledge_point": info["kp_id"], "question_difficulty": info["difficulty"],
            "mastery_before": info["previous_mastery"], "mastery_after": info["mastery"],
            "mastery_gain": info["mastery"] - info["previous_mastery"],
            **{f"q_value_action_{a}": q_values[a] for a in range(3)},
            **{f"reward_{key}": value for key, value in info["reward_components"].items()},
        })
        state = next_state
    summary = {
        "episode": episode,
        "evaluation_seed": evaluation_seed, "steps": len(records),
        "total_reward": sum(row["reward"] for row in records),
        "mean_step_reward": sum(row["reward"] for row in records) / len(records),
        "correct_rate": sum(row["correct"] for row in records) / len(records),
        "average_attempts": sum(row["attempts"] for row in records) / len(records),
        "average_response_time": sum(row["response_time"] for row in records) / len(records),
        "initial_mastery": initial_mastery, "final_mastery": records[-1]["mastery"],
        "mastery_gain": records[-1]["mastery"] - initial_mastery,
        **{f"action_{a}_count": sum(row["action"] == a for row in records) for a in range(3)},
    }
    return summary, records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty result table.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(explicit_na(row) for row in rows)


def explicit_na(value):
    """Serialize non-applicable values explicitly, including nested manifest fields."""
    if value is None:
        return "NA"
    if isinstance(value, dict):
        return {key: explicit_na(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [explicit_na(item) for item in value]
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(experiment_id: str, policy: str, training_seed: int | None,
             config: ExperimentConfig, checkpoint: Path | None) -> dict:
    environment = make_environment(config.evaluation_seeds[0], config)
    source_paths = [
        "agent/dqn.py", "agent/double_dqn.py", "agent/replay_buffer.py",
        "config/config.py", "env/environment.py", "env/state_builder.py",
        "student/simulator.py", "knowledge/knowledge_tree.json", "knowledge/hint_bank.json",
        "knowledge/loader.py", "evaluation/baseline.py", "evaluation/policies.py",
        "training/run_episode.py", "utils/reproducibility.py",
        "experiments/journal_hint_policy_experiment.py",
    ]
    result = {
        "schema_version": 2, "experiment_id": experiment_id, "policy": policy,
        "training_seed": training_seed, "evaluation_seed_set": list(config.evaluation_seeds),
        "number_of_training_episodes": config.training_episodes if checkpoint else 0,
        "steps_per_episode": config.steps_per_episode,
        "network_architecture": [6, 64, 64, 3], "hidden_activation": "ReLU",
        "optimizer": "Adam", "loss": "MSE", "learning_rate": config.learning_rate,
        "gamma": config.gamma, "replay_capacity": config.replay_capacity,
        "batch_size": config.batch_size, "target_update_interval": config.target_update_interval,
        "target_update_unit": "optimizer_updates", "updates_per_interaction": 1,
        "exploration_mode": config.exploration_mode,
        "epsilon_start": config.epsilon_max, "epsilon_end": config.epsilon_min,
        "epsilon_parameters": {"start": config.epsilon_max, "end": config.epsilon_min,
                               "schedule_unit": "episode", "one_episode_value": config.epsilon_min},
        "training_settings_apply": checkpoint is not None,
        "evaluation_exploration": "uniform_random" if policy == "random" else "none",
        "fixed_action": 1 if policy == "fixed" else None,
        "checkpoint_path": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint and checkpoint.exists() else None,
        "checkpoint_hash": sha256(checkpoint) if checkpoint and checkpoint.exists() else None,
        "checkpoint_selection": "final_training_episode" if checkpoint else "not_applicable",
        "environment_configuration": {
            "class": "env.environment.TutoringEnvironment", "state_dim": 6, "action_dim": 3,
            "state_order": ["mastery", "difficulty", "recent_accuracy", "response_time",
                            "attempts", "knowledge_point"],
            "action_mapping": {0: "level_1_strong", 1: "level_2_partial_guidance", 2: "level_3_weak"},
            "max_response_time": defaults.MAX_RESPONSE_TIME, "max_attempts": defaults.MAX_ATTEMPTS,
            "recent_accuracy_window": 3, "initial_ability": 0.5,
            "steps_per_episode": 50, "question_order": [
                item["question"]["question_id"] for item in environment.question_sequence],
            "curriculum": "ordered_24_questions_cycled_from_KP1_low",
        },
        "simulator_parameters": {
            "ability": environment.student.ability, "learning_rate": environment.student.learning_rate,
            "forgetting_rate": environment.student.forgetting_rate,
            "base_response_time": environment.student.base_response_time,
            "difficulty_values": DIFFICULTY_VALUES, "hint_effects": HINT_EFFECTS,
            "response_noise_sd": 5.0, "minimum_response_time": 5.0,
            "equations_source": "student/simulator.py (see source_sha256)",
        },
        "reward_parameters": dict(defaults.REWARD_WEIGHTS),
        "reward_source": "env/environment.py:_calculate_reward",
        "rng_protocol": {
            "training_environment_seed": training_seed,
            "exploration_seed": stream_seed(training_seed, "exploration") if training_seed is not None else None,
            "replay_seed": stream_seed(training_seed, "replay") if training_seed is not None else None,
            "random_policy_seeds": {str(s): stream_seed(s, "random_policy") for s in config.evaluation_seeds},
            "evaluation": "fresh environment per seed; original conditional RNG consumption preserved",
            "pairing_limit": "same starting scenario seeds, not identical per-step noise after policies diverge",
        },
        "configuration": asdict(config), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "device": config.device,
        "source_sha256": {name: sha256(ROOT / name) for name in source_paths},
    }
    if checkpoint is None:
        for key in ("network_architecture", "hidden_activation", "optimizer", "loss",
                    "learning_rate", "gamma", "replay_capacity", "batch_size",
                    "target_update_interval", "target_update_unit", "updates_per_interaction",
                    "exploration_mode", "epsilon_start", "epsilon_end", "epsilon_parameters",
                    "checkpoint_selection"):
            result[key] = None
    return result


def run_experiment(config: ExperimentConfig, output_root: Path) -> Path:
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_dir = output_root.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    # Baselines are evaluated once, not copied per training seed as false replicates.
    runs = [(name, None) for name in POLICIES[:3]] + [
        (name, seed) for seed in config.training_seeds for name in POLICIES[3:]]
    for name, seed in runs:
        run_id = name if seed is None else f"{name}_seed_{seed}"
        run_dir = output_dir / run_id
        run_dir.mkdir()
        checkpoint = run_dir / "checkpoint.pt" if seed is not None else None
        meta = manifest(experiment_id, name, seed, config, checkpoint)
        meta.update(run_id=run_id, status="running")
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(explicit_na(meta), indent=2), encoding="utf-8")
        try:
            learned_policy = None
            if seed is not None:
                agent = make_agent(name, seed, config)
                replay = ReplayBuffer(config.replay_capacity,
                                      rng=random.Random(stream_seed(seed, "replay")))
                environment = make_environment(seed, config)
                logs = []
                for episode in range(1, config.training_episodes + 1):
                    agent.set_training_progress(episode, config.training_episodes)
                    updates_before = agent.update_count
                    entropy_before = len(agent.entropy_history)
                    summary = run_episode(environment, agent, replay,
                                          batch_size=config.batch_size, train=True)
                    measured_entropy = agent.entropy_history[entropy_before:]
                    summary["average_entropy"] = (sum(measured_entropy) / len(measured_entropy)
                                                  if measured_entropy else None)
                    if agent.update_count == updates_before:
                        summary["average_loss"] = None
                    logs.append({"run_id": run_id, "policy": name, "training_seed": seed,
                                 "episode": episode, **summary,
                                 "mean_step_reward": summary["total_reward"] / summary["steps"],
                                 "final_mastery": environment.student.mastery})
                write_csv(run_dir / "training.csv", logs)
                agent.save(checkpoint)
                # Evaluate the exact saved weights, using a new agent and explicit path.
                evaluation_agent = make_agent(name, seed, config)
                evaluation_agent.online_network.load_state_dict(
                    torch.load(checkpoint, map_location=config.device, weights_only=True))
                evaluation_agent.update_target_network()
                learned_policy = GreedyQPolicy(evaluation_agent)
                meta["checkpoint_sha256"] = sha256(checkpoint)
                meta["checkpoint_hash"] = meta["checkpoint_sha256"]
            episodes, steps = [], []
            for episode, eval_seed in enumerate(config.evaluation_seeds, start=1):
                if name == "random":
                    policy = RandomHintPolicy(rng=random.Random(stream_seed(eval_seed, "random_policy")))
                elif name == "fixed":
                    policy = FixedHintPolicy()
                elif name == "rule_based":
                    policy = RuleBasedHintPolicy()
                else:
                    policy = learned_policy
                summary, records = evaluate_policy(policy, eval_seed, config, episode=episode)
                identity = {"experiment_id": experiment_id, "run_id": run_id,
                            "policy": name, "training_seed": seed}
                episodes.append({**identity, **summary})
                steps.extend({**identity, "checkpoint_id": meta["checkpoint_hash"], **row}
                             for row in records)
            write_csv(run_dir / "evaluation_episodes.csv", episodes)
            write_csv(run_dir / "evaluation_steps.csv", steps)
            meta["status"] = "complete"
        except Exception as exc:
            meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            manifest_path.write_text(json.dumps(explicit_na(meta), indent=2), encoding="utf-8")
        print(f"Completed {run_id}: {run_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute; otherwise only print the configuration.")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(10000, 10020)))
    parser.add_argument("--training-episodes", type=int, default=500)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "journal")
    args = parser.parse_args()
    config = ExperimentConfig(training_seeds=tuple(args.training_seeds),
                              evaluation_seeds=tuple(args.evaluation_seeds),
                              training_episodes=args.training_episodes)
    if args.run:
        print(run_experiment(config, args.output_root))
    else:
        print(json.dumps(asdict(config), indent=2))
        print("Configuration only. No experiment executed. Add --run to execute.")


if __name__ == "__main__":
    main()
