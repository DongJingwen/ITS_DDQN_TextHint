"""Focused verification; only tiny runs in temporary directories, never full experiments."""

from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from agent.double_dqn import DoubleDQNAgent
from agent.dqn import DQNAgent
from agent.replay_buffer import ReplayBuffer
from env.environment import TutoringEnvironment
from evaluation.policies import FixedHintPolicy, GreedyQPolicy, RandomHintPolicy, RuleBasedHintPolicy
from experiments.journal_hint_policy_experiment import (
    ROOT, ExperimentConfig, evaluate_policy, make_agent, make_environment,
    run_experiment, sha256,
)


class ConstantQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.expand(len(states), -1)


class JournalVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.config = ExperimentConfig(training_episodes=2, evaluation_seeds=(10000,))

    def target_values(self, cls):
        agent = cls(gamma=0.9, device="cpu")
        agent.online_network = ConstantQ([9, 2, 1])
        agent.target_network = ConstantQ([2, 7, 4])
        return agent.compute_td_targets(torch.tensor([1., -0.5]),
                                        torch.zeros(2, 6), torch.tensor([0., 1.]))

    def test_01_dqn_target_including_terminal_mask(self):
        actual = self.target_values(DQNAgent)
        torch.testing.assert_close(actual, torch.tensor([7.3, -0.5]))
        self.assertFalse(actual.requires_grad)

    def test_02_ddqn_target_including_terminal_mask(self):
        torch.testing.assert_close(self.target_values(DoubleDQNAgent), torch.tensor([2.8, -0.5]))

    def test_03_random_reproducibility(self):
        a = RandomHintPolicy(rng=random.Random(71))
        b = RandomHintPolicy(rng=random.Random(71))
        state = np.zeros(6)
        actions = [a.select_action(state) for _ in range(100)]
        self.assertEqual(actions, [b.select_action(state) for _ in range(100)])
        self.assertEqual(set(actions), {0, 1, 2})

    def test_04_fixed_partial_guidance(self):
        for state in np.random.RandomState(1).rand(100, 6):
            self.assertEqual(FixedHintPolicy().select_action(state), 1)

    def test_05_rule_valid_and_existing_boundaries(self):
        rule = RuleBasedHintPolicy()
        for state in np.random.RandomState(2).rand(1000, 6):
            self.assertIn(rule.select_action(state), (0, 1, 2))
        self.assertEqual(rule.select_action(np.array([.8, 0, 1, 0, 2/3, 0])), 2)
        self.assertEqual(rule.select_action(np.array([.8, 0, 1, 0, 1, 0])), 0)
        self.assertEqual(rule.select_action(np.array([.8, .5, 1, 0, 0, 0])), 1)

    def test_06_evaluation_does_not_update_networks_optimizer_or_history(self):
        for name in ("dqn", "ddqn"):
            agent = make_agent(name, 42, self.config)
            before = [copy.deepcopy(net.state_dict()) for net in (agent.online_network, agent.target_network)]
            optimizer = copy.deepcopy(agent.optimizer.state_dict())
            rng_state = agent.rng.getstate()
            with patch.object(agent, "update", side_effect=AssertionError("evaluation updated")), \
                 patch.object(agent, "select_action", side_effect=AssertionError("exploration called")):
                summary, rows = evaluate_policy(GreedyQPolicy(agent), 10000, self.config)
            for net, saved in zip((agent.online_network, agent.target_network), before):
                for key, value in net.state_dict().items():
                    self.assertTrue(torch.equal(value, saved[key]))
            self.assertEqual(agent.optimizer.state_dict(), optimizer)
            self.assertEqual(agent.rng.getstate(), rng_state)
            self.assertEqual(agent.update_count, 0)
            self.assertEqual(agent.epsilon_history, [])
            self.assertEqual(agent.loss_history, [])
            self.assertEqual(summary["steps"], 50)
            self.assertEqual(sum(row["done"] for row in rows), 1)

    def test_07_evaluation_cannot_write_or_sample_replay(self):
        buffer = ReplayBuffer(10)
        buffer.push(np.zeros(6), 1, 0, np.ones(6), False)
        original = list(buffer.buffer)
        with patch.object(ReplayBuffer, "push", side_effect=AssertionError("replay write")), \
             patch.object(ReplayBuffer, "sample", side_effect=AssertionError("replay sample")):
            for policy in (FixedHintPolicy(), RuleBasedHintPolicy(),
                           RandomHintPolicy(rng=random.Random(1)),
                           GreedyQPolicy(make_agent("dqn", 42, self.config)),
                           GreedyQPolicy(make_agent("ddqn", 42, self.config))):
                evaluate_policy(policy, 10000, self.config)
        self.assertEqual(len(buffer), 1)
        self.assertIs(buffer.buffer[0], original[0])

    def test_08_shared_environment_and_agent_configuration(self):
        dqn = make_agent("dqn", 42, self.config)
        ddqn = make_agent("ddqn", 42, self.config)
        self.assertIs(DQNAgent.update, DoubleDQNAgent.update)
        self.assertIs(DQNAgent.__init__, DoubleDQNAgent.__init__)
        self.assertEqual(str(dqn.online_network), str(ddqn.online_network))
        self.assertEqual(dqn.optimizer.defaults, ddqn.optimizer.defaults)
        for key in dqn.online_network.state_dict():
            self.assertTrue(torch.equal(dqn.online_network.state_dict()[key], ddqn.online_network.state_dict()[key]))
        env = make_environment(10000, self.config)
        reference = TutoringEnvironment(seed=10000, max_steps=50, initial_ability=.5)
        self.assertIs(type(env), TutoringEnvironment)
        self.assertEqual(len(env.question_sequence), 24)
        np.testing.assert_array_equal(env.reset(), reference.reset())
        for step in range(50):
            actual, expected = env.step(step % 3), reference.step(step % 3)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])

    def test_09_shared_linear_exploration_endpoints(self):
        agents = [make_agent(name, 42, self.config) for name in ("dqn", "ddqn")]
        for episode, expected in ((1, .8), (3, .425), (5, .05)):
            for agent in agents:
                agent.set_training_progress(episode, 5)
                self.assertEqual(agent.epsilon_mode, "linear_decay")
                self.assertAlmostEqual(agent.compute_epsilon(torch.tensor([1., 5., 8.])), expected)
        with self.assertRaises(ValueError):
            ExperimentConfig(exploration_mode="dynamic_entropy")

    def test_10_reject_seed_overlap_and_changed_horizon(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(evaluation_seeds=(42,))
        with self.assertRaises(ValueError):
            ExperimentConfig(steps_per_episode=3)

    def test_11_ablation_entropy_writer(self):
        # Extract only the writer to avoid importing matplotlib or creating output folders.
        tree = ast.parse((ROOT / "evaluation/ablation.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_write_episode_log")
        namespace = {"csv": csv, "Path": Path, "Any": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "ablation_writer", "exec"), namespace)
        fields = ["episode", "total_reward", "average_loss", "average_epsilon", "average_entropy",
                  "average_attempts", "average_response_time", "action_0_count", "action_1_count",
                  "action_2_count", "steps"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ablation.csv"
            namespace["_write_episode_log"](path, [{**dict.fromkeys(fields, 0), "epsilon_mode": "fixed"}])
            with path.open() as handle:
                self.assertIn("average_entropy", csv.DictReader(handle).fieldnames)

    def test_12_tiny_end_to_end_reproducibility_and_provenance(self):
        # Two training episodes per learned policy, one evaluation episode per policy.
        # Run twice only to verify deterministic artifacts, all under an auto-cleaned temp directory.
        with tempfile.TemporaryDirectory() as directory:
            first = run_experiment(self.config, Path(directory))
            second = run_experiment(self.config, Path(directory))
            for name in ("random", "fixed", "rule_based", "dqn_seed_42", "ddqn_seed_42"):
                manifest = json.loads((first / name / "manifest.json").read_text())
                required_manifest = """experiment_id run_id policy training_seed evaluation_seed_set
                    checkpoint_path checkpoint_hash number_of_training_episodes steps_per_episode
                    network_architecture learning_rate gamma replay_capacity batch_size
                    target_update_interval exploration_mode epsilon_start epsilon_end
                    environment_configuration simulator_parameters reward_parameters status""".split()
                self.assertTrue(set(required_manifest) <= set(manifest))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["evaluation_seed_set"], [10000])
                self.assertEqual(manifest["environment_configuration"]["steps_per_episode"], 50)
                for filename in ("evaluation_steps.csv", "evaluation_episodes.csv"):
                    def read(path):
                        with path.open() as handle:
                            return [{k: v for k, v in row.items() if k != "experiment_id"}
                                    for row in csv.DictReader(handle)]
                    self.assertEqual(read(first / name / filename), read(second / name / filename))
                with (first / name / "evaluation_steps.csv").open() as handle:
                    step_rows = list(csv.DictReader(handle))
                required_steps = """experiment_id run_id policy training_seed evaluation_seed checkpoint_id
                    episode step simulated_student_id knowledge_point question_difficulty mastery_before
                    recent_accuracy previous_response_time previous_attempts action hint_level correct
                    response_time attempts reward mastery_after mastery_gain done q_value_action_0
                    q_value_action_1 q_value_action_2""".split()
                self.assertTrue(set(required_steps) <= set(step_rows[0]))
                self.assertNotIn("epsilon", step_rows[0])
                for i, row in enumerate(step_rows):
                    self.assertNotIn("", row.values())
                    self.assertEqual(row["simulated_student_id"], "sim_student")
                    self.assertEqual(row["episode"], "1")
                    self.assertAlmostEqual(float(row["mastery_gain"]),
                                           float(row["mastery_after"]) - float(row["mastery_before"]))
                    expected_time = float(step_rows[i-1]["response_time"]) if i else 0
                    expected_attempts = int(step_rows[i-1]["attempts"]) if i else 0
                    self.assertEqual(float(row["previous_response_time"]), expected_time)
                    self.assertEqual(int(row["previous_attempts"]), expected_attempts)
                    recent = step_rows[max(0, i-3):i]
                    self.assertEqual(float(row["recent_accuracy"]),
                                     sum(r["correct"] == "True" for r in recent) / len(recent) if recent else 0)
                    self.assertEqual(row["checkpoint_id"], manifest["checkpoint_hash"])
                    if name.startswith(("dqn", "ddqn")):
                        values = [float(row[f"q_value_action_{a}"]) for a in range(3)]
                        self.assertEqual(int(row["action"]), int(np.argmax(values)))
                    else:
                        self.assertEqual(row["training_seed"], "NA")
                        self.assertTrue(all(row[f"q_value_action_{a}"] == "NA" for a in range(3)))
                with (first / name / "evaluation_episodes.csv").open() as handle:
                    episode_row = next(csv.DictReader(handle))
                required_episode = """run_id policy training_seed evaluation_seed episode total_reward
                    mean_step_reward correct_rate average_attempts average_response_time initial_mastery
                    final_mastery mastery_gain action_0_count action_1_count action_2_count steps""".split()
                self.assertTrue(set(required_episode) <= set(episode_row))
                self.assertAlmostEqual(float(episode_row["total_reward"]), sum(float(r["reward"]) for r in step_rows))
                self.assertAlmostEqual(float(episode_row["mean_step_reward"]), float(episode_row["total_reward"]) / 50)
                if name.startswith(("dqn", "ddqn")):
                    self.assertEqual(manifest["checkpoint_sha256"], sha256(Path(manifest["checkpoint_path"])))
                    self.assertEqual((first / name / "training.csv").read_text(),
                                     (second / name / "training.csv").read_text())
                    with (first / name / "training.csv").open() as handle:
                        rows = list(csv.DictReader(handle))
                    required_training = """run_id policy training_seed episode total_reward mean_step_reward
                        average_loss average_epsilon average_entropy average_attempts average_response_time
                        action_0_count action_1_count action_2_count final_mastery steps""".split()
                    self.assertTrue(set(required_training) <= set(rows[0]))
                    self.assertEqual(rows[0]["average_loss"], "NA")
                    for row in rows:
                        self.assertEqual(row["average_entropy"], "NA")
                        self.assertNotIn("", row.values())
                        self.assertAlmostEqual(float(row["mean_step_reward"]), float(row["total_reward"]) / 50)
                        self.assertTrue(0 <= float(row["final_mastery"]) <= 1)
                    self.assertAlmostEqual(float(rows[0]["average_epsilon"]), .8)
                    self.assertAlmostEqual(float(rows[1]["average_epsilon"]), .05)
                    self.assertGreater(float(rows[1]["average_loss"]), 0)


if __name__ == "__main__":
    unittest.main()
