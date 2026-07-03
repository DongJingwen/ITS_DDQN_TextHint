"""Fast smoke tests for the DDQN text-hint ITS project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent.double_dqn import DoubleDQNAgent
from agent.replay_buffer import ReplayBuffer
from config.config import ACTION_DIM, BATCH_SIZE, STATE_DIM
from env.environment import TutoringEnvironment
from knowledge import loader as KnowledgeLoader
from student.simulator import StudentSimulator


def _print_result(name: str, passed: bool, error: Exception | None = None) -> None:
    status = "PASS" if passed else "FAIL"
    if error is None:
        print(f"{status}: {name}")
    else:
        print(f"{status}: {name} - {error}")


def _run_test(name: str, test_func) -> bool:
    try:
        test_func()
    except Exception as exc:  # pragma: no cover - smoke-test reporting path
        _print_result(name, False, exc)
        return False

    _print_result(name, True)
    return True


def test_knowledge_tree_loads() -> None:
    knowledge_points = KnowledgeLoader.get_knowledge_points()
    assert len(knowledge_points) == 8
    assert knowledge_points[0]["kp_id"] == "KP1"


def test_hint_bank_loads() -> None:
    question = KnowledgeLoader.get_question("KP1", "low")
    hint = KnowledgeLoader.get_hint(question["question_id"], "level_1")
    assert isinstance(hint, str)
    assert hint


def test_student_simulator_response() -> None:
    simulator = StudentSimulator(ability=0.5)
    result = simulator.simulate_interaction(difficulty="medium", action=1)
    assert "correct" in result
    assert result["response_time"] > 0
    assert result["attempts"] >= 1


def test_environment_reset() -> None:
    environment = TutoringEnvironment(max_steps=3, seed=1)
    state = environment.reset()
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32


def test_environment_step_all_actions() -> None:
    for action in range(ACTION_DIM):
        environment = TutoringEnvironment(max_steps=1, seed=action)
        environment.reset()
        next_state, reward, done, info = environment.step(action)
        assert next_state.shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert done is True
        assert info["action"] == action


def test_replay_buffer_push_sample() -> None:
    replay_buffer = ReplayBuffer(capacity=10)
    for index in range(5):
        state = np.full(STATE_DIM, index, dtype=np.float32)
        replay_buffer.push(
            state=state,
            action=index % ACTION_DIM,
            reward=float(index),
            next_state=state + 1,
            done=False,
        )

    batch = replay_buffer.sample(batch_size=3)
    assert batch[0].shape == (3, STATE_DIM)
    assert batch[1].shape == (3,)


def test_agent_select_action() -> None:
    agent = DoubleDQNAgent(device="cpu")
    state = np.zeros(STATE_DIM, dtype=np.float32)
    action = agent.select_action(state)
    assert action in range(ACTION_DIM)
    assert len(agent.epsilon_history) == 1


def test_one_training_update() -> None:
    agent = DoubleDQNAgent(device="cpu")
    batch_size = min(4, BATCH_SIZE)
    batch = (
        np.random.rand(batch_size, STATE_DIM).astype(np.float32),
        np.random.randint(0, ACTION_DIM, size=batch_size),
        np.random.randn(batch_size).astype(np.float32),
        np.random.rand(batch_size, STATE_DIM).astype(np.float32),
        np.zeros(batch_size, dtype=np.float32),
    )
    loss = agent.update(batch)
    assert isinstance(loss, float)
    assert len(agent.loss_history) == 1


def test_output_directories() -> None:
    for directory in (
        PROJECT_ROOT / "outputs" / "logs",
        PROJECT_ROOT / "outputs" / "models",
        PROJECT_ROOT / "outputs" / "figures",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        assert directory.exists()


def main() -> None:
    tests = [
        ("knowledge tree loads correctly", test_knowledge_tree_loads),
        ("hint bank loads correctly", test_hint_bank_loads),
        ("StudentSimulator can generate a response", test_student_simulator_response),
        ("TutoringEnvironment reset() works", test_environment_reset),
        ("TutoringEnvironment step(action) works for action 0, 1, 2", test_environment_step_all_actions),
        ("ReplayBuffer can push and sample", test_replay_buffer_push_sample),
        ("DoubleDQNAgent can select an action", test_agent_select_action),
        ("one mini training update can run", test_one_training_update),
        ("output directories exist or can be created", test_output_directories),
    ]

    results = [_run_test(name, test_func) for name, test_func in tests]
    passed = sum(results)
    total = len(results)
    print(f"\nSmoke test summary: {passed}/{total} passed")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
