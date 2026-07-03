# README Project Design

## Project Title

Reinforcement Learning-Based Intelligent Tutoring System for Text Hint Recommendation in Quadratic Equations

## Research Purpose

This project develops a modular Python research prototype for an Intelligent Tutoring System (ITS) that recommends text-based hints to students solving middle school quadratic equation problems.

The recommendation task is formulated as a reinforcement learning problem. A Double DQN agent learns to recommend one of three text hint support levels according to the student's current learning state.

The first research goal is to generate experimental evidence showing whether a Double DQN agent can learn an effective personalised hint recommendation policy using simulated students before later real-user evaluation.

## Current Scope

This repository is designed as a runnable experimental system, not a web application.

The first working version should support:

- quadratic equation knowledge and hint loading
- simulated student interaction
- tutoring environment reset and step logic
- three-action text hint recommendation
- Double DQN training
- dynamic epsilon-greedy exploration based on Q-value uncertainty
- training logs
- saved model files
- thesis-ready plots

The first version does not need:

- a web frontend
- full SQLite integration
- real student login
- production deployment
- real-user experiment workflow

## Important Design Decision

The system recommends text hint levels only.

It does not recommend different hint formats such as image, video, or audio.

The action space is fixed:

| Action | Hint Type | Support Level |
| --- | --- | --- |
| 0 | Low-level text hint | Strong guidance / explicit hint |
| 1 | Medium-level text hint | Partial guidance |
| 2 | High-level text hint | Minimal guidance / weak scaffold |

The Double DQN agent selects the hint level. After the action is selected, the system retrieves the corresponding text hint from the current question's hint set.

## Learning Domain

The learning domain is middle school mathematics.

The selected chapter is quadratic equations.

Knowledge points:

| ID | Knowledge Point |
| --- | --- |
| KP1 | Concept of Quadratic Equations |
| KP2 | Standard Form |
| KP3 | Solving by Direct Square Root |
| KP4 | Solving by Factoring |
| KP5 | Solving by Completing the Square |
| KP6 | Solving by Quadratic Formula |
| KP7 | Discriminant |
| KP8 | Applications of Quadratic Equations |

Each knowledge point should contain low, medium, and high difficulty questions.

Each question should have exactly three text hint levels.

Target data structure:

```text
Knowledge Point
  -> Question Difficulty
    -> Question
      -> Hint Level 1
      -> Hint Level 2
      -> Hint Level 3
```

## Reinforcement Learning Formulation

| RL Component | Project Meaning |
| --- | --- |
| State | Student learning condition |
| Action | Recommended text hint level |
| Reward | Learning effectiveness and efficiency |
| Agent | Double DQN |
| Environment | Quadratic equation tutoring environment |
| Student | Simulated student for training; real student later |

## State Design

The state is a fixed-size numerical vector:

```text
[
  mastery_level,
  question_difficulty,
  recent_accuracy,
  normalized_response_time,
  attempts,
  knowledge_point_index
]
```

State variables:

| Variable | Meaning |
| --- | --- |
| mastery_level | Estimated student mastery, between 0 and 1 |
| question_difficulty | Low, medium, or high difficulty, numerically encoded |
| recent_accuracy | Accuracy over recent attempts, preferably last 3 questions |
| normalized_response_time | Normalized response time |
| attempts | Number of attempts used |
| knowledge_point_index | Current KP index from KP1 to KP8 |

The state dimension must remain fixed and consistent across modules.

## Reward Design

The reward function belongs in the environment module, not the agent.

The reward should reflect both learning effectiveness and learning efficiency.

Positive reward components:

- correct answer after receiving a hint
- improvement compared with previous attempt
- fewer attempts
- faster response time
- appropriate hint level for the student's state

Negative reward components:

- incorrect answer
- too many attempts
- excessive hint dependence
- overly strong hints when the student is already capable
- long response time

Reward weights should be explicit and easy to modify in `config/config.py`.

## Dynamic Epsilon-Greedy Strategy

The project uses dynamic epsilon-greedy exploration, not only fixed epsilon decay.

For a state, the online Q-network outputs Q-values for the three actions:

```text
Q(s, a1), Q(s, a2), Q(s, a3)
```

The Q-values are converted to a probability distribution using softmax:

```text
p_i = softmax(Q(s, a_i) / temperature)
```

State uncertainty is computed using entropy or an entropy-like measure:

```text
H(s) = -sum_i p_i * log(p_i)
```

or:

```text
H(s) = 1 - sum_i p_i^2
```

Dynamic epsilon:

```text
epsilon(s) = epsilon_min + lambda_entropy * H(s)
```

Then epsilon is clipped to:

```text
[epsilon_min, epsilon_max]
```

The agent module should include:

```text
compute_dynamic_epsilon(q_values)
select_action(state)
```

Training must record epsilon values so an epsilon curve can be plotted for the thesis.

## Double DQN Requirement

The agent must implement Double DQN rather than vanilla DQN.

Required components:

- online Q-network
- target Q-network
- replay buffer
- target network update
- mini-batch training
- Bellman target using Double DQN logic

Double DQN target logic:

```text
next_action = argmax(Q_online(next_state))
target_q = reward + gamma * Q_target(next_state, next_action)
```

This design supports the thesis argument that Double DQN reduces overestimation bias compared with standard DQN.

## Student Simulator Design

The simulator should model:

- student ability / mastery level
- probability of answering correctly
- question difficulty effect
- hint level effect
- response time
- number of attempts
- mastery update after learning

An acceptable initial model:

```text
student_ability in [0, 1]

question_difficulty:
  Low = 0.3
  Medium = 0.6
  High = 0.9

hint_effect:
  Action 0 = +0.30
  Action 1 = +0.20
  Action 2 = +0.10

P(correct) = sigmoid(student_ability - question_difficulty + hint_effect)
```

Response time can be simulated as:

```text
base_time * (1 + question_difficulty - student_ability) + noise
```

The simulator should be simple but realistic enough to produce meaningful learning curves.

## Planned Project Structure

```text
ITS_DDQN_TextHint/
├── README_PROJECT_DESIGN.md
├── requirements.txt
├── main.py
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── knowledge/
│   ├── __init__.py
│   ├── knowledge_tree.json
│   ├── hint_bank.json
│   └── loader.py
│
├── env/
│   ├── __init__.py
│   ├── environment.py
│   └── state_builder.py
│
├── student/
│   ├── __init__.py
│   ├── simulator.py
│   └── real_user_interface.py
│
├── agent/
│   ├── __init__.py
│   ├── double_dqn.py
│   └── replay_buffer.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── run_episode.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── baseline.py
│   └── evaluator.py
│
├── visualization/
│   ├── __init__.py
│   ├── plot_reward.py
│   ├── plot_loss.py
│   ├── plot_efficiency.py
│   └── plot_hint_distribution.py
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py
│   └── schema.sql
│
├── outputs/
│   ├── logs/
│   ├── models/
│   └── figures/
│
└── tests/
    ├── test_environment.py
    └── test_agent.py
```

## Module Responsibilities

### `config/config.py`

Stores hyperparameters and reward weights.

Expected settings include:

- learning rate
- gamma
- batch size
- number of episodes
- epsilon minimum
- epsilon maximum
- entropy coefficient
- softmax temperature
- target update frequency
- reward weights

### `knowledge/knowledge_tree.json`

Stores the quadratic equation knowledge tree.

It should include KP1 to KP8 and low, medium, and high difficulty questions for each knowledge point.

### `knowledge/hint_bank.json`

Stores text-only hints.

Each question should have:

- Hint Level 1: explicit guidance
- Hint Level 2: partial explanation
- Hint Level 3: minimal scaffold

### `knowledge/loader.py`

Loads the knowledge tree and hint bank.

Provides clean retrieval functions for:

- knowledge points
- questions
- hints by question and hint level

### `env/environment.py`

Implements the tutoring environment.

Required interface:

```python
reset() -> state
step(action) -> next_state, reward, done, info
```

The environment should call the student simulator, knowledge loader, state builder, and reward function.

### `env/state_builder.py`

Builds fixed-size numerical state vectors from student and question information.

### `student/simulator.py`

Simulates student behavior during training.

It should generate:

- correctness
- response time
- attempts
- updated mastery

### `student/real_user_interface.py`

Placeholder for later real-user experiment integration.

This module should stay minimal until real-user evaluation is required.

### `agent/double_dqn.py`

Implements the Double DQN agent.

Required features:

- online Q-network
- target Q-network
- dynamic epsilon computation
- action selection
- Double DQN update logic
- model save and load support

### `agent/replay_buffer.py`

Implements replay buffer storage and mini-batch sampling.

### `training/train.py`

Main training loop.

Responsible for:

- running training episodes
- saving CSV logs
- saving trained model files
- recording reward, loss, epsilon, actions, attempts, response time, and hint usage

### `training/run_episode.py`

Runs one training or evaluation episode cleanly.

### `evaluation/metrics.py`

Computes learning and RL metrics.

Expected metrics:

- accuracy
- learning gain
- mastery improvement
- correct-after-hint rate
- hint usefulness rate
- hint level distribution
- average response time
- average attempts
- hints used
- time to mastery
- cumulative reward
- average reward
- loss curve data
- Q-value convergence data if available
- dynamic epsilon curve data

### `evaluation/baseline.py`

Implements simple comparison policies:

- random hint policy
- rule-based hint policy
- optional vanilla DQN later if feasible

### `evaluation/evaluator.py`

Compares Double DQN with baseline policies.

### `visualization/`

Generates thesis-ready figures using matplotlib.

Required figures:

- training reward curve
- loss curve
- dynamic epsilon curve
- hint level recommendation distribution
- average response time comparison
- average attempts comparison
- baseline vs Double DQN reward comparison
- correct-after-hint rate comparison if available

### `database/`

Optional and later-stage only.

SQLite may be used for real-user evaluation logs, but database logic must remain separate from the RL training pipeline.

### `main.py`

Simple entry point for training and plot generation.

## Output Policy

All generated research artifacts should be saved under `outputs/`.

Expected output folders:

- `outputs/logs/`
- `outputs/models/`
- `outputs/figures/`

## Implementation Order

Implementation should proceed one module at a time:

1. `config/config.py`
2. `knowledge/knowledge_tree.json`
3. `knowledge/hint_bank.json`
4. `knowledge/loader.py`
5. `student/simulator.py`
6. `env/state_builder.py`
7. `env/environment.py`
8. `agent/replay_buffer.py`
9. `agent/double_dqn.py`
10. `training/train.py`
11. visualization plots
12. evaluation metrics
13. baselines
14. optional SQLite logging

For each step:

- implement only the requested module
- avoid unrelated edits
- run or explain a minimal test
- keep interfaces consistent

## Coding Principles

- Keep the project modular.
- Do not mix environment logic with agent logic.
- Keep the action space fixed at three actions.
- Keep all hints text-only.
- Use simulated students for initial training.
- Do not require a database for the training demo.
- Store configurable values in `config/config.py`.
- Keep reward weights explicit.
- Record dynamic epsilon values during training.
- Save all generated files under `outputs/`.
- Prefer a simple working research prototype over over-engineered infrastructure.

## Current Status

Initial scaffold only.

No implementation code has been added yet.
