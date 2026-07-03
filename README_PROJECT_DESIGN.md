# DDQN Text-Hint Intelligent Tutoring System

## 1. Project Overview

This project is a modular Python research prototype for a Reinforcement Learning-based Intelligent Tutoring System (ITS).

The system recommends one of three text-based hint levels to students solving middle school quadratic equation problems. It is designed for simulation training first, then later real-user evaluation with a small participant group.

This is not a web application. The goal is to generate training logs, model files, evaluation results, and thesis-ready figures for a PhD research workflow.

## 2. Research Design

The hint recommendation problem is formulated as reinforcement learning:

| Component | Meaning |
| --- | --- |
| State | Student learning state |
| Action | Text hint support level |
| Reward | Learning improvement and efficiency |
| Agent | Double DQN |
| Environment | Quadratic equation tutoring environment |
| Student | Simulated students for training; real students later |

The learning domain is quadratic equations, covering KP1 to KP8:

- KP1 Concept of Quadratic Equations
- KP2 Standard Form
- KP3 Solving by Direct Square Root
- KP4 Solving by Factoring
- KP5 Solving by Completing the Square
- KP6 Solving by Quadratic Formula
- KP7 Discriminant
- KP8 Applications of Quadratic Equations

## 3. System Architecture

```text
ITS_DDQN_TextHint/
├── agent/              # Double DQN and replay buffer
├── config/             # Hyperparameters and reward weights
├── database/           # Optional SQLite logging and export
├── env/                # Tutoring environment and state builder
├── evaluation/         # Baselines, ablation, summaries, real-user metrics
├── knowledge/          # Knowledge tree, hint bank, loaders
├── student/            # Student simulator and real-user session wrapper
├── training/           # Training loop and episode runner
├── utils/              # Reproducibility helpers
├── visualization/      # Training, baseline, ablation, real-user plots
├── outputs/            # Logs, models, figures, exports
└── tests/              # Smoke test
```

## 4. State, Action, Reward

State vector:

```text
[
  mastery_level,
  normalized_question_difficulty,
  recent_accuracy,
  normalized_response_time,
  normalized_attempts,
  normalized_knowledge_point_index
]
```

Action space:

| Action | Meaning |
| --- | --- |
| 0 | Strong text hint / explicit guidance |
| 1 | Medium text hint / partial guidance |
| 2 | Weak text hint / minimal scaffold |

Reward is calculated in `env/environment.py` and includes:

- correctness
- mastery improvement
- response time efficiency
- attempt penalty
- hint usage penalty
- unnecessary strong hint penalty

Reward weights are configured in `config/config.py`.

## 5. Dynamic Entropy-based Epsilon Strategy

The default exploration strategy is `dynamic_entropy`.

For each state, the online Q-network produces Q-values. These are converted to probabilities:

```text
p_i = softmax(Q_i / temperature)
```

Entropy is computed:

```text
H = -sum(p_i log p_i)
```

Dynamic epsilon:

```text
epsilon = epsilon_min + lambda_entropy * H
```

Then epsilon is clipped into `[epsilon_min, epsilon_max]`.

The project also supports ablation modes:

- `dynamic_entropy`
- `fixed`
- `linear_decay`

## 6. How to Install

Recommended from the project root:

```bash
cd ITS_DDQN_TextHint
python3 -m pip install torch matplotlib numpy
```

The project uses only standard Python libraries plus:

- NumPy
- PyTorch
- Matplotlib

SQLite support uses Python's built-in `sqlite3`.

## 7. How to Run Training

From the parent workspace:

```bash
python3 ITS_DDQN_TextHint/training/train.py
```

Short test run:

```bash
python3 ITS_DDQN_TextHint/training/train.py \
  --episodes 5 \
  --max-steps 10 \
  --batch-size 4 \
  --buffer-size 100 \
  --seed 42
```

Or use the main entry point:

```bash
python3 ITS_DDQN_TextHint/main.py train --episodes 100
```

Training saves:

- CSV logs under `outputs/logs/`
- model weights under `outputs/models/`
- config snapshots under `outputs/logs/`

## 8. How to Generate Figures

Generate all standard training figures:

```bash
python3 ITS_DDQN_TextHint/main.py plot
```

Individual scripts:

```bash
python3 ITS_DDQN_TextHint/visualization/plot_reward.py
python3 ITS_DDQN_TextHint/visualization/plot_loss.py
python3 ITS_DDQN_TextHint/visualization/plot_efficiency.py
python3 ITS_DDQN_TextHint/visualization/plot_hint_distribution.py
```

Generated figures include:

- reward curve
- loss curve
- dynamic epsilon curve
- average attempts curve
- average response time curve
- hint action distribution

## 9. How to Run Baselines

Run baseline comparison:

```bash
python3 ITS_DDQN_TextHint/evaluation/evaluator.py --episodes 20
```

Generate baseline comparison figures:

```bash
python3 ITS_DDQN_TextHint/visualization/plot_baseline_comparison.py
```

Baselines:

- Random Hint
- Rule-based Hint
- trained Double DQN

## 10. How to Run Ablation Study

Run epsilon strategy ablation:

```bash
python3 ITS_DDQN_TextHint/evaluation/ablation.py --episodes 50
```

Compared strategies:

- Double DQN + dynamic entropy epsilon
- Double DQN + fixed epsilon
- Double DQN + linear decay epsilon

Outputs:

- logs under `outputs/logs/ablation/`
- figures under `outputs/figures/ablation/`

## 11. How to Use SQLite for Real User Experiments

Initialize the optional real-user database:

```bash
python3 ITS_DDQN_TextHint/main.py init-db
```

Run a minimal real-user session demo:

```bash
python3 ITS_DDQN_TextHint/main.py real-demo
```

Export SQLite tables to CSV:

```bash
python3 ITS_DDQN_TextHint/database/export_data.py
```

Compute real-user metrics:

```bash
python3 ITS_DDQN_TextHint/evaluation/real_user_metrics.py
```

Generate real-user figures:

```bash
python3 ITS_DDQN_TextHint/visualization/plot_real_user_results.py
```

SQLite is optional and does not affect simulation training.

## 12. Output Files

Main output folders:

```text
outputs/logs/        # training logs, baseline results, summaries, SQLite DBs
outputs/models/      # saved DDQN model weights
outputs/figures/     # generated thesis figures
outputs/logs/exports # exported SQLite CSV files
```

Useful generated files:

- `training_log_*.csv`
- `double_dqn_*.pt`
- `baseline_comparison.csv`
- `real_user_metrics.csv`
- `final_result_summary.csv`
- `final_result_summary.md`

Generate final automatic summary:

```bash
python3 ITS_DDQN_TextHint/evaluation/result_summary.py
```

## 13. Reproducibility

Training now sets global seeds for:

- Python `random`
- NumPy
- PyTorch
- CUDA if available

Run with a fixed seed:

```bash
python3 ITS_DDQN_TextHint/training/train.py --seed 123
```

Each training run saves a config snapshot:

```text
outputs/logs/experiment_config_*.json
```

Use the same config, command-line parameters, and seed to reproduce an experiment.

## 14. Current Limitations

- The current training system uses simulated students, not real student answer input.
- The real-user interface is a Python session wrapper, not a complete classroom UI.
- The knowledge bank currently contains a small structured quadratic-equation dataset.
- The simulator is probabilistic and simplified.
- Results from very short runs are only smoke-test evidence, not thesis-level evidence.
- SQLite logging is optional and not part of the simulation training loop.

## 15. Next Steps

- Expand the question bank for each knowledge point and difficulty level.
- Run longer simulation experiments with multiple seeds.
- Add statistical comparison across repeated runs.
- Refine the student simulator using real pilot data.
- Conduct small-scale real-user evaluation.
- Use exported SQLite data for final thesis analysis.
- Add richer reporting tables for supervisor meetings.
