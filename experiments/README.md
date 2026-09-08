# Primary journal hint-policy experiment

This simulation-only entry point compares Random, Fixed (action 1, Partial Guidance),
the unchanged Rule-based policy, conventional DQN, and DDQN. It never imports GUI
services or reads real-user/demo databases.

The environment, reward, six states, three actions, simulator equations, ordered
24-question curriculum, and 50-step horizon are unchanged. DQN subclasses DDQN and
overrides only the terminal-masked TD target. Both use identical initialization
seeds, architecture, Adam/MSE, replay settings, update budgets, target synchronization,
and episode-based linear epsilon decay from 0.80 to 0.05. Epsilon is constant within
an episode. A one-episode test uses 0.05, matching the existing schedule convention.
Entropy exploration is excluded from this entry point and remains a separate ablation.

Run from the project root. Preview configuration (no output files or training):

```bash
python3 -B -m experiments.journal_hint_policy_experiment
```

Execute only when the experiment budget and seed lists have been decided:

```bash
python3 -B -m experiments.journal_hint_policy_experiment --run \
  --training-seeds 42 43 44 --evaluation-seeds 10000 10001 10002 \
  --training-episodes 500
```

The seed counts above are syntax examples, not a statistical sample-size recommendation.
The default single training seed is also not a journal replication design. Training
and evaluation seed sets must be nonempty, unique, and disjoint. CPU is the default.
No checkpoint is selected automatically: the final checkpoint of each named run
is explicitly reloaded for greedy evaluation and its absolute path and SHA-256 saved.
Checkpoint files contain inference weights, not resumable optimizer/replay state.

Each invocation creates `outputs/journal/<unique experiment ID>/`. Each baseline
has one directory; each learned policy has one directory per training seed. Baselines
are not duplicated across training seeds. Each directory contains a manifest,
`evaluation_episodes.csv`, and `evaluation_steps.csv`; learned policies additionally
contain `training.csv` and `checkpoint.pt`. Manifests record completion/failure,
configuration, RNG stream seeds, environment/question order, simulator/reward settings,
source/content hashes, runtime versions, and checkpoint provenance. Baseline manifests
mark training settings as inapplicable and record zero actual training episodes.

Every evaluation seed creates a fresh original environment. The Random policy has
its own reproducible per-scenario RNG. Training exploration and replay have independent
RNG streams. Original simulator RNG consumption is preserved: matching scenario seeds
does not imply identical step-level noise after different policies take different
branches. All policies still use exactly the same simulator transition distribution.

Evaluation never uses the training episode runner, replay buffer, exploratory action
method, or optimizer. It reports raw return, correctness, attempts, response time,
action counts, and initial/final mastery. Step records include observations,
next observations, terminal flags, question IDs, mastery, and reward components.
For later statistics, preserve the training-run/scenario structure; do not treat
individual transitions or training episodes as independent trained-model replicates.

Focused tests (temporary two-episode training runs only; no persistent results):

```bash
python3 -B -m unittest discover -s tests -p 'test_journal_experiment.py' -v
```

## Logging schema version 2

All missing/non-applicable values serialize as the literal `NA` in CSV and JSON.
For baselines, training seed, checkpoint fields, and top-level neural training
settings are NA; the shared experiment configuration remains available under
`configuration`. Baselines have zero training episodes and no training CSV.
`status` is the manifest completion status. `checkpoint_hash` is SHA-256 and
equals the retained `checkpoint_sha256` field. `epsilon_start`/`epsilon_end`
expose the existing nested epsilon parameters without changing the schedule.

Training and held-out evaluation remain separate files. Training CSV fields:

```text
run_id, policy, training_seed, episode, total_reward, average_loss,
average_epsilon, average_entropy, average_attempts, average_response_time,
action_0_count, action_1_count, action_2_count, steps, mean_step_reward, final_mastery
```

Entropy is NA for the primary linear schedule: it is not measured and is never
reconstructed from epsilon. Loss is NA when an episode has no optimizer updates.
`mean_step_reward` always equals `total_reward / steps`.

Held-out evaluation step fields:

```text
experiment_id, run_id, policy, training_seed, checkpoint_id, episode,
simulated_student_id, recent_accuracy, previous_response_time, previous_attempts,
evaluation_seed, step, state, next_state, action, reward, done, question_id, kp_id,
difficulty, hint_level, correct, probability_correct, response_time, attempts,
previous_mastery, mastery, knowledge_point, question_difficulty, mastery_before,
mastery_after, mastery_gain, q_value_action_0, q_value_action_1, q_value_action_2,
reward_correctness, reward_improvement, reward_response_time, reward_attempts,
reward_hint_usage, reward_unnecessary_strong_hint
```

`checkpoint_id` is the checkpoint content SHA-256, or NA for baselines. Episode
is the one-based position in the evaluation seed list (one episode per seed).
`simulated_student_id` is the actual simulator ID, currently `sim_student`, not
a fabricated unique student identifier. Join observations using experiment/run,
evaluation seed, episode, and step. Recent accuracy is captured before the action;
previous time/attempts are the environment's raw seconds/count, initially zero.
Mastery aliases preserve the original columns for compatibility. Step mastery gain
is directly calculated as after minus before. Learned-policy Q-values are direct
online-network outputs at the pre-action state, read without gradients; baselines
have NA Q-values. No evaluation epsilon metric is recorded.

Held-out evaluation episode fields:

```text
experiment_id, run_id, policy, training_seed, episode, evaluation_seed, steps,
total_reward, mean_step_reward, correct_rate, average_attempts,
average_response_time, initial_mastery, final_mastery, mastery_gain,
action_0_count, action_1_count, action_2_count
```

The schema supports later analysis of the saved simulation outcomes. It does not
create new human-student measurements or add any GUI/database data.
