"""Execute the existing framework unchanged, with observational validation guards."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path('/Users/jingwendong/Documents/UPM_Phd/backup/ITS_DDQN_TextHint')
sys.path.insert(0, str(ROOT))
import torch
import experiments.journal_hint_policy_experiment as journal

torch.set_num_threads(1)
config = journal.ExperimentConfig(training_seeds=tuple(range(42, 52)),
                                  evaluation_seeds=tuple(range(10000, 10100)),
                                  training_episodes=500)
audit = dict(training_interactions=0, evaluation_interactions=0,
             guarded_learned_evaluation_episodes=0, failed_or_repeated_runs=[],
             torch_intraop_threads=torch.get_num_threads())
evaluating = False
original_step = journal.TutoringEnvironment.step
original_eval = journal.evaluate_policy
original_episode = journal.run_episode

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8), (a, b)

def guarded_step(env, action):
    assert action in (0, 1, 2)
    result = original_step(env, action)
    state, reward, done, info = result
    assert 0 <= info['previous_mastery'] <= 1 and 0 <= info['mastery'] <= 1
    assert 0 <= float(state[0]) <= 1
    close(reward, sum(info['reward_components'].values()))
    audit['evaluation_interactions' if evaluating else 'training_interactions'] += 1
    return result

def forbidden(*args, **kwargs):
    raise AssertionError('Forbidden learning/replay/exploration operation during evaluation')

def guarded_eval(policy, seed, cfg, episode=1):
    global evaluating
    evaluating = True
    agent = policy.agent if isinstance(policy, journal.GreedyQPolicy) else None
    if agent:
        before = [copy.deepcopy(net.state_dict()) for net in (agent.online_network, agent.target_network)]
        optimizer_before = copy.deepcopy(agent.optimizer.state_dict())
        history_before = (agent.update_count, len(agent.loss_history), len(agent.epsilon_history),
                          len(agent.entropy_history), agent.rng.getstate())
    try:
        with patch.object(journal.ReplayBuffer, 'push', forbidden), \
             patch.object(journal.ReplayBuffer, 'sample', forbidden), \
             patch.object(journal.DoubleDQNAgent, 'update', forbidden), \
             patch.object(journal.DoubleDQNAgent, 'select_action', forbidden), \
             patch.object(torch.optim.Adam, 'step', forbidden):
            result = original_eval(policy, seed, cfg, episode=episode)
        if agent:
            for net, saved in zip((agent.online_network, agent.target_network), before):
                assert all(torch.equal(value, saved[key]) for key, value in net.state_dict().items())
            assert agent.optimizer.state_dict() == optimizer_before
            assert history_before == (agent.update_count, len(agent.loss_history), len(agent.epsilon_history),
                                       len(agent.entropy_history), agent.rng.getstate())
            for row in result[1]:
                qs = [row['q_value_action_' + str(a)] for a in range(3)]
                assert row['action'] == max(range(3), key=lambda a: qs[a])
            audit['guarded_learned_evaluation_episodes'] += 1
        return result
    finally:
        evaluating = False

def progress_episode(*args, **kwargs):
    result = original_episode(*args, **kwargs)
    agent = args[1]
    if agent.current_episode % 50 == 0:
        print(type(agent).__name__, 'training episode', agent.current_episode, '/ 500', flush=True)
    return result

def read_csv(path, required):
    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert set(required.split()) <= set(reader.fieldnames), str(path)
        rows = list(reader)
    assert rows and all(all(v is not None and v != '' for v in row.values()) for row in rows)
    return rows

def validate(directory):
    expected = {'random', 'fixed', 'rule_based'} | {
        name + '_seed_' + str(seed) for seed in range(42, 52) for name in ('dqn', 'ddqn')}
    assert {p.name for p in directory.iterdir() if p.is_dir()} == expected
    training_total = evaluation_total = 0
    learned_settings = None
    for run_id in sorted(expected):
        run = directory / run_id
        meta = json.loads((run / 'manifest.json').read_text())
        assert meta['status'] == 'complete'
        assert meta['run_id'] == run_id and meta['experiment_id'] == directory.name
        assert meta['evaluation_seed_set'] == list(range(10000, 10100))
        required_manifest = '''experiment_id run_id policy training_seed evaluation_seed_set checkpoint_path
            checkpoint_hash number_of_training_episodes steps_per_episode network_architecture learning_rate
            gamma replay_capacity batch_size target_update_interval exploration_mode epsilon_start epsilon_end
            environment_configuration simulator_parameters reward_parameters status'''.split()
        assert set(required_manifest) <= set(meta)
        assert not set(meta['configuration']['training_seeds']) & set(meta['evaluation_seed_set'])
        assert meta['steps_per_episode'] == 50
        for name, digest in meta['source_sha256'].items():
            assert journal.sha256(ROOT / name) == digest
        learned = meta['policy'] in ('dqn', 'ddqn')
        if learned:
            assert meta['training_seed'] in range(42, 52)
            assert meta['number_of_training_episodes'] == 500
            checkpoint = Path(meta['checkpoint_path'])
            assert checkpoint == run / 'checkpoint.pt'
            assert journal.sha256(checkpoint) == meta['checkpoint_hash'] == meta['checkpoint_sha256']
            keys = '''network_architecture optimizer learning_rate gamma replay_capacity batch_size
                target_update_interval exploration_mode epsilon_start epsilon_end'''.split()
            settings = {key: meta[key] for key in keys}
            if learned_settings is None:
                learned_settings = settings
            assert settings == learned_settings
            assert settings['exploration_mode'] == 'linear_decay'
            assert settings['epsilon_start'] == .8 and settings['epsilon_end'] == .05
            rows = read_csv(run / 'training.csv', '''run_id policy training_seed episode total_reward
                mean_step_reward average_loss average_epsilon average_entropy average_attempts
                average_response_time action_0_count action_1_count action_2_count final_mastery steps''')
            assert len(rows) == 500
            for i, row in enumerate(rows, 1):
                assert int(row['episode']) == i and int(row['steps']) == 50
                assert row['run_id'] == run_id and row['policy'] == meta['policy']
                assert int(row['training_seed']) == meta['training_seed']
                assert 0 <= float(row['final_mastery']) <= 1
                assert row['average_entropy'] == 'NA'
                close(row['average_epsilon'], .8 - (i-1) / 499 * .75)
                close(row['mean_step_reward'], float(row['total_reward']) / 50)
                assert sum(int(row['action_' + str(a) + '_count']) for a in range(3)) == 50
                training_total += 50
        episodes = read_csv(run / 'evaluation_episodes.csv', '''run_id policy training_seed evaluation_seed
            episode total_reward mean_step_reward correct_rate average_attempts average_response_time
            initial_mastery final_mastery mastery_gain action_0_count action_1_count action_2_count steps''')
        rows = read_csv(run / 'evaluation_steps.csv', '''experiment_id run_id policy training_seed evaluation_seed
            checkpoint_id episode step simulated_student_id knowledge_point question_difficulty mastery_before
            recent_accuracy previous_response_time previous_attempts action hint_level correct response_time
            attempts reward mastery_after mastery_gain done q_value_action_0 q_value_action_1 q_value_action_2''')
        assert len(episodes) == 100 and len(rows) == 5000
        assert 'epsilon' not in rows[0]
        components = ['reward_' + key for key in ('correctness', 'improvement', 'response_time',
                                                 'attempts', 'hint_usage', 'unnecessary_strong_hint')]
        for i, summary in enumerate(episodes, 1):
            group = rows[(i-1)*50:i*50]
            assert int(summary['evaluation_seed']) == 9999 + i and int(summary['episode']) == i
            assert int(summary['steps']) == 50
            for step, row in enumerate(group, 1):
                assert row['experiment_id'] == directory.name and row['run_id'] == run_id
                assert row['policy'] == meta['policy'] and row['training_seed'] == str(meta['training_seed'])
                assert int(row['episode']) == i and int(row['evaluation_seed']) == 9999 + i
                assert int(row['step']) == step and (row['done'] == 'True') == (step == 50)
                assert row['checkpoint_id'] == meta['checkpoint_hash']
                assert int(row['action']) in (0, 1, 2)
                for key in ('mastery_before', 'mastery_after', 'previous_mastery', 'mastery'):
                    assert 0 <= float(row[key]) <= 1
                for key in ('state', 'next_state'):
                    state = json.loads(row[key])
                    assert len(state) == 6 and all(0 <= x <= 1 for x in state)
                close(row['mastery_gain'], float(row['mastery_after']) - float(row['mastery_before']))
                close(row['reward'], sum(float(row[key]) for key in components))
                if meta['policy'] == 'fixed':
                    assert int(row['action']) == 1
                if learned:
                    qs = [float(row['q_value_action_' + str(a)]) for a in range(3)]
                    assert all(math.isfinite(q) for q in qs)
                    assert int(row['action']) == max(range(3), key=lambda a: qs[a])
                else:
                    assert all(row['q_value_action_' + str(a)] == 'NA' for a in range(3))
            close(summary['total_reward'], sum(float(row['reward']) for row in group))
            close(summary['mean_step_reward'], float(summary['total_reward']) / 50)
            for key in ('initial_mastery', 'final_mastery'):
                assert 0 <= float(summary[key]) <= 1
            close(summary['initial_mastery'], group[0]['mastery_before'])
            close(summary['final_mastery'], group[-1]['mastery_after'])
            close(summary['mastery_gain'], float(summary['final_mastery']) - float(summary['initial_mastery']))
            for a in range(3):
                assert int(summary['action_' + str(a) + '_count']) == sum(int(r['action']) == a for r in group)
        evaluation_total += len(rows)
    assert training_total == audit['training_interactions'] == 500000
    assert evaluation_total == audit['evaluation_interactions'] == 115000
    assert audit['guarded_learned_evaluation_episodes'] == 2000
    return dict(experiment_id=directory.name, completed_dqn_runs=10, completed_ddqn_runs=10,
                total_training_interactions=training_total, total_held_out_evaluation_interactions=evaluation_total,
                all_validation_checks_passed=True, validation_checks={
                    key: 'passed' for key in ('all_runs_completed', 'seed_separation', 'checkpoint_traceability',
                    'mastery_bounds', 'valid_actions', 'fixed_action_1', 'greedy_evaluation',
                    'evaluation_model_immutability', 'required_logging_fields', 'reward_reconciliation')},
                runtime_audit=audit)

journal.TutoringEnvironment.step = guarded_step
journal.evaluate_policy = guarded_eval
journal.run_episode = progress_episode
directory = journal.run_experiment(config, ROOT / 'outputs/journal_experiment/final')
print('VALIDATING', directory, flush=True)
report = validate(directory)
(directory / 'validation_report.json').write_text(json.dumps(report, indent=2))
(directory / 'execution_validation.py').write_text(Path(__file__).read_text())
print(json.dumps(report, indent=2), flush=True)
