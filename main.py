"""Simple command-line entry point for the DDQN text-hint ITS prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.double_dqn import DoubleDQNAgent
from config.config import BATCH_SIZE, BUFFER_SIZE, MAX_STEPS_PER_EPISODE, NUM_EPISODES
from database.db_manager import DatabaseManager
from env.environment import TutoringEnvironment
from evaluation.evaluator import evaluate
from student.real_user_interface import RealUserSession
from training.train import train
from visualization.plot_efficiency import plot_attempts_curve, plot_response_time_curve
from visualization.plot_hint_distribution import plot_hint_distribution
from visualization.plot_loss import plot_epsilon_curve, plot_loss_curve
from visualization.plot_reward import plot_reward_curve


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_user_experiment.db"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def run_training(args: argparse.Namespace) -> Path:
    result = train(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )
    print(f"Training log saved to: {result['log_path']}")
    print(f"Model saved to: {result['model_path']}")
    args.training_log_path = result["log_path"]
    return result["model_path"]


def run_plots(args: argparse.Namespace) -> list[Path]:
    log_path = args.log
    saved_paths = [
        plot_reward_curve(log_path),
        plot_loss_curve(log_path),
        plot_epsilon_curve(log_path),
        plot_attempts_curve(log_path),
        plot_response_time_curve(log_path),
        plot_hint_distribution(log_path),
    ]

    for path in saved_paths:
        print(f"Figure saved to: {path}")

    return saved_paths


def run_evaluation(args: argparse.Namespace) -> None:
    results = evaluate(
        model_path=args.model,
        num_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )

    for result in results:
        print(
            f"{result['policy']}: "
            f"average_reward={result['average_reward']:.3f}, "
            f"correct_after_hint_rate={result['correct_after_hint_rate']:.3f}, "
            f"average_attempts={result['average_attempts']:.3f}, "
            f"average_response_time={result['average_response_time']:.3f}"
        )


def initialize_database(args: argparse.Namespace) -> Path:
    db = DatabaseManager(args.db_path)
    db.initialize_database()
    db.close()
    print(f"Database initialized at: {args.db_path}")
    return args.db_path


def _latest_model_path() -> Path:
    model_files = sorted(MODEL_DIR.glob("double_dqn_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No DDQN model files found in {MODEL_DIR}")
    return model_files[-1]


def run_real_user_demo(args: argparse.Namespace) -> None:
    db = DatabaseManager(args.db_path)
    db.initialize_database()

    agent = DoubleDQNAgent(device="cpu")
    agent.load(args.model or _latest_model_path())
    environment = TutoringEnvironment(max_steps=args.max_steps, seed=args.seed)

    session = RealUserSession(
        student_id=args.student_id,
        student_name=args.student_name,
        agent=agent,
        environment=environment,
        db_manager=db,
    )

    estimated_level, initial_difficulty = session.run_pretest(args.pretest_score)
    learning_summary = session.start_learning_session(max_steps=args.max_steps)
    learning_gain = session.run_posttest(args.posttest_score)
    final_summary = session.get_summary()

    print(f"Estimated level: {estimated_level:.3f}")
    print(f"Initial difficulty: {initial_difficulty}")
    print(f"Learning session summary: {learning_summary}")
    print(f"Learning gain: {learning_gain:.3f}")
    print(f"Final summary: {final_summary}")
    print(f"Database saved at: {args.db_path}")
    db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training, plotting, and evaluation for the DDQN text-hint ITS."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Double DQN model.")
    train_parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    train_parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    train_parser.add_argument("--seed", type=int, default=42)

    plot_parser = subparsers.add_parser("plot", help="Generate all training figures.")
    plot_parser.add_argument("--log", type=Path, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Run baseline comparison.")
    eval_parser.add_argument("--model", type=Path, default=None)
    eval_parser.add_argument("--eval-episodes", type=int, default=20)
    eval_parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    eval_parser.add_argument("--seed", type=int, default=100)

    db_parser = subparsers.add_parser("init-db", help="Initialize optional SQLite database.")
    db_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)

    demo_parser = subparsers.add_parser("real-demo", help="Run a minimal real-user session demo.")
    demo_parser.add_argument("--student-id", type=str, default="DEMO001")
    demo_parser.add_argument("--student-name", type=str, default="Demo Student")
    demo_parser.add_argument("--pretest-score", type=float, default=0.62)
    demo_parser.add_argument("--posttest-score", type=float, default=0.76)
    demo_parser.add_argument("--max-steps", type=int, default=5)
    demo_parser.add_argument("--seed", type=int, default=21)
    demo_parser.add_argument("--model", type=Path, default=None)
    demo_parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "real_user_session_demo.db",
    )

    all_parser = subparsers.add_parser("all", help="Train, plot, and evaluate.")
    all_parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    all_parser.add_argument("--eval-episodes", type=int, default=20)
    all_parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    all_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    all_parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    all_parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "plot":
        run_plots(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "init-db":
        initialize_database(args)
    elif args.command == "real-demo":
        run_real_user_demo(args)
    elif args.command == "all":
        model_path = run_training(args)
        args.log = args.training_log_path
        args.model = model_path
        run_plots(args)
        run_evaluation(args)


if __name__ == "__main__":
    main()
