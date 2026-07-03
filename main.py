"""Simple command-line entry point for the DDQN text-hint ITS prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.config import BATCH_SIZE, BUFFER_SIZE, MAX_STEPS_PER_EPISODE, NUM_EPISODES
from evaluation.evaluator import evaluate
from training.train import train
from visualization.plot_efficiency import plot_attempts_curve, plot_response_time_curve
from visualization.plot_hint_distribution import plot_hint_distribution
from visualization.plot_loss import plot_epsilon_curve, plot_loss_curve
from visualization.plot_reward import plot_reward_curve


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
    elif args.command == "all":
        model_path = run_training(args)
        args.log = None
        args.model = model_path
        run_plots(args)
        run_evaluation(args)


if __name__ == "__main__":
    main()
