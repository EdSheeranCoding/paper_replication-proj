"""
Main entry point for Klein (2021) replication.

Usage:
  python main.py                          # baseline run (3 runs for testing)
  python main.py --num-runs 1000          # full replication
  python main.py --num-runs 1000 --figures # full replication + all figures
  python main.py --load results.npz --punish  # punishment test on saved results
  python main.py --load results.npz --summary # print metrics summary
"""

import argparse
import numpy as np
from pathlib import Path

from config import Config
from train import train, run_single
from metrics import summarise, punishment_test_avg, profitability_per_firm, is_nash
from visuals import figure1, figure2, figure3, figure4, ensure_output_dir

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def save_results(results: list[dict], path: Path):
    """Save trained Q-tables and run metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    q_tables = []
    final_profits = []
    final_prices = []

    for r in results:
        q_tables.append([a.Q for a in r["agents"]])
        final_profits.append(r["final_profit"])
        final_prices.append(r["final_prices"])

    config = results[0]["config"]

    np.savez(
        path,
        q_tables=np.array(q_tables),            # (num_runs, 2, k+1, k+1)
        final_profits=np.array(final_profits),   # (num_runs, 2)
        final_prices=np.array(final_prices),     # (num_runs, 2)
        # Config as scalars
        T=config.T,
        k=config.k,
        gamma=config.gamma,
        lr=config.lr,
        num_runs=len(results),
    )
    print(f"Saved {len(results)} runs to {path}")


def load_results(path: Path) -> list[dict]:
    """Load saved results back into the format metrics/visuals expect."""
    from agent import Agent

    data = np.load(path)
    k = int(data["k"])
    gamma = float(data["gamma"])
    lr = float(data["lr"])
    T = int(data["T"])
    config = Config(T=T, k=k, gamma=gamma, lr=lr)

    results = []
    num_runs = int(data["num_runs"])

    for i in range(num_runs):
        # Reconstruct agents with loaded Q-tables
        agents = []
        for j in range(2):
            agent = Agent(id=j, config=config)
            agent.Q = data["q_tables"][i, j]
            agent.eps = 0.0  # fully trained
            agents.append(agent)

        results.append({
            "final_profit": list(data["final_profits"][i]),
            "final_prices": list(data["final_prices"][i].astype(int)),
            "profits": [[], []],  # not saved (too large)
            "agents": agents,
            "config": config,
        })

    print(f"Loaded {num_runs} runs from {path} (k={k}, T={T})")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Klein (2021) Q-learning collusion replication"
    )

    # Training
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of independent runs (paper uses 1000)")
    parser.add_argument("--T", type=int, default=500_000,
                        help="Learning periods per run")
    parser.add_argument("--k", type=int, default=6,
                        help="Price granularity (k+1 prices)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Learning rate α")
    parser.add_argument("--delta", type=float, default=0.95,
                        help="Discount factor δ")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")

    # Output
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save results (.npz)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to load saved results (.npz)")

    # Analysis modes
    parser.add_argument("--summary", action="store_true",
                        help="Print metrics summary")
    parser.add_argument("--punish", action="store_true",
                        help="Run punishment test (Figure 3)")
    parser.add_argument("--figures", action="store_true",
                        help="Generate all baseline figures (1-4)")

    args = parser.parse_args()

    # Either load or train
    if args.load:
        results = load_results(Path(args.load))
        config = results[0]["config"]
    else:
        config = Config(
            T=args.T, k=args.k, gamma=args.delta,
            lr=args.alpha, num_runs=args.num_runs,
        )
        print(f"Config: T={config.T}, k={config.k}, α={config.lr}, "
              f"δ={config.gamma}, runs={args.num_runs}")
        print("=" * 60)

        results = train(config=config, num_runs=args.num_runs, base_seed=args.seed)

        # Save by default
        save_path = args.save or str(OUTPUT_DIR / "results.npz")
        save_results(results, Path(save_path))

    # Analysis
    if args.summary or not (args.punish or args.figures):
        summarise(results, k=config.k, gamma=config.gamma)

    if args.punish:
        pt = punishment_test_avg(results, config.k, config.gamma)
        if pt:
            figure3(results, k=config.k, gamma=config.gamma)

    if args.figures:
        figure2(results, k=config.k, gamma=config.gamma)
        figure3(results, k=config.k, gamma=config.gamma)
        # These run their own training at multiple T / k values:
        figure1(num_runs=args.num_runs)
        figure4(num_runs=args.num_runs)


if __name__ == "__main__":
    main()
