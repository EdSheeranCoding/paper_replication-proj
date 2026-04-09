"""
Analyze results from the CUDA collusion simulation.

Usage:
    python analyze.py                              # print profit summary
    python analyze.py --summary                    # full metrics (profitability, optimality, Nash)
    python analyze.py --figures                    # generate Figure 2 & 3
    python analyze.py --punish                     # punishment/deviation test (Figure 3 only)
    python analyze.py --summary --nash-tol 0.1     # relax Nash tolerance
    python analyze.py results.bin                  # specify path
"""

import sys
import numpy as np
from pathlib import Path

# Add src_py to path so we can reuse metrics/visuals
sys.path.insert(0, str(Path(__file__).parent.parent / "src_py"))


def load_cuda_results(path: str = "results.bin") -> dict:
    """Load the binary file written by the CUDA kernel."""
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        num_runs, T, k, num_prices = header

        profits = np.fromfile(f, dtype=np.float32, count=num_runs * 2).reshape(num_runs, 2)
        prices = np.fromfile(f, dtype=np.int32, count=num_runs * 2).reshape(num_runs, 2)
        qtables = np.fromfile(f, dtype=np.float32,
                              count=num_runs * 2 * num_prices * num_prices
                              ).reshape(num_runs, 2, num_prices, num_prices)

    print(f"Loaded {num_runs} runs from {path} (k={k}, T={T}, num_prices={num_prices})")
    return {
        "num_runs": num_runs, "T": T, "k": k, "num_prices": num_prices,
        "profits": profits, "prices": prices, "qtables": qtables,
    }


def to_py_results(data: dict) -> list:
    """Convert CUDA results to the format expected by src_py metrics/visuals."""
    from agent import Agent
    from config import Config

    k = int(data["k"])
    config = Config(T=int(data["T"]), k=k)

    results = []
    for i in range(data["num_runs"]):
        agents = []
        for j in range(2):
            agent = Agent(id=j, config=config)
            agent.Q = data["qtables"][i, j].astype(np.float64)
            agent.eps = 0.0
            agents.append(agent)

        results.append({
            "final_profit": list(data["profits"][i].astype(float)),
            "final_prices": list(data["prices"][i].astype(int)),
            "profits": [[], []],
            "agents": agents,
            "config": config,
        })

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze CUDA collusion results")
    parser.add_argument("path", nargs="?", default="results.bin",
                        help="Path to results.bin")
    parser.add_argument("--summary", action="store_true",
                        help="Print full metrics summary (profitability, optimality, Nash)")
    parser.add_argument("--figures", action="store_true",
                        help="Generate Figure 2 (profit distribution) & Figure 3 (punishment)")
    parser.add_argument("--punish", action="store_true",
                        help="Run punishment/deviation test and generate Figure 3")
    parser.add_argument("--nash-tol", type=float, default=1e-5,
                        help="Tolerance for Nash equilibrium check (default: 1e-5)")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor (default: 0.95)")
    args = parser.parse_args()

    data = load_cuda_results(args.path)

    # Always print basic profit summary
    mean_p = np.mean(data["profits"])
    std_p = np.std(np.mean(data["profits"], axis=1))
    print(f"\nMean profit:  {mean_p:.4f}")
    print(f"Std profit:   {std_p:.4f}")
    print(f"Collusive:    0.1250")
    print(f"Competitive: ~0.0611")

    needs_results = args.summary or args.figures or args.punish
    if needs_results:
        results = to_py_results(data)
        k = int(data["k"])

    if args.summary:
        from metrics import summarise
        summarise(results, k=k, gamma=args.gamma, tol=args.nash_tol)

    if args.figures:
        from visuals import figure2, figure3
        figure2(results, k=k, gamma=args.gamma)
        figure3(results, k=k, gamma=args.gamma)

    if args.punish and not args.figures:
        # Only run separately if --figures didn't already generate it
        from visuals import figure3
        figure3(results, k=k, gamma=args.gamma)
