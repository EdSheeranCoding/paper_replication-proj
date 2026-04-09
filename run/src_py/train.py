"""
Training script for Klein (2021) replication.
"Autonomous algorithmic collusion: Q-learning under sequential pricing."
"""

import numpy as np
from config import Config
from agent import Agent

# Defaults
SEED = 42
PRINT_EVERY = 50_000


def compute_demand(p_i: float, p_j: float) -> float:
    """Demand for firm i given firm prices, (eq. 3)."""
    if p_i < p_j:
        return 1.0 - p_i
    elif p_i == p_j:
        return 0.5 * (1.0 - p_i)
    else:
        return 0.0


def compute_profit(p_i: float, p_j: float) -> float:
    """Profit for firm i, (eq. 1)."""
    return p_i * compute_demand(p_i, p_j)


def price_grid(k: int) -> np.ndarray:
    """Discrete price set P = {0, 1/k, ..., 1}."""
    return np.linspace(0, 1, k + 1)


def run_single(config: Config, seed: int = SEED) -> dict:
    """Run a single T-period simulation of the sequential pricing game.

    Follows the pseudocode on p.545 of the paper:
      1. Initialise Q-tables and random prices for t={1,2}
      2. At each period t, the mover:
         - Updates Q for the action taken 2 periods ago (now that next state is known)
         - Picks a new price via ε-greedy
         - The other firm's price persists
    """

    P = price_grid(config.k)
    rng = np.random.default_rng(seed)

    agent0 = Agent(id=0, config=config, seed=seed)
    agent1 = Agent(id=1, config=config, seed=seed + 1)
    agents = [agent0, agent1]

    # Random initial prices (indices into price grid)
    p = [rng.integers(0, config.k + 1), rng.integers(0, config.k + 1)]

    # Track profits for final metrics
    profits = [[], []]

    for t in range(config.T):
        # Which agent moves this period
        mover = t % 2
        other = 1 - mover

        # State for mover = opponent's current price index
        state = p[other]

        # Mover picks a new price
        action = agents[mover].get_action(state)

        # Profit for mover at new price vs opponent's current price
        profit = compute_profit(P[action], P[p[other]])

        # Q-learning update for the PREVIOUS mover (delayed by 1 period).
        # We now know the next state they will face = the new price just chosen.
        # Pseudocode line 5: Update Q_i(p_{i,t-2}, p_{j,t-2}) using eq. (5)
        if t > 0:
            prev_mover = other
            pm = agents[prev_mover]
            next_state = action  # the price just set = what prev_mover faces next
            pm.update(
                state=pm.last_state,
                action=pm.last_action,
                profit=pm.last_profit,
                continuation_profit=compute_profit(
                    P[pm.last_action], P[action]
                ),
                next_state=next_state,
            )

        # Update mover's price
        p[mover] = action

        # Record profit for both firms this period
        profits[mover].append(profit)
        profits[other].append(compute_profit(P[p[other]], P[p[mover]]))

        # Store info for the delayed update next period
        agents[mover].last_action = action
        agents[mover].last_profit = profit
        agents[mover].last_state = state

        # Decay epsilon for mover
        agents[mover].decay_epsilon()

        if (t + 1) % PRINT_EVERY == 0:
            avg0 = np.mean(profits[0][-1000:]) if len(profits[0]) >= 1000 else np.mean(profits[0])
            avg1 = np.mean(profits[1][-1000:]) if len(profits[1]) >= 1000 else np.mean(profits[1])
            print(
                f"  t={t+1:>7d} | "
                f"avg profit: [{avg0:.4f}, {avg1:.4f}] | "
                f"prices: [{P[p[0]]:.2f}, {P[p[1]]:.2f}] | "
                f"eps: [{agents[0].eps:.4f}, {agents[1].eps:.4f}]"
            )

    # Final metrics
    final_profit_0 = np.mean(profits[0][-1000:])
    final_profit_1 = np.mean(profits[1][-1000:])

    return {
        "final_profit": [final_profit_0, final_profit_1],
        "final_prices": [p[0], p[1]],
        "profits": profits,
        "agents": agents,
        "config": config,
    }


def train(config: Config = None, num_runs: int = None, base_seed: int = SEED):
    """Run the full experiment: num_runs independent simulations."""

    if config is None:
        config = Config()
    if num_runs is None:
        num_runs = config.num_runs

    all_results = []

    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}")
        result = run_single(config, seed=base_seed + i)
        all_results.append(result)
        avg = np.mean(result["final_profit"])
        print(f"  -> final avg profit: {avg:.4f} (collusive=0.125, competitive~0.061)\n")

    # Summary across runs
    avg_profits = [np.mean(r["final_profit"]) for r in all_results]
    print("=" * 50)
    print(f"Results over {num_runs} runs:")
    print(f"  Mean profit:   {np.mean(avg_profits):.4f}")
    print(f"  Std profit:    {np.std(avg_profits):.4f}")
    print(f"  Collusive:     0.1250")
    print(f"  Competitive:  ~0.0611")

    return all_results


if __name__ == "__main__":
    results = train(num_runs=3)
