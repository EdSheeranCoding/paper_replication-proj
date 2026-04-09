"""
Metrics for Klein (2021) replication.
Profitability (eq. 7), Optimality Γ (eq. 8), Nash equilibrium share,
and forced-deviation punishment test (Figure 3).
"""

import numpy as np
from train import compute_profit, price_grid


def _profit_matrix(k: int) -> np.ndarray:
    """Precompute profit for all (p_i, p_j) pairs as a (k+1, k+1) matrix."""
    num_prices = k + 1
    P = np.linspace(0, 1, num_prices)
    M = np.zeros((num_prices, num_prices))
    for i in range(num_prices):
        for j in range(num_prices):
            M[i, j] = compute_profit(P[i], P[j])
    return M


def compute_optimal_q(opponent_agent, k: int, gamma: float,
                      max_iter: int = 5000, tol: float = 1e-12,
                      profit_mat: np.ndarray = None) -> np.ndarray:
    """
    Compute Q* for a firm facing the opponent's FIXED learned strategy.

    From p.547: "Q* can be computed exactly by keeping the competitor
    Q-function fixed and looping over all action-state pairs until
    Equation (5) converges."

    For each state s (opponent's last price) and action a (my price):
      - The opponent responds with their greedy policy: a_opp = argmax Q_opp[a, :]
        (since after I set price a, the opponent's state becomes a)
      - Q*(s, a) = π(a, s) + δ·π(a, a_opp) + δ²·max_a' Q*(a_opp, a')
    """
    num_prices = k + 1
    gamma2 = gamma * gamma

    if profit_mat is None:
        profit_mat = _profit_matrix(k)

    Q_star = np.zeros((num_prices, num_prices))

    # Precompute opponent's greedy response to each possible state
    opp_response = np.argmax(opponent_agent.Q, axis=1)  # shape (num_prices,)

    # Precompute immediate and continuation profits as matrices
    # pi_now[s, a] = profit(P[a], P[s]) = profit_mat[a, s]
    pi_now = profit_mat.T  # (num_prices, num_prices) where [s, a] = profit(a, s)

    # pi_cont[a] = profit(P[a], P[opp_response[a]])
    pi_cont = profit_mat[np.arange(num_prices), opp_response]  # shape (num_prices,)

    # Base reward: pi_now[s, a] + gamma * pi_cont[a]  (doesn't change across iterations)
    base = pi_now + gamma * pi_cont[np.newaxis, :]  # (num_prices, num_prices)

    for _ in range(max_iter):
        Q_old = Q_star.copy()

        # max_q_next[a] = max over actions of Q_star[opp_response[a], :]
        max_q_next = np.max(Q_star[opp_response], axis=1)  # shape (num_prices,)

        Q_star = base + gamma2 * max_q_next[np.newaxis, :]

        if np.max(np.abs(Q_star - Q_old)) < tol:
            break

    return Q_star


def profitability(results: list[dict]) -> list[float]:
    """
    Average profit per run over the final 1,000 periods (eq. 7).
    Returns one value per run (mean across both firms).
    """
    return [np.mean(r["final_profit"]) for r in results]


def profitability_per_firm(results: list[dict]) -> list[list[float]]:
    """
    Per-firm profitability for each run. For Figure 2 scatter plots.
    Returns list of [firm0_profit, firm1_profit] per run.
    """
    return [r["final_profit"] for r in results]


def _compute_all_optimality(results: list[dict], k: int, gamma: float):
    """
    Compute optimality ratios for all agents in all runs in a single pass.
    Returns (ratios, q_stars) where:
      - ratios[run_idx] = [Γ_agent0, Γ_agent1]
      - q_stars[(run_idx, agent_idx)] = Q* array (cached for Nash check)
    """
    profit_mat = _profit_matrix(k)
    ratios = []
    q_stars = {}

    for r_idx, r in enumerate(results):
        agents = r["agents"]
        fp = r["final_prices"]
        run_ratios = []

        for i in range(2):
            agent = agents[i]
            opponent = agents[1 - i]
            s = fp[1 - i]
            a = fp[i]

            Q_star = compute_optimal_q(opponent, k, gamma, profit_mat=profit_mat)
            q_stars[(r_idx, i)] = Q_star

            q_learned = agent.Q[s, a]
            q_star_best = np.max(Q_star[s])

            if q_star_best > 0:
                run_ratios.append(q_learned / q_star_best)
            else:
                run_ratios.append(0.0)

        ratios.append(run_ratios)

    return ratios, q_stars


def optimality(results: list[dict], k: int, gamma: float) -> list[list[float]]:
    """
    Optimality ratio Γ for each agent in each run (eq. 8).
    Returns list of [Γ_agent0, Γ_agent1] per run.
    """
    ratios, _ = _compute_all_optimality(results, k, gamma)
    return ratios


def is_nash(result: dict, k: int, gamma: float, tol: float = 1e-5) -> bool:
    """Check if both agents in a run have Γ = 1 (within tolerance)."""
    agents = result["agents"]
    fp = result["final_prices"]
    profit_mat = _profit_matrix(k)

    for i in range(2):
        agent = agents[i]
        opponent = agents[1 - i]
        s = fp[1 - i]
        a = fp[i]

        Q_star = compute_optimal_q(opponent, k, gamma, profit_mat=profit_mat)
        q_learned = agent.Q[s, a]
        q_star_best = np.max(Q_star[s])

        if q_star_best > 0 and abs(q_learned / q_star_best - 1.0) > tol:
            return False
    return True


def nash_share(results: list[dict], k: int, gamma: float,
               tol: float = 1e-5) -> float:
    """Fraction of runs at Nash equilibrium (Γ=1 for both firms)."""
    count = sum(1 for r in results if is_nash(r, k, gamma, tol))
    return count / len(results)


def collusive_nash_runs(results: list[dict], k: int, gamma: float,
                        tol: float = 1e-5) -> list[dict]:
    """
    Filter runs that reached Nash equilibrium at the collusive price.
    Collusive price = 0.5 → index k//2 for even k.
    Used for the punishment test (Figure 3).
    """
    collusive_idx = k // 2  # p=0.5 → index 3 for k=6
    profit_mat = _profit_matrix(k)
    out = []
    for r in results:
        fp = r["final_prices"]
        if fp[0] == collusive_idx and fp[1] == collusive_idx:
            # Inline Nash check with shared profit_mat
            is_n = True
            for i in range(2):
                agent = r["agents"][i]
                opponent = r["agents"][1 - i]
                s = fp[1 - i]
                a = fp[i]
                Q_star = compute_optimal_q(opponent, k, gamma, profit_mat=profit_mat)
                q_learned = agent.Q[s, a]
                q_star_best = np.max(Q_star[s])
                if q_star_best > 0 and abs(q_learned / q_star_best - 1.0) > tol:
                    is_n = False
                    break
            if is_n:
                out.append(r)
    return out


def punishment_test(result: dict, k: int, pre_moves: int = 5,
                    post_moves: int = 15) -> dict:
    """
    Forced-deviation test for a single run (Figure 3).

    Each data point = one sequential move (one firm acts per move),
    matching the paper's Figure 3 exactly. The x-axis counts moves
    relative to the forced deviation at move 0.

    At move 0, firm 0 is forced to undercut by one price step.
    All other moves use greedy (learned) policies.

    Returns market price and two-period profit per firm at each move.
    """
    P = price_grid(k)
    agents = result["agents"]
    collusive_idx = k // 2
    deviate_idx = collusive_idx - 1  # one step below collusive

    market_prices = []
    firm_profits = [[], []]
    movers = []

    # Start both prices at collusive
    p = [collusive_idx, collusive_idx]

    total_moves = pre_moves + 1 + post_moves
    # Firm 0 moves at even steps, firm 1 at odd steps.
    # We want firm 0 to be the mover at the deviation move (t_rel=0),
    # so we need pre_moves % 2 == 0. If odd, add 1 extra pre-move.
    orig_pre = pre_moves
    if pre_moves % 2 != 0:
        pre_moves += 1
        total_moves += 1

    for step in range(total_moves):
        t_rel = step - pre_moves
        mover = step % 2  # 0 or 1
        other = 1 - mover

        state = p[other]

        if t_rel == 0:
            # Forced deviation: firm 0 undercuts
            action = deviate_idx
        else:
            action = agents[mover].greedy_action(state)

        p[mover] = action

        market_prices.append(P[action])
        movers.append(mover)

        for f in range(2):
            firm_profits[f].append(compute_profit(P[p[f]], P[p[1 - f]]))

    periods = list(range(-orig_pre, post_moves + 1))
    # Trim extra pre-move we added for alignment
    if len(periods) < len(market_prices):
        market_prices = market_prices[1:]
        firm_profits[0] = firm_profits[0][1:]
        firm_profits[1] = firm_profits[1][1:]
        movers = movers[1:]
    elif len(periods) > len(market_prices):
        periods = periods[:len(market_prices)]

    return {
        "periods": periods,
        "market_prices": market_prices,
        "firm_profits": firm_profits,
        "movers": movers,
    }


def punishment_test_avg(results: list[dict], k: int, gamma: float,
                        pre_moves: int = 5, post_moves: int = 15) -> dict:
    """
    Average punishment test across all collusive Nash equilibrium runs.
    Returns averaged market price and per-firm profits for Figure 3.
    """
    nash_runs = collusive_nash_runs(results, k, gamma)
    if not nash_runs:
        print("No collusive Nash equilibrium runs found for punishment test.")
        return None

    all_tests = [punishment_test(r, k, pre_moves, post_moves) for r in nash_runs]
    periods = all_tests[0]["periods"]
    movers = all_tests[0]["movers"]  # same for all runs

    avg_market_price = np.mean([t["market_prices"] for t in all_tests], axis=0)
    avg_profit_0 = np.mean([t["firm_profits"][0] for t in all_tests], axis=0)
    avg_profit_1 = np.mean([t["firm_profits"][1] for t in all_tests], axis=0)

    print(f"Punishment test: averaged over {len(nash_runs)} collusive Nash runs")

    return {
        "periods": periods,
        "avg_market_price": avg_market_price,
        "avg_profit_firm0": avg_profit_0,
        "avg_profit_firm1": avg_profit_1,
        "movers": movers,
        "num_runs": len(nash_runs),
    }


def summarise(results: list[dict], k: int, gamma: float, tol: float = 1e-5):
    """Print a summary of Figure 1 metrics. Computes optimality and Nash in one pass."""
    profs = profitability(results)

    # Single pass: compute Q* once per agent, get both optimality and Nash
    ratios, q_stars = _compute_all_optimality(results, k, gamma)

    # Nash check using cached Q* values
    nash_count = 0
    for r_idx, r in enumerate(results):
        fp = r["final_prices"]
        is_n = True
        for i in range(2):
            s = fp[1 - i]
            a = fp[i]
            q_learned = r["agents"][i].Q[s, a]
            q_star_best = np.max(q_stars[(r_idx, i)][s])
            if q_star_best > 0 and abs(q_learned / q_star_best - 1.0) > tol:
                is_n = False
                break
        if is_n:
            nash_count += 1

    nash = nash_count / len(results)
    flat_opts = [g for pair in ratios for g in pair]

    print("=" * 50)
    print("Figure 1 Metrics")
    print("=" * 50)
    print(f"  Profitability (mean):   {np.mean(profs):.4f}")
    print(f"  Profitability (std):    {np.std(profs):.4f}")
    print(f"  Optimality Γ (mean):    {np.mean(flat_opts):.4f}")
    print(f"  Optimality Γ (std):     {np.std(flat_opts):.4f}")
    print(f"  Nash equilibrium share: {nash:.4f} ({nash*100:.1f}%)")
    print(f"  Collusive benchmark:    0.1250")
    print(f"  Competitive benchmark: ~0.0611")
