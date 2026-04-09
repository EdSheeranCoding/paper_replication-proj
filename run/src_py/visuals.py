"""
Visualisations for Klein (2021) replication.
Reproduces Figures 1-4 (and optionally 8-9 for robustness checks).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from train import train
from metrics import (
    profitability, profitability_per_firm, optimality, nash_share,
    punishment_test_avg, is_nash,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "figures"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Figure 1: Baseline performance vs T
# ============================================================================

def figure1(num_runs: int = 1000, save: bool = True):
    """
    Average Profitability, Average Optimality Γ, and Share with Nash Equilibrium
    under different learning durations T.
    Baseline: k=6, α=0.3, δ=0.95.
    """
    T_values = [1000, 5000, 10_000, 25_000, 50_000, 75_000,
                100_000, 150_000, 200_000, 250_000, 300_000,
                350_000, 400_000, 450_000, 500_000]

    avg_profs = []
    avg_opts = []
    nash_shares = []

    for T in T_values:
        print(f"\n--- Figure 1: T = {T} ---")
        config = Config(T=T, num_runs=num_runs)
        results = train(config=config, num_runs=num_runs)

        profs = profitability(results)
        opts = optimality(results, config.k, config.gamma)
        ns = nash_share(results, config.k, config.gamma)

        avg_profs.append(np.mean(profs))
        flat_opts = [g for pair in opts for g in pair]
        avg_opts.append(np.mean(flat_opts))
        nash_shares.append(ns)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Profitability
    axes[0].plot(T_values, avg_profs, 'o-', color='steelblue', markersize=5)
    axes[0].axhline(y=0.125, color='gray', linestyle='--', label='Joint-Profit Maximizing')
    axes[0].axhline(y=0.0611, color='gray', linestyle=':', label='Competitive Benchmark')
    axes[0].set_ylabel('Average Profitability Π')
    axes[0].set_title('Average Profitability Π')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 0.15)

    # Optimality
    axes[1].plot(T_values, avg_opts, 'o-', color='steelblue', markersize=5)
    axes[1].set_ylabel('Average Optimality Γ')
    axes[1].set_title('Average Optimality Γ')
    axes[1].set_ylim(0, 1.05)

    # Nash share
    axes[2].plot(T_values, [ns * 100 for ns in nash_shares], 'o-',
                 color='steelblue', markersize=5)
    axes[2].set_ylabel('Share with Nash Equilibrium (%)')
    axes[2].set_title('Share with Nash Equilibrium')
    axes[2].set_xlabel('Total Learning Duration T')
    axes[2].set_ylim(0, 105)

    fig.tight_layout()

    if save:
        ensure_output_dir()
        fig.savefig(OUTPUT_DIR / "figure1.png", dpi=150)
        print(f"Saved to {OUTPUT_DIR / 'figure1.png'}")

    plt.show()
    return fig


# ============================================================================
# Figure 2: Joint distribution of profitability
# ============================================================================

def figure2(results: list[dict], k: int = 6, gamma: float = 0.95,
            save: bool = True):
    """
    Joint distribution of per-firm profitability.
    Left: all runs. Right: only runs with Nash equilibrium.
    """
    per_firm = profitability_per_firm(results)
    p0 = [pf[0] for pf in per_firm]
    p1 = [pf[1] for pf in per_firm]

    # Split into Nash vs non-Nash
    nash_mask = [is_nash(r, k, gamma) for r in results]
    p0_nash = [p0[i] for i in range(len(results)) if nash_mask[i]]
    p1_nash = [p1[i] for i in range(len(results)) if nash_mask[i]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All runs
    axes[0].hist2d(p0, p1, bins=30, cmap='Blues')
    axes[0].plot(0.125, 0.125, 's', color='orange', markersize=10,
                 label='Joint-Profit Maximizing')
    axes[0].set_xlabel('Firm 1')
    axes[0].set_ylabel('Firm 2')
    axes[0].set_title('All Runs')
    axes[0].set_xlim(0, 0.15)
    axes[0].set_ylim(0, 0.15)
    axes[0].legend(fontsize=8)

    # Nash runs only
    if p0_nash:
        axes[1].hist2d(p0_nash, p1_nash, bins=30, cmap='Blues')
    axes[1].plot(0.125, 0.125, 's', color='orange', markersize=10,
                 label='Joint-Profit Maximizing')
    axes[1].set_xlabel('Firm 1')
    axes[1].set_title(f'Runs with Nash Equilibrium ({sum(nash_mask)} out of {len(results)})')
    axes[1].set_xlim(0, 0.15)
    axes[1].set_ylim(0, 0.15)
    axes[1].legend(fontsize=8)

    fig.tight_layout()

    if save:
        ensure_output_dir()
        fig.savefig(OUTPUT_DIR / "figure2.png", dpi=150)
        print(f"Saved to {OUTPUT_DIR / 'figure2.png'}")

    plt.show()
    return fig


# ============================================================================
# Figure 3: Reward-punishment after forced deviation
# ============================================================================

def figure3(results: list[dict], k: int = 6, gamma: float = 0.95,
            save: bool = True):
    """
    Average market price and two-period profit after a forced deviation
    by firm 1, for runs that converged to collusive Nash equilibrium.
    """
    pt = punishment_test_avg(results, k, gamma)
    if pt is None:
        print("Cannot generate Figure 3: no collusive Nash runs.")
        return None

    periods = np.array(pt["periods"])
    movers = np.array(pt["movers"])
    prices = np.array(pt["avg_market_price"])
    prof0 = np.array(pt["avg_profit_firm0"])
    prof1 = np.array(pt["avg_profit_firm1"])

    # Masks for which firm moved at each data point
    m0 = movers == 0  # firm 0 (deviator) moved
    m1 = movers == 1  # firm 1 moved

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Market price — different markers per firm so you can see the alternation
    axes[0].plot(periods, prices, '-', color='gray', linewidth=0.8, alpha=0.5)
    axes[0].plot(periods[m0], prices[m0], 'o', color='steelblue',
                 markersize=6, label='Firm 1 (Deviator) moves')
    axes[0].plot(periods[m1], prices[m1], '^', color='orange',
                 markersize=6, label='Firm 2 moves')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Average Market Price')
    axes[0].set_title('Average Market Price After Forced Deviation')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.0)

    # Profits — same marker distinction
    axes[1].plot(periods, prof0, '-', color='steelblue', linewidth=0.8, alpha=0.5)
    axes[1].plot(periods[m0], prof0[m0], 'o', color='steelblue',
                 markersize=6, label='Firm 1 (Deviator)')
    axes[1].plot(periods[m1], prof0[m1], 'o', color='steelblue',
                 markersize=4, alpha=0.3)
    axes[1].plot(periods, prof1, '-', color='orange', linewidth=0.8, alpha=0.5)
    axes[1].plot(periods[m1], prof1[m1], '^', color='orange',
                 markersize=6, label='Firm 2')
    axes[1].plot(periods[m0], prof1[m0], '^', color='orange',
                 markersize=4, alpha=0.3)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Move Relative to Forced Deviation')
    axes[1].set_ylabel('Average Two-Period Profit')
    axes[1].set_title('Average Two-Period Profit After Forced Deviation')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 0.20)

    fig.tight_layout()

    if save:
        ensure_output_dir()
        fig.savefig(OUTPUT_DIR / "figure3.png", dpi=150)
        print(f"Saved to {OUTPUT_DIR / 'figure3.png'}")

    plt.show()
    return fig


# ============================================================================
# Figure 4: Performance under different k
# ============================================================================

def figure4(num_runs: int = 1000, save: bool = True):
    """
    Performance under different amounts of pricing intervals k (6, 12, 24)
    and learning durations T.
    """
    k_values = [6, 12, 24]
    colors = ['steelblue', 'orange', 'green']
    T_values = [1000, 5000, 10_000, 25_000, 50_000, 75_000,
                100_000, 150_000, 200_000, 250_000, 300_000,
                350_000, 400_000, 450_000, 500_000]

    all_profs = {}
    all_opts = {}
    all_nash = {}

    for k in k_values:
        all_profs[k] = []
        all_opts[k] = []
        all_nash[k] = []
        for T in T_values:
            print(f"\n--- Figure 4: k={k}, T={T} ---")
            config = Config(T=T, k=k, num_runs=num_runs)
            results = train(config=config, num_runs=num_runs)

            profs = profitability(results)
            opts = optimality(results, k, config.gamma)
            ns = nash_share(results, k, config.gamma)

            all_profs[k].append(np.mean(profs))
            flat_opts = [g for pair in opts for g in pair]
            all_opts[k].append(np.mean(flat_opts))
            all_nash[k].append(ns)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for k, color in zip(k_values, colors):
        axes[0].plot(T_values, all_profs[k], 'o-', color=color,
                     markersize=4, label=f'k={k}')
        axes[1].plot(T_values, all_opts[k], 'o-', color=color,
                     markersize=4, label=f'k={k}')
        axes[2].plot(T_values, [ns * 100 for ns in all_nash[k]], 'o-',
                     color=color, markersize=4, label=f'k={k}')

    axes[0].axhline(y=0.125, color='gray', linestyle='--', label='Joint-Profit Maximizing')
    axes[0].axhline(y=0.0611, color='gray', linestyle=':', label='Competitive Benchmark')
    axes[0].set_ylabel('Average Profitability Π')
    axes[0].set_title('Average Profitability Π')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 0.15)

    axes[1].set_ylabel('Average Optimality Γ')
    axes[1].set_title('Average Optimality Γ')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)

    axes[2].set_ylabel('Share with Nash Equilibrium (%)')
    axes[2].set_title('Share with Nash Equilibrium')
    axes[2].set_xlabel('Total Learning Duration T')
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 105)

    fig.tight_layout()

    if save:
        ensure_output_dir()
        fig.savefig(OUTPUT_DIR / "figure4.png", dpi=150)
        print(f"Saved to {OUTPUT_DIR / 'figure4.png'}")

    plt.show()
    return fig


# ============================================================================
# Figure 8: Robustness — varying step-size α
# ============================================================================

def figure8(num_runs: int = 1000, save: bool = True):
    """Performance under different step-size parameters α."""
    alpha_values = np.arange(0.05, 1.01, 0.05)
    T_configs = [100_000, 500_000]
    colors = ['steelblue', 'orange']

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for T, color in zip(T_configs, colors):
        profs, opts, nashes = [], [], []
        for alpha in alpha_values:
            print(f"\n--- Figure 8: α={alpha:.2f}, T={T} ---")
            config = Config(T=T, lr=alpha, num_runs=num_runs)
            results = train(config=config, num_runs=num_runs)

            profs.append(np.mean(profitability(results)))
            o = optimality(results, config.k, config.gamma)
            opts.append(np.mean([g for pair in o for g in pair]))
            nashes.append(nash_share(results, config.k, config.gamma))

        axes[0].plot(alpha_values, profs, 'o-', color=color, markersize=4,
                     label=f'T={T:,}')
        axes[1].plot(alpha_values, opts, 'o-', color=color, markersize=4,
                     label=f'T={T:,}')
        axes[2].plot(alpha_values, [n * 100 for n in nashes], 'o-',
                     color=color, markersize=4, label=f'T={T:,}')

    axes[0].axhline(y=0.125, color='gray', linestyle='--', label='Joint-Profit Maximizing')
    axes[0].axhline(y=0.0611, color='gray', linestyle=':', label='Competitive Benchmark')
    axes[0].set_title('Average Profitability Π')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 0.15)

    axes[1].set_title('Average Optimality Γ')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)

    axes[2].set_title('Share with Nash Equilibrium')
    axes[2].set_xlabel('Stepsize Parameter α')
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 105)

    fig.tight_layout()

    if save:
        ensure_output_dir()
        fig.savefig(OUTPUT_DIR / "figure8.png", dpi=150)
        print(f"Saved to {OUTPUT_DIR / 'figure8.png'}")

    plt.show()
    return fig


# ============================================================================
# Figure 9: Robustness — varying discount factor δ
# ============================================================================

def figure9(num_runs: int = 1000, save: bool = True):
    """Performance under different discount factors δ."""
    delta_values = np.arange(0.60, 1.001, 0.025)

    profs, opts, nashes = [], [], []

    for delta in delta_values:
        print(f"\n--- Figure 9: δ={delta:.3f} ---")
        config = Config(gamma=delta, num_runs=num_runs)
        results = train(config=config, num_runs=num_runs)

        profs.append(np.mean(profitability(results)))
        o = optimality(results, config.k, delta)
        opts.append(np.mean([g for pair in o for g in pair]))
        nashes.append(nash_share(results, config.k, delta))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(delta_values, profs, 'o-', color='steelblue', markersize=5)
    axes[0].axhline(y=0.125, color='gray', linestyle='--', label='Joint-Profit Maximizing')
    axes[0].axhline(y=0.0611, color='gray', linestyle=':', label='Competitive Benchmark')
    axes[0].set_title('Average Profitability Π')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 0.15)

    axes[1].plot(delta_values, opts, 'o-', color='steelblue', markersize=5)
    axes[1].set_title('Average Optimality Γ')
    axes[1].set_ylim(0, 1.05)

    axes[2].plot(delta_values, [n * 100 for n in nashes], 'o-',
                 color='steelblue', markersize=5)
    axes[2].set_title('Share with Nash Equilibrium')
    axes[2].set_xlabel('Discount factor δ')
    axes[2].set_ylim(0, 105)

    fig.tight_layout()

    if save:
        ensure_output_dir()
        fig.savefig(OUTPUT_DIR / "figure9.png", dpi=150)
        print(f"Saved to {OUTPUT_DIR / 'figure9.png'}")

    plt.show()
    return fig


# ============================================================================
# Run all baseline figures
# ============================================================================

def run_baseline(num_runs: int = 1000):
    """Run the baseline experiment and generate Figures 1-3."""
    ensure_output_dir()

    # Baseline run at T=500,000
    print("=" * 60)
    print("Running baseline: k=6, α=0.3, δ=0.95, T=500,000")
    print("=" * 60)
    config = Config()
    results = train(config=config, num_runs=num_runs)

    # Figures 2 and 3 use the baseline results
    figure2(results, k=config.k, gamma=config.gamma)
    figure3(results, k=config.k, gamma=config.gamma)

    # Figure 1 needs multiple T values (runs its own training)
    figure1(num_runs=num_runs)

    return results


if __name__ == "__main__":
    # Quick test with small num_runs
    run_baseline(num_runs=10)
