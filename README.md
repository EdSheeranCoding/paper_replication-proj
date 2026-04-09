# Algorithmic Collusion via Q-Learning

Replication of Klein (2021), *"Autonomous algorithmic collusion: Q-learning under sequential pricing"* (The RAND Journal of Economics).

This is a day project I did to practice my cuda skills and to introduce myself to MARL (lightly). I hope to contribute more paper replications to this repo in the future.

This project includes:
- Aa Python implementation of the sequential pricing duopoly environment with tabular Q-learning agents, epsilon-greedy exploration with exponential decay, and the paper's two-period Q-update rule
- A CUDA implementation that parallelizes all 1,000 independent training runs across GPU threads.
- 
- Figure replication for Figures 1-9
- A punishment analysis pipeline that forces a deviation from the learned collusive price and tracks the opponent's retaliatory response over subsequent moves. (see Fig. 3 p.12)
- The paper.

## Project structure

```
run/
  src_py/          Python implementation
    main.py        CLI entry point
    agent.py       Tabular Q-learning agent
    train.py       Training loop & environment
    metrics.py     Profitability, optimality, Nash equilibrium checks
    visuals.py     Figure generation (replicates Figures 1-4, 8-9)
    config.py      Hyperparameter configuration
  src_cu/          CUDA implementation for GPU-accelerated parallel runs
    collusion.cu   CUDA kernel (all runs execute in parallel)
    analyze.py     Post-processing for CUDA results
    Makefile
  outputs/         Saved results and generated figures
docs/              Paper PDF, research notes, and project context
```

## Getting started

```bash
pip install -r run/requirements.txt

# Quick test (3 runs)
python run/src_py/main.py

# Full replication (1000 runs, matches paper)
python run/src_py/main.py --num-runs 1000

# Generate figures
python run/src_py/main.py --figures

# Load saved results
python run/src_py/main.py --load run/outputs/results.npz --summary
python run/src_py/main.py --load run/outputs/results.npz --punish
```

For GPU acceleration (optional, requires CUDA toolkit):

```bash
cd run/src_cu
make
make run-full
```

## Reference

Klein, T. (2021). Autonomous algorithmic collusion: Q-learning under sequential pricing. *The RAND Journal of Economics*, 52(3), 538-558.


# AI Use Discolsure
Claude Opus 4.5 was used for spell checking the README, formating code and light bug fixing.