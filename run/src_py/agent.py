"""
Tabular Q-learning agent for Klein (2021) sequential pricing game.
"""

import numpy as np
from config import Config


class Agent:
    def __init__(self, id: int, config: Config = None, seed: int = 42):
        if config is None:
            config = Config()

        self.id = id
        self.num_prices = config.k + 1          # |P| = k+1 prices: {0, 1/k, ..., 1}
        self.gamma = config.gamma               # discount factor δ
        self.lr = config.lr                     # learning rate α

        # Epsilon-greedy with exponential decay: ε_t = (1-θ)^t
        # At T/2, ε = 0.001 (0.1%). At T, ε = 0.000001 (0.0001%).
        # (1-θ)^(T/2) = 0.001 → θ = 1 - 0.001^(2/T)
        self.eps = 1.0
        self.eps_decay = 1 - 0.001 ** (2 / config.T)

        # Q-table: rows = state (opponent's price index), cols = action (own price index)
        self.Q = np.zeros((self.num_prices, self.num_prices))

        self.last_action = None
        self.last_profit = None
        self.last_state = None

        self.rng = np.random.default_rng(seed)

    def get_action(self, state: int) -> int:
        """ε-greedy action selection (eq. 6). State = opponent's price index."""
        if self.rng.random() < self.eps:
            return self.rng.integers(0, self.num_prices)
        else:
            # Ties broken randomly (paper: "randomizes over all perceived optimal actions")
            max_q = np.max(self.Q[state])
            best_actions = np.where(self.Q[state] == max_q)[0]
            return self.rng.choice(best_actions)

    def greedy_action(self, state: int) -> int:
        """Pure greedy action (no exploration). For post-training analysis."""
        max_q = np.max(self.Q[state])
        best_actions = np.where(self.Q[state] == max_q)[0]
        return self.rng.choice(best_actions)

    def update(self, state: int, action: int, profit: float,
               continuation_profit: float, next_state: int):
        """
        Q-learning update (eq. 5).

        Q(s, a) ← (1-α)·Q(s, a) + α·[π + δ·π_cont + δ²·max_a' Q(s', a')]

        - profit: π earned when action was taken
        - continuation_profit: π earned next period while price persists
        - next_state: opponent's price when this agent next moves
        """
        target = (
            profit
            + self.gamma * continuation_profit
            + self.gamma ** 2 * np.max(self.Q[next_state])
        )
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def decay_epsilon(self):
        """Decay exploration rate: ε *= (1 - θ)."""
        self.eps *= (1 - self.eps_decay)

    def reset(self, seed: int = None):
        """Reset agent for a new run."""
        self.Q = np.zeros((self.num_prices, self.num_prices))
        self.eps = 1.0
        self.last_action = None
        self.last_profit = None
        self.last_state = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
