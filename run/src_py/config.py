from __future__ import annotations
from typing import TypeAlias, Optional
from dataclasses import dataclass

@dataclass
class Config:
    T: int = 500_000
    num_runs: int = 1000
    k: int = 6
    gamma: float = 0.95
    lr: float = 0.3

