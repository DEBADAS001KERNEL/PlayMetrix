"""
PlayMetrix - Base Agent Interface (Layer 3: Agent)
Abstract interface defining the lifecycle and contract of any PlayMetrix agent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Agent Layer:
    Decides which action to perform given the current mathematical state.
    Independent of execution logic and validation.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an action given state observation."""
        pass

    @abstractmethod
    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Store experience in memory."""
        pass

    @abstractmethod
    def train_step(self) -> Optional[float]:
        """Perform one gradient update step. Returns loss if updated."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model weights."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load model weights."""
        pass
