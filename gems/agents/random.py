"""RandomAgent: picks uniformly from legal actions using provided RNG."""
from typing import Optional, Sequence, Any
import random

from .core import Agent
from ..actions import Action
from ..state import GameState


class RandomAgent(Agent):
  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: Optional[float] = None) -> Action:
    if not legal_actions:
      raise ValueError("No legal actions available")
    return self.rng.choice(legal_actions)


__all__ = ["RandomAgent"]
