"""RandomAgent: picks uniformly from legal actions using provided RNG."""
from typing import Optional, Sequence, Any
import random

from .core import Agent


class RandomAgent(Agent):
  def act(self, state: Any, legal_actions: Sequence[Any], *, timeout: Optional[float] = None) -> Any:
    if not legal_actions:
      raise ValueError("No legal actions available")
    return self.rng.choice(legal_actions)


__all__ = ["RandomAgent"]
