"""Core agent base class for gems agents.

Follows the contract described in AGENTS.md. Keep this file minimal.
All Python code in this repo uses 2-space indentation.
"""
import random
from typing import Optional, Sequence, Any


class Agent:
  def __init__(self, seat_id: int, rng: Optional[random.Random] = None):
    self.seat_id = seat_id
    # Use a local RNG instance to guarantee reproducible behavior
    self.rng = rng or random.Random()

  def reset(self, seed: Optional[int] = None) -> None:
    if seed is not None:
      self.rng.seed(seed)

  def observe(self, state: Any) -> None:
    # optional hook for receiving state updates
    pass

  def act(self, state: Any, legal_actions: Sequence[Any], *, timeout: Optional[float] = None) -> Any:
    """Return one element from legal_actions.

    Must be overridden by subclasses.
    """
    raise NotImplementedError()


__all__ = ["Agent"]
