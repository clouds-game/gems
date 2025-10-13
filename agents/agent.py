"""Agent implementations for the gems engine.

This module follows the contract described in the repository's AGENTS.md.

Classes:
  Agent: base class
  RandomAgent: picks uniformly from legal actions using the provided RNG
  GreedyAgent: fast heuristic-based agent using quick_score

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


class RandomAgent(Agent):
  def act(self, state: Any, legal_actions: Sequence[Any], *, timeout: Optional[float] = None) -> Any:
    if not legal_actions:
      raise ValueError("No legal actions available")
    return self.rng.choice(legal_actions)


def quick_score(state: Any, seat_id: int, action: Any) -> float:
  """Quick, cheap heuristic for GreedyAgent.

  This is intentionally minimal: it returns 0.0 for unknown actions. A
  repository-specific heuristic can replace or extend this function.
  """
  # TODO: implement a domain-specific heuristic using engine accessors
  return 0.0


class GreedyAgent(Agent):
  def act(self, state: Any, legal_actions: Sequence[Any], *, timeout: Optional[float] = None) -> Any:
    if not legal_actions:
      raise ValueError("No legal actions available")
    best = None
    best_score = float('-inf')
    for a in legal_actions:
      s = quick_score(state, self.seat_id, a)
      if s > best_score:
        best_score = s
        best = a
    return best


__all__ = ["Agent", "RandomAgent", "GreedyAgent", "quick_score"]
