"""GreedyAgent and a tiny quick_score heuristic.

GreedyAgent uses quick_score to evaluate legal actions and picks the best.
"""

from typing import Sequence, Any, Optional

from .core import Agent
from ..actions import Action
from ..state import GameState


def quick_score(state: GameState, seat_id: int, action: Action) -> float:
  """Quick, cheap heuristic for GreedyAgent.

  This is intentionally minimal: it returns 0.0 for unknown actions. A
  repository-specific heuristic can replace or extend this function.
  """
  # TODO: implement a domain-specific heuristic using engine accessors
  return 0.0


class GreedyAgent(Agent):
  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: Optional[float] = None) -> Action:
    if not legal_actions:
      raise ValueError("No legal actions available")
    best = None
    best_score = float('-inf')
    for a in legal_actions:
      s = quick_score(state, self.seat_id, a)
      if s > best_score:
        best_score = s
        best = a
    # At this point `best` is guaranteed to be set because `legal_actions`
    # is non-empty, but help the type-checker by asserting not None.
    assert best is not None
    return best


__all__ = ["GreedyAgent", "quick_score"]
