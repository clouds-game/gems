"""GreedyAgent and a tiny quick_score heuristic.

GreedyAgent uses quick_score to evaluate legal actions and picks the best.
"""

from typing import Sequence, Any, Optional

from .core import Agent


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


__all__ = ["GreedyAgent", "quick_score"]
