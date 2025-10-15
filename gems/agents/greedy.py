"""GreedyAgent and a tiny quick_score heuristic.

GreedyAgent uses quick_score to evaluate legal actions and picks the best.
"""

from collections.abc import Sequence

from gems.typings import Gem

from .core import Agent
from ..actions import Action, NoopAction, Take3Action, Take2Action, BuyCardAction, ReserveCardAction
from ..state import GameState


def quick_score(state: GameState, seat_id: int, action: Action) -> float:
  """Quick, cheap heuristic for GreedyAgent.

  This is intentionally minimal: it returns 0.0 for unknown actions. A
  repository-specific heuristic can replace or extend this function.
  """
  # TODO: implement a domain-specific heuristic using engine accessors
  player = state.players[seat_id]

  if isinstance(action, Take3Action):
    get_num = len(action.gems)
    drop_num = action.ret.count() if action.ret else 0
    return (get_num - drop_num) * 10
  elif isinstance(action, Take2Action):
    get_num = action.count
    drop_num = action.ret.count() if action.ret else 0
    return (get_num - drop_num) * 10
  elif isinstance(action, BuyCardAction):
    card = action.card
    score = (card.points + 1) * 50 + (5 if card.bonus is not None else 0)
    payment_cost = action.payment.get(Gem.GOLD) * 2 + \
        sum(action.payment.get(g) for g in Gem if g != Gem.GOLD)
    return float(score) - float(payment_cost)
  elif isinstance(action, ReserveCardAction):
    score = 0
    if action.take_gold:
      score += 15
    if action.ret:
      score -= 10
    return score
  elif isinstance(action, NoopAction):
    return -100.0
  return 0.0


class GreedyAgent(Agent):
  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:
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
