from collections.abc import Sequence
from random import Random
from gems.typings import Card, Gem, GemList

from .core import AGENT_METADATA_HISTORY_ROUND, Agent
from ..actions import Action, NoopAction, Take3Action, Take2Action, BuyCardAction, ReserveCardAction
from ..state import GameState


def gem_extra_score(player_gems: GemList, cost: GemList, gems: Sequence[Gem]) -> float:
  score = 0.0
  total_cost_num = cost.count()
  # 如果目标卡牌需求的宝石，玩家当前不满足，则提高分数
  # 缺的越多分数越高 （因为更需要这些宝石）
  # 缺的宝石总需求量越高分数越高
  for g in gems:
    if (g_cost := cost.get(g)) > 0 and (g_have := player_gems.get(g)) < g_cost:
      score += (g_cost - g_have) * g_cost / total_cost_num * 5
  return score


def quick_score(state: GameState, seat_id: int, target_card: Card | None, action: Action) -> float:

  player = state.players[seat_id]
  cost = target_card.cost if target_card is not None else GemList()

  if isinstance(action, Take3Action):
    get_num = len(action.gems)
    drop_num = action.ret.count() if action.ret else 0
    score = (get_num - drop_num) * 10
    score += gem_extra_score(player.gems, cost, action.gems)
    if action.ret:
      score -= gem_extra_score(player.gems, cost, action.ret.flatten())
    return score
  elif isinstance(action, Take2Action):
    get_num = action.count
    drop_num = action.ret.count() if action.ret else 0
    score = (get_num - drop_num) * 10
    score += gem_extra_score(player.gems, cost, [action.gem] * get_num)
    if action.ret:
      score -= gem_extra_score(player.gems, cost, action.ret.flatten())
    return score
  elif isinstance(action, BuyCardAction):
    card = action.card
    assert card is not None
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
      score -= gem_extra_score(player.gems, cost, [action.ret])
    return score
  elif isinstance(action, NoopAction):
    return -100.0
  return 0.0


class TargetAgent(Agent):
  target_card: Card | None = None
  card_history: list[Card | None] = []
  debug: bool = False

  def __init__(self, seat_id: int, rng: Random | None = None, debug: bool = False):
    super().__init__(seat_id, rng)
    self.debug = debug

  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:

    if not legal_actions:
      raise ValueError("No legal actions available")
    self.update(state)
    self.card_history.append(self.target_card)

    action_score = [
        (a, quick_score(state, self.seat_id, self.target_card, a)) for a in legal_actions
    ]
    best_score = max(score for _, score in action_score)
    best_actions = [a for a, score in action_score if score == best_score]

    best = self.rng.choice(best_actions)
    # At this point `best` is guaranteed to be set because `legal_actions`
    # is non-empty, but help the type-checker by asserting not None.
    assert best is not None

    if self.debug:
      print(f"[TargetAgent] seat_id={self.seat_id} target_card={self.target_card}")
      print("  Legal actions and scores:")
      action_score.sort(key=lambda x: x[1], reverse=True)
      for a, score in action_score:
        print(f"    Action: {a}, Score: {score}")
    return best

  def _reset(self) -> None:
    self.target_card = None
    self.card_history = []

  def update(self, state: GameState) -> None:
    """Update the agent's internal state.

    This method is called after each game state update. The agent can
    use this opportunity to choose a new target card.
    """
    if self.target_card is None or self.target_card not in state.visible_cards:
      visible_cards = list(state.visible_cards) + list(state.players[self.seat_id].reserved_cards)
      if self.target_card not in visible_cards:
        self.target_card = self.choose_target_card(state)

  def choose_target_card(self, state: GameState) -> Card | None:
    """Choose a target card from the currently visible cards.

    Selects uniformly at random from `state.visible_cards` using the
    agent's `self.rng`. If no visible cards are present the target is set
    to None.
    """
    visible = list(state.visible_cards) + list(state.players[self.seat_id].reserved_cards)
    if not visible:
      return None
    # Use the agent's RNG for determinism when seeded
    return self.rng.choice(visible)

  def metadata(self) -> dict:
    return {
        "type": self.__class__.__name__,
        "seat_id": self.seat_id,
        # "card_history": self.card_history,
        AGENT_METADATA_HISTORY_ROUND : [str(card) if card else "None" for card in self.card_history],
    }


__all__ = ["TargetAgent", "quick_score"]
