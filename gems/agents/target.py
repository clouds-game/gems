from collections.abc import Sequence
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import asdict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from gems.typings import Card, Gem, GemList
from .core import AGENT_METADATA_HISTORY_ROUND, Agent
from ..actions import Action, NoopAction, Take3Action, Take2Action, BuyCardAction, ReserveCardAction
from ..state import GameState


@pydantic_dataclass(frozen=True)
class BaseConfig():
  pass


C = TypeVar("C", bound=BaseConfig)


@pydantic_dataclass(frozen=True)
class BaseEvaluation(ABC, Generic[C]):
  config: C

  @abstractmethod
  def quick_score(self, state: GameState, seat_id: int, target_card: Card | None, action: Action) -> float:
    pass


@pydantic_dataclass(frozen=True)
class TargetAgentEvaluationV1Config(BaseConfig):
  # 每个宝石的基础分数
  gem_score: float = 10.0
  # 每个金宝石的基础分数
  gold_score: float = 15.0
  # 每张卡牌额外分数
  extra_point_per_card: float = 1.0
  # 每分基础分数
  point_score: float = 50.0
  # 减费奖励分数
  bonus_score: float = 5.0
  # 金宝石花费减分
  gold_cost_score: float = 2.0
  # 普通宝石花费减分
  gem_cost_score: float = 2.0


@pydantic_dataclass(frozen=True)
class TargetAgentEvaluationV1(BaseEvaluation[TargetAgentEvaluationV1Config]):
  config: TargetAgentEvaluationV1Config = TargetAgentEvaluationV1Config()

  def _gem_extra_score(self, player_gems: GemList, cost: GemList, gems: Sequence[Gem]) -> float:
    score = 0.0
    total_cost_num = cost.count()
    # 如果目标卡牌需求的宝石，玩家当前不满足，则提高分数
    # 缺的越多分数越高 （因为更需要这些宝石）
    # 缺的宝石总需求量越高分数越高
    for g in gems:
      if (g_cost := cost.get(g)) > 0 and (g_have := player_gems.get(g)) < g_cost:
        score += (g_cost - g_have) * g_cost / total_cost_num * 5
    return score

  def quick_score(self, state: GameState, seat_id: int, target_card: Card | None, action: Action) -> float:
    player = state.players[seat_id]
    cost = target_card.cost if target_card is not None else GemList()

    if isinstance(action, Take3Action):
      get_num = len(action.gems)
      drop_num = action.ret.count() if action.ret else 0
      score = (get_num - drop_num) * self.config.gem_score
      score += self._gem_extra_score(player.gems, cost, action.gems)
      if action.ret:
        score -= self._gem_extra_score(player.gems, cost, action.ret.flatten())
      return score
    elif isinstance(action, Take2Action):
      get_num = action.count
      drop_num = action.ret.count() if action.ret else 0
      score = (get_num - drop_num) * self.config.gem_score
      score += self._gem_extra_score(player.gems, cost, [action.gem] * get_num)
      if action.ret:
        score -= self._gem_extra_score(player.gems, cost, action.ret.flatten())
      return score
    elif isinstance(action, BuyCardAction):
      card = action.card
      assert card is not None
      score = (card.points + self.config.extra_point_per_card) * \
          self.config.point_score + (self.config.bonus_score if card.bonus is not None else 0)
      payment_cost = action.payment.get(Gem.GOLD) * self.config.gold_cost_score + \
          sum(action.payment.get(g) for g in Gem if g != Gem.GOLD) * self.config.gem_cost_score
      return float(score) - float(payment_cost)
    elif isinstance(action, ReserveCardAction):
      score = 0
      if action.take_gold:
        score += self.config.gold_score
      if action.ret:
        score -= self.config.gem_score
        score -= self._gem_extra_score(player.gems, cost, [action.ret])
      return score
    elif isinstance(action, NoopAction):
      return -100.0
    return 0.0


class TargetAgent(Agent):
  evaluation: BaseEvaluation
  target_card: Card | None = None
  card_history: list[Card | None] = []
  debug: bool = False

  def __init__(self, seat_id: int, seed: int | None = None, evaluation: BaseEvaluation = TargetAgentEvaluationV1(), debug: bool = False):
    super().__init__(seat_id, seed=seed)
    self.evaluation = evaluation
    self.debug = debug

  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:

    if not legal_actions:
      raise ValueError("No legal actions available")
    self.update(state)
    self.card_history.append(self.target_card)

    action_score = [
        (a, self.evaluation.quick_score(state, self.seat_id, self.target_card, a)) for a in legal_actions
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

    This method is called at the beginning of function `act`. The agent can
    use this opportunity to choose a new target card.
    """
    if self.target_card is None:
      visible_cards = list(state.visible_cards) + list(state.players[self.seat_id].reserved_cards)
      if self.target_card not in visible_cards:
        self.target_card = self.rng.choice(visible_cards) if len(visible_cards) > 0 else None

  def _metadata(self) -> dict:
    return {
        # "card_history": self.card_history,
        AGENT_METADATA_HISTORY_ROUND: [str(card) if card else "None" for card in self.card_history],
        "evaluation": self.evaluation.__class__.__name__,
        "evaluation_config": asdict(self.evaluation.config),
    }


__all__ = ["TargetAgent", "TargetAgentEvaluationV1", "TargetAgentEvaluationV1Config"]
