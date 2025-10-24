"""GreedyAgent and a tiny quick_score heuristic.

GreedyAgent uses quick_score to evaluate legal actions and picks the best.
"""

from collections.abc import Sequence
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import asdict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from gems.typings import Gem
from .core import Agent
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
  def quick_score(self, state: GameState, seat_id: int, action: Action) -> float:
    pass


@pydantic_dataclass(frozen=True)
class GreedyAgentEvaluationV1Config(BaseConfig):
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
class GreedyAgentEvaluationV1(BaseEvaluation[GreedyAgentEvaluationV1Config]):
  config: GreedyAgentEvaluationV1Config = GreedyAgentEvaluationV1Config()

  def quick_score(self, state: GameState, seat_id: int, action: Action) -> float:
    """Quick, cheap heuristic for GreedyAgent.

    This is intentionally minimal: it returns 0.0 for unknown actions. A
    repository-specific heuristic can replace or extend this function.
    """
    # TODO: implement a domain-specific heuristic using engine accessors
    player = state.players[seat_id]

    if isinstance(action, Take3Action):
      get_num = len(action.gems)
      drop_num = action.ret.count() if action.ret else 0
      return (get_num - drop_num) * self.config.gem_score
    elif isinstance(action, Take2Action):
      get_num = action.count
      drop_num = action.ret.count() if action.ret else 0
      return (get_num - drop_num) * self.config.gem_score
    elif isinstance(action, BuyCardAction):
      card = action.card
      assert card is not None
      score = (card.points + self.config.extra_point_per_card) * self.config.point_score + \
          (self.config.bonus_score if card.bonus is not None else 0)
      payment_cost = action.payment.get(Gem.GOLD) * self.config.gold_cost_score + \
          sum(action.payment.get(g) for g in Gem if g != Gem.GOLD) * self.config.gem_cost_score
      return float(score) - float(payment_cost)
    elif isinstance(action, ReserveCardAction):
      score = 0
      if action.take_gold:
        score += self.config.gold_score
      if action.ret:
        score -= self.config.gem_score
      return score
    elif isinstance(action, NoopAction):
      return -100.0
    return 0.0


class GreedyAgent(Agent):
  evaluation: BaseEvaluation

  def __init__(self, seat_id: int, seed: int | None = None, evaluation: BaseEvaluation = GreedyAgentEvaluationV1(), debug: bool = False):
    super().__init__(seat_id, seed=seed)
    self.evaluation = evaluation

  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:
    if not legal_actions:
      raise ValueError("No legal actions available")

    action_score = [
        (a, self.evaluation.quick_score(state, self.seat_id, a)) for a in legal_actions
    ]
    best_score = max(score for _, score in action_score)
    best_actions = [a for a, score in action_score if score == best_score]

    best = self.rng.choice(best_actions)

    # At this point `best` is guaranteed to be set because `legal_actions`
    # is non-empty, but help the type-checker by asserting not None.
    assert best is not None
    return best

  def _metadata(self) -> dict:
    return {
        "evaluation": self.evaluation.__class__.__name__,
        "evaluation_config": asdict(self.evaluation.config),
    }


__all__ = ["GreedyAgent", "GreedyAgentEvaluationV1Config", "GreedyAgentEvaluationV1"]
