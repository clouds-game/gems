"""Core agent base class for gems agents.

Follows the contract described in AGENTS.md. Keep this file minimal.
All Python code in this repo uses 2-space indentation.
"""
from dataclasses import asdict, dataclass
from pydantic import BaseModel, field_validator
import random
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

from ..state import PlayerState, GameState
from ..actions import Action

AGENT_METADATA_HISTORY_ROUND = "history_round"
def AGENT_SEED_GENERATOR(): return random.Random().randint(0, 2**31 - 1)


@dataclass
class Agent:
  seat_id: int
  name: str

  agent_name_to_cls: ClassVar[dict[str, type["Agent"]]] = {}

  def __init__(self, seat_id: int, *, seed: int | None = None, name: str | None) -> None:
    self.seat_id = seat_id
    self.name = name if name is not None else self.__class__.__name__
    # Use a local RNG instance to guarantee reproducible behavior
    if seed is None:
      seed = AGENT_SEED_GENERATOR()
    self._seed = seed
    self.rng = random.Random(seed)

  @classmethod
  def __init_subclass__(cls):
    # 将子类添加到注册表
    cls.agent_name_to_cls[cls.__name__] = cls
    super().__init_subclass__()

  def reset(self, seed: int | None = None) -> None:
    if seed is None:
      seed = AGENT_SEED_GENERATOR()
    self._seed = seed
    self.rng.seed(seed)
    self._reset()

  def _reset(self) -> None:
    """Internal reset hook called before each game.

    Subclasses may override this to reset any internal state.
    Default implementation is a no-op.
    """
    pass

  def observe(self, player: PlayerState, state: GameState) -> None:
    """Optional hook: receive an update about `player` and the full `state`.

    Default implementation is a no-op. Subclasses may override this to
    maintain lightweight per-player bookkeeping.
    """
    pass

  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:
    """Return one element from `legal_actions`.

    Subclasses must return one of the provided `legal_actions` (by identity
    or value). Use `timeout` for interruption-aware planners.
    """
    raise NotImplementedError()

  def metadata(self) -> dict:
    """Return optional metadata about the agent's internal state.

    This is recorded during simulations for later analysis.
    """

    metadata = {
        "type": self.__class__.__name__,
        "seat_id": self.seat_id,
        "seed": self._seed,
    }
    if (extra := self._metadata()):
      metadata.update(extra)
    return metadata

  def _metadata(self) -> dict:
    """
    Internal hook for subclasses to provide extra metadata.
    Default implementation returns an empty dict. Subclasses may override.
    """
    return {}

  @classmethod
  def print_metadata_round(cls, agents_metadata: list[dict[str, str]], round: int) -> None:
    print("Agent Metadata:")
    for data in agents_metadata:
      if (history := data.get(AGENT_METADATA_HISTORY_ROUND)):
        if round < len(history):
          type_name = data.get("type", "Agent")
          seat_id = data.get("seat_id", "unknown")
          print(f"  [{type_name}] seat_id={seat_id} {history[round]})")


BaseAgent = TypeVar('BaseAgent', bound=Agent)


class AgentBuilder(BaseModel):
  """Simple factory for constructing agents used in simulations.
  """

  cls_name: str
  seat_id: int
  name: str | None = None
  kwargs: dict | None = None  # evaluation=GreedyAgentEvaluationV1()
  config: dict = {}  # gem_score=1.0, ...

  @field_validator('kwargs', mode='after')  # mode='after' 确保在类型验证后执行
  def set_kwargs_to_None(cls, v):
    # 强制将 kwargs 设为 None，忽略 JSON 中的值
    if v is not None:
      evaluation = v.get("evaluation")
      if evaluation is not None and hasattr(evaluation, 'config'):
        return v
    return None

  def build(self) -> Agent:
    """Instantiate the configured agent.

    Returns an instance of the configured agent class. `seat_id` and `name`
    are forwarded as keyword arguments, together with any extra `kwargs`.
    """
    if self.kwargs is None:
      raise ValueError("AgentBuilder.kwargs must be set to build an agent")
    agent_cls = Agent.agent_name_to_cls.get(self.cls_name)
    if agent_cls is None:
      raise ValueError(f"Unknown agent class name: {self.cls_name}")
    return agent_cls(seat_id=self.seat_id, name=self.name, **self.kwargs)


__all__ = ["Agent", "AgentBuilder", "BaseAgent"]
