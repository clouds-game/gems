"""Core agent base class for gems agents.

Follows the contract described in AGENTS.md. Keep this file minimal.
All Python code in this repo uses 2-space indentation.
"""
import random
from collections.abc import Sequence
from typing import TypeVar

from ..state import PlayerState, GameState
from ..actions import Action

AGENT_METADATA_HISTORY_ROUND = "history_round"

class Agent:
  def __init__(self, seat_id: int, rng: random.Random | None = None):
    self.seat_id = seat_id
    # Use a local RNG instance to guarantee reproducible behavior
    self.rng = rng or random.Random()

  def reset(self, seed: int | None = None) -> None:
    if seed is not None:
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
__all__ = ["Agent", "BaseAgent"]
