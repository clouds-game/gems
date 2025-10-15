"""Core agent base class for gems agents.

Follows the contract described in AGENTS.md. Keep this file minimal.
All Python code in this repo uses 2-space indentation.
"""
import random
from collections.abc import Sequence
from typing import TypeVar

from ..state import PlayerState, GameState
from ..actions import Action


class Agent:
  def __init__(self, seat_id: int, rng: random.Random | None = None):
    self.seat_id = seat_id
    # Use a local RNG instance to guarantee reproducible behavior
    self.rng = rng or random.Random()

  def reset(self, seed: int | None = None) -> None:
    if seed is not None:
      self.rng.seed(seed)

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

BaseAgent = TypeVar('BaseAgent', bound=Agent)
__all__ = ["Agent", "BaseAgent"]
