from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple, Iterable


@dataclass(frozen=True)
class Action:
  """A small, opaque Action object the engine and agents pass around.

  Keep it intentionally simple: a string `type` and a mapping-like `payload`.
  It's frozen so agents/engine treat it as an immutable value object.
  """
  type: str
  # payload is stored as a tuple of (key, value) pairs to keep the
  # object fully immutable.
  payload: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)


class Gem(Enum):
  """Enumeration of gem/resource types used across the engine.

  Values are the lowercase resource names used in serialization and
  external APIs.
  """
  RED = "red"
  BLUE = "blue"
  WHITE = "white"
  BLACK = "black"
  GREEN = "green"
  GOLD = "gold"

  def __str__(self) -> str:  # pragma: no cover - tiny convenience
    return self.value


@dataclass(frozen=True)
class PlayerState:
  """Public-per-player snapshot inside GameState.

  Minimal fields used by agents/engine. Implementations may extend this.
  """
  seat_id: int
  name: Optional[str] = None
  gems: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)
  score: int = 0
  reserved_cards: Tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GameState:
  """A read-only view of the full public game state.

  Fields are converted to immutable tuples so agents can safely treat the
  object as read-only.
  """
  players: Tuple[PlayerState, ...]
  # bank is represented as an immutable tuple of (resource, amount).
  bank: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)
  visible_cards: Tuple[Any, ...] = field(default_factory=tuple)
  turn: int = 0
  last_action: Optional[Action] = None

  def __post_init__(self):
    # normalize inputs into tuples where appropriate so the public
    # API is always immutable. Allow callers to provide dicts or
    # iterables; we try to be forgiving.
    def _to_kv_tuple(v: Iterable):
      if isinstance(v, tuple):
        return v
      if isinstance(v, dict):
        # sort dict items by stringified key so callers may pass either
        # string keys or Gem enum keys without causing a TypeError from
        # comparing Enum instances.
        return tuple(sorted(v.items(), key=lambda kv: str(kv[0])))
      # assume iterable of pairs
      return tuple(v)

    object.__setattr__(self, 'bank', _to_kv_tuple(self.bank))
    object.__setattr__(self, 'players', tuple(self.players))
    object.__setattr__(self, 'visible_cards', tuple(self.visible_cards))
