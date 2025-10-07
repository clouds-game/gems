from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import Any, Optional, Tuple, Iterable, Mapping, Union


def _to_kv_tuple(v: Iterable):
  """Normalize a dict or iterable of pairs into a stable tuple of pairs."""
  if isinstance(v, tuple):
    return v
  if isinstance(v, dict):
    # sort dict items by stringified key so callers may pass either
    # string keys or Gem enum keys without causing a TypeError from
    # comparing Enum instances.
    return tuple(sorted(v.items(), key=lambda kv: str(kv[0])))
  return tuple(v)


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
    object.__setattr__(self, 'bank', _to_kv_tuple(self.bank))
    object.__setattr__(self, 'players', tuple(self.players))
    object.__setattr__(self, 'visible_cards', tuple(self.visible_cards))

@dataclass(frozen=True)
class Card:
  """Represents a purchasable card in the game.

  - `level` is a game-defined tier (1..3).
  - `points` is the victory points the card provides.
  - `bonus` is an optional Gem awarded permanently after purchase.
  - `cost` is an immutable tuple of (Gem, amount) pairs.
  """
  id: Optional[str] = None
  name: Optional[str] = None
  level: int = 1
  points: int = 0
  bonus: Optional[Gem] = None
  # Accept iterator/mapping inputs at construction time; store immutable
  # tuple-backed attributes for consumers.
  cost_in: InitVar[Union[Iterable[Tuple[Gem, int]], Mapping[Gem, int]]] = ()
  metadata_in: InitVar[Union[Iterable[Tuple[str, Any]], Mapping[str, Any]]] = ()
  cost: Tuple[Tuple[Gem, int], ...] = field(init=False, default_factory=tuple)
  face_up: bool = True
  metadata: Tuple[Tuple[str, Any], ...] = field(init=False, default_factory=tuple)

  def __post_init__(self, cost_in, metadata_in):
    # normalize cost and metadata into tuples so Card is always immutable
    object.__setattr__(self, 'cost', _to_kv_tuple(cost_in))
    object.__setattr__(self, 'metadata', _to_kv_tuple(metadata_in))

  def to_dict(self) -> dict:
    """Return a JSON-serializable dict representation of the Card."""
    return {
      'id': self.id,
      'name': self.name,
      'level': self.level,
      'points': self.points,
      'bonus': self.bonus.value if self.bonus is not None else None,
      'cost': [(g.value, n) for g, n in self.cost],
      'face_up': self.face_up,
      'metadata': list(self.metadata),
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'Card':
    bonus = Gem(d['bonus']) if d.get('bonus') is not None else None
    cost = tuple((Gem(g), n) for g, n in d.get('cost', ()))
    metadata = tuple(d.get('metadata', ()))
    return cls(id=d.get('id'), name=d.get('name'), level=d.get('level', 1),
               points=d.get('points', 0), bonus=bonus, cost_in=cost,
               face_up=d.get('face_up', True), metadata_in=metadata)


@dataclass(frozen=True)
class Role:
  """Represents a special role/noble with requirements and point reward.

  The engine treats Role as read-only metadata awarded when a player
  satisfies the `requirements` (a mapping of Gem -> required amount).
  """
  id: Optional[str] = None
  name: Optional[str] = None
  points: int = 0
  # Accept iterator or mapping at construction time via InitVar; the
  # public attributes `requirements` and `metadata` are always tuples.
  requirements_in: InitVar[Union[Iterable[Tuple[Gem, int]], Mapping[Gem, int]]] = ()
  metadata_in: InitVar[Union[Iterable[Tuple[str, Any]], Mapping[str, Any]]] = ()
  requirements: Tuple[Tuple[Gem, int], ...] = field(init=False, default_factory=tuple)
  metadata: Tuple[Tuple[str, Any], ...] = field(init=False, default_factory=tuple)

  def __post_init__(self, requirements_in, metadata_in):
    # normalize and store the tuple-backed public attributes
    object.__setattr__(self, 'requirements', _to_kv_tuple(requirements_in))
    object.__setattr__(self, 'metadata', _to_kv_tuple(metadata_in))

  def to_dict(self) -> dict:
    return {
      'id': self.id,
      'name': self.name,
      'points': self.points,
      'requirements': [(g.value, n) for g, n in self.requirements],
      'metadata': list(self.metadata),
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'Role':
    reqs = tuple((Gem(g), n) for g, n in d.get('requirements', ()))
    metadata = tuple(d.get('metadata', ()))
    return cls(id=d.get('id'), name=d.get('name'), points=d.get('points', 0),
               requirements_in=reqs, metadata_in=metadata)
