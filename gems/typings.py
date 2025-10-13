from dataclasses import MISSING, dataclass, field, InitVar
from enum import Enum
from typing import Any, Optional, Tuple, Iterable, Mapping, Union, Iterator

from .utils import _to_kv_tuple

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

  def short_str(self) -> str:  # pragma: no cover - tiny convenience
    if self == Gem.BLACK:
      return 'K'  # avoid confusion with BLUE
    if self == Gem.GOLD:
      return 'D'  # avoid confusion with GREEN
    return self.value[0].upper()


@dataclass(frozen=True)
class GemList:
  """Immutable list-like wrapper for a sequence of (Gem, int) pairs.

  Purpose: replace the frequent usage of Tuple[Tuple[Gem, int], ...].

  Construction accepts a tuple, dict, or any iterable of (Gem, int).
  Provides lightweight helpers to convert to/from dict and to iterate.
  """
  _pairs: Tuple[Tuple['Gem', int], ...] = field(default_factory=tuple)

  def __init__(self, vals: Union[Iterable[Tuple['Gem', int]], Mapping['Gem', int]] = ()):  # pragma: no cover - simple wrapper
    # normalize via the existing helper
    pairs = _to_kv_tuple(vals)
    object.__setattr__(self, '_pairs', pairs)

  def __iter__(self) -> Iterator[Tuple['Gem', int]]:
    return iter(self._pairs)

  def __len__(self) -> int:
    return len(self._pairs)

  def __getitem__(self, i):
    return self._pairs[i]

  def to_dict(self) -> dict:
    return {g: n for g, n in self._pairs}

  def as_tuple(self) -> Tuple[Tuple['Gem', int], ...]:
    return self._pairs

  def __repr__(self) -> str:  # pragma: no cover - convenience
    return f"GemList({self._pairs!r})"

  def __str__(self) -> str:  # pragma: no cover - convenience
    return "".join(f"{g.short_str()}{n}" for g, n in self._pairs if n > 0) or "Na"


class ActionType(Enum):
  TAKE_3_DIFFERENT = "take_3_different"
  TAKE_2_SAME = "take_2_same"
  BUY_CARD = "buy_card"
  RESERVE_CARD = "reserve_card"
  NOOP = "noop"

  def __str__(self) -> str:
    return self.value


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
  cost: GemList = field(init=False, default_factory=GemList)
  metadata_in: InitVar[Union[Iterable[Tuple[str, Any]], Mapping[str, Any]]] = ()
  metadata: Tuple[Tuple[str, Any], ...] = field(init=False, default_factory=tuple)

  def __post_init__(self, cost_in, metadata_in):
    # normalize cost and metadata into tuples so Card is always immutable
    object.__setattr__(self, 'cost', GemList(_to_kv_tuple(cost_in)))
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
        'metadata': list(self.metadata),
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'Card':
    bonus = Gem(d['bonus']) if d.get('bonus') is not None else None
    cost = tuple((Gem(g), n) for g, n in d.get('cost', ()))
    metadata = tuple(d.get('metadata', ()))
    return cls(id=d.get('id'), name=d.get('name'), level=d.get('level', 1),
               points=d.get('points', 0), bonus=bonus, cost_in=cost,
               metadata_in=metadata)

  def __str__(self) -> str:  # pragma: no cover - tiny convenience
    bonus = self.bonus.short_str() if self.bonus else "N"
    points = f"{self.points}" if self.points > 0 else ""
    costs = "".join(f"{g.short_str()}{n}" for g, n in self.cost)
    return f"Card([{self.level}]{bonus}{points}:{costs})"


@dataclass(frozen=True)
class CardList:
  """Immutable list-like wrapper for a sequence of Card objects.

  Minimal helper used to represent visible/reserved/purchased card lists
  in the public API. Mirrors the shape of `GemList` but for `Card`.
  """
  _items: Tuple['Card', ...] = field(default_factory=tuple)

  def __init__(self, vals: Iterable['Card'] = ()):  # pragma: no cover - simple wrapper
    items = tuple(vals)
    object.__setattr__(self, '_items', items)

  def __iter__(self):
    return iter(self._items)

  def __len__(self) -> int:
    return len(self._items)

  def __getitem__(self, i):
    return self._items[i]

  def as_tuple(self) -> Tuple['Card', ...]:
    return self._items

  def to_list(self) -> list:
    return list(self._items)

  def __repr__(self) -> str:  # pragma: no cover - convenience
    return f"CardList({self._items!r})"


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
  requirements: GemList = field(init=False, default_factory=GemList)
  metadata: Tuple[Tuple[str, Any], ...] = field(init=False, default_factory=tuple)

  def __post_init__(self, requirements_in, metadata_in):
    # normalize and store the tuple-backed public attributes
    object.__setattr__(self, 'requirements', GemList(_to_kv_tuple(requirements_in)))
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
