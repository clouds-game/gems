from dataclasses import InitVar
from enum import Enum
from typing import Any, TypeAlias
from pydantic import field_validator, model_validator, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from collections.abc import Mapping, Iterator, Sequence

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

  def color_circle(self) -> str:  # pragma: no cover - tiny convenience
    """Return a colored circle emoji representing this gem."""
    if self == Gem.RED:
      return "🔴"
    if self == Gem.BLUE:
      return "🔵"
    if self == Gem.WHITE:
      return "⚪"
    if self == Gem.BLACK:
      return "⚫"
    if self == Gem.GREEN:
      return "🟢"
    if self == Gem.GOLD:
      return "🟡"
    return "⭕"  # fallback to HEAVY LARGE CIRCLE

@pydantic_dataclass(frozen=True)
class GemList:
  """Immutable list-like wrapper for a sequence of (Gem, int) pairs.

  Purpose: replace the frequent usage of tuple[tuple[Gem, int], ...].

  Construction accepts a tuple, dict, or any iterable of (Gem, int).
  Provides lightweight helpers to convert to/from dict and to iterate.
  """
  COLOR_ORDER = (Gem.BLUE, Gem.WHITE, Gem.BLACK, Gem.RED, Gem.GREEN, Gem.GOLD)
  _pairs_in: InitVar[Mapping['Gem', int] | tuple[tuple['Gem', int], ...] | list[tuple['Gem', int]]] = Field(default_factory=dict, alias='_pairs')
  _pairs: dict[Gem, int] = Field(init=False, default_factory=dict)

  @field_validator('_pairs_in', mode='before')
  @classmethod
  def validate_pairs(cls, vals):
    # normalize via the existing helper
    if isinstance(vals, GemList):
      pairs = vals._pairs
    else:
      pairs = dict(vals)
    return pairs

  def __post_init__(self, _pairs_in):
    assert isinstance(_pairs_in, dict)
    object.__setattr__(self, '_pairs', _pairs_in)

  def __iter__(self) -> Iterator[tuple['Gem', int]]:
    return iter(self._pairs.items())

  def __len__(self) -> int:
    return len(self._pairs)

  def __getitem__(self, i):
    return self._pairs[i]

  def __hash__(self) -> int:
    return hash(frozenset(self.normalized()._pairs.items()))

  def normalized(self) -> 'GemList':
    """Return a new GemList with all gems in COLOR_ORDER and zero counts removed."""
    counts = self._pairs
    normalized_pairs = tuple((g, counts[g]) for g in self.COLOR_ORDER if counts.get(g, 0) > 0)
    return GemList(normalized_pairs)

  def count(self) -> int:
    """Return the total count of all gems in this GemList."""
    return sum(self._pairs.values())

  def count_distinct(self) -> int:
    """Return the count of distinct gem types in this GemList."""
    return len(set(g for g, n in self._pairs.items() if n > 0))

  def get(self, gem: Gem) -> int:
    return self._pairs.get(gem, 0)

  def to_dict(self) -> dict[Gem, int]:
    return dict(self._pairs)

  def as_tuple(self) -> tuple[tuple[Gem, int], ...]:
    return tuple(self._pairs.items())

  def flatten(self) -> list[Gem]:
    """Return a flat list of Gem, repeating each gem according to its count."""
    result = []
    for g, n in self._pairs.items():
      result.extend([g] * n)
    return result

  def __repr__(self) -> str:  # pragma: no cover - convenience
    return f"GemList({self._pairs!r})"

  def __str__(self) -> str:  # pragma: no cover - convenience
    return "".join(f"{n}{g.color_circle()}" for g, n in self) or "⭕"
GemListInput: TypeAlias = GemList | Mapping[Gem, int] | Sequence[tuple[Gem, int]]


class ActionType(Enum):
  TAKE_3_DIFFERENT = "take_3_different"
  TAKE_2_SAME = "take_2_same"
  BUY_CARD = "buy_card"
  RESERVE_CARD = "reserve_card"
  NOOP = "noop"

  def __str__(self) -> str:
    return self.value


@pydantic_dataclass(frozen=True)
class Card:
  """Represents a purchasable card in the game.

  - `level` is a game-defined tier (1..3).
  - `points` is the victory points the card provides.
  - `bonus` is an optional Gem awarded permanently after purchase.
  - `cost` is an immutable tuple of (Gem, amount) pairs.
  """
  id: str
  name: str | None = None
  level: int = 1
  points: int = 0
  bonus: Gem | None = None
  # Accept iterator/mapping inputs at construction time; store immutable
  # tuple-backed attributes for consumers.
  cost_in: InitVar[GemListInput] = Field(default_factory=GemList, alias='cost')
  cost: GemList = Field(init=False, default_factory=GemList)
  metadata_in: InitVar[Mapping[str, Any] | Sequence[tuple[str, Any]]] = Field(default_factory=tuple, alias='metadata')
  metadata: tuple[tuple[str, Any], ...] = Field(init=False, default_factory=tuple)

  def __post_init__(self, cost_in, metadata_in):
    # normalize cost and metadata into tuples so Card is always immutable
    object.__setattr__(self, 'cost', GemList(cost_in))
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
    cost = GemList([(Gem(g), n) for g, n in d.get('cost', ())])
    metadata = tuple(d.get('metadata', ()))
    # Ensure id is always a string (use empty string as default)
    cid = d.get('id')
    if not cid:
      raise Exception("Card id is required")
    return cls(id=cid, name=d.get('name'), level=d.get('level', 1),
               points=d.get('points', 0), bonus=bonus, cost=cost,
               metadata=metadata)

  def __str__(self) -> str:  # pragma: no cover - tiny convenience
    bonus = self.bonus.color_circle() if self.bonus else "⭕"
    points = f"[{self.points}]" if self.points > 0 else ""
    return f"Card{self.level}(<{self.id}>{points}{bonus}:{self.cost})"


@pydantic_dataclass(frozen=True)
class CardList(Sequence['Card']):
  """Immutable list-like wrapper for a sequence of Card objects.

  Minimal helper used to represent visible/reserved/purchased card lists
  in the public API. Mirrors the shape of `GemList` but for `Card`.
  """
  _items_in: InitVar[Sequence['Card']] = Field(default_factory=tuple, alias='items')
  _items: tuple['Card', ...] = Field(init=False, default_factory=tuple)

  def __post_init__(self, _items_in: Sequence['Card'] = ()):
    object.__setattr__(self, '_items', tuple(_items_in))

  def __iter__(self):
    return iter(self._items)

  def __len__(self) -> int:
    return len(self._items)

  def __getitem__(self, i):
    return self._items[i]

  def as_tuple(self) -> tuple['Card', ...]:
    return self._items

  def to_list(self) -> list:
    return list(self._items)

  def __repr__(self) -> str:  # pragma: no cover - convenience
    return f"CardList({self._items!r})"

  def get_level(self, level: int) -> 'CardList':
    """Return a new CardList containing only cards with the given level."""
    return CardList([c for c in self._items if c.level == level])

  def find(self, card_id: str) -> Card | None:
    """Return the Card with the given ID, or None if not found."""
    for c in self._items:
      if c.id == card_id:
        return c
    return None

  def merge(self, *others: Sequence['Card']) -> 'CardList':
    """Return a new CardList combining this CardList with other iterables of Card.

    Each element in `others` may be a `CardList` or any iterable of `Card`.
    The original CardList instances are not mutated.
    """
    items = list(self._items)
    for o in others:
      if isinstance(o, CardList):
        items.extend(o._items)
      else:
        items.extend(tuple(o))
    return CardList(items)

  def __add__(self, other: Sequence['Card']) -> 'CardList':
    """Support `CardList + other` to produce a new merged CardList."""
    return self.merge(other)


@pydantic_dataclass(frozen=True)
class CardIdx:
  """Represents an index referring to a card location.

  Exactly one of the following should be non-None:
  - `visible_idx`: index into the public visible cards (int)
  - `reserve_idx`: index into a player's reserved cards (int)
  - `deck_head_level`: an int indicating the top card of the deck for that level

  The class validates on construction that exactly one of the three is provided.
  """
  visible_idx: int | None = None
  reserve_idx: int | None = None
  deck_head_level: int | None = None

  @model_validator(mode='after')
  def validate(self):
    # enforce exactly one non-null field
    vals = (self.visible_idx, self.reserve_idx, self.deck_head_level)
    non_null = sum(1 for v in vals if v is not None)
    if non_null != 1:
      raise ValueError("CardIdx requires exactly one of visible_idx, reserve_idx, deck_head_level to be set")
    return self

  def __str__(self) -> str:
    return self.to_str()

  def to_str(self, id: str | None = None) -> str:
    id_part = f"={id}" if id is not None else ""
    if self.visible_idx is not None:
      return f"<[{self.visible_idx}]{id_part}>"
    if self.reserve_idx is not None:
      return f"<R[{self.reserve_idx}]{id_part}>"
    if self.deck_head_level is not None:
      return f"<D[{self.deck_head_level}]{id_part}>"
    return f"<[Invalid]{id_part}>"


@pydantic_dataclass(frozen=True)
class Role:
  """Represents a special role/noble with requirements and point reward.

  The engine treats Role as read-only metadata awarded when a player
  satisfies the `requirements` (a mapping of Gem -> required amount).
  """
  id: str | None = None
  name: str | None = None
  points: int = 0
  # Accept iterator or mapping at construction time via InitVar; the
  # public attributes `requirements` and `metadata` are always tuples.
  requirements_in: InitVar[GemListInput] = Field(default_factory=GemList, alias='requirements')
  requirements: GemList = Field(init=False, default_factory=GemList)
  metadata_in: InitVar[Mapping[str, Any] | Sequence[tuple[str, Any]]] = Field(default_factory=tuple, alias='metadata')
  metadata: tuple[tuple[str, Any], ...] = Field(init=False, default_factory=tuple)

  def __post_init__(self, requirements_in, metadata_in):
    # normalize and store the tuple-backed public attributes
    object.__setattr__(self, 'requirements', GemList(requirements_in))
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
    reqs = GemList([(Gem(g), int(n)) for g, n in d.get('requirements', ())])
    metadata = tuple(d.get('metadata', ()))
    return cls(id=d.get('id'), name=d.get('name'), points=d.get('points', 0),
               requirements=reqs, metadata=metadata)
