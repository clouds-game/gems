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


class ActionType(Enum):
  TAKE_3_DIFFERENT = "take_3_different"
  TAKE_2_SAME = "take_2_same"
  BUY_CARD = "buy_card"
  RESERVE_CARD = "reserve_card"

  def __str__(self) -> str:
    return self.value


@dataclass(frozen=True)
class Action:
  """Minimal base Action used as a type tag for polymorphism.

  Concrete action types should subclass this and add strongly-typed
  fields. Keeping this minimal lets engine and agents switch on
  `type` safely.
  """
  type: ActionType

  # Backwards-compatible convenience constructors forwarding to the
  # concrete action dataclasses. These preserve prior call-site
  # ergonomics while returning the new concrete types.
  @classmethod
  def take3(cls, *gems: Gem) -> 'Take3Action':
    return Take3Action.create(*gems)

  @classmethod
  def take2(cls, gem: Gem, count: int = 2) -> 'Take2Action':
    return Take2Action.create(gem, count)

  @classmethod
  def buy(cls, card_id: str, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    return BuyCardAction.create(card_id, payment=payment)

  @classmethod
  def reserve(cls, card_id: str, take_gold: bool = True) -> 'ReserveCardAction':
    return ReserveCardAction.create(card_id, take_gold=take_gold)


@dataclass(frozen=True)
class Take3Action(Action):
  gems: Tuple[Gem, ...] = field(default_factory=tuple)

  @classmethod
  def create(cls, *gems: Gem) -> 'Take3Action':
    return cls(type=ActionType.TAKE_3_DIFFERENT, gems=tuple(gems))


@dataclass(frozen=True)
class Take2Action(Action):
  gem: Gem
  count: int = 2

  @classmethod
  def create(cls, gem: Gem, count: int = 2) -> 'Take2Action':
    return cls(type=ActionType.TAKE_2_SAME, gem=gem, count=count)


@dataclass(frozen=True)
class BuyCardAction(Action):
  card_id: str = ''
  payment: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)

  @classmethod
  def create(cls, card_id: str, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    pay = _to_kv_tuple(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, card_id=card_id, payment=pay)


@dataclass(frozen=True)
class ReserveCardAction(Action):
  card_id: str = ''
  take_gold: bool = True

  @classmethod
  def create(cls, card_id: str, take_gold: bool = True) -> 'ReserveCardAction':
    return cls(type=ActionType.RESERVE_CARD, card_id=card_id, take_gold=bool(take_gold))


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

  def __str__(self) -> str:  # pragma: no cover - tiny convenience
    bonus = self.bonus.short_str() if self.bonus else "N"
    points = f"{self.points}" if self.points > 0 else ""
    costs = "".join(f"{g.short_str()}{n}" for g, n in self.cost)
    return f"Card([{self.level}]{bonus}{points}:{costs})"


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


@dataclass(frozen=True)
class PlayerState:
  """Public-per-player snapshot inside GameState.

  Minimal fields used by agents/engine. Implementations may extend this.
  """
  seat_id: int
  name: Optional[str] = None
  gems: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)
  score: int = 0
  # cards the player has reserved but not yet purchased
  reserved_cards: Tuple[Card, ...] = field(default_factory=tuple)
  # cards the player has purchased (permanent bonuses / points)
  purchased_cards_in: InitVar[Iterable[Card]] = ()
  purchased_cards: Tuple[Card, ...] = field(init=False, default_factory=tuple)
  # per-gem permanent discounts are derived from purchased_cards: each
  # purchased card may have a `bonus` Gem that gives a permanent -1 cost
  # for that Gem. `discounts` stores the aggregated counts as an
  # immutable tuple of (Gem, amount) pairs.
  discounts: Tuple[Tuple[Gem, int], ...] = field(init=False, default_factory=tuple)

  def __post_init__(self, purchased_cards_in):
    # normalize reserved_cards and purchased_cards into tuples so the
    # public attributes remain immutable even when callers pass lists.
    object.__setattr__(self, 'reserved_cards', tuple(self.reserved_cards))
    purchased = tuple(purchased_cards_in)
    object.__setattr__(self, 'purchased_cards', purchased)

    # Build discounts by counting bonuses on purchased cards.
    counts: dict = {}
    for c in purchased:
      if getattr(c, 'bonus', None) is not None:
        counts[c.bonus] = counts.get(c.bonus, 0) + 1
    # normalize into the same stable tuple-of-pairs format used elsewhere
    object.__setattr__(self, 'discounts', _to_kv_tuple(counts))



@dataclass(frozen=True)
class GameState:
  """A read-only view of the full public game state.

  Fields are converted to immutable tuples so agents can safely treat the
  object as read-only.
  """
  players: Tuple[PlayerState, ...]
  # bank is represented as an immutable tuple of (resource, amount).
  bank: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)
  visible_cards: Tuple[Card, ...] = field(default_factory=tuple)
  turn: int = 0
  last_action: Optional[Action] = None

  def __post_init__(self):
    # normalize inputs into tuples where appropriate so the public
    # API is always immutable. Allow callers to provide dicts or
    # iterables; we try to be forgiving.
    object.__setattr__(self, 'bank', _to_kv_tuple(self.bank))
    object.__setattr__(self, 'players', tuple(self.players))
    object.__setattr__(self, 'visible_cards', tuple(self.visible_cards))
