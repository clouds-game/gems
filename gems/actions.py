from dataclasses import dataclass, field
from typing import Optional, Mapping, Tuple

from .typings import Gem, ActionType, _to_kv_tuple


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

  def __str__(self) -> str:
    gem_str = ''.join(g.short_str() for g in self.gems)
    return f"Action.Take3(gems=[{gem_str}])"


@dataclass(frozen=True)
class Take2Action(Action):
  gem: Gem
  count: int = 2

  @classmethod
  def create(cls, gem: Gem, count: int = 2) -> 'Take2Action':
    return cls(type=ActionType.TAKE_2_SAME, gem=gem, count=count)

  def __str__(self) -> str:
    if self.count != 2:
      return f"Action.Take2({self.gem.short_str()}{self.count})"
    return f"Action.Take2({self.gem.short_str()})"


@dataclass(frozen=True)
class BuyCardAction(Action):
  card_id: str = ''
  payment: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)

  @classmethod
  def create(cls, card_id: str, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    pay = _to_kv_tuple(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, card_id=card_id, payment=pay)

  def __str__(self) -> str:
    pay_str = ''.join(f"{g.short_str()}{n}" for g, n in self.payment)
    return f"Action.Buy({self.card_id}, {pay_str})"


@dataclass(frozen=True)
class ReserveCardAction(Action):
  card_id: str = ''
  take_gold: bool = True

  @classmethod
  def create(cls, card_id: str, take_gold: bool = True) -> 'ReserveCardAction':
    return cls(type=ActionType.RESERVE_CARD, card_id=card_id, take_gold=bool(take_gold))

  def __str__(self) -> str:
    if self.take_gold:
      return f"Action.Reserve({self.card_id}, D)"
    return f"Action.Reserve({self.card_id})"
