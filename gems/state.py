from dataclasses import InitVar, dataclass, field
from typing import Iterable, Mapping, Optional, List, Dict, TYPE_CHECKING

from .typings import Gem, GemList, Card, GameState, _to_kv_tuple

if TYPE_CHECKING:
  from .actions import Action


@dataclass(frozen=True)
class PlayerState:
  """Per-player snapshot with small helper methods.

  This class mirrors the previous definition from `typings.py` but adds
  convenience methods `get_legal_actions` and `can_afford` so callers can
  ask a player about their options directly.
  """
  seat_id: int
  name: Optional[str] = None
  gems_in: InitVar[Iterable[tuple[Gem, int]] | Mapping[Gem, int] | GemList | None] = None
  gems: GemList = field(default_factory=GemList)
  score: int = 0
  reserved_cards_in: InitVar[Iterable[Card]] = ()
  reserved_cards: tuple[Card, ...] = field(init=False, default_factory=tuple)
  purchased_cards_in: InitVar[Iterable[Card]] = ()
  purchased_cards: tuple[Card, ...] = field(init=False, default_factory=tuple)
  discounts: GemList = field(init=False, default_factory=GemList)

  def __post_init__(self, gems_in, reserved_cards_in, purchased_cards_in):
    if gems_in is not None:
      object.__setattr__(self, 'gems', GemList(_to_kv_tuple(gems_in) if not isinstance(gems_in, GemList) else gems_in))
    object.__setattr__(self, 'reserved_cards', tuple(reserved_cards_in))
    purchased = tuple(purchased_cards_in)
    object.__setattr__(self, 'purchased_cards', purchased)

    counts: dict = {}
    for c in purchased:
      if getattr(c, 'bonus', None) is not None:
        counts[c.bonus] = counts.get(c.bonus, 0) + 1
    object.__setattr__(self, 'discounts', GemList(tuple(counts.items())))

  def can_afford(self, card: Card) -> List[Dict[Gem, int]]:
    """Return all exact payment dicts this player could use to buy `card`.

    Mirrors the previous `can_afford` helper but scoped to this player's
    available gems (`self.gems`).
    """
    player_gems: Dict[Gem, int] = {g: amt for g, amt in self.gems}
    gold_available = player_gems.get(Gem.GOLD, 0)

    requirements = list(card.cost)
    if not requirements:
      return [{}]

    ranges = []
    gems_order: List[Gem] = []
    req_amounts: List[int] = []
    for g, req in requirements:
      gems_order.append(g)
      req_amounts.append(req)
      have = player_gems.get(g, 0)
      max_colored = min(have, req)
      ranges.append(range(0, max_colored + 1))

    from itertools import product

    payments: List[Dict[Gem, int]] = []
    for combo in product(*ranges):
      deficit = 0
      for spend, req in zip(combo, req_amounts):
        if spend < req:
          deficit += (req - spend)
      if deficit <= gold_available:
        pay: Dict[Gem, int] = {}
        for g, spend in zip(gems_order, combo):
          if spend > 0:
            pay[g] = spend
        if deficit > 0:
          pay[Gem.GOLD] = deficit
        payments.append(pay)

    return payments

  def get_legal_actions(self, state: GameState) -> List["Action"]:
    from .actions import (
      Action,
      Take3Action,
      Take2Action,
      BuyCardAction,
      ReserveCardAction,
    )
    """Enumerate a permissive set of legal actions for this player.

    This is a port of the `Engine.get_legal_actions` logic to be available
    directly from a PlayerState instance. It does not mutate `state`.
    """
    player = self

    bank = {g: amt for g, amt in state.bank}

    actions: List[Action] = []

    available_gems = [g for g, amt in bank.items() if amt > 0 and g != Gem.GOLD]
    from itertools import combinations
    if len(available_gems) > 3:
      for combo in combinations(available_gems, 3):
        actions.append(Take3Action.create(*combo))
    elif len(available_gems) != 0:
      actions.append(Take3Action.create(*available_gems))

    for g, amt in bank.items():
      if g == Gem.GOLD:
        continue
      if amt >= 4:
        actions.append(Take2Action.create(g, 2))

    gold_in_bank = bank.get(Gem.GOLD, 0)
    for card in state.visible_cards:
      card_id = getattr(card, 'id', None)
      if card_id is None:
        continue
      take_gold = gold_in_bank > 0
      if len(player.reserved_cards) < 3:
        actions.append(ReserveCardAction.create(card_id, take_gold=take_gold))
      payments = player.can_afford(card)
      for payment in payments:
        actions.append(BuyCardAction.create(card_id, payment=payment))

    if not actions:
      return [Action.noop()]

    return actions
