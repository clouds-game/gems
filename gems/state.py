from dataclasses import InitVar, dataclass, field
from typing import Iterable, Mapping, Optional, List, Dict, TYPE_CHECKING

from .typings import Gem, GemList, Card, CardList
from .utils import _to_kv_tuple

if TYPE_CHECKING:
  from .actions import Action


def _apply_discounts(cost: Iterable[tuple["Gem", int]], discounts: "GemList") -> List[tuple["Gem", int]]:
  """Return a list of (Gem, effective_amount) after applying discounts.

  Discounts are a GemList of (Gem, amt) pairs representing permanent
  bonuses from purchased cards. Effective amounts are floored at zero.
  """
  discount_map: Dict["Gem", int] = {g: n for g, n in discounts}
  effective: List[tuple["Gem", int]] = []
  for g, req in cost:
    disc = discount_map.get(g, 0)
    eff = req - disc
    if eff < 0:
      eff = 0
    effective.append((g, eff))
  return effective


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
  reserved_cards: CardList = field(init=False, default_factory=CardList)
  purchased_cards_in: InitVar[Iterable[Card]] = ()
  purchased_cards: CardList = field(init=False, default_factory=CardList)
  discounts: GemList = field(init=False, default_factory=GemList)

  def __post_init__(self, gems_in, reserved_cards_in, purchased_cards_in):
    if gems_in is not None:
      object.__setattr__(self, 'gems', GemList(_to_kv_tuple(
          gems_in) if not isinstance(gems_in, GemList) else gems_in))
    object.__setattr__(self, 'reserved_cards', CardList(tuple(reserved_cards_in)))
    purchased = tuple(purchased_cards_in)
    object.__setattr__(self, 'purchased_cards', CardList(purchased))

    counts: dict = {}
    for c in purchased:
      if getattr(c, 'bonus', None) is not None:
        counts[c.bonus] = counts.get(c.bonus, 0) + 1
    object.__setattr__(self, 'discounts', GemList(tuple(counts.items())))

  def check_afford(self, card: Card, payment: Dict[Gem, int]) -> bool:
    """Check if the given payment dict is a valid way to afford the card."""
    # TODO: improve performance
    possible_payments = self.can_afford(card)
    return payment in possible_payments

  def can_afford(self, card: Card) -> List[Dict[Gem, int]]:
    """Return all exact payment dicts this player could use to buy `card`.

    Mirrors the previous `can_afford` helper but scoped to this player's
    available gems (`self.gems`).
    """
    player_gems: Dict[Gem, int] = {g: amt for g, amt in self.gems}
    gold_available = player_gems.get(Gem.GOLD, 0)

    # Apply permanent discounts from purchased cards to the card cost
    requirements = list(card.cost)
    effective_requirements = _apply_discounts(requirements, self.discounts)

    # If no effective requirements remain, the card is free (via discounts)
    if not effective_requirements or all(req == 0 for _, req in effective_requirements):
      return [{}]

    ranges = []
    gems_order: List[Gem] = []
    req_amounts: List[int] = []
    for g, req in effective_requirements:
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

  def can_reserve(self) -> bool:
    """Return whether this player can reserve another card."""
    return len(self.reserved_cards) < 3

  def get_legal_actions(self, state: "GameState") -> List["Action"]:
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
    # visible_cards = state.visible_cards + self.reserved_cards

    for card in state.visible_cards + self.reserved_cards:

      payments = player.can_afford(card)
      for payment in payments:
        actions.append(BuyCardAction.create(card, payment=payment))

    if self.can_reserve():
      for card in state.visible_cards:
        take_gold = gold_in_bank > 0
        actions.append(ReserveCardAction.create(card, take_gold=take_gold))

    if not actions:
      return [Action.noop()]

    return actions


@dataclass(frozen=True)
class GameState:
  """A read-only view of the full public game state.

  Fields are converted to immutable tuples so agents can safely treat the
  object as read-only.
  """
  players: tuple["PlayerState", ...]
  # bank is represented as an immutable tuple of (resource, amount).
  bank_in: InitVar[Iterable[tuple[Gem, int]] | Mapping[Gem, int] | GemList | None] = None
  bank: GemList = field(default_factory=GemList)
  visible_cards_in: InitVar[Iterable[Card] | CardList | None] = None
  visible_cards: CardList = field(default_factory=CardList)
  turn: int = 0
  round: int = 0
  last_action: Optional["Action"] = None

  def __post_init__(self, bank_in, visible_cards_in):
    # normalize inputs into tuples where appropriate so the public
    # API is always immutable. Allow callers to provide dicts or
    # iterables; we try to be forgiving.
    if bank_in is not None:
      object.__setattr__(self, 'bank', GemList(_to_kv_tuple(bank_in)))
    if visible_cards_in is not None:
      object.__setattr__(self, 'visible_cards', CardList(visible_cards_in))
    object.__setattr__(self, 'players', tuple(self.players))
    object.__setattr__(self, 'visible_cards', CardList(self.visible_cards))

    num_players = len(self.players)
    if num_players <= 0:
      raise ValueError("GameState must have at least one player")
    object.__setattr__(self, 'round', self.turn // num_players)

  def advance_turn(self, decks_by_level: Optional[Dict[int, List[Card]]] = None, per_level: int = 4) -> 'GameState':
    """Return a new GameState with the turn advanced by one.

    If `decks_by_level` is provided (a mutable mapping level->list[Card])
    the method will attempt to top up visible cards so that there are
    `per_level` cards for each level present. This operation will pop
    cards from the provided decks (mutating them) similar to how the
    `Engine` draws cards. If `decks_by_level` is omitted no visible-card
    refilling is performed.
    """
    # Start with current visible cards as a list we can extend
    visible = list(self.visible_cards)

    if decks_by_level is not None:
      # Count how many visible cards we currently have per level
      counts: Dict[int, int] = {}
      for c in visible:
        lvl = c.level
        counts[lvl] = counts.get(lvl, 0) + 1

      # For each known level in the decks, draw up to per_level
      for lvl, deck in decks_by_level.items():
        need = per_level - counts.get(lvl, 0)
        for _ in range(min(need, len(deck))):
          # pop from the end (deck treated LIFO with end as top)
          visible.append(deck.pop())

    # Return a new GameState with incremented turn and updated visible_cards
    return GameState(players=self.players, bank=self.bank,
                     visible_cards_in=visible, turn=self.turn + 1,
                     last_action=self.last_action)
