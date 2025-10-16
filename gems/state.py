from dataclasses import InitVar, dataclass, field
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from gems.consts import GameConfig

from .typings import Gem, GemList, Card, CardList, Role

if TYPE_CHECKING:
  from .actions import Action


def _apply_discounts(cost: Iterable[tuple["Gem", int]], discounts: "GemList") -> list[tuple["Gem", int]]:
  """Return a list of (Gem, effective_amount) after applying discounts.

  Discounts are a GemList of (Gem, amt) pairs representing permanent
  bonuses from purchased cards. Effective amounts are floored at zero.
  """
  discount_map: dict["Gem", int] = {g: n for g, n in discounts}
  effective: list[tuple["Gem", int]] = []
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
  name: str | None = None
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
      object.__setattr__(self, 'gems', GemList(gems_in))
    object.__setattr__(self, 'reserved_cards', CardList(tuple(reserved_cards_in)))
    purchased = tuple(purchased_cards_in)
    object.__setattr__(self, 'purchased_cards', CardList(purchased))

    counts: dict = {}
    for c in purchased:
      if getattr(c, 'bonus', None) is not None:
        counts[c.bonus] = counts.get(c.bonus, 0) + 1
    object.__setattr__(self, 'discounts', GemList(tuple(counts.items())))

  def check_afford(self, card: Card, payment: dict[Gem, int]) -> bool:
    """Check if the given payment dict is a valid way to afford the card."""
    # TODO: improve performance
    possible_payments = self.can_afford(card)
    return payment in possible_payments

  def can_afford(self, card: Card) -> list[dict[Gem, int]]:
    """Return all exact payment dicts this player could use to buy `card`.

    Mirrors the previous `can_afford` helper but scoped to this player's
    available gems (`self.gems`).
    """
    player_gems: dict[Gem, int] = {g: amt for g, amt in self.gems}
    gold_available = player_gems.get(Gem.GOLD, 0)

    # Apply permanent discounts from purchased cards to the card cost
    requirements = list(card.cost)
    effective_requirements = _apply_discounts(requirements, self.discounts)

    # If no effective requirements remain, the card is free (via discounts)
    if not effective_requirements or all(req == 0 for _, req in effective_requirements):
      return [{}]

    ranges = []
    gems_order: list[Gem] = []
    req_amounts: list[int] = []
    for g, req in effective_requirements:
      gems_order.append(g)
      req_amounts.append(req)
      have = player_gems.get(g, 0)
      max_colored = min(have, req)
      ranges.append(range(0, max_colored + 1))

    from itertools import product

    payments: list[dict[Gem, int]] = []
    for combo in product(*ranges):
      deficit = 0
      for spend, req in zip(combo, req_amounts):
        if spend < req:
          deficit += (req - spend)
      if deficit <= gold_available:
        pay: dict[Gem, int] = {}
        for g, spend in zip(gems_order, combo):
          if spend > 0:
            pay[g] = spend
        if deficit > 0:
          pay[Gem.GOLD] = deficit
        payments.append(pay)

    return payments

  def can_reserve(self, config: GameConfig) -> bool:
    """Return whether this player can reserve another card."""
    return len(self.reserved_cards) < config.card_max_count_reserved

  def get_legal_actions(self, state: "GameState") -> list["Action"]:
    from .actions import (
        Action,
        Take3Action,
        Take2Action,
        BuyCardAction,
        ReserveCardAction,
        NoopAction,
    )
    """Enumerate a permissive set of legal actions for this player.

    Delegates to each Action subclass' `_get_legal_actions` classmethod so
    logic is co-located with the action definitions. Falls back to a
    single NoopAction if no actions are available.
    """
    config = state.config
    actions: list[Action] = []
    # Gather from each action type
    actions.extend(Take3Action._get_legal_actions(self, state, config))
    actions.extend(Take2Action._get_legal_actions(self, state, config))
    actions.extend(BuyCardAction._get_legal_actions(self, state, config))
    actions.extend(ReserveCardAction._get_legal_actions(self, state, config))
    # Fallback
    if not actions:
      return [Action.noop()]
    return actions


@dataclass(frozen=True)
class GameState:
  """A read-only view of the full public game state.

  Fields are converted to immutable tuples so agents can safely treat the
  object as read-only.
  """
  config: GameConfig
  players: tuple["PlayerState", ...]
  # bank is represented as an immutable tuple of (resource, amount).
  bank_in: InitVar[Iterable[tuple[Gem, int]] | Mapping[Gem, int] | GemList | None] = None
  bank: GemList = field(default_factory=GemList)
  visible_cards_in: InitVar[Iterable[Card] | CardList | None] = None
  visible_cards: CardList = field(default_factory=CardList)
  visible_roles_in: InitVar[Iterable[Role] | list[Role] | None] = None
  visible_roles: tuple[Role, ...] = field(default_factory=tuple)
  turn: int = 0
  round: int = 0
  last_action: "Action | None" = None

  def __post_init__(self, bank_in, visible_cards_in, visible_roles_in):
    # normalize inputs into tuples where appropriate so the public
    # API is always immutable. Allow callers to provide dicts or
    # iterables; we try to be forgiving.
    if bank_in is not None:
      object.__setattr__(self, 'bank', GemList(bank_in))
    if visible_cards_in is not None:
      object.__setattr__(self, 'visible_cards', CardList(visible_cards_in))
    if visible_roles_in is not None:
      object.__setattr__(self, 'visible_roles', tuple(visible_roles_in))
    object.__setattr__(self, 'players', tuple(self.players))
    object.__setattr__(self, 'visible_cards', CardList(self.visible_cards))

    num_players = len(self.players)
    if num_players <= 0:
      raise ValueError("GameState must have at least one player")
    object.__setattr__(self, 'round', self.turn // num_players)

  def advance_turn(self, decks_by_level: dict[int, list[Card]] | None = None, per_level: int = 4) -> 'GameState':
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
      counts: dict[int, int] = {}
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
    return GameState(config=self.config, players=self.players, bank=self.bank,
                     visible_cards_in=visible, turn=self.turn + 1,
                     last_action=self.last_action)
