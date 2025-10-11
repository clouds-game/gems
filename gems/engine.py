"""Game engine helpers: initialization and summarization utilities.

This module provides a small, well-documented public API the rest of the
project (and agents) can import: `init_game` and `print_summary`.

The functions intentionally mirror the simple helpers used during
development in the repository root so callers have a stable, package-level
entrypoint.
"""

from typing import List, Optional, Dict, Sequence

from .typings import GameState, PlayerState, Gem, Card, Role
from .action import (
  Action,
  Take3Action,
  Take2Action,
  BuyCardAction,
  ReserveCardAction,
)
from pathlib import Path
import random


class Engine:
  """A tiny, stateful wrapper around the engine helpers.

  This class is intentionally small and suitable for development, tests,
  and simple interactive sessions. It stores the current `GameState` and
  exposes a couple of convenience methods:

  - `reset(...)` to re-initialize the game
  - `get_state()` to access the current immutable GameState
  - `print_summary()` to display a human readable summary
  """

  def __init__(self, num_players: int = 2, names: Optional[List[str]] = None):
    self._num_players = num_players
    self._names = names
    self._state = self.create_game(num_players, names)
    # load and shuffle assets, then draw initial visible cards and roles
    # default seed omitted for non-deterministic startup; callers can
    # re-seed by calling `load_and_shuffle_assets` explicitly.
    self.load_and_shuffle_assets()
    # draw 4 cards for each level (1..3)
    visible = []
    for lvl in (1, 2, 3):
      drawn = self.draw_from_deck(lvl, 4)
      # drawn are popped in LIFO order; present them as top-first by
      # reversing the drawn list so consumers see the top card first.
      visible.extend(reversed(drawn))
    # draw num_players + 1 roles and store separately
    roles_to_draw = (num_players or 2) + 1
    self.visible_roles = []
    for _ in range(min(roles_to_draw, len(self.roles_deck))):
      self.visible_roles.append(self.roles_deck.pop())
    # update GameState.visible_cards to include the visible cards
    self._state = GameState(players=self._state.players, bank=self._state.bank,
                            visible_cards=tuple(visible), turn=self._state.turn)

  @staticmethod
  def create_game(num_players: int = 2, names: Optional[List[str]] = None) -> GameState:
    """Create and return a minimal starting GameState.

    - num_players: between 2 and 4 (inclusive).
    - names: optional list of player display names; defaults to "Player 1"...
    """
    if not (2 <= num_players <= 4):
      raise ValueError("num_players must be between 2 and 4")

    names = names or [f"Player {i + 1}" for i in range(num_players)]
    if len(names) < num_players:
      # extend with default names if caller provided too few
      names = names + [f"Player {i + 1}" for i in range(len(names), num_players)]

    players = [PlayerState(seat_id=i, name=names[i]) for i in range(num_players)]

    # Typical gem counts for a 2-4 player game (simple heuristic):
    bank = (
        (Gem.RED, 7),
        (Gem.BLUE, 7),
        (Gem.WHITE, 7),
        (Gem.BLACK, 7),
        (Gem.GREEN, 7),
        (Gem.GOLD, 5),
    )

    visible_cards = tuple()

    return GameState(players=tuple(players), bank=bank, visible_cards=visible_cards, turn=0)

  def reset(self, num_players: Optional[int] = None, names: Optional[List[str]] = None) -> None:
    """Reset the engine's internal GameState.

    If `num_players` or `names` are omitted the values provided at
    construction time are used.
    """
    if num_players is None:
      num_players = self._num_players
    if names is None:
      names = self._names
    self._state = self.create_game(num_players, names)
    self._num_players = num_players
    self._names = names

  def get_state(self) -> GameState:
    """Return the current (immutable) GameState object."""
    return self._state

  def print_summary(self) -> None:
    """Print a short, human-readable summary of the given GameState.

    This is a convenience for development and quick debugging; callers should
    avoid parsing the printed output in tests.
    """
    print(f"Turn: {self._state.turn}")
    print("Players:")
    for p in self._state.players:
      print(f"  seat={p.seat_id} name={p.name!r} score={p.score} gems={p.gems}")
    print("Bank:")
    for g, amt in self._state.bank:
      print(f"  {g}: {amt}")
    print(f"Visible cards: {", ".join(str(c) for c in self._state.visible_cards)}")

  def load_and_shuffle_assets(self, path: Optional[str] = None, seed: Optional[int] = None) -> None:
    """Load assets from disk and shuffle them into decks on this Engine.

    - path: optional path to config file (falls back to package assets)
    - seed: optional RNG seed to make shuffling deterministic
    """
    cards, roles = load_assets(path)
    rng = random.Random(seed)
    levels, roles_list = shuffle_assets(cards, roles, rng=rng)
    # store on instance for consumers
    self.decks_by_level = levels
    self.roles_deck = roles_list
    # keep RNG for reproducibility if callers want to do more shuffling
    self._rng = rng

  def get_deck(self, level: int) -> List[Card]:
    return list(self.decks_by_level.get(level, []))

  def get_roles(self) -> List[Role]:
    return list(self.roles_deck)

  def draw_from_deck(self, level: int, n: int = 1) -> List[Card]:
    """Remove and return up to `n` cards from the deck of `level`.

    Pops from the end of the level list (treating the end as the top of the
    deck) which is efficient for Python lists.
    """
    deck = self.decks_by_level.get(level, [])
    drawn: List[Card] = []
    for _ in range(min(n, len(deck))):
      drawn.append(deck.pop())
    return drawn

  def peek_deck(self, level: int, n: int = 1) -> List[Card]:
    """Return up to `n` cards from the top of the deck without removing them."""
    deck = self.decks_by_level.get(level, [])
    if not deck:
      return []
    return list(deck[-n:]) if n > 0 else []

  def get_legal_actions(self, seat_id: Optional[int] = None) -> List[Action]:
    """Return a list of legal `Action` objects for the given player seat.

    This is a lightweight implementation used by tests and agents to
    enumerate plausible moves. It intentionally implements a permissive
    set of actions:

    - take_3_different: any triple of distinct gem types with at least 1 in
      the bank.
    - take_2_same: any gem type with at least 4 tokens in the bank (caller may
      choose a stricter rule if desired).
    - reserve_card: reserve any visible card (present on `visible_cards`).
    - buy_card: a permissive buy action for visible cards (detailed payment
      validation is left to the engine's apply/validation logic).

    The method does not mutate engine state.
    """
    state = self._state
    if seat_id is None:
      seat_id = state.turn % len(state.players)

    player = state.players[seat_id]

    # Build a simple bank lookup
    bank = {g: amt for g, amt in state.bank}

    actions: List[Action] = []

    # take_3_different: any combination of 3 distinct gems with at least 1
    available_gems = [g for g, amt in bank.items() if amt > 0 and g != Gem.GOLD]
    from itertools import combinations
    if len(available_gems) > 3:
      # simple approach: choose any 3-combination (order not important)
      for combo in combinations(available_gems, 3):
        actions.append(Take3Action.create(*combo))
    elif len(available_gems) != 0:
      # if fewer than 3 types available, allow taking all available types
      actions.append(Take3Action.create(*available_gems))

    # take_2_same: allow gems with at least 4 tokens in bank
    for g, amt in bank.items():
      if g == Gem.GOLD:
        continue
      if amt >= 4:
        actions.append(Take2Action.create(g, 2))

    # reserve_card and buy_card for visible cards (if card has id); filter buy_card by affordability
    gold_in_bank = bank.get(Gem.GOLD, 0)
    for card in state.visible_cards:
      card_id = getattr(card, 'id', None)
      if card_id is None:
        continue
      # take a gold only if available
      take_gold = gold_in_bank > 0
      actions.append(ReserveCardAction.create(card_id, take_gold=take_gold))
      payments = can_afford(card, player)
      for payment in payments:
        actions.append(BuyCardAction.create(card_id, payment=payment))

    return actions


def can_afford(card: Card, player: PlayerState) -> List[Dict[Gem, int]]:
  '''
  Return all possible exact payment combinations for purchasing `card` with
  the given `player`'s gem tokens, including gold (wild) substitutions.

  A payment combination is represented as a dict mapping `Gem` -> int token
  count spent of that gem (including `Gem.GOLD` if wilds are used). Each
  combination pays the card's cost exactly (no overpay) and uses no more
  than the player's available tokens.

  Rules / assumptions:
  - Colored gems must cover some portion (possibly all) of each required
    color. Gold wilds may substitute for any remaining unmet requirement.
  - We allow (and enumerate) the use of gold even if the player has enough
    colored gems; while not always strategically optimal it is typically
    a legal payment in wildcard-based games.
  - No combination intentionally overpays or substitutes gold when colored
    gems are unused beyond the card's requirement (i.e. you cannot pay extra).

  Returns: List[Dict[Gem, int]] (may be empty if unaffordable).
  '''
  # build player's gem lookup
  player_gems: Dict[Gem, int] = {g: amt for g, amt in player.gems}
  gold_available = player_gems.get(Gem.GOLD, 0)

  # cost requirements as list of (gem, required)
  requirements = list(card.cost)
  if not requirements:
    # nothing to pay
    return [{}]

  # For each required color, determine max colored gems the player can spend
  ranges = []  # list of ranges for each required gem indicating possible colored spends
  gems_order: List[Gem] = []
  req_amounts: List[int] = []
  for g, req in requirements:
    gems_order.append(g)
    req_amounts.append(req)
    have = player_gems.get(g, 0)
    max_colored = min(have, req)
    # allow spending 0..max_colored colored gems for this color
    ranges.append(range(0, max_colored + 1))

  from itertools import product

  payments: List[Dict[Gem, int]] = []
  # enumerate all combinations of colored spends
  for combo in product(*ranges):
    # combo is tuple of colored spends aligned with gems_order
    deficit = 0
    for spend, req in zip(combo, req_amounts):
      if spend < req:
        deficit += (req - spend)
    if deficit <= gold_available:
      # build payment mapping: include colored spends and gold used
      pay: Dict[Gem, int] = {}
      for g, spend in zip(gems_order, combo):
        if spend > 0:
          pay[g] = spend
      if deficit > 0:
        pay[Gem.GOLD] = deficit
      payments.append(pay)

  return payments


def load_assets(path: Optional[str] = None):
  """Load cards and roles from a JSON config file and return (cards, roles).

  The config file is expected to contain top-level `cards` and `roles` arrays
  matching the `Card.from_dict` / `Role.from_dict` shapes.
  """
  import yaml
  p = Path(path) if path is not None else Path(__file__).parent / "assets" / "config.yaml"
  with p.open('r', encoding='utf8') as fh:
    j = yaml.safe_load(fh)

  cards = [Card.from_dict(c) for c in j.get('cards', [])]
  roles = [Role.from_dict(r) for r in j.get('roles', [])]
  return cards, roles


def shuffle_assets(cards: Sequence[Card], roles: Sequence[Role], rng: Optional[random.Random] = None):
  """Shuffle cards by level and shuffle roles.

  Returns a dict mapping level->list[Card] and a list of roles. The RNG may
  be a `random.Random` instance; if omitted a new one is created.
  """
  rng = rng or random.Random()
  # group cards by level
  levels: Dict[int, List[Card]] = {}
  for c in cards:
    levels.setdefault(c.level, []).append(c)

  # shuffle each level's deck
  for lvl, lst in levels.items():
    rng.shuffle(lst)

  # shuffle roles
  roles_list = list(roles)
  rng.shuffle(roles_list)

  return levels, roles_list
