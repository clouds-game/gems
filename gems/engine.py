"""Game engine helpers: initialization and summarization utilities.

This module provides a small, well-documented public API the rest of the
project (and agents) can import: `init_game` and `print_summary`.

The functions intentionally mirror the simple helpers used during
development in the repository root so callers have a stable, package-level
entrypoint.
"""

from typing import List, Optional, Dict, Sequence, TypeVar

from .agents.core import Agent
from .consts import CARD_LEVELS, COIN_DEFAULT_INIT, COIN_GOLD_INIT, DEFAULT_PLAYERS, CARD_VISIBLE_COUNT

from .typings import ActionType, Gem, Card, Role
from .state import PlayerState, GameState
from .actions import Action
from pathlib import Path
import random

BaseAgent = TypeVar('BaseAgent', bound=Agent)


class Engine:
  """A tiny, stateful wrapper around the engine helpers.

  This class is intentionally small and suitable for development, tests,
  and simple interactive sessions. It stores the current `GameState` and
  exposes a couple of convenience methods:

  - `reset(...)` to re-initialize the game
  - `get_state()` to access the current immutable GameState
  - `print_summary()` to display a human readable summary
  """

  _num_players: int
  _state: GameState
  decks_by_level: Dict[int, List[Card]]
  roles_deck: List[Role]
  _rng: random.Random

  def __init__(
      self,
      *,
      num_players: int,
      names: Optional[List[str]],
      state: GameState,
      decks_by_level: Dict[int, List[Card]],
      roles_deck: List[Role],
      rng: random.Random,
      all_noops_last_round: bool = False,
      action_history: Optional[Sequence[Action]] = None,
  ) -> None:
    self._num_players = num_players
    self._names = names
    self._state = state
    self.decks_by_level = decks_by_level
    self.roles_deck = roles_deck
    self._rng = rng
    self._all_noops_last_round = all_noops_last_round
    self._action_history = list(action_history) if action_history is not None else []

  @staticmethod
  def new(num_players: int = DEFAULT_PLAYERS, names: Optional[List[str]] = None, seed: Optional[int] = None) -> "Engine":
    if not (1 <= num_players <= 4):
      raise ValueError("num_players must be between 1 and 4")
    state = Engine.create_game(num_players, names)
    engine = Engine(
        num_players=num_players,
        names=names,
        state=state,
        decks_by_level={},
        roles_deck=[],
        rng=random.Random(),
    )
    engine.load_and_shuffle_assets(seed=seed)
    visible_cards = []
    for lvl in CARD_LEVELS:
      drawn = engine.draw_from_deck(lvl, CARD_VISIBLE_COUNT)
      visible_cards.extend(reversed(drawn))
    roles_to_draw = (num_players or DEFAULT_PLAYERS) + 1
    visible_roles = []
    for _ in range(min(roles_to_draw, len(engine.roles_deck))):
      visible_roles.append(engine.roles_deck.pop())
    engine._state = GameState(players=engine._state.players, bank=engine._state.bank,
                              visible_cards_in=visible_cards, visible_roles_in=visible_roles,
                              turn=engine._state.turn)
    engine._all_noops_last_round = False
    engine._action_history = []
    return engine

  def clone(self, seed=None) -> "Engine":
    engine = Engine(
        num_players=self._num_players,
        names=self._names,
        state=self._state,
        decks_by_level={lvl: list(deck) for lvl, deck in self.decks_by_level.items()},
        roles_deck=list(self.roles_deck),
        rng=random.Random(seed),  # new RNG instance
        all_noops_last_round=self._all_noops_last_round,
        action_history=list(self._action_history),
    )
    return engine

  @staticmethod
  def create_game(num_players: int = DEFAULT_PLAYERS, names: Optional[List[str]] = None) -> GameState:
    """Create and return a minimal starting GameState.

    - num_players: between 2 and 4 (inclusive).
    - names: optional list of player display names; defaults to "Player 1"...
    """
    # if not (2 <= num_players <= 4):
    #   raise ValueError("num_players must be between 2 and 4")

    names = names or [f"Player {i + 1}" for i in range(num_players)]
    if len(names) < num_players:
      # extend with default names if caller provided too few
      names = names + [f"Player {i + 1}" for i in range(len(names), num_players)]

    players = [PlayerState(seat_id=i, name=names[i]) for i in range(num_players)]

    coin_default_init = COIN_DEFAULT_INIT[min(num_players, DEFAULT_PLAYERS) - 1]
    # Typical gem counts for a 2-4 player game (simple heuristic):
    bank = (
        (Gem.RED, coin_default_init),
        (Gem.BLUE, coin_default_init),
        (Gem.WHITE, coin_default_init),
        (Gem.BLACK, coin_default_init),
        (Gem.GREEN, coin_default_init),
        (Gem.GOLD, COIN_GOLD_INIT),
    )

    visible_cards = tuple()

    return GameState(players=tuple(players), bank_in=bank, visible_cards_in=visible_cards, turn=0)

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
    print(f"Round: {self._state.round} Turn: {self._state.turn}")
    print("Players:")
    for p in self._state.players:
      print(f"  seat={p.seat_id} name={p.name!r} score={p.score} gems={p.gems} cards={len(p.purchased_cards)} reserved={len(p.reserved_cards)}")
    print(f"Bank: {self._state.bank}")
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
    seat_id = seat_id if seat_id is not None else self._state.turn % len(self._state.players)
    return self._state.players[seat_id].get_legal_actions(self._state)

  def advance_turn(self) -> None:
    """Advance the turn to the next player.

    This is a convenience method that updates the internal GameState's
    `turn` counter. It does not mutate any other part of the state.
    """
    self._state = self._state.advance_turn(self.decks_by_level)

  def play_one_round(self, agents: List[BaseAgent], debug=True) -> None:
    """Play a full round (one turn per player) using specific Agents.

    This is a convenience for quick simulations and testing. It does not
    check for game end conditions.
    """
    all_noops = True
    num_players = len(self._state.players)
    for seat in range(num_players):
      state = self.get_state()
      agent = agents[seat]
      actions = self.get_legal_actions(seat)
      if not all(a.type == ActionType.NOOP for a in actions):
        all_noops = False
      action = agent.act(state, actions)
      if debug:
        print(f"Turn {state.turn} — player {seat} performs: {action}")
      # apply action and update engine state
      self._state = action.apply(state)
      self._action_history.append(action)
      # print a brief summary after the move
      if debug:
        self.print_summary()
      self.advance_turn()
    if all_noops:
      self._all_noops_last_round = True

  def game_end(self) -> bool:
    """Return True if any player has reached the winning score (15 points)."""
    if self._all_noops_last_round:
      return True
    winners = self.game_winners()
    return len(winners) > 0

  def game_winners(self) -> List[PlayerState]:
    """Return a list of players who have reached the winning score (15 points)."""
    return [p for p in self._state.players if p.score >= 15]


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
