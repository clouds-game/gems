"""Game engine helpers: initialization and summarization utilities.

This module provides a small, well-documented public API the rest of the
project (and agents) can import: `init_game` and `print_summary`.

The functions intentionally mirror the simple helpers used during
development in the repository root so callers have a stable, package-level
entrypoint.
"""

from typing import List, Optional, Dict, Sequence

from .typings import GameState, PlayerState, Gem, Card, Role
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

  @staticmethod
  def create_game(num_players: int = 2, names: Optional[List[str]] = None) -> GameState:
    """Create and return a minimal starting GameState.

    - num_players: between 2 and 4 (inclusive).
    - names: optional list of player display names; defaults to "Player 1"...
    """
    if not (2 <= num_players <= 4):
      raise ValueError("num_players must be between 2 and 4")

    names = names or [f"Player {i+1}" for i in range(num_players)]
    if len(names) < num_players:
      # extend with default names if caller provided too few
      names = names + [f"Player {i+1}" for i in range(len(names), num_players)]

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
    print(f"Visible cards: {len(self._state.visible_cards)}")

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
