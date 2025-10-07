"""Game engine helpers: initialization and summarization utilities.

This module provides a small, well-documented public API the rest of the
project (and agents) can import: `init_game` and `print_summary`.

The functions intentionally mirror the simple helpers used during
development in the repository root so callers have a stable, package-level
entrypoint.
"""

from typing import List, Optional

from .typings import GameState, PlayerState, Gem


def init_game(num_players: int = 2, names: Optional[List[str]] = None) -> GameState:
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


def print_summary(state: GameState) -> None:
  """Print a short, human-readable summary of the given GameState.

  This is a convenience for development and quick debugging; callers should
  avoid parsing the printed output in tests.
  """
  print(f"Turn: {state.turn}")
  print("Players:")
  for p in state.players:
    print(f"  seat={p.seat_id} name={p.name!r} score={p.score} gems={p.gems}")
  print("Bank:")
  for g, amt in state.bank:
    print(f"  {g}: {amt}")
  print(f"Visible cards: {len(state.visible_cards)}")


if __name__ == "__main__":
  # example: initialize a 3-player game and show a brief summary
  gs = init_game(3, ["Alice", "Bob", "Cara"])
  print("Initialized game state:\n")
  print_summary(gs)


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
    self._state = init_game(num_players, names)

  def reset(self, num_players: Optional[int] = None, names: Optional[List[str]] = None) -> None:
    """Reset the engine's internal GameState.

    If `num_players` or `names` are omitted the values provided at
    construction time are used.
    """
    if num_players is None:
      num_players = self._num_players
    if names is None:
      names = self._names
    self._state = init_game(num_players, names)
    self._num_players = num_players
    self._names = names

  def get_state(self) -> GameState:
    """Return the current (immutable) GameState object."""
    return self._state

  def print_summary(self) -> None:
    """Print a human-friendly summary of the current GameState."""
    print_summary(self._state)
