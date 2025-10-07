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
