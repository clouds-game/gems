"""Small runner to initialize a GameState for development and testing.

This creates a minimal, valid GameState using the datatypes in
`gems.engine` so other tools and agents can exercise a fresh game.
"""

from gems.engine import Engine
from gems.agents.random import RandomAgent
import random

from gems.typings import ActionType


if __name__ == "__main__":
  # initialize a 3-player game and run a short random-play simulation
  engine = Engine.new(num_players=3, names=["Alice", "Bob", "Cara"])
  # create one RandomAgent per seat and seed deterministically for reproducibility
  rng = random.Random(0)
  agents = [RandomAgent(seat_id=i, rng=random.Random(100 + i)) for i in range(3)]

  print("Initialized game state:\n")
  engine.print_summary()

  # Play until any player reaches 15 points (win condition) or no legal
  # actions are available for the current player.

  while not engine.game_end():
    engine.play_one_round(agents=agents)
  winners = engine.game_winners()
  if winners:
    print("Game finished — winner(s):")
    for w in winners:
      print(f"  seat={w.seat_id} name={w.name!r} score={w.score} cards={len(w.purchased_cards)} reserved={len(w.reserved_cards)}")
  else:
    print("All players have only noop actions. Ending game.")
