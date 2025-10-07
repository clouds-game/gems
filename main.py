"""Small runner to initialize a GameState for development and testing.

This creates a minimal, valid GameState using the datatypes in
`gems.engine` so other tools and agents can exercise a fresh game.
"""

from gems.engine import Engine


if __name__ == "__main__":
  # example: initialize a 3-player game and show a brief summary
  engine = Engine(num_players=3, names=["Alice", "Bob", "Cara"])
  print("Initialized game state:\n")
  engine.print_summary()
