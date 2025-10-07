"""Small runner to initialize a GameState for development and testing.

This creates a minimal, valid GameState using the datatypes in
`gems.engine` so other tools and agents can exercise a fresh game.
"""

from gems import engine


if __name__ == "__main__":
  # example: initialize a 3-player game and show a brief summary
  gs = engine.init_game(3, ["Alice", "Bob", "Cara"])
  print("Initialized game state:\n")
  engine.print_summary(gs)
