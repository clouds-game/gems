"""Small runner to initialize a GameState for development and testing.

This creates a minimal, valid GameState using the datatypes in
`gems.engine` so other tools and agents can exercise a fresh game.
"""

from gems.engine import Engine
import random


if __name__ == "__main__":
  # initialize a 3-player game and run a short random-play simulation
  engine = Engine(num_players=3, names=["Alice", "Bob", "Cara"])
  rng = random.Random()

  print("Initialized game state:\n")
  engine.print_summary()

  # Play until any player reaches 15 points (win condition) or no legal
  # actions are available for the current player.
  while True:
    state = engine.get_state()
    # check for winner
    scores = [p.score for p in state.players]
    if any(s >= 15 for s in scores):
      winners = [p for p in state.players if p.score >= 15]
      print("Game finished — winner(s):")
      for w in winners:
        print(f"  seat={w.seat_id} name={w.name!r} score={w.score}")
      break

    seat = state.turn % len(state.players)
    actions = engine.get_legal_actions(seat)
    if not actions:
      print(f"No legal actions available for player {seat}. Ending game.")
      break

    action = rng.choice(actions)
    print(f"Turn {state.turn} — player {seat} performs: {action}")
    # apply action and update engine state
    engine._state = action.apply(state)
    # print a brief summary after the move
    engine.print_summary()
