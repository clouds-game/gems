# %%
"""Single-player search / simulation helper.

This module provides a minimal loop to play a single-player variant of
the gems game until the (only) player's score reaches 15 points or the
player has no non-noop legal actions.

The regular `Engine` enforces 2-4 players; for experimentation we add a
`SinglePlayerEngine` that permits 1 player and reuses all existing
action/payment logic. Visible card refilling still occurs each turn so
the solo player can progress.

Usage (from repository root):

	python -m gems.single_player_search

Or directly:

	python single_player_search.py

Implementation notes:
	- We subclass `Engine` only to relax the num_players check.
	- The loop seeds a deterministic RNG for reproducibility.
	- Uses `GreedyAgent` by default (configurable).
	- Stops when score >= 15 or only NOOP actions remain.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import random

from gems.engine import Engine
from gems.agents.greedy import GreedyAgent, quick_score
from gems.typings import ActionType


def expand_search(engine: Engine, top_n: int = 3) -> list[Engine]:
  if engine.game_end():
    return []
  actions = engine.get_legal_actions()
  if all(a.type == ActionType.NOOP for a in actions):
    return []
  action_score = [(a, quick_score(engine._state, 0, a)) for a in actions]
  action_score.sort(key=lambda x: x[1], reverse=True)
  res: list[Engine] = []
  for a, s in action_score[:top_n]:
    new_engine = deepcopy(engine)
    new_engine._state = a.apply(new_engine._state)
    new_engine._action_history.append(a)
    new_engine.advance_turn()
    res.append(new_engine)
  return res


def _play_to_end(engine: Engine, debug=False) -> Engine:
  agent = GreedyAgent(seat_id=0, rng=random.Random(100))
  while not engine.game_end():
    engine.play_one_round(agents=[agent], debug=debug)
  return engine


def play_to_end(engines: list[Engine]) -> list[Engine]:
  end_engines: list[Engine] = []
  for i, e in enumerate(engines):
    if i % 100 == 0:
      print(f" {i} / {len(engines)}")
    end_engines.append(_play_to_end(e))
  return end_engines


def single_player_search(all_depth=5) -> list[Engine]:
  """Run a single-player search/simulation until win or no actions."""
  engine = Engine(num_players=1, names=["Solo"], seed=20)
  agent = GreedyAgent(seat_id=0, rng=random.Random(100))

  depth_engine_map = defaultdict(list)
  depth_engine_map[0].append(engine)
  depth = 0
  while depth < all_depth:
    next_depth = depth + 1
    current_engines = depth_engine_map[depth]
    print(f"current depth: {depth} engine nums: {len(current_engines)}")
    for e in current_engines:
      depth_engine_map[next_depth].extend(expand_search(e, top_n=5))
    depth += 1
  print("expand finish")
  return depth_engine_map[depth]


def single_play():
  engine = Engine(num_players=1, names=["Solo"])
  agent = GreedyAgent(seat_id=0, rng=random.Random(100))
  print("Initialized game state:\n")
  engine.print_summary()

  while not engine.game_end():
    engine.play_one_round(agents=[agent])

  winners = engine.game_winners()
  if winners:
    print("Game finished — winner(s):")
    for w in winners:
      print(f"  seat={w.seat_id} name={w.name!r} score={w.score} cards={len(w.purchased_cards)} reserved={len(w.reserved_cards)}")
  else:
    print("All players have only noop actions. Ending game.")


# %%
start_engines = single_player_search()
end_engines = play_to_end(start_engines)


# %%
engine_score_list: list[tuple[Engine, int]] = []
for i, e in enumerate(end_engines):
  if i % 100 == 0:
    print(f" {i} / {len(end_engines)}")
  player = e._state.players[0]
  score = player.score
  engine_score_list.append((e, score))
engine_score_list.sort(key=lambda x: (x[1], -x[0]._state.turn), reverse=True)
for e, score in engine_score_list[:20:2]:
  print("==" * 20)
  print(f"score: {score}, turns: {e._state.turn}")
# %%
for a in engine_score_list[0][0]._action_history:
  print(a)

print("--" * 20)
for a in engine_score_list[1][0]._action_history:
  print(a)
# %%
if __name__ == "__main__":
  single_player_search()
