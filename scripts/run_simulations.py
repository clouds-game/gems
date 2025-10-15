# %%
from _common import RES_DIR

import json
from pathlib import Path

from gems.agents.core import Agent, BaseAgent
from gems.engine import Engine
from gems.agents.random import RandomAgent
from gems.agents.greedy import GreedyAgent
import matplotlib.pyplot as plt
# %%

Simulation_Dir = RES_DIR / "simulations"
RandomAgentFile = Simulation_Dir / "RandomAgent.json"
GreedyAgentFile = Simulation_Dir / "GreedyAgent.json"


def run_simulations(n: int, agents: list[BaseAgent]) -> list[Engine]:
  """Run `n` independent games using `agents` for each seat.
  - n: number of full rounds to play (each run starts a fresh Engine)
  - agents: list of agents, one per player seat.
  """

  num_players = len(agents)
  engines = []
  for i in range(n):
    seed = i
    engine = Engine.new(num_players=num_players, seed=seed)
    while not engine.game_end():
      engine.play_one_round(agents=agents)
    engines.append(engine)
  return engines


def save_engines(engines: list[Engine], output_file: Path):
  output_file.parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump([e.serialize() for e in engines], f, indent=2)


def load_engines(input_file: Path) -> list[Engine]:
  with open(input_file, "r", encoding="utf-8") as f:
    engines_data = json.load(f)
  return [Engine.deserialize(data) for data in engines_data]

# %%


def play_and_save(n_games: int, agent_cls):
  n_games = 5
  agents = [agent_cls(seat_id=0)]
  engines = run_simulations(n_games, agents)

  save_file: Path | None = None
  match agent_cls:
    case _ if issubclass(agent_cls, RandomAgent):
      save_file = RandomAgentFile
    case _ if issubclass(agent_cls, GreedyAgent):
      save_file = GreedyAgentFile
    case _:
      raise ValueError(f"Unsupported agent class: {agent_cls}")
  if not save_file:
    raise ValueError("No save file specified")
  save_engines(engines, save_file)


def _display_cards(engine: Engine):
  print("Visible cards:")
  for card in engine._state.visible_cards:
    print(card)
  print("Decks")
  for lvl, cards in engine.decks_by_level.items():
    print(f"Level {lvl}: {len(cards)} cards")
    for card in cards:
      print(f"  {card}")


def get_score_lists(path: Path):
  engines = load_engines(path)
  score_lists: list[list[int]] = []
  for i, engine in enumerate(engines):
    print(f"Replaying game {i + 1}/{len(engines)}")
    scores: list[int] = []
    for action in engine._actions_to_replay:
      state_before = engine.get_state()
      engine._state = action.apply(state_before)
      engine._action_history.append(action)
      engine.advance_turn()
      scores.append(engine._state.players[0].score)
    engine._actions_to_replay = []
    score_lists.append(scores)
  return score_lists
# %%
# play_and_save(5, GreedyAgent)


# %%


def plot_score_lists(score_lists: list[list[int]] | list[list[float]], labels: list[str] | None = None) -> None:
  """Plot score progress for each replay using matplotlib.
  - score_lists: list of score sequences (one per game)
  """
  if not labels:
    labels = [f"Game {i + 1}" for i in range(len(score_lists))]
  if len(labels) != len(score_lists):
    raise ValueError("Length of labels must match length of score_lists")

  # plt.figure(figsize=(8, 4 + len(score_lists) * 0.5))
  for scores, label in zip(score_lists, labels):
    x = list(range(1, len(scores) + 1))
    plt.plot(x, scores, marker="o", label=label)

  plt.xlabel("Turn")
  plt.ylabel("Score")
  plt.title("Score over turn")
  plt.legend(loc="upper left")
  plt.grid(True, linestyle="--", alpha=0.4)

  plt.show()


# %%


def average_score_lists(score_lists: list[list[int]]) -> list[float]:
  """Compute per-turn average scores across multiple games.
  - score_lists: list of score sequences (one per game). Games may have
    different lengths; shorter games contribute only to their existing turns.

  Returns a list of floats where element i is the average score at turn i+1.
  """
  if not score_lists:
    return []
  # find the maximum length among the provided score sequences
  max_len = max(len(s) for s in score_lists)

  averages: list[float] = []
  for turn in range(max_len):
    # gather scores for this turn from games that lasted at least this long
    vals: list[int] = [s[turn] if len(s) > turn else s[-1] for s in score_lists]
    averages.append(sum(vals) / len(vals))
  return averages


# %%
greedy_score_lists = get_score_lists(GreedyAgentFile)
greedy_average_scores = average_score_lists(greedy_score_lists)

random_score_lists = get_score_lists(RandomAgentFile)
random_average_scores = average_score_lists(random_score_lists)
plot_score_lists(greedy_score_lists)
plot_score_lists(random_score_lists)
plot_score_lists([greedy_average_scores, random_average_scores], labels=["Greedy", "Random"])

# %%
