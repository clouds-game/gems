# %%
from tqdm.notebook import tqdm
from _common import RES_DIR

import json
from pathlib import Path

from gems.agents.core import Agent, BaseAgent
from gems.consts import GameConfig
from gems.engine import Engine
from gems.agents.random import RandomAgent
from gems.agents.greedy import GreedyAgent
import matplotlib.pyplot as plt

from gems.state import GameState
# %%

Simulation_Dir = RES_DIR / "simulations"
RandomAgentFile = Simulation_Dir / "RandomAgent.jsonl"
GreedyAgentFile = Simulation_Dir / "GreedyAgent.jsonl"


def run_simulations(n: int, config: GameConfig, agents: list[BaseAgent], debug: bool = False) -> list[Engine]:
  """Run `n` independent games using `agents` for each seat.
  - n: number of full rounds to play (each run starts a fresh Engine)
  - agents: list of agents, one per player seat.
  """

  num_players = config.num_players
  assert len(agents) == num_players
  engines = []
  for i in tqdm(range(n), desc="Running simulations"):
    seed = 1234 + i
    engine: Engine = Engine.new(num_players=num_players, seed=seed, config=config)
    while not engine.game_end():
      engine.play_one_round(agents=agents, debug=debug)
    engines.append(engine)
  return engines


def save_engines(engines: list[Engine], output_file: Path):
  output_file.parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, "a", encoding="utf-8") as f:
    for e in engines:
      json.dump(e.serialize(), f, ensure_ascii=False)
      f.write("\n")


def load_engines(input_file: Path) -> list[Engine]:
  with open(input_file, "r", encoding="utf-8") as f:
    engines_data = [json.loads(line) for line in f]
  return [Engine.deserialize(data) for data in engines_data]

# %%


# def _get_save_file(agent_cls: type[BaseAgent], num_players: int) -> Path:
#   """Get the save file path for a specific agent class and number of players."""
#   match agent_cls:
#     case _ if issubclass(agent_cls, RandomAgent):
#       return Simulation_Dir / f"random_agent_{num_players}_players.jsonl"
#     case _ if issubclass(agent_cls, GreedyAgent):
#       return Simulation_Dir / f"greedy_agent_{num_players}_players.jsonl"
#     case _:
#       raise ValueError(f"Unsupported agent class: {agent_cls}")


def play_and_save(n_games: int, num_players: int, agents: list[BaseAgent], path: Path) -> None:
  config = GameConfig(num_players=num_players)
  engines = run_simulations(n_games, config, agents)
  save_engines(engines, path)


def load_and_replay(path: Path) -> list[list[GameState]]:
  engines = load_engines(path)
  states_list: list[list[GameState]] = []
  for engine in tqdm(engines, desc="replay game"):
    states = engine.replay()
    states_list.append(states)
  return states_list


# %%
# play_and_save(5, GreedyAgent, num_players=1)
# play_and_save(5, RandomAgent, num_players=1)


# %%

def _extract_scores(states_list: list[list[GameState]], seat_id: int) -> list[list[int]]:
  scores_list: list[list[int]] = []
  for states in states_list:
    scores: list[int] = [state.players[seat_id].score for state in states]
    scores_list.append(scores)
  return scores_list


def single_player_extract_scores(states_list: list[list[GameState]]) -> list[list[int]]:
  return _extract_scores(states_list, seat_id=0)


def _average_scores(scores_list: list[list[int]]) -> list[float]:
  max_len = max(len(scores) for scores in scores_list)
  averages: list[float] = []
  for turn in range(max_len):
    # gather scores for this turn from games that lasted at least this long
    vals: list[int] = [scores[turn] if len(scores) > turn else scores[-1] for scores in scores_list]
    averages.append(sum(vals) / len(vals))
  return averages


def single_player_extract_average_scores(states_list: list[list[GameState]]) -> list[float]:
  """Compute per-turn average scores across multiple games.
  - states_list: list of game states (one per game). Games may have
    different lengths; shorter games contribute only to their existing turns.

  Returns a list of floats where element i is the average score at turn i+1.
  """
  if not states_list:
    return []
  scores_list = single_player_extract_scores(states_list)
  return _average_scores(scores_list)


def multiplayer_extract_average_scores(states_list: list[list[GameState]]) -> list[list[float]]:
  """Compute per-turn average scores for each player across multiple games.
  - states_list: list of game states (one per game). Games may have
    different lengths; shorter games contribute only to their existing turns.

  Returns a list of lists where element i is a list of average scores
  for each player at turn i+1.
  """
  if not states_list:
    return []
  num_players = len(states_list[0][0].players)

  avg_scores_list = []
  for seat_id in range(num_players):
    scores_list = _extract_scores(states_list, seat_id)
    avg_scores = _average_scores(scores_list)
    avg_scores_list.append(avg_scores)
  return avg_scores_list

# %%


def plot_scores(score_lists: list[list[int]] | list[list[float]], labels: list[str] | None = None) -> None:
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
    plt.plot(x, scores, marker=".", label=label)

  plt.xlabel("Turn")
  plt.ylabel("Score")
  plt.title("Score over turn")
  plt.legend(loc="upper left")
  plt.grid(True, linestyle="--", alpha=0.4)
  plt.show()

# %%


# %%
greedy_states_list = load_and_replay(Simulation_Dir / "greedy_agent_1_players.jsonl")
random_states_list = load_and_replay(Simulation_Dir / "random_agent_1_players.jsonl")
plot_scores(single_player_extract_scores(greedy_states_list))
plot_scores(single_player_extract_scores(random_states_list))
plot_scores([
    single_player_extract_average_scores(greedy_states_list),
    single_player_extract_average_scores(random_states_list)
], labels=["Greedy", "Random"])

# %%
plot_scores(multiplayer_extract_average_scores(load_and_replay(Simulation_Dir /
            "greedy_agent_3_players.jsonl")), labels=["Player 1", "Player 2", "Player 3"])
# %%
