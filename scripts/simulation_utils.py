# %%
from collections.abc import Callable
from dataclasses import dataclass
from tqdm.notebook import tqdm
from _common import RES_DIR

import json
from pathlib import Path

from gems.agents.core import Agent, BaseAgent
from gems.agents.target import TargetAgent
from gems.consts import GameConfig
from gems.engine import Engine
from gems.agents.random import RandomAgent
from gems.agents.greedy import GreedyAgent
import matplotlib.pyplot as plt
import tomllib


from gems.state import GameState
# %%

Simulation_Dir = RES_DIR / "simulations"


@dataclass(frozen=True)
class RunConfig:
  agents: list[str]
  filename: str
  n_games: int
  mode: str


@dataclass(frozen=True)
class ScoreConfig:
  filenames: list[str]
  extractor: str
  labels: list[str] | None


@dataclass(frozen=True)
class FinishRoundConfig:
  filenames: list[str]


@dataclass(frozen=True)
class WinrateConfig:
  filename: str


@dataclass(frozen=True)
class ActionConfig:
  filename: str
  line: int = 0
  seat_id: int = 0


@dataclass()
class SimulationConfig:
  run_config: RunConfig
  score_config: ScoreConfig
  finish_round_config: FinishRoundConfig
  winrate_config: WinrateConfig
  action_config: ActionConfig

  def __init__(self, config_data: dict):
    self.run_config = RunConfig(**config_data["run"])
    config_data["score"]["labels"] = config_data["score"].get("labels", None)
    self.score_config = ScoreConfig(**config_data["score"])
    self.finish_round_config = FinishRoundConfig(**config_data["finish_round"])
    self.winrate_config = WinrateConfig(**config_data["winrate"])
    self.action_config = ActionConfig(**config_data["action"])


def get_simulation_config() -> SimulationConfig:
  with open("simulation_config.toml", "rb") as f:
    data = tomllib.load(f)
  return SimulationConfig(data)


def instantiate_agents(agent_names: list[str]) -> list[Agent]:
  """Instantiate agents from their class names."""
  agents: list[Agent] = []
  for seat_id, name in enumerate(agent_names):
    match name:
      case "RandomAgent":
        agents.append(RandomAgent(seat_id=seat_id))
      case "GreedyAgent":
        agents.append(GreedyAgent(seat_id=seat_id))
      case "TargetAgent":
        agents.append(TargetAgent(seat_id=seat_id))
      case _:
        raise ValueError(f"Unsupported agent name: {name}")
  return agents

# %%


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


def save_engines(engines: list[Engine], output_file: Path, mode="a"):
  output_file.parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, mode, encoding="utf-8") as f:
    for e in engines:
      json.dump(e.serialize(), f, ensure_ascii=False)
      f.write("\n")


def load_engines(input_file: Path, start: int | None = None, end: int | None = None) -> list[Engine]:
  with open(input_file, "r", encoding="utf-8") as f:
    engines_data = [json.loads(line) for line in f]
  if end is not None:
    engines_data = engines_data[:end]
  if start is not None:
    engines_data = engines_data[start:]
  res = []
  for data in tqdm(engines_data, desc="Loading engines"):
    res.append(Engine.deserialize(data))
  return res

# %%


def play_and_save(run_config: RunConfig) -> None:
  num_players = len(run_config.agents)
  game_config = GameConfig(num_players=num_players)

  agents = instantiate_agents(run_config.agents)
  engines = run_simulations(run_config.n_games, game_config, agents)
  output_file = Simulation_Dir / run_config.filename
  save_engines(engines, output_file, mode=run_config.mode)


def load_and_replay(path: Path) -> tuple[list[list[GameState]], list[Engine]]:
  engines = load_engines(path)
  states_list: list[list[GameState]] = []
  for engine in tqdm(engines, desc="replay game"):
    states = engine.replay()
    states_list.append(states)
  return states_list, engines


# %%


# %%
EXTRACTORS: dict[str, tuple[Callable, bool]] = {}


def extractor(need_group=False):
  def decorator(func):
    EXTRACTORS[func.__name__] = (func, need_group)

    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)
    return wrapper
  return decorator


def _extract_scores(states_list: list[list[GameState]], seat_id: int) -> list[list[int]]:
  scores_list: list[list[int]] = []
  for states in states_list:
    scores: list[int] = [state.players[seat_id].score for state in states]
    scores_list.append(scores)
  return scores_list


@extractor()
def single_player_extract_scores(states_list: list[list[GameState]]) -> list[list[int]]:
  return _extract_scores(states_list, seat_id=0)


def _average_scores(scores_list: list[list[int]]) -> list[float]:
  max_len = max(len(scores) for scores in scores_list)
  averages: list[float] = []
  for turn in range(max_len):
    # gather scores for this turn from games that lasted at least this long
    vals: list[int] = [scores[turn] for scores in scores_list if len(scores) > turn]
    averages.append(sum(vals) / len(vals))
  return averages


@extractor(True)
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


@extractor()
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


def plot_rounds(finish_rounds: list[int], label: str | None = None) -> None:
  plt.hist(finish_rounds, bins=range(0, max(finish_rounds) + 2, 1))
  plt.xlabel("Number of rounds to finish")
  plt.ylabel("Number of games")
  label = label or "Distribution of number of rounds to finish games"
  plt.title(label)
  plt.show()

# %%
# greedy_states_list = load_and_replay(Simulation_Dir / "greedy_agent_1_players.jsonl")
# random_states_list = load_and_replay(Simulation_Dir / "random_agent_1_players.jsonl")
# plot_scores(single_player_extract_scores(greedy_states_list))
# plot_scores(single_player_extract_scores(random_states_list))
# plot_scores([
#     single_player_extract_average_scores(greedy_states_list),
#     single_player_extract_average_scores(random_states_list)
# ], labels=["Greedy", "Random"])

# # %%
# plot_scores(multiplayer_extract_average_scores(load_and_replay(Simulation_Dir /
#             "greedy_agent_3_players.jsonl")), labels=["Player 1", "Player 2", "Player 3"])
# %%
