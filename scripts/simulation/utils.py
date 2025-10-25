
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import TypeAlias

from tqdm import tqdm

from gems.agents.core import Agent, BaseAgent
from gems.agents.greedy import GreedyAgent
from gems.agents.random import RandomAgent
from gems.agents.target import TargetAgent
from gems.consts import GameConfig
from gems.engine import Engine, Replay
from gems.state import GameState
from .core import SimulationResult, export_to_replay, apply_replays, run_simulations

PathLike: TypeAlias = str | Path


def get_simulation_config(filename: PathLike = "simulation_config.toml"):
  from .config import SimulationConfig
  with open(filename, "rb") as f:
    data = tomllib.load(f)
  return SimulationConfig(data)


def play_and_save(agents: list[BaseAgent], game_config: GameConfig, *, count: int = 100, output_file: PathLike) -> None:
  output_file = Path(output_file)
  if output_file.exists():
    return
  results = run_simulations(count, game_config, agents)
  replays = [r.replay for r in results]
  save_replays(replays, output_file)


def load_and_replay(path: PathLike, *, start: int | None = None, end: int | None = None) -> list[SimulationResult]:
  replays = load_replays(path, start=start, end=end)
  return apply_replays(replays)


def save_replays(replays: list[Replay], output_file: PathLike, mode="a"):
  output_file = Path(output_file)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, mode, encoding="utf-8") as f:
    for r in replays:
      jsonl = r.model_dump_json()
      f.write(f"{jsonl}\n")


def load_replays(input_file: PathLike, start: int | None = None, end: int | None = None) -> list[Replay]:
  with open(input_file, "r", encoding="utf-8") as f:
    lines = list(f)
  if end is not None:
    lines = lines[:end]
  if start is not None:
    lines = lines[start:]
  res = []
  for data in tqdm(lines, desc="Loading replays"):
    r = Replay.model_validate_json(data)
    res.append(r)
  return res


# TODO change to list
def get_win_counts(engines: list[Engine]) -> dict[int, int]:
  """Calculate win counts for each seat_id from a list of engines."""
  win_counts: dict[int, int] = {}
  for i in range(engines[0].config.num_players):
    win_counts[i] = 0

  for engine in engines:
    winners = engine.game_winners()
    for winner in winners:
      win_counts[winner.seat_id] += 1
  return win_counts


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
