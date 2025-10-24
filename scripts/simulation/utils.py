
import json
from pathlib import Path
import tomllib
from typing import TypeAlias

from tqdm import tqdm

from gems.agents.core import Agent
from gems.agents.greedy import GreedyAgent
from gems.agents.random import RandomAgent
from gems.agents.target import TargetAgent
from gems.consts import GameConfig
from gems.engine import Engine, Replay
from gems.state import GameState
from .core import export_to_replay, replay_engine, run_simulations

PathLike: TypeAlias = str | Path


def get_simulation_config(filename: PathLike = "simulation_config.toml"):
  from .config import SimulationConfig
  with open(filename, "rb") as f:
    data = tomllib.load(f)
  return SimulationConfig(data)


def play_and_save(agents: list[Agent], game_config: GameConfig, *, count: int = 100, output_file: PathLike) -> None:
  output_file = Path(output_file)
  if output_file.exists():
    return
  engines, agent_metadata_list = run_simulations(count, game_config, agents)
  replays = [export_to_replay(engine, metadata)
             for engine, metadata in zip(engines, agent_metadata_list)]
  save_replays(replays, output_file)


def load_and_replay(path: PathLike) -> tuple[list[list[GameState]], list[Engine]]:
  replays = load_replays(path)
  states_list, engines = replay_engine(replays)
  return states_list, engines


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



def get_win_counts(engines: list[Engine]) -> dict[int, int]:
  """Calculate win counts for each seat_id from a list of engines."""
  win_counts: dict[int, int] = {}

  for engine in engines:
    winners = engine.game_winners()
    for winner in winners:
      win_counts[winner.seat_id] = win_counts.get(winner.seat_id, 0) + 1
  return win_counts
