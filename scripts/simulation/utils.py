
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
from gems.engine import Engine
from gems.state import GameState
from .core import replay_engine, run_simulations, save_simulation_result

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
  save_simulation_result(engines, output_file, mode="w")


def load_and_replay(path: PathLike) -> tuple[list[list[GameState]], list[Engine]]:
  engines = load_engines(path)
  states_list, engines = replay_engine(engines)
  return states_list, engines


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


def save_engines(engines: list[Engine], output_file: PathLike, mode="a"):
  output_file = Path(output_file)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  with open(output_file, mode, encoding="utf-8") as f:
    for e in engines:
      json.dump(e.serialize(), f, ensure_ascii=False)
      f.write("\n")

def load_engines(input_file: PathLike, start: int | None = None, end: int | None = None) -> list[Engine]:
  with open(input_file, "r", encoding="utf-8") as f:
    lines = list(f)
  if end is not None:
    lines = lines[:end]
  if start is not None:
    lines = lines[start:]
  res = []
  for data in tqdm(lines, desc="Loading engines"):
    res.append(Engine.deserialize(json.loads(data)))
  return res
