
from pathlib import Path
import tomllib
from typing import TypeAlias

from gems.agents.core import Agent
from gems.agents.greedy import GreedyAgent
from gems.agents.random import RandomAgent
from gems.agents.target import TargetAgent
from gems.consts import GameConfig
from .core import run_simulations, save_engines
from .config import RunConfig, SimulationConfig

PathLike: TypeAlias = str | Path

def get_simulation_config(base_dir: PathLike = ".") -> SimulationConfig:
  with open(Path(base_dir) / "simulation_config.toml", "rb") as f:
    data = tomllib.load(f)
  return SimulationConfig(data)


def play_and_save(run_config: RunConfig, base_dir: PathLike) -> None:
  num_players = len(run_config.agents)
  game_config = GameConfig(num_players=num_players)

  agents = instantiate_agents(run_config.agents)
  engines = run_simulations(run_config.n_games, game_config, agents)
  output_file = Path(base_dir) / run_config.filename
  save_engines(engines, output_file, mode=run_config.mode)


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
