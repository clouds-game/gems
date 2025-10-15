from scripts._common import RES_DIR

import json
from pathlib import Path

from gems.agents.core import Agent, BaseAgent
from gems.engine import Engine
from gems.agents.random import RandomAgent
from gems.agents.greedy import GreedyAgent

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
  output_file.mkdir(parents=True, exist_ok=True)
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump([e.serialize() for e in engines], f, indent=2)


def load_engines(output_file: Path) -> list[Engine]:
  with open(output_file, "r", encoding="utf-8") as f:
    engines_data = json.load(f)
  return [Engine.deserialize(data) for data in engines_data]

# %%


def play_and_save():
  n_games = 50
  agents = [RandomAgent(seat_id=0)]
  engines = run_simulations(n_games, agents)
  save_engines(engines, RandomAgentFile)


# %%
play_and_save()
