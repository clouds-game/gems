

from tqdm import tqdm
from gems.agents.core import Agent
from gems.consts import GameConfig
from gems.engine import Engine
from gems.state import GameState


def run_simulations(n: int, config: GameConfig, agents: list[Agent], debug: bool = False) -> list[Engine]:
  """Run `n` independent games using `agents` for each seat.
  - n: number of full rounds to play (each run starts a fresh Engine)
  - agents: list of agents, one per player seat.
  """

  num_players = config.num_players
  assert len(agents) == num_players
  engines = []
  for i in tqdm(range(n), desc="Running simulations"):
    seed = 1234 + i
    engine = Engine.new(num_players=num_players, seed=seed, config=config)
    while not engine.game_end():
      engine.play_one_round(agents=agents, debug=debug)
    engines.append(engine)
  return engines



def replay_engine(engines: list[Engine]) -> tuple[list[list[GameState]], list[Engine]]:
  states_list: list[list[GameState]] = []
  for engine in tqdm(engines, desc="replay game"):
    states = engine.replay()
    states_list.append(states)
  return states_list, engines
