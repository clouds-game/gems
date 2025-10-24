

from tqdm import tqdm
from gems.agents.core import Agent
from gems.consts import GameConfig
from gems.engine import Engine
from gems.state import GameState


def run_simulations(n: int, config: GameConfig, agents: list[Agent], debug: bool = False) -> tuple[list[Engine], list[list[dict]]]:
  """Run `n` independent games using `agents` for each seat.
  - n: number of full rounds to play (each run starts a fresh Engine)
  - agents: list of agents, one per player seat.
  """

  num_players = config.num_players
  assert len(agents) == num_players
  engines = []
  agent_metadata_list: list[list[dict]] = []
  for i in tqdm(range(n), desc="Running simulations"):
    for agent in agents:
      agent.reset()
    seed = 1234 + i
    engine = Engine.new(num_players=num_players, seed=seed, config=config)
    while not engine.game_end():
      engine.play_one_round(agents=agents, debug=debug)
    engines.append(engine)
    agent_metadata_list.append([agent.metadata() for agent in agents])
  return engines, agent_metadata_list


def replay_engine(engines: list[Engine]) -> tuple[list[list[GameState]], list[Engine]]:
  states_list: list[list[GameState]] = []
  for engine in tqdm(engines, desc="replay game"):
    states = engine.replay()
    states_list.append(states)
  return states_list, engines

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
