

from tqdm import tqdm
from gems.agents.core import Agent
from gems.consts import GameConfig
from gems.engine import Engine, Replay
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


def export_to_replay(engine: Engine, agent_metadata: list[dict]) -> Replay:
  replay = engine.export()
  replay.metadata["agents"] = agent_metadata
  return replay


def replay_engine(replays: list[Replay]) -> tuple[list[list[GameState]], list[Engine]]:
  states_list: list[list[GameState]] = []

  engines: list[Engine] = []
  for replay in tqdm(replays, desc="replay game"):
    states, engine = replay.replay()
    states_list.append(states)
    engines.append(engine)
  return states_list, engines
