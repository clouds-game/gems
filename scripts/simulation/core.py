

from dataclasses import dataclass
from tqdm import tqdm
from gems.agents.core import Agent, BaseAgent
from gems.consts import GameConfig
from gems.engine import Engine, Replay
from gems.state import GameState


@dataclass
class SimulationResult:
  states: list[GameState]
  engine: Engine
  replay: Replay
  agent_metadata: list[dict]
  filename: str | None

  @property
  def action_history(self) -> list:
    return self.replay.action_history

  @property
  def num_players(self) -> int:
    return self.engine.config.num_players

  @property
  def num_rounds(self) -> int:
    return len(self.action_history) // self.num_players

  @property
  def config(self) -> GameConfig:
    return self.engine.config


def run_simulations(n: int, config: GameConfig, agents: list[BaseAgent], debug: bool = False) -> list[SimulationResult]:
  """Run `n` independent games using `agents` for each seat.
  - n: number of full rounds to play (each run starts a fresh Engine)
  - agents: list of agents, one per player seat.
  """

  num_players = config.num_players
  assert len(agents) == num_players
  result: list[SimulationResult] = []
  for i in tqdm(range(n), desc="Running simulations"):
    for agent in agents:
      agent.reset()
    seed = 1234 + i
    engine = Engine.new(num_players=num_players, seed=seed, config=config)
    states = []
    while not engine.game_end():
      state = engine.play_one_round(agents=agents, debug=debug)
      states.append(state)
    agent_metadata = [agent.metadata() for agent in agents]
    replay = export_to_replay(engine, agent_metadata)
    result.append(SimulationResult(
        states=states,
        engine=engine,
        replay=replay,
        agent_metadata=agent_metadata,
        filename=None,
    ))
  return result



def export_to_replay(engine: Engine, agent_metadata: list[dict]) -> Replay:
  replay = engine.export()
  replay.metadata["agents"] = agent_metadata
  return replay


def apply_replays(replays: list[Replay]) -> list[SimulationResult]:
  result = []
  for replay in tqdm(replays, desc="replay game"):
    result.append(apply_replay(replay))
  return result


def apply_replay(replay: Replay, *, filename: str | None = None) -> SimulationResult:
  states, engine = replay.replay()
  return SimulationResult(
      states=states,
      engine=engine,
      replay=replay,
      agent_metadata=replay.metadata.get("agents", []),
      filename=filename,
  )
