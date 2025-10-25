
from _common import RES_DIR

from gems.agents.core import Agent
from simulation import get_simulation_config, load_and_replay


Simulation_Dir = RES_DIR / "simulations"

def display_actions():
  config = get_simulation_config().action_config
  result = load_and_replay(
      Simulation_Dir / config.filename, start=config.line, end=config.line + 1)[0]

  states = result.states
  engine = result.engine
  replay = result.replay
  num_players = engine.config.num_players
  actions_list = [replay.action_history[i:i + num_players]
                  for i in range(0, len(replay.action_history), num_players)]

  # states = engine.replay()
  # action_history = engine._action_history
  # num_players = engine.config.num_players

  # assert len(states) == len(actions_list) + 1
  # score_list = [[] for _ in range(num_players)]
  # for state in states:
  #   for seat_id in range(num_players):
  #     score_list[seat_id].append(state.players[seat_id].score)
  # plot_scores(score_list, labels=[f"Player {i}" for i in range(num_players)])

  # display details
  states[0].print_summary()
  for i, (state, actions) in enumerate(zip(states[1:], actions_list)):
    print("==" * 20)
    Agent.print_metadata_round(result.agent_metadata, i)
    print(f"Actions:")
    for action in actions:
      print(f"  {action}")
    state.print_summary()


# %%
import _common
from gems.agents.target import TargetAgent, TargetAgentEvaluationV1, TargetAgentEvaluationV1Config
from gems.engine import Engine


def test_target_agent():
  config = TargetAgentEvaluationV1Config(gem_score=11.0)
  evaluation = TargetAgentEvaluationV1(config=config)
  agents = [TargetAgent(seat_id=0, evaluation=evaluation, debug=True)]
  agents[0].reset(seed=42)
  engine = Engine.new(num_players=1, seed=42)
  while not engine.game_end():
    engine.play_one_round(agents=agents)
    print("===" * 20)
