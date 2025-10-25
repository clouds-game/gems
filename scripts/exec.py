# %%
from _common import RES_DIR
from gems.agents.greedy import GreedyAgent
from gems.agents.random import RandomAgent
from gems.consts import GameConfig
from scripts.simulation.core import run_simulations
from scripts.simulation.extractors import multiplayer_extract_average_scores, single_player_extract_average_scores
from scripts.simulation.plot import plot_winrate
from scripts.simulation.utils import get_win_counts
from simulation import get_simulation_config, play_and_save, load_and_replay, plot_rounds, plot_scores, EXTRACTORS
from gems.agents.core import AGENT_METADATA_HISTORY_ROUND, Agent
import json

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


# %%

CANDIDATES = [RandomAgent, TargetAgent, GreedyAgent]
SIMULATION_NUM = 5

# TODO generate filename??
# TODO pass candidates and num of games as parameters?
# TODO TargetAgent with different configs?
# single player


def self_play(all_visiable: bool = False):
  for AgentClass in CANDIDATES:
    agents = [AgentClass(seat_id=0)]
    game_config = GameConfig(num_players=1, card_visible_count=100) if all_visiable else GameConfig(
        num_players=1)

    filename = f"{AgentClass.__name__}_self_play{"_all_visiable" if all_visiable else ""}.jsonl"
    output_file = Simulation_Dir / filename
    play_and_save(agents, game_config, count=SIMULATION_NUM, output_file=output_file)


# two players


def pairwise_play():
  for AgentClass1 in CANDIDATES:
    for AgentClass2 in CANDIDATES:
      agents = [AgentClass1(seat_id=0), AgentClass2(seat_id=1)]
      game_config = GameConfig(num_players=2)

      output_file = Simulation_Dir / f"{AgentClass1.__name__}_vs_{AgentClass2.__name__}.jsonl"
      play_and_save(agents, game_config, count=SIMULATION_NUM, output_file=output_file)

# %%


def self_play_statistics():
  for AgentClass in CANDIDATES:
    stem = f"{AgentClass.__name__}_self_play"
    data_file = Simulation_Dir / f"{stem}.jsonl"

    average_scores_fig_file = Simulation_Dir / f"{stem}_average_scores.png"
    finish_round_fig_file = Simulation_Dir / f"{stem}_finish_round_distribution.png"

    if average_scores_fig_file.exists():
      print(f"{stem} statistics exists, skip.")
    else:

      print(f"Statistics for {stem} simulation:")
      results = load_and_replay(data_file)

      states_list = [result.states for result in results]
      scores_list = single_player_extract_average_scores(states_list)
      fig = plot_scores([scores_list], labels=[f"{AgentClass.__name__}"])
      fig.savefig(average_scores_fig_file)

      finish_rounds = [result.num_rounds for result in results]

      fig = plot_rounds(finish_rounds, label=f"{stem}")
      fig.savefig(finish_round_fig_file)


def pairwise_play_statistics():
  for AgentClass in CANDIDATES:
    for OpponentClass in CANDIDATES:
      stem = f"{AgentClass.__name__}_vs_{OpponentClass.__name__}"
      data_file = Simulation_Dir / f"{stem}.jsonl"
      average_scores_fig_file = Simulation_Dir / f"{stem}_average_scores.png"
      winrate_fig_file = Simulation_Dir / f"{stem}_winrate.png"

      engines = []
      if average_scores_fig_file.exists():
        print(f"{stem} statistics exists, skip.")
      else:
        print(f"Statistics for {stem} simulation:")
        results = load_and_replay(data_file)

        states_list = [result.states for result in results]
        scores_list = multiplayer_extract_average_scores(states_list)

        fig = plot_scores(scores_list,
                          labels=[f"{AgentClass.__name__}", f"{OpponentClass.__name__}"])
        fig.savefig(average_scores_fig_file)

        win_counts = get_win_counts(engines)
        fig = plot_winrate(win_counts, total_games=len(engines),
                           player_labels=[f"{AgentClass.__name__}", f"{OpponentClass.__name__}"])
        fig.savefig(winrate_fig_file)

# %%


def main():
  self_play(all_visiable=False)
  self_play(all_visiable=True)
  pairwise_play()
  self_play_statistics()
  pairwise_play_statistics()

# %%
