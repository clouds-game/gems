# %%
from _common import RES_DIR
from scripts.simulation.utils import load_and_replay
from simulation import get_simulation_config, load_simulation_result, replay_engine, load_engines, play_and_save, load_and_replay, plot_rounds, plot_scores, EXTRACTORS
from gems.agents.core import AGENT_METADATA_HISTORY_ROUND, Agent
import json

Simulation_Dir = RES_DIR / "simulations"


def run():
  config = get_simulation_config().run_config
  config.exec(base_dir=Simulation_Dir)


def display_scores():
  config = get_simulation_config().score_config

  if config.extractor not in EXTRACTORS:
    raise ValueError(f"Extractor '{config.extractor}' not found.")
  extractor, need_group = EXTRACTORS[config.extractor]
  print(f"Using extractor: {config.extractor}, need_group={need_group}")
  if need_group:
    score_lists = []
    for filename in config.filenames:
      states_list, _ = load_and_replay(Simulation_Dir / filename)
      score_lists.append(extractor(states_list))
    plot_scores(score_lists, labels=config.labels)
  else:
    states_list, _ = load_and_replay(Simulation_Dir / config.filenames[0])
    plot_scores(extractor(states_list), labels=config.labels)


def display_finish_round_distribution():
  config = get_simulation_config().finish_round_config
  for filename in config.filenames:
    input_file = Simulation_Dir / filename
    with open(input_file, "r", encoding="utf-8") as f:
      engines_data = [json.loads(line) for line in f]
    game_rounds: list[int] = [len(data["action_history"]) // data['num_players']
                              for data in engines_data]
    plot_rounds(game_rounds, label=filename)


# %%


def winrate():
  config = get_simulation_config().winrate_config
  _, engines = load_and_replay(Simulation_Dir / config.filename)
  res = {}
  for i, engine in enumerate(engines):
    winners = engine.game_winners()
    for winner in winners:
      res[winner.seat_id] = res.get(winner.seat_id, 0) + 1
  print("total games:", len(engines))
  print("win rates:", {k: v / len(engines) for k, v in res.items()})


def display_actions():
  config = get_simulation_config().action_config
  engines, agents_metadata_list = load_simulation_result(
      Simulation_Dir / config.filename, start=config.line, end=config.line + 1)

  assert len(engines) == 1
  assert len(agents_metadata_list) == 1
  engine = engines[0]
  agents_metadata = agents_metadata_list[0]

  states = engine.replay()
  action_history = engine._action_history
  num_players = engine.config.num_players
  actions_list = [action_history[i:i + num_players]
                  for i in range(0, len(action_history), num_players)]

  assert len(states) == len(actions_list) + 1
  score_list = [[] for _ in range(num_players)]
  for state in states:
    for seat_id in range(num_players):
      score_list[seat_id].append(state.players[seat_id].score)
  plot_scores(score_list, labels=[f"Player {i}" for i in range(num_players)])

  # display details
  states[0].print_summary()
  for i, (state, actions) in enumerate(zip(states[1:], actions_list)):
    print("==" * 20)
    Agent.print_metadata_round(agents_metadata, i)
    print(f"Actions:")
    for action in actions:
      print(f"  {action}")
    state.print_summary()


# %%
import _common
from gems.agents.target import TargetAgent
from gems.engine import Engine


def test_target_agent():
  agents = [TargetAgent(seat_id=0, debug=True)]
  agents[0].reset(seed=42)
  engine = Engine.new(num_players=1, seed=42)
  while not engine.game_end():
    engine.play_one_round(agents=agents)
    print("===" * 20)


# %%
from _common import RES_DIR
import json
input_file = RES_DIR / "simulations/greedy_vs_target.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
  engines_data = [json.loads(line) for line in f]
lengths = [len(data["action_history"]) for data in engines_data]
lengths.sort()
max_length = max(lengths)
datas = [(i, data) for i, data in enumerate(engines_data)
         if len(data["action_history"]) == max_length]
avg: float = sum(lengths) / len(lengths)
avg

# %%
run()
# %%
display_scores()
# %%
display_finish_round_distribution()
# %%
winrate()
# %%
display_actions()
# %%
