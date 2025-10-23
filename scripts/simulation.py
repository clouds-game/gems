# %%
from simulation_utils import get_simulation_config, load_engines, play_and_save, load_and_replay, Simulation_Dir, plot_scores, EXTRACTORS
from gems.agents.core import Agent


def run():
  config = get_simulation_config().run_config
  play_and_save(config)


def display_scores():
  config = get_simulation_config().score_config

  if config.extractor not in EXTRACTORS:
    raise ValueError(f"Extractor '{config.extractor}' not found.")
  extractor, need_group = EXTRACTORS[config.extractor]
  if need_group:
    score_lists = []
    for filename in config.filenames:
      states_list, _ = load_and_replay(Simulation_Dir / filename)
      score_lists.append(extractor(states_list))
    plot_scores(score_lists)
  else:
    states_list, _ = load_and_replay(Simulation_Dir / config.filenames[0])
    plot_scores(extractor(states_list), labels=config.labels)

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
  engines = load_engines(Simulation_Dir / config.filename, start=config.line, end=config.line + 1)

  assert len(engines) == 1
  engine = engines[0]

  states = engine.replay()
  action_history = engine._action_history
  agent_metadata = engine._metadata.agent_metadata
  num_players = engine.config.num_players
  actions_list = [action_history[i:i + num_players]
                  for i in range(0, len(action_history), num_players)]
  agent_metadata_list = [agent_metadata[i:i + num_players]
                         for i in range(0, len(agent_metadata), num_players)]

  assert len(states) == len(actions_list) + 1
  states[0].print_summary()
  for state, actions, metadata in zip(states[1:], actions_list, agent_metadata_list):
    print("==" * 20)
    print("Agent Metadata:")
    for meta in metadata:
      if meta:
        print(f"  {Agent.metadata_str(meta)}")
    print(f"Actions:")
    for action in actions:
      print(f"  {action}")
    state.print_summary()


# %%
# import _common
# from gems.agents.target import TargetAgent
# from gems.engine import Engine

# agents = [TargetAgent(seat_id=0, debug=True)]
# engine = Engine.new(num_players=1, seed=42)
# while not engine.game_end():
#   engine.play_one_round(agents=agents)
#   print("===" * 20)
