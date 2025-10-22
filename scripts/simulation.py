# %%
from simulation_utils import get_simulation_config, play_and_save, load_and_replay, Simulation_Dir, plot_scores, EXTRACTORS


def run():
  config = get_simulation_config().run_config
  play_and_save(config)


def display():
  config = get_simulation_config().display_config

  if config.extractor not in EXTRACTORS:
    raise ValueError(f"Extractor '{config.extractor}' not found.")
  extractor, need_group = EXTRACTORS[config.extractor]
  if need_group:
    score_lists = []
    for filename in config.filenames:
      states_list = load_and_replay(Simulation_Dir / filename)
      score_lists.append(extractor(states_list))
    plot_scores(score_lists)
  else:
    states_list = load_and_replay(Simulation_Dir / config.filenames[0])
    plot_scores(extractor(states_list), labels=config.labels)

# %%
