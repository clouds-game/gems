# %%
from collections.abc import Sequence
from pathlib import Path
from _common import RES_DIR
from gems.agents.greedy import GreedyAgent
from gems.agents.random import RandomAgent
from gems.agents.target import TargetAgent
from gems.consts import GameConfig
from simulation.core import SimulationResult
from simulation.extractors import multiplayer_extract_average_scores, single_player_extract_average_scores
from simulation.plot import plot_winrate
from simulation.utils import get_win_counts
from simulation import play_and_save, load_and_replay, plot_rounds, plot_scores, EXTRACTORS
from gems.agents.core import Agent, BaseAgent

Simulation_Dir = RES_DIR / "simulations"
Output_Dir = RES_DIR / "output"

CANDIDATES = [RandomAgent, TargetAgent, GreedyAgent]
SIMULATION_NUM = 5

# %%



def init_agents(agent_cls: Sequence[type[BaseAgent]]) -> list[BaseAgent]:
  return [cls(seat_id=i, name=None) for i, cls in enumerate(agent_cls)]

def play(agents: list[BaseAgent], all_visiable: bool = False, count: int = SIMULATION_NUM, output_dir = Simulation_Dir):
  if all_visiable:
    game_config = GameConfig(num_players=len(agents), card_visible_count=100)
  else:
    game_config = GameConfig(num_players=len(agents))

  names = "".join([f"[{agent.name}]" for agent in agents])

  filename = f"run_{names}{"_all_visiable" if all_visiable else ""}.jsonl"
  output_file = output_dir / filename
  play_and_save(agents, game_config, count=count, output_file=output_file)
  return output_file

agents_one_player = [
  (AgentClass,)
  for AgentClass in CANDIDATES
]
agents_two_players = [
  (AgentClass1, AgentClass2)
  for AgentClass1 in CANDIDATES
  for AgentClass2 in CANDIDATES
]

files: list[Path] = []
for agent_cls in agents_one_player + agents_two_players:
  files.append(play(init_agents(agent_cls)))
  files.append(play(init_agents(agent_cls), all_visiable=True))

# %%
results = [load_and_replay(f) for f in files]

# %%
def analysis(results: list[SimulationResult], labels: list[str], output_dir: Path):
  states_list = [result.states for result in results]
  if results[0].num_players == 1:
    scores_list = [single_player_extract_average_scores(states_list)]
  else:
    scores_list = multiplayer_extract_average_scores(states_list)
  fig = plot_scores(scores_list, labels=labels)
  fig.savefig(output_dir / f"average_scores.png")

  finish_rounds = [result.num_rounds for result in results]

  fig = plot_rounds(finish_rounds)
  fig.savefig(output_dir / f"finish_round_distribution.png")

  win_counts = get_win_counts([result.engine for result in results])
  fig = plot_winrate(win_counts, total_games=len(results), player_labels=labels)
  fig.savefig(output_dir / f"winrate.png")

for (result, file) in zip(results, files):
  filename = result[0].filename
  if filename is None:
    filename = file.stem
  else:
    filename = Path(filename).stem
  print(f"Loaded {len(result)} simulations from {filename}")
  labels = [f"Player {i}" for i in range(result[0].engine.config.num_players)]
  output_dir = Output_Dir / filename
  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
  if (output_dir / "average_scores.png").exists():
    print(f"Analysis for {filename} exists, skip.")
    continue
  num_players = result[0].engine.config.num_players
  analysis(result, labels=labels, output_dir=output_dir)

#%%
