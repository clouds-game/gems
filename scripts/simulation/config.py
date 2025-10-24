from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
  agents: list[str]
  filename: str
  n_games: int
  mode: str

@dataclass(frozen=True)
class ScoreConfig:
  filenames: list[str]
  extractor: str
  labels: list[str] | None


@dataclass(frozen=True)
class FinishRoundConfig:
  filenames: list[str]


@dataclass(frozen=True)
class WinrateConfig:
  filename: str

@dataclass(frozen=True)
class ActionConfig:
  filename: str
  line: int = 0
  seat_id: int = 0

@dataclass()
class SimulationConfig:
  run_config: RunConfig
  score_config: ScoreConfig
  finish_round_config: FinishRoundConfig
  winrate_config: WinrateConfig
  action_config: ActionConfig

  def __init__(self, config_data: dict):
    self.run_config = RunConfig(**config_data["run"])
    config_data["score"]["labels"] = config_data["score"].get("labels", None)
    self.score_config = ScoreConfig(**config_data["score"])
    self.finish_round_config = FinishRoundConfig(**config_data["finish_round"])
    self.winrate_config = WinrateConfig(**config_data["winrate"])
    self.action_config = ActionConfig(**config_data["action"])
