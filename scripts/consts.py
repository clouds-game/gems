from gems.agents.greedy import GreedyAgent, GreedyAgentEvaluationV1, GreedyAgentEvaluationV1Config
from gems.agents.target import TargetAgent, TargetAgentEvaluationV1, TargetAgentEvaluationV1Config


GREEDY1_1 = GreedyAgentEvaluationV1Config()
GREEDY1_2 = GreedyAgentEvaluationV1Config(
    extra_point_per_card=0.5,
    point_score=40.0,
    bonus_score=10.0,
    gold_cost_score=1.0,
    gem_cost_score=2.0,
)
TARGET1_1 = TargetAgentEvaluationV1Config()
TARGET1_2 = TargetAgentEvaluationV1Config(
    extra_point_per_card=0.5,
    point_score=40.0,
    bonus_score=10.0,
    gold_cost_score=1.0,
    gem_cost_score=2.0,
)


def get_greedy_evaluation_config(major: int, minor: int) -> GreedyAgentEvaluationV1Config:
  const_name = f"GREEDY{major}_{minor}"
  try:
    return globals()[const_name]
  except KeyError:
    raise ValueError(f"Unknown GreedyAgent evaluation config: {const_name}")


def get_greedy_agent(major: int, minor: int, seat_id: int) -> GreedyAgent:
  assert major == 1
  config = get_greedy_evaluation_config(major, minor)
  return GreedyAgent(seat_id=seat_id, evaluation=GreedyAgentEvaluationV1(config=config), name=f"GreedyAgentV{major}_{minor}")


def get_target_evaluation_config(major: int, minor: int) -> TargetAgentEvaluationV1Config:
  const_name = f"TARGET{major}_{minor}"
  try:
    return globals()[const_name]
  except KeyError:
    raise ValueError(f"Unknown TargetAgent evaluation config: {const_name}")


def get_target_agent(major: int, minor: int, seat_id: int) -> TargetAgent:
  assert major == 1
  config = get_target_evaluation_config(major, minor)
  return TargetAgent(seat_id=seat_id, evaluation=TargetAgentEvaluationV1(config=config), name=f"TargetAgentV{major}_{minor}")
