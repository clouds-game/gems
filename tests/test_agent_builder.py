from dataclasses import asdict
import pytest

from gems.agents.core import AgentBuilder, Agent
from gems.agents.greedy import GreedyAgent, GreedyAgentEvaluationV1


def test_agent_builder_build_greedy():
  evaluation = GreedyAgentEvaluationV1()
  builder = AgentBuilder(
      cls_name="GreedyAgent",
      seat_id=2,
      kwargs={"evaluation": evaluation},
      config=asdict(evaluation.config),
  )
  agent = builder.build()
  assert isinstance(agent, GreedyAgent)
  assert agent.seat_id == 2
  # evaluation object should be preserved
  assert agent.evaluation.__class__.__name__ == "GreedyAgentEvaluationV1"
  assert asdict(agent.evaluation.config) == asdict(evaluation.config)


def test_agent_builder_missing_kwargs_raises():
  # kwargs None -> ValueError during build
  builder = AgentBuilder(cls_name="GreedyAgent", seat_id=0)
  with pytest.raises(ValueError):
    builder.build()


def test_agent_builder_unknown_class_raises():
  evaluation = GreedyAgentEvaluationV1()
  builder = AgentBuilder(cls_name="NotExistAgent", seat_id=0, kwargs={"evaluation": evaluation})
  with pytest.raises(ValueError):
    builder.build()


def test_agent_builder_json_roundtrip():
  evaluation = GreedyAgentEvaluationV1()
  builder = AgentBuilder(
      cls_name="GreedyAgent",
      seat_id=1,
      kwargs={"evaluation": evaluation},
      config=asdict(evaluation.config),
  )
  data = builder.model_dump_json()
  builder2 = AgentBuilder.model_validate_json(data)
  assert builder2.cls_name == builder.cls_name
  assert builder2.seat_id == builder.seat_id
  assert builder2.kwargs is None
  assert builder2.config is not None
  assert builder2.config == builder.config
