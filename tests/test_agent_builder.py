from dataclasses import asdict
import pytest

from gems.agents.core import AgentBuilder, Agent
from gems.agents.greedy import GreedyAgent, GreedyAgentEvaluationV1


def test_agent_builder_build_greedy():
  evaluation = GreedyAgentEvaluationV1()
  builder = AgentBuilder(
      cls_name="GreedyAgent",
      seat_id=2,
      name="GreedyAgent",
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
  builder = AgentBuilder(cls_name="GreedyAgent", name="GreedyAgent", seat_id=0)
  with pytest.raises(ValueError):
    builder.build()


def test_agent_builder_unknown_class_raises():
  evaluation = GreedyAgentEvaluationV1()
  builder = AgentBuilder(cls_name="NotExistAgent", seat_id=0,
                         name="NotExistAgent", kwargs={"evaluation": evaluation})
  with pytest.raises(ValueError):
    builder.build()


def test_agent_builder_json_roundtrip():
  evaluation = GreedyAgentEvaluationV1()
  builder = AgentBuilder(
      cls_name="GreedyAgent",
      seat_id=1,
      name="GreedyAgent",
      kwargs={"evaluation": evaluation},
      config=asdict(evaluation.config),
  )
  data = builder.model_dump_json()
  builder2 = AgentBuilder.model_validate_json(data)
  assert builder2.cls_name == builder.cls_name
  assert builder2.seat_id == builder.seat_id
  assert builder2.kwargs is None
  assert builder2.config == builder.config


def test_agent_builder_method_on_agent_instance():
  # Ensure calling agent.builder() yields a usable AgentBuilder
  agent = GreedyAgent(seat_id=3, seed=999, evaluation=GreedyAgentEvaluationV1())
  builder = agent.builder()
  # Basic properties
  assert builder.cls_name == agent.__class__.__name__
  assert builder.seat_id == agent.seat_id
  assert builder.kwargs is not None, "builder.kwargs should contain evaluation for GreedyAgent"
  assert "evaluation" in builder.kwargs
  # Config roundtrip
  assert builder.config == asdict(agent.evaluation.config)
  # Build a new agent from builder and compare evaluation config
  built_agent = builder.build()
  assert isinstance(built_agent, GreedyAgent)
  assert built_agent.seat_id == agent.seat_id
  assert asdict(built_agent.evaluation.config) == asdict(agent.evaluation.config)
  # Evaluation object identity may or may not be preserved; ensure at least same type
  assert built_agent.evaluation.__class__ is agent.evaluation.__class__
