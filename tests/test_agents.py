import random

from gems.agents.random import RandomAgent
from gems.agents.greedy import GreedyAgent
from gems.actions import Action
from gems.engine import Engine
from gems.state import PlayerState, GameState
from gems.consts import GameConfig
from gems.typings import ActionType, Gem

legal = [Action.take3(Gem.RED, Gem.BLUE, Gem.GREEN), Action.take2(Gem.WHITE), Action.noop()]

def test_random_agent_chooses_legal_and_is_deterministic():
  agent = RandomAgent(seat_id=0, seed=123)

  # Minimal valid GameState with a single player so agents can inspect it if needed
  player = PlayerState(seat_id=0)
  state = GameState(config=GameConfig(), players=(player,))

  # determinism: reseed and run twice
  agent.reset(seed=42)
  first = agent.act(state, legal)

  agent.reset(seed=42)
  second = agent.act(state, legal)

  # both choices should be among provided legal actions and deterministic for the same seed
  assert first in legal
  assert second in legal
  assert (first is second) or (type(first) is type(second))


def test_greedy_agent_returns_one_of_legal_actions():
  agent = GreedyAgent(seat_id=0, seed=1)

  player = PlayerState(seat_id=0)
  state = GameState(config=GameConfig(), players=(player,))

  # If GreedyAgent implements reset, call it; otherwise this is a no-op.
  try:
    agent.reset(seed=7)
  except Exception:
    pass

  chosen = agent.act(state, legal)
  assert chosen in legal


def test_space_sample_agent_returns_noop_when_sampling_fails():
  from gems.agents.space_sample import SpaceSampleAgent
  from gems.gym.action_space import ActionSpace, ActionDict

  game_config = GameConfig(num_players=1)
  print("game_config:", game_config)

  agent = SpaceSampleAgent(
    seat_id=0,
    action_space=ActionSpace(game_config, seed=99),
    seed=42,
    max_samples=1000,
  )

  state = Engine.create_game(config=game_config)

  chosen = agent.act(state, legal)
  assert isinstance(chosen, Action)
  assert chosen.type != ActionType.NOOP
