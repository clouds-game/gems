import numpy as np
import pytest

from gems import Engine
from gems.state import GameState
from gems.gym_env import GemEnv


def test_engine_init_and_get_state():
  e = Engine.new(2, ["P1", "P2"])
  state = e.get_state()
  assert isinstance(state, GameState)
  assert len(state.players) == 2
  assert state.turn == 0


def test_engine_reset_changes_state():
  e = Engine.new(2, ["A", "B"])
  e.reset(3, ["X", "Y", "Z"])
  s = e.get_state()
  assert len(s.players) == 3


def test_init_game_invalid_count_raises():
  with pytest.raises(ValueError):
    Engine.new(0)


def test_serialize_deserialize_replay_roundtrip():
  from gems.actions import Action
  from gems.typings import Gem

  e1 = Engine.new(2, ["P1", "P2"], seed=123)
  # pick a deterministic simple action: take 3 different gems
  act = Action.take3(Gem.RED, Gem.BLUE, Gem.WHITE)
  # apply action and record it as engine would
  s0 = e1.get_state()
  e1._state = act.apply(s0)
  e1.advance_turn()
  e1._action_history.append(act)

  data = e1.serialize()
  e2 = Engine.deserialize(data)
  # after deserialize the actions should be staged for replay
  assert hasattr(e2, '_actions_to_replay')
  assert len(e2._actions_to_replay) == 1
  # now apply replay and ensure states and histories match
  e2.apply_replay()
  assert e2.get_state() == e1.get_state()


def test_serialize_deserialize_assets_roundtrip():
  # ensure that decks_by_level and roles_deck are the same after
  # serialize -> deserialize (assets are shuffled deterministically
  # using the provided seed)
  e1 = Engine.new(3, ["A", "B", "C"], seed=42)
  # copy the structures for comparison
  decks1 = {lvl: list(deck) for lvl, deck in e1.decks_by_level.items()}
  roles1 = list(e1.roles_deck)

  data = e1.serialize()
  e2 = Engine.deserialize(data)
  # after deserialize the assets should be available on the engine
  # (Engine.new called during deserialize uses the same seed)
  assert isinstance(e2.decks_by_level, dict)
  assert isinstance(e2.roles_deck, list)

  # decks_by_level should have same levels and same card identities/order
  for lvl, deck in decks1.items():
    assert lvl in e2.decks_by_level
    assert list(e2.decks_by_level[lvl]) == deck

  # roles_deck should match
  assert list(e2.roles_deck) == roles1


def test_gym_env_reset_shapes_and_mask():
  env = GemEnv(num_players=3, seat_id=1, max_actions=64, seed=123)
  try:
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.int32

    assert info['max_actions'] == env.max_actions
    legal_count = info['legal_action_count']
    assert legal_count > 0

    mask = info['action_mask']
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (env.max_actions,)
    assert mask.dtype == np.int8
    assert mask[:legal_count].all()
    if legal_count < env.max_actions:
      assert not mask[legal_count:].any()
  finally:
    env.close()


def test_gym_env_deterministic_step_with_seed():
  env1 = GemEnv(num_players=3, seat_id=0, max_actions=64, seed=777)
  env2 = GemEnv(num_players=3, seat_id=0, max_actions=64, seed=777)
  try:
    obs1, info1 = env1.reset()
    obs2, info2 = env2.reset()
    assert np.array_equal(obs1, obs2)
    assert info1['legal_action_count'] == info2['legal_action_count']
    assert np.array_equal(info1['action_mask'], info2['action_mask'])

    step1 = env1.step(0)
    step2 = env2.step(0)

    assert step1 is not None
    assert step2 is not None

    obs_a, reward_a, term_a, trunc_a, info_a = step1
    obs_b, reward_b, term_b, trunc_b, info_b = step2

    assert np.array_equal(obs_a, obs_b)
    assert reward_a == reward_b
    assert term_a == term_b
    assert trunc_a == trunc_b
    assert info_a['legal_action_count'] == info_b['legal_action_count']
    assert np.array_equal(info_a['action_mask'], info_b['action_mask'])
  finally:
    env1.close()
    env2.close()


def test_gym_env_step_out_of_range_defaults_to_first_action():
  env = GemEnv(num_players=2, seat_id=0, max_actions=32, seed=2024)
  try:
    env.reset()
    assert env._engine is not None
    legal = env._engine.get_legal_actions(env.seat_id)
    assert legal
    first_type = legal[0].type.value

    result = env.step(len(legal) + 5)
    assert result is not None
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info['chosen_action_index'] == len(legal) + 5
    assert info['action_applied_type'] == first_type
  finally:
    env.close()
