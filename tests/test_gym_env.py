import numpy as np
from typing import cast
from gymnasium import spaces

from gems.gym_env import GemEnv

def test_gym_env_reset_shapes_and_mask():
  env = GemEnv(num_players=3, seat_id=1, max_actions=64, seed=123)
  try:
    obs, info = env.reset()
    assert isinstance(obs, dict)
    dict_space = cast(spaces.Dict, env.observation_space)
    assert set(obs.keys()) == set(dict_space.spaces.keys())
    for key, space in dict_space.spaces.items():
      value = obs[key]
      # Handle nested Dict (visible_cards)
      if isinstance(space, spaces.Dict):
        nested = cast(spaces.Dict, space)
        assert isinstance(value, dict)
        assert set(value.keys()) == set(nested.spaces.keys())
        for nk, nspace in nested.spaces.items():
          nval = value[nk]
          assert isinstance(nval, np.ndarray)
          assert nval.shape == nspace.shape
          assert nval.dtype == np.int32
      elif isinstance(space, spaces.Discrete):
        # expect a 0-d numpy scalar
        assert isinstance(value, np.ndarray)
        assert value.shape == ()
      else:
        # Box and others
        assert isinstance(value, np.ndarray)
        assert value.shape == space.shape
        assert value.dtype == np.int32

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
    assert obs1.keys() == obs2.keys()
    for key in obs1:
      v1 = obs1[key]
      v2 = obs2[key]
      if isinstance(v1, dict) and isinstance(v2, dict):
        for nk in v1:
          assert np.array_equal(v1[nk], v2[nk])
      else:
        assert not isinstance(v1, dict)
        assert not isinstance(v2, dict)
        assert np.array_equal(v1, v2)
    assert info1['legal_action_count'] == info2['legal_action_count']
    assert np.array_equal(info1['action_mask'], info2['action_mask'])

    step1 = env1.step(0)
    step2 = env2.step(0)

    assert step1 is not None
    assert step2 is not None

    obs_a, reward_a, term_a, trunc_a, info_a = step1
    obs_b, reward_b, term_b, trunc_b, info_b = step2

    assert obs_a.keys() == obs_b.keys()
    for key in obs_a:
      v1 = obs_a[key]
      v2 = obs_b[key]
      if isinstance(v1, dict) and isinstance(v2, dict):
        for nk in v1:
          assert np.array_equal(v1[nk], v2[nk])
      else:
        assert not isinstance(v1, dict)
        assert not isinstance(v2, dict)
        assert np.array_equal(v1, v2)
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

    result = env.step(len(legal) + 5)
    assert result is not None
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, dict)
    dict_space = cast(spaces.Dict, env.observation_space)
    for key, space in dict_space.spaces.items():
      value = obs[key]
      if isinstance(space, spaces.Dict):
        nested = cast(spaces.Dict, space)
        assert isinstance(value, dict)
        for nk, nspace in nested.spaces.items():
          nval = value[nk]
          assert isinstance(nval, np.ndarray)
          assert nval.shape == nspace.shape
      elif isinstance(space, spaces.Discrete):
        assert isinstance(value, np.ndarray)
        assert value.shape == ()
      else:
        assert isinstance(value, np.ndarray)
        assert value.shape == space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info['chosen_action'].type.value == "noop"
    assert info['action_applied_type'] == "noop"
    assert reward == -0.1
  finally:
    env.close()
