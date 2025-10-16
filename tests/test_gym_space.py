import numpy as np
from typing import cast

from gymnasium import spaces

from gems.gym_env import StateSpace, ActionSpace, GemEnv
from gems.typings import Gem
from gems.actions import Action


def test_action_space_cardidx_flatten_unflatten():
  aspace = ActionSpace()
  # visible index
  from gems.typings import CardIdx
  v = CardIdx(visible_idx=2)
  flat_v = aspace._flatten_card_idx(v)
  round_v = aspace._unflatten_card_idx(flat_v)
  assert isinstance(round_v, CardIdx)
  assert round_v.visible_idx == 2

  # reserve index
  r = CardIdx(reserve_idx=1)
  flat_r = aspace._flatten_card_idx(r)
  round_r = aspace._unflatten_card_idx(flat_r)
  assert isinstance(round_r, CardIdx)
  assert round_r.reserve_idx == 1

  # deck head level
  d = CardIdx(deck_head_level=3)
  flat_d = aspace._flatten_card_idx(d)
  round_d = aspace._unflatten_card_idx(flat_d)
  assert isinstance(round_d, CardIdx)
  assert round_d.deck_head_level == 3

  # out of range / negative -> None
  assert aspace._unflatten_card_idx(-5) is None
  assert aspace._unflatten_card_idx(aspace._max_card_index + 10) is None


def test_action_space_decode_invalid_type_raises():
  aspace = ActionSpace()
  d = aspace.empty()
  # set an invalid type index
  d['type'][...] = len(aspace._type_order) + 5
  import pytest
  with pytest.raises(ValueError):
    aspace.decode(d)


def test_action_space_decode_take2_no_gem_raises():
  aspace = ActionSpace()
  d = aspace.empty()
  from gems.typings import ActionType
  # set type to TAKE_2_SAME but leave gem vector empty
  d['type'][...] = aspace._type_index[ActionType.TAKE_2_SAME]
  d['take2']['gem'][...] = 0
  import pytest
  with pytest.raises(ValueError):
    aspace.decode(d)


def test_gemenv_reset_and_legal_action_mask():
  env = GemEnv(num_players=2, seat_id=0, max_actions=10, seed=7)
  mask = env._legal_action_mask(3)
  assert mask.shape == (10,)
  assert int(mask.sum()) == 3

  obs, info = env.reset(seed=7)
  assert 'action_mask' in info
  assert len(info['action_mask']) == 10
  assert isinstance(obs, dict)


def test_state_space_empty_obs_shapes():
  # create a StateSpace and request an observation with engine=None
  ss = StateSpace(per_card_feats=2 + (len(Gem) + 1) + len(Gem), num_players=2, visible_card_count=8)
  obs = ss.make_obs(None, seat_id=0)
  assert isinstance(obs, dict)
  # validate top-level keys
  expected_keys = {'bank', 'player_gems', 'player_discounts', 'player_score', 'turn_mod_players', 'visible_cards'}
  assert set(obs.keys()) == expected_keys
  # bank and player arrays
  assert isinstance(obs['bank'], np.ndarray)
  assert obs['bank'].shape == (len(Gem),)
  assert obs['bank'].dtype == np.int32
  assert isinstance(obs['player_score'], np.ndarray)
  assert obs['player_score'].shape == (1,)

  # visible cards nested structure
  vc = obs['visible_cards']
  assert isinstance(vc, dict)
  assert set(vc.keys()) == {'level', 'points', 'bonus', 'costs'}
  assert vc['level'].shape[0] == 8
  assert vc['costs'].shape == (8, len(Gem))


def test_action_space_encode_decode_roundtrip():
  aspace = ActionSpace()
  # noop
  action_list = [
    Action.take3(Gem.RED, Gem.BLUE, Gem.GREEN),
    Action.take3(Gem.RED, Gem.BLUE, Gem.WHITE, ret_map={Gem.RED: 1}),
    Action.take2(Gem.WHITE),
    Action.take2(Gem.BLACK, count=2, ret_map={Gem.WHITE: 1}),
    Action.buy(card=None, visible_idx=0, payment={Gem.BLACK: 1, Gem.GREEN: 1}),
    # buy by reserve_idx
    Action.buy(card=None, reserve_idx=0, payment={Gem.BLUE: 2}),
    Action.reserve(card=None, visible_idx=1, take_gold=True),
    Action.reserve(card=None, visible_idx=2, take_gold=False),
    Action.noop(),
  ]
  for a in action_list:
    enc = aspace.encode(a)
    dec = aspace.decode(enc)
    assert isinstance(dec, Action)
    assert dec.type == a.type
    assert dec == a
