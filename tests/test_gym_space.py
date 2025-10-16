import numpy as np
from typing import cast

from gymnasium import spaces

from gems.gym_env import StateSpace, ActionSpace, GemEnv
from gems.typings import Gem
from gems.actions import Action


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
    Action.take2(Gem.WHITE),
    Action.buy(card=None, visible_idx=0, payment={Gem.BLACK: 1, Gem.GREEN: 1}),
    Action.reserve(card=None, visible_idx=1, take_gold=True),
    Action.noop(),
  ]
  for a in action_list:
    enc = aspace.encode(a)
    dec = aspace.decode(enc)
    assert isinstance(dec, Action)
    assert dec.type == a.type
    assert dec == a
