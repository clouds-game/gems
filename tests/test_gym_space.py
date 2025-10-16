import numpy as np
from typing import cast

from gymnasium import spaces

from gems.consts import GameConfig
from gems.gym import StateSpace, ActionSpace, GemEnv
from gems.typings import Gem
from gems.actions import Action


def test_action_space_cardidx_flatten_unflatten():
  config = GameConfig()
  aspace = ActionSpace(config)
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
  config = GameConfig()
  aspace = ActionSpace(config)
  d = aspace.empty()
  # set an invalid type index
  d['type'][...] = len(aspace._type_order) + 5
  import pytest
  with pytest.raises(ValueError):
    aspace.decode(d)


def test_action_space_decode_take2_no_gem_raises():
  config = GameConfig()
  aspace = ActionSpace(config)
  d = aspace.empty()
  from gems.typings import ActionType
  # set type to TAKE_2_SAME but leave gem vector empty
  d['type'][...] = aspace._type_index[ActionType.TAKE_2_SAME]
  d['take2']['gem'][...] = 0
  import pytest
  with pytest.raises(ValueError):
    aspace.decode(d)


def test_state_space_empty_obs_shapes():
  # create a StateSpace and request an observation with engine=None
  config = GameConfig(num_players=2)
  ss = StateSpace(config)
  ss._visible_card_count = 8
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
  config = GameConfig()
  aspace = ActionSpace(config)
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


def test_state_space_obs():
  # build a deterministic engine and set a small custom state
  from gems.engine import Engine
  from gems.typings import Card
  from gems.state import PlayerState, GameState

  engine = Engine.new(num_players=2, seed=123)
  # construct a visible card with a bonus and costs
  cards = [
    Card(id='c1', level=1, points=2, bonus=Gem.BLACK, cost_in=[(Gem.RED, 1), (Gem.BLUE, 2)]),
    Card(id='c2', level=2, points=3, bonus=None, cost_in=[(Gem.WHITE, 1)]),
  ]

  # bank and player gems mapping
  bank = {g: 5 for g in Gem}
  bank[Gem.GOLD] = 2

  # player 0 has some gems and a purchased card that grants a discount (bonus)
  p0 = PlayerState(seat_id=0, name='A', gems_in={Gem.RED: 1, Gem.BLUE: 0, Gem.GOLD: 1}, score=7,
                   reserved_cards_in=(), purchased_cards_in=(cards[0],))
  # player 1 default
  p1 = engine.get_state().players[1]

  new_state = GameState(config=engine.config, players=(p0, p1), bank_in=bank, visible_cards_in=cards, turn=3)
  engine._state = new_state

  ss = StateSpace(config=engine.config)
  obs = ss.make_obs(engine, seat_id=0)

  # bank vector checks (same Gem order as enum)
  expected_bank = np.array([bank[g] for g in Gem], dtype=np.int32)
  assert np.array_equal(obs['bank'], expected_bank)

  # player gems and score
  expected_player_gems = np.array([p0.gems.get(g) for g in Gem], dtype=np.int32)
  assert np.array_equal(obs['player_gems'], expected_player_gems)
  assert int(obs['player_score'][0]) == p0.score

  # visible cards: first two positions populated, rest zeros
  vc = obs['visible_cards']
  # levels are stored as level-1
  for i, c in enumerate(cards):
    assert int(vc['level'][i]) == c.level - 1
    assert int(vc['points'][i]) == c.points
    # bonus encoded as GemIndex+1 (0 == none)
    gem_order = list(Gem)
    # c1.bonus is set above; cast to Gem for typing
    bonus_index = gem_order.index(cast(Gem, c.bonus)) + 1 if c.bonus is not None else 0
    assert int(vc['bonus'][i]) == bonus_index
    # costs matrix
    # find index positions for gems
    for g, amt in c.cost:
      gi = gem_order.index(g)
      assert int(vc['costs'][i, gi]) == amt
