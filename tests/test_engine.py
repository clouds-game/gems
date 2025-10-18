import pytest

from gems import Engine
from gems.state import GameState


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
  # ensure config field present and contains expected num_players
  assert 'config' in data
  assert data['config']['num_players'] == 2
  e2 = Engine.deserialize(data)
  # after deserialize the actions should be staged for replay
  assert hasattr(e2, '_actions_to_replay')
  assert len(e2._actions_to_replay) == 1
  # now apply replay and ensure states and histories match
  states = e2.replay()  # replay pending action(s)
  # should return initial + one new state
  assert len(states) == 2
  assert states[-1] == e1.get_state()
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
  assert 'config' in data
  assert data['config']['num_players'] == 3
  e2 = Engine.deserialize(data)
  # config roundtrip: serialized then deserialized config should match values
  cfg1 = e1.config
  cfg2 = e2.config
  assert cfg1.num_players == cfg2.num_players
  assert cfg1.coin_init == cfg2.coin_init
  assert cfg1.card_levels == cfg2.card_levels
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
