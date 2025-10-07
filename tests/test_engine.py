import pytest

from gems import Engine, init_game
from gems.typings import GameState


def test_engine_init_and_get_state():
  e = Engine(2, ["P1", "P2"])
  state = e.get_state()
  assert isinstance(state, GameState)
  assert len(state.players) == 2
  assert state.turn == 0


def test_engine_reset_changes_state():
  e = Engine(2, ["A", "B"])
  e.reset(3, ["X", "Y", "Z"])
  s = e.get_state()
  assert len(s.players) == 3


def test_init_game_invalid_count_raises():
  with pytest.raises(ValueError):
    init_game(1)
