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
  e.reset(["X", "Y"])
  s = e.get_state()
  assert s.players[0].name == "X"
  assert s.players[1].name == "Y"


def test_init_game_invalid_count_raises():
  with pytest.raises(ValueError):
    Engine.new(0)
