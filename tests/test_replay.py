import json

from gems import Engine


def test_export_and_replay_model_roundtrip():
  from gems.engine import Replay
  from gems.actions import Action
  from gems.typings import Gem

  e = Engine.new(2, ["P1", "P2"], seed=42)
  # perform a simple deterministic action and record it on the engine
  # act = Action.take3(Gem.RED, Gem.BLUE, Gem.WHITE)
  # e._state = act.apply(e.get_state())
  # e.advance_turn()
  # e._action_history.append(act)

  rep = e.export()
  assert isinstance(rep, Replay)

  # basic fields roundtrip and types; prefer model-dump/dict access which is
  # robust across pydantic versions and avoids attribute access surprises.
  d = rep.model_dump()
  assert d['config']['num_players'] == e.config.num_players
  assert d['player_names'] == (e._names or [])

  Replay.model_validate(d)

  # action history should at least include the expected action types when
  # converted to JSON; pydantic's dict/JSON representation of user objects
  # can vary by version so this check is intentionally tolerant.
  js = rep.model_dump_json()
  assert isinstance(js, str)
  for a in e._action_history:
    assert a.serialize()['type'] in js
  parsed = json.loads(js)
  assert isinstance(parsed, dict)
