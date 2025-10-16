from gems.actions import (
  Action,
  NoopAction,
  Take3Action,
  Take2Action,
  BuyCardAction,
  ReserveCardAction,
)
from gems.typings import (
  ActionType,
  CardIdx,
  Gem,
  Card,
)


def test_action_constructors_basic():
  a0 = Action.noop()
  assert isinstance(a0, Action)
  assert isinstance(a0, NoopAction)
  assert a0.type == ActionType.NOOP
  assert str(a0) == "Action.Noop()"

  a1 = Action.take3(Gem.RED, Gem.BLUE, Gem.GREEN)
  assert isinstance(a1, Action)
  assert isinstance(a1, Take3Action)
  assert a1.type == ActionType.TAKE_3_DIFFERENT
  assert a1.gems == (Gem.RED, Gem.BLUE, Gem.GREEN)
  assert str(a1) == "Action.Take3(🔴🔵🟢)"

  a2 = Action.take2(Gem.WHITE)
  assert isinstance(a2, Action)
  assert isinstance(a2, Take2Action)
  assert a2.type == ActionType.TAKE_2_SAME
  assert a2.gem == Gem.WHITE
  assert a2.count == 2
  assert str(a2) == "Action.Take2(2⚪)"

  card1 = Card(id='card-1', level=1, cost_in={Gem.BLACK: 1})
  a3 = Action.buy(card1, payment={Gem.BLACK: 1, Gem.GREEN: 1}, visible_idx=0)
  assert isinstance(a3, Action)
  assert isinstance(a3, BuyCardAction)
  assert a3.type == ActionType.BUY_CARD
  assert a3.card is not None
  assert a3.card is card1
  assert a3.card.id == 'card-1'
  assert a3.idx == CardIdx(visible_idx=0)
  assert any(g == Gem.BLACK for g, _ in a3.payment)
  assert str(a3) == "Action.Buy(<[0]=card-1>, 1⚫1🟢)"

  card2 = Card(id='card-2', level=1)
  a4 = Action.reserve(card2, take_gold=True, visible_idx=0)
  assert isinstance(a4, Action)
  assert isinstance(a4, ReserveCardAction)
  assert a4.type == ActionType.RESERVE_CARD
  assert a4.take_gold is True
  assert str(a4) == "Action.Reserve(<[0]=card-2>, 🟡)"

  a5 = Action.take3(Gem.RED, Gem.BLUE, Gem.GREEN, ret_map = {Gem.WHITE: 2})
  assert isinstance(a5, Action)
  assert isinstance(a5, Take3Action)
  assert a5.type == ActionType.TAKE_3_DIFFERENT
  assert a5.gems == (Gem.RED, Gem.BLUE, Gem.GREEN)
  assert a5.ret is not None
  assert a5.ret.to_dict() == {Gem.WHITE: 2}
  assert str(a5) == "Action.Take3(🔴🔵🟢-2⚪)"

  a6 = Action.take2(Gem.WHITE, ret_map= {Gem.RED: 1})
  assert isinstance(a6, Action)
  assert isinstance(a6, Take2Action)
  assert a6.type == ActionType.TAKE_2_SAME
  assert a6.gem == Gem.WHITE
  assert a6.count == 2
  assert a6.ret is not None
  assert a6.ret.to_dict() == {Gem.RED: 1}
  assert str(a6) == "Action.Take2(2⚪-1🔴)"


def test_action_serialize_roundtrip():
  # create a variety of actions and ensure serialize->deserialize roundtrips
  card1 = Card(id='card-serialize-1', level=2, cost_in={Gem.BLACK: 1, Gem.GREEN: 2}, points=1, bonus=Gem.BLUE)

  actions = [
    Action.noop(),
    Action.take3(Gem.RED, Gem.BLUE, Gem.GREEN),
    Action.take2(Gem.WHITE),
    Action.buy(card1, payment={Gem.BLACK: 1, Gem.GOLD: 1}, visible_idx=0),
    Action.reserve(card1, take_gold=False, visible_idx=1),
    BuyCardAction.create(None, card1, payment={Gem.BLACK: 1, Gem.GOLD: 1}),
    ReserveCardAction.create(None, card1, take_gold=False),
    Action.take3(Gem.RED, Gem.BLUE, Gem.GREEN, ret_map={Gem.WHITE: 2}),
    Action.take2(Gem.WHITE, ret_map={Gem.RED: 1}),
  ]

  for a in actions:
    d = a.serialize()
    b = Action.deserialize(d)
    # compare serialized forms to avoid relying on object equality (GemList has no eq)
    assert b.serialize() == d
