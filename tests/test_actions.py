from gems.typings import (
    Action,
    ActionType,
    Gem,
)


def test_action_constructors_basic():
  a1 = Action.take_3_different(Gem.RED, Gem.BLUE, Gem.GREEN)
  assert isinstance(a1, Action)
  assert a1.type == ActionType.TAKE_3_DIFFERENT
  assert ('gems', (Gem.RED, Gem.BLUE, Gem.GREEN)) in a1.payload

  a2 = Action.take_2_same(Gem.WHITE)
  assert a2.type == ActionType.TAKE_2_SAME
  assert ('gem', Gem.WHITE) in a2.payload

  a3 = Action.buy_card('card-1', payment={Gem.BLACK: 1, Gem.GOLD: 1})
  assert a3.type == ActionType.BUY_CARD
  assert ('card_id', 'card-1') in a3.payload
  assert any(k == 'payment' for k, _ in a3.payload)

  a4 = Action.reserve_card('card-2', take_gold=True)
  assert a4.type == ActionType.RESERVE_CARD
  assert ('take_gold', True) in a4.payload
