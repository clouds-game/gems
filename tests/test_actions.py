from gems.typings import (
    Action,
    Gem,
)


def test_action_constructors_basic():
  a1 = Action.take_3_different([Gem.RED, Gem.BLUE, Gem.GREEN])
  assert isinstance(a1, Action)
  assert a1.type == 'take_3_different'
  assert ('gems', [Gem.RED, Gem.BLUE, Gem.GREEN]) in a1.payload

  a2 = Action.take_2_same(Gem.WHITE)
  assert a2.type == 'take_2_same'
  assert ('gem', Gem.WHITE) in a2.payload

  a3 = Action.buy_card('card-1', payment={Gem.BLACK: 1, Gem.GOLD: 1})
  assert a3.type == 'buy_card'
  assert ('card_id', 'card-1') in a3.payload
  assert any(k == 'payment' for k, _ in a3.payload)

  a4 = Action.reserve_card('card-2', take_gold=True)
  assert a4.type == 'reserve_card'
  assert ('take_gold', True) in a4.payload
