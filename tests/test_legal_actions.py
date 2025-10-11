from gems import Engine
from gems.typings import Action, ActionType


def test_get_legal_actions_basic():
  e = Engine(2)
  # ensure assets are loaded by constructor
  actions = e.get_legal_actions(seat_id=0)
  assert isinstance(actions, list)
  # we expect at least one take_3_different or take_2_same or reserve_card
  types = {a.type for a in actions}
  assert ActionType.TAKE_3_DIFFERENT in types or ActionType.TAKE_2_SAME in types or ActionType.RESERVE_CARD in types


def test_buy_card_included_when_affordable():
  # create engine and replace state with a known affordable visible card
  from gems.typings import Card, Gem, PlayerState, GameState

  e = Engine(2)
  card = Card(id='buy-1', cost_in=[(Gem.BLACK, 2)])
  p0 = PlayerState(seat_id=0, gems=((Gem.BLACK, 2),))
  p1 = PlayerState(seat_id=1, gems=())
  # reuse the bank from the current engine state
  state = GameState(players=(p0, p1), bank=e.get_state().bank, visible_cards=(card,), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  assert any(a.type == ActionType.BUY_CARD for a in actions)


def test_buy_card_not_included_when_unaffordable_but_included_with_gold():
  # unaffordable without gold
  from gems.typings import Card, Gem, PlayerState, GameState

  e = Engine(2)
  card = Card(id='buy-2', cost_in=[(Gem.BLACK, 3)])
  p0 = PlayerState(seat_id=0, gems=((Gem.BLACK, 2),))
  p1 = PlayerState(seat_id=1, gems=())
  state = GameState(players=(p0, p1), bank=e.get_state().bank, visible_cards=(card,), turn=0)
  e._state = state
  actions = e.get_legal_actions(seat_id=0)
  assert not any(a.type == ActionType.BUY_CARD for a in actions)

  # now give player a gold to allow substitution
  p0_with_gold = PlayerState(seat_id=0, gems=((Gem.BLACK, 2), (Gem.GOLD, 1)))
  state2 = GameState(players=(p0_with_gold, p1), bank=e.get_state().bank,
                     visible_cards=(card,), turn=0)
  e._state = state2
  actions2 = e.get_legal_actions(seat_id=0)
  assert any(a.type == ActionType.BUY_CARD for a in actions2)


def test_gold_allows_multiple_payment_combinations():
  from gems.typings import Card, Gem, PlayerState, GameState
  from gems.engine import can_afford

  e = Engine(2)
  # card requires 2 red and 2 blue
  card = Card(id='multi-1', cost_in=[(Gem.RED, 2), (Gem.BLUE, 2)])
  # player has 2 red, 2 blue and 1 gold -> multiple ways to pay (use gold for either color or not at all)
  p0 = PlayerState(seat_id=0, gems=((Gem.RED, 2), (Gem.BLUE, 2), (Gem.GOLD, 1)))
  p1 = PlayerState(seat_id=1, gems=())
  state = GameState(players=(p0, p1), bank=e.get_state().bank, visible_cards=(card,), turn=0)
  e._state = state

  payments = can_afford(card, p0)
  # expect at least the three combinations: full colored, gold used for one of the colors
  expected = [
      {Gem.RED: 2, Gem.BLUE: 2},
      {Gem.RED: 2, Gem.BLUE: 1, Gem.GOLD: 1},
      {Gem.RED: 1, Gem.BLUE: 2, Gem.GOLD: 1},
  ]

  for exp in expected:
    assert exp in payments

  # ensure buy_card action is included
  actions = e.get_legal_actions(seat_id=0)
  assert any(a.type == ActionType.BUY_CARD for a in actions)
