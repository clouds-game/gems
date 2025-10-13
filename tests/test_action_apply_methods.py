from gems.actions import (
  Take3Action,
  Take2Action,
  BuyCardAction,
  ReserveCardAction,
  NoopAction,
  apply_action_and_advance,
)
from gems.typings import Gem, GameState, PlayerState, Card


def make_basic_state():
  # two players, bank with some tokens, one visible card
  players = (
    PlayerState(seat_id=0, name='P0', gems_in=((Gem.RED, 0), (Gem.BLUE, 0))),
    PlayerState(seat_id=1, name='P1', gems_in=((Gem.RED, 0), (Gem.BLUE, 0))),
  )
  bank = {Gem.RED: 4, Gem.BLUE: 4, Gem.GOLD: 1, Gem.WHITE: 4, Gem.BLACK: 4, Gem.GREEN: 4}
  card = Card(id='c1', level=1, points=1, bonus=Gem.RED, cost_in={Gem.RED: 1})
  state = GameState(players=players, bank_in=bank.items(), visible_cards=(card,), turn=0)
  return state


def test_take3_apply():
  state = make_basic_state()
  action = Take3Action.create(Gem.RED, Gem.BLUE, Gem.WHITE)
  new_state = action.apply(state)
  # player 0 should have gained those gems
  p0 = new_state.players[0]
  gems = dict(p0.gems)
  assert gems.get(Gem.RED, 0) == 1
  assert gems.get(Gem.BLUE, 0) == 1
  assert gems.get(Gem.WHITE, 0) == 1
  # bank reduced
  bank = dict(new_state.bank)
  assert bank[Gem.RED] == 3


def test_take2_apply():
  state = make_basic_state()
  action = Take2Action.create(Gem.GREEN)
  new_state = action.apply(state)
  p0 = new_state.players[0]
  gems = dict(p0.gems)
  assert gems.get(Gem.GREEN, 0) == 2
  bank = dict(new_state.bank)
  assert bank[Gem.GREEN] == 2


def test_reserve_apply():
  state = make_basic_state()
  action = ReserveCardAction.create('c1', take_gold=True)
  new_state = action.apply(state)
  p0 = new_state.players[0]
  # reserved card added
  assert any(getattr(c, 'id', None) == 'c1' for c in p0.reserved_cards)
  # gold given if available
  gems = dict(p0.gems)
  assert gems.get(Gem.GOLD, 0) == 1


def test_noop_and_advance_adjacent_to_reserve():
  # ensure noop.apply does not change turn but apply_action_and_advance does
  state = make_basic_state()
  noop = NoopAction.create()
  applied = noop.apply(state)
  # apply should not advance the turn
  assert applied.turn == state.turn
  # last_action should be set to the noop
  assert getattr(applied, 'last_action') == noop

  # apply_action_and_advance should advance the turn by 1
  advanced = apply_action_and_advance(state, noop)
  assert advanced.turn == state.turn + 1
  assert getattr(advanced, 'last_action') == noop


def test_buy_apply_from_visible():
  state = make_basic_state()
  # Give player a red gem to pay for the card
  p0 = state.players[0]
  p0_with_gems = PlayerState(seat_id=0, name='P0', gems_in=((Gem.RED, 1),))
  players = (p0_with_gems, state.players[1])
  state = GameState(players=players, bank=state.bank, visible_cards=state.visible_cards, turn=0)
  action = BuyCardAction.create('c1', payment={Gem.RED: 1})
  new_state = action.apply(state)
  p0n = new_state.players[0]
  assert any(getattr(c, 'id', None) == 'c1' for c in p0n.purchased_cards)
  # score incremented by card points
  assert p0n.score == 1
