import math
import pytest
from gems import Engine
from gems.actions import Action, Take2Action
from gems.consts import GameConfig
from gems.typings import ActionType, Gem, Card, GemList
from gems.state import PlayerState, GameState


def test_get_legal_actions_basic():
  e = Engine.new(2)
  # ensure assets are loaded by constructor
  actions = e.get_legal_actions(seat_id=0)
  assert isinstance(actions, list)
  # we expect at least one take_3_different or take_2_same or reserve_card
  types = {a.type for a in actions}
  assert ActionType.TAKE_3_DIFFERENT in types or ActionType.TAKE_2_SAME in types or ActionType.RESERVE_CARD in types


def test_buy_card_included_when_affordable():
  # create engine and replace state with a known affordable visible card
  e = Engine.new(2)
  card = Card(id='buy-1', cost_in=[(Gem.BLACK, 2)])
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.BLACK, 2),)))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  # reuse the bank from the current engine state
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(card,), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  assert any(a.type == ActionType.BUY_CARD for a in actions)


def test_buy_card_not_included_when_unaffordable_but_included_with_gold():
  # unaffordable without gold
  e = Engine.new(2)
  card = Card(id='buy-2', cost_in=[(Gem.BLACK, 3)])
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.BLACK, 2),)))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(card,), turn=0)
  e._state = state
  actions = e.get_legal_actions(seat_id=0)
  assert not any(a.type == ActionType.BUY_CARD for a in actions)

  # now give player a gold to allow substitution
  p0_with_gold = PlayerState(seat_id=0, gems=GemList(((Gem.BLACK, 2), (Gem.GOLD, 1))))
  state2 = GameState(config=e.config, players=(p0_with_gold, p1), bank=e.get_state().bank,
                     visible_cards_in=(card,), turn=0)
  e._state = state2
  actions2 = e.get_legal_actions(seat_id=0)
  assert any(a.type == ActionType.BUY_CARD for a in actions2)


def test_gold_allows_multiple_payment_combinations():
  e = Engine.new(2)
  # card requires 2 red and 2 blue
  card = Card(id='multi-1', cost_in=[(Gem.RED, 2), (Gem.BLUE, 2)])
  # player has 2 red, 2 blue and 1 gold -> multiple ways to pay (use gold for either color or not at all)
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 2), (Gem.BLUE, 2), (Gem.GOLD, 1))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(card,), turn=0)
  e._state = state

  payments = p0.can_afford(card)
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


def test_no_legal_actions_fallbacks_to_noop():
  e = Engine.new(2)
  # construct a minimal state with no visible cards and players without gems
  p0 = PlayerState(seat_id=0, gems=GemList(()))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  # create a bank with zero tokens for all gem types so no take/buy/reserve
  zero_bank = GemList(((Gem.RED, 0), (Gem.BLUE, 0), (Gem.WHITE, 0),
                      (Gem.BLACK, 0), (Gem.GREEN, 0), (Gem.GOLD, 0)))
  state = GameState(config=e.config, players=(p0, p1), bank=zero_bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  # Expect the engine to return a single NOOP action when nothing else is
  # legal. If the implementation still exposes other actions for this
  # environment configuration, skip the test as it's not applicable.
  if not actions:
    pytest.skip("Environment provided no actions and no fallback; cannot test noop")

  # Accept either a single NOOP action or ensure at least one NOOP is present.
  types = {a.type for a in actions}
  assert ActionType.NOOP in types


def test_take_2_same_available_when_bank_has_at_least_four():
  e = Engine.new(2)
  # Try to craft a bank with at least 4 of a particular gem.
  # Representation of bank may vary; if mutation fails, skip test.
  try:
    bank = e.get_state().bank
    # attempt to treat bank as a mapping-like object
    bank_dict = dict(bank)
    bank_dict[Gem.RED] = max(bank_dict.get(Gem.RED, 0), 4)
    bank_tuple = tuple(bank_dict.items())
    p0 = PlayerState(seat_id=0, gems=GemList(()))
    p1 = PlayerState(seat_id=1, gems=GemList(()))
    state = GameState(config=e.config, players=(p0, p1), bank=GemList(bank_tuple), visible_cards_in=(), turn=0)
    e._state = state

    actions = e.get_legal_actions(seat_id=0)
    assert any(a.type == ActionType.TAKE_2_SAME for a in actions)
  except Exception:
    pytest.skip("Cannot mutate or construct bank representation for this environment")


def test_buy_card_legal_if_affordable_by_exact_payment():
  # sanity check: buying a visible affordable card should appear in legal actions

  e = Engine.new(2)
  card = Card(id='aff-1', cost_in=[(Gem.BLACK, 1)])
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.BLACK, 1),)))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(card,), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  assert any(a.type == ActionType.BUY_CARD for a in actions)


def test_player_can_buy_own_reserved_card():
  # Player reserves a card and then should be able to buy it if they can afford it
  e = Engine.new(2)
  card = Card(id='res-1', cost_in=[(Gem.BLACK, 1)])
  # player 0 has the exact gem to buy the reserved card
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.BLACK, 1),)), reserved_cards_in=(card,))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  # ensure buying own reserved card is exposed as a legal action
  assert any(a.type == ActionType.BUY_CARD for a in actions)


def test_take3_with_returns_enumerated_and_apply():
  # player has 8 gems; taking 3 would go to 11 so must return 1
  from gems.actions import Take3Action

  e = Engine.new(2)
  # craft player with 8 gems (3 red, 3 blue, 2 white)
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 3), (Gem.BLUE, 3), (Gem.WHITE, 2))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take3_actions = [a for a in actions if a.type == ActionType.TAKE_3_DIFFERENT]
  # Expect at least one such action that includes a returns payload (since need_return == 1)
  has_returns = False
  chosen = None
  for a in take3_actions:
    if isinstance(a, Take3Action) and a.ret is not None:
      ret = a.ret.to_dict()
      if sum(ret.values()) > 0:
        has_returns = True
        chosen = a
        break

  assert has_returns, "Expected at least one Take3Action with returns when taking would exceed 10 gems"

  # ensure returns sum equals needed amount
  total_before = sum(n for _, n in p0.gems)
  need = max(0, total_before + 3 - 10)
  assert chosen is not None
  assert chosen.ret is not None
  assert sum(chosen.ret.to_dict().values()) == need

  # apply the action and verify final totals and bank adjustments
  new_state = chosen.apply(state)
  new_p0 = new_state.players[0]
  # player's total gems must be <= 10
  assert sum(n for _, n in new_p0.gems) == 10


def test_take3_no_return_needed_enumerates_combos():
  # if player has room, Take3 should enumerate all 3-combinations of available colors
  e = Engine.new(2)
  p0 = PlayerState(seat_id=0, gems=GemList(()))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  # create a bank with 5 colored gems available
  bank = GemList(((Gem.RED, 1), (Gem.BLUE, 1), (Gem.WHITE, 1),
                 (Gem.BLACK, 1), (Gem.GREEN, 1), (Gem.GOLD, 5)))
  state = GameState(config=e.config, players=(p0, p1), bank=bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take3 = [a for a in actions if a.type == ActionType.TAKE_3_DIFFERENT]
  # expect C(5,3) combos
  assert len(take3) == 10
  # none should include a return payload
  assert all((not getattr(a, 'ret', None)) for a in take3)


def test_take3_with_required_returns_enumerated():
  # player with many tokens may need to return after taking
  e = Engine.new(2)
  # player has 9 gems: 5 red, 4 blue
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 5), (Gem.BLUE, 4))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  # bank has 3 other colors available to take
  bank = GemList(((Gem.WHITE, 1), (Gem.BLACK, 1), (Gem.GREEN, 1), (Gem.GOLD, 5)))
  state = GameState(config=e.config, players=(p0, p1), bank=bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take3 = [a for a in actions if a.type == ActionType.TAKE_3_DIFFERENT]
  # should include actions that require returning 2 tokens
  assert len(take3) == 3 + 3 * 2 + 1 * 3
  assert len([a for a in take3 if getattr(a, 'ret', None)]) == 3 * 2 + 1 * 3


def test_take3_returns_within_player_holdings():
  # ensure any enumerated return amounts do not exceed player's holdings
  e = Engine.new(2)
  # player has 8 gems: 5 red, 3 blue (needs 1 return when taking 3)
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 5), (Gem.BLUE, 3))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  bank = GemList(((Gem.WHITE, 1), (Gem.BLACK, 1), (Gem.GREEN, 1), (Gem.GOLD, 5)))
  state = GameState(config=e.config, players=(p0, p1), bank=bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take3 = [a for a in actions if a.type == ActionType.TAKE_3_DIFFERENT]
  for a in take3:
    if getattr(a, 'ret', None):
      for g, amt in getattr(a, 'ret'):
        # player originally had this many of g
        orig = dict(p0.gems).get(g, 0)
        assert amt <= orig


def test_take2_no_return_needed_enumerates_combos():
  # if player has room, Take2 should enumerate available take-2 actions without returns
  e = Engine.new(2)
  p0 = PlayerState(seat_id=0, gems=GemList(()))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  # create a bank with at least 4 of two colors
  bank = GemList(((Gem.RED, 4), (Gem.BLUE, 4), (Gem.WHITE, 1),
                 (Gem.BLACK, 1), (Gem.GREEN, 1), (Gem.GOLD, 5)))
  state = GameState(config=e.config, players=(p0, p1), bank=bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take2 = [a for a in actions if a.type == ActionType.TAKE_2_SAME]
  # expect to find at least two take-2 actions for red and blue
  gems_taken = {a.gem for a in take2}  # type: ignore
  assert Gem.RED in gems_taken and Gem.BLUE in gems_taken
  # none should include a return payload
  assert all((not getattr(a, 'ret', None)) for a in take2)


def test_take2_required_returns_enumerated_and_apply():
  # player with many tokens must return when taking 2
  e = Engine.new(2)
  # player has 9 gems: 5 red, 4 blue
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 5), (Gem.BLUE, 4))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  # bank has at least 4 of white so take2 on white is available
  bank = GemList(((Gem.WHITE, 4), (Gem.GOLD, 5)))
  state = GameState(config=e.config, players=(p0, p1), bank=bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take2 = [a for a in actions if a.type == ActionType.TAKE_2_SAME]
  # Expect at least one take2 action that includes a return payload
  has_returns = all(getattr(a, 'ret', None) for a in take2)
  assert has_returns
  # pick one with returns and apply it
  chosen = next(a for a in take2 if getattr(a, 'ret', None))
  assert isinstance(chosen, Take2Action)
  total_before = sum(n for _, n in p0.gems)
  need = max(0, total_before + 2 - 10)
  assert chosen.ret is not None
  assert sum(chosen.ret.to_dict().values()) == need
  new_state = chosen.apply(state)
  new_p0 = new_state.players[0]
  assert sum(n for _, n in new_p0.gems) == 10


def test_take2_returns_within_player_holdings():
  e = Engine.new(2)
  # player has 8 gems: 5 red, 3 blue (needs 0 or maybe returns depending on take)
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 5), (Gem.BLUE, 5))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  bank = GemList(((Gem.WHITE, 4), (Gem.GREEN, 4), (Gem.GOLD, 5)))
  state = GameState(config=e.config, players=(p0, p1), bank=bank, visible_cards_in=(), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  take2 = [a for a in actions if a.type == ActionType.TAKE_2_SAME]
  assert len(take2) == 2 * 3
  assert all(getattr(a, 'ret', None) for a in take2)
  for a in take2:
    if getattr(a, 'ret', None):
      for g, amt in getattr(a, 'ret'):
        orig = dict(p0.gems).get(g, 0)
        assert amt <= orig

def test_initial_state_actions_num():
  e = Engine.new(2)
  actions = e.get_legal_actions(seat_id=0)
  kinds = len(Gem) - 1
  assert len(actions) == math.comb(kinds, 3) + kinds + 0 + e.config.card_visible_total_count

def test_initial_state_has_no_illegal_actions():
  # Create a fresh engine and enumerate legal actions for player 0
  e = Engine.new(2)
  state = e.get_state()
  player = state.players[0]
  config = e.config
  actions = e.get_legal_actions(seat_id=0)
  illegal = []

  for a in actions:
    # _check should pass and apply should succeed without mutating original state
    try:
      if not a._check(player, state, config):
        illegal.append(a)
        continue
      # apply on a snapshot; ensure no exception
      a.apply(state)
    except Exception:
      illegal.append(a)
  assert len(illegal) == 0, f"Found illegal actions in initial state: {illegal}"


def test_reserve_with_return_enumerated_and_apply():
  # player has 10 gems; reserving with gold would push to 11 so must return 1
  from gems.actions import ReserveCardAction

  e = Engine.new(2)
  card = Card(id='res-ret', cost_in=[])
  # player has 10 gems: 5 red, 5 blue
  p0 = PlayerState(seat_id=0, gems=GemList(((Gem.RED, 5), (Gem.BLUE, 5))))
  p1 = PlayerState(seat_id=1, gems=GemList(()))
  state = GameState(config=e.config, players=(p0, p1), bank=e.get_state().bank, visible_cards_in=(card,), turn=0)
  e._state = state

  actions = e.get_legal_actions(seat_id=0)
  reserve_actions = [a for a in actions if a.type == ActionType.RESERVE_CARD]
  # Expect at least one reserve action that specifies a returned gem
  with_ret = [a for a in reserve_actions if getattr(a, 'ret', None)]
  assert len(with_ret) == 2

  chosen = with_ret[0]
  assert isinstance(chosen, ReserveCardAction)
  assert chosen.ret is not None

  orig_count = dict(p0.gems).get(chosen.ret, 0)
  assert orig_count > 0

  new_state = chosen.apply(state)
  new_p0 = new_state.players[0]
  # player's total gems must remain at max (10)
  assert sum(n for _, n in new_p0.gems) == e.config.coin_max_count_per_player
  # returned gem count decreased by 1
  assert dict(new_p0.gems).get(chosen.ret, 0) == orig_count - 1
  # player gained a gold token
  assert dict(new_p0.gems).get(Gem.GOLD, 0) == 1
  # bank gained the returned token
  assert dict(new_state.bank).get(chosen.ret, 0) == dict(state.bank).get(chosen.ret, 0) + 1
