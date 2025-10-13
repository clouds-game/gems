from gems.typings import Card
from gems.state import PlayerState, GameState


def make_card(level: int, id: str):
  return Card(id=id, level=level)


def test_advance_turn_no_decks():
  p = PlayerState(seat_id=0)
  card = make_card(1, 'a')
  gs = GameState(players=(p,), visible_cards_in=(card,), turn=0)

  gs2 = gs.advance_turn()
  assert gs2.turn == 1
  # visible cards unchanged when no decks provided
  assert [c.id for c in gs2.visible_cards] == ['a']


def test_advance_turn_with_decks_refill():
  p = PlayerState(seat_id=0)
  gs = GameState(players=(p,), visible_cards_in=(), turn=5)

  # decks: level 1 has two cards, level 2 has one
  d1 = [make_card(1, 'l1c1'), make_card(1, 'l1c2')]
  d2 = [make_card(2, 'l2c1')]
  decks = {1: d1, 2: d2, 3: []}

  gs2 = gs.advance_turn(decks_by_level=decks, per_level=4)
  # turn advanced
  assert gs2.turn == 6
  # visible cards should include IDs from the decks
  ids = sorted(c.id for c in gs2.visible_cards) # type: ignore
  assert ids == sorted(['l1c1', 'l1c2', 'l2c1'])
  # decks should have been mutated (popped)
  assert decks[1] == []
  assert decks[2] == []
