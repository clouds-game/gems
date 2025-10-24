from gems import Engine


def test_deterministic_shuffle_same_seed():
  e1 = Engine.new(2, seed=12345)
  e2 = Engine.new(2, seed=12345)

  # For each level the ordering should be identical when using the same seed
  for lvl in (1, 2, 3):
    d1 = e1.get_deck(lvl)
    d2 = e2.get_deck(lvl)
    assert [c.id for c in d1] == [c.id for c in d2]


def test_draw_and_peek_behavior():
  e = Engine.new(2, seed=42)

  lvl = 1
  deck_before = e.get_deck(lvl)
  assert len(deck_before) > 2

  top_peek = e.peek_deck(lvl, 2)
  # draw returns popped items (last element first)
  drawn = e.draw_from_deck(lvl, 2)

  # drawn should be the reverse of the peeked slice
  assert drawn == list(reversed(top_peek))

  # deck size should be reduced by 2
  assert len(e.get_deck(lvl)) == len(deck_before) - 2
