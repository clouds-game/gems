import pytest

from gems.typings import Card, CardList


def make_card(level: int, id: str) -> Card:
  return Card(id=id, level=level)


def test_get_level_filters():
  cards = [make_card(1, 'a'), make_card(2, 'b'), make_card(1, 'c'), make_card(3, 'd')]
  cl = CardList(cards)

  lvl1 = cl.get_level(1)
  assert isinstance(lvl1, CardList)
  assert [c.id for c in lvl1] == ['a', 'c']

  lvl2 = cl.get_level(2)
  assert [c.id for c in lvl2] == ['b']

  lvl_missing = cl.get_level(99)
  assert list(lvl_missing) == []


def test_index_and_len_and_iteration():
  cards = [make_card(1, 'x'), make_card(2, 'y')]
  cl = CardList(cards)

  assert len(cl) == 2
  assert cl[0].id == 'x'
  assert cl[1].id == 'y'

  ids = [c.id for c in cl]
  assert ids == ['x', 'y']


def test_as_tuple_and_to_list_and_empty():
  cards = [make_card(3, 'z')]
  cl = CardList(cards)

  tup = cl.as_tuple()
  assert isinstance(tup, tuple)
  assert tup[0].id == 'z'

  lst = cl.to_list()
  assert isinstance(lst, list)
  assert lst[0].id == 'z'

  empty = CardList([])
  assert len(empty) == 0
  assert list(empty) == []
