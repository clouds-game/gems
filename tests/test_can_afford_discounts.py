import pytest
from gems.typings import Gem, Card, GemList
from gems.state import PlayerState


def test_can_afford_with_discounts_happy_path():
  # Card costs 2 black and 2 blue, player has discounts: 1 black
  card = Card(id='d-1', cost_in=[(Gem.BLACK, 2), (Gem.BLUE, 2)])
  # player has one discount in BLACK from purchased cards
  p = PlayerState(seat_id=0, gems=GemList(((Gem.BLACK, 1), (Gem.BLUE, 2), (Gem.GOLD, 0))), purchased_cards_in=())
  # manually set discounts to simulate one purchased black bonus
  # Construct PlayerState with purchased_cards that produce discounts would be easier,
  # but we can directly rely on the `discounts` attribute for this focused test.
  # Replace discounts with GemList containing (BLACK, 1)
  object.__setattr__(p, 'discounts', GemList(((Gem.BLACK, 1),)))

  payments = p.can_afford(card)
  # After discount, effective cost is BLACK:1, BLUE:2; player has exactly those colored gems
  assert {Gem.BLACK: 1, Gem.BLUE: 2} in payments


def test_can_afford_with_discounts_makes_card_free():
  # Card costs 1 red and 1 blue, player has discounts that cover both
  card = Card(id='d-2', cost_in=[(Gem.RED, 1), (Gem.BLUE, 1)])
  p = PlayerState(seat_id=0, gems=GemList(()), purchased_cards_in=())
  object.__setattr__(p, 'discounts', GemList(((Gem.RED, 1), (Gem.BLUE, 1))))

  payments = p.can_afford(card)
  # Card is free thanks to discounts -> empty payment dict expected
  assert payments == [{}]
