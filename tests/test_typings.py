from gems.typings import Card, Role, Gem


def test_card_roundtrip():
  c = Card(id='c1', name='C', level=2, points=2, bonus=Gem.GREEN, cost_in=[(Gem.RED, 2)], metadata_in=[('foo', 'bar')])
  d = c.to_dict()
  c2 = Card.from_dict(d)
  assert c2.id == c.id
  assert c2.level == c.level
  assert c2.points == c.points
  assert c2.bonus == c.bonus
  assert tuple(c2.cost) == tuple(c.cost)
  assert str(c) == "Card([c1][2]G2:R2)"


def test_role_roundtrip():
  r = Role(id='r1', name='R', points=3, requirements_in={Gem.BLUE:3}, metadata_in=[('k','v')])
  d = r.to_dict()
  r2 = Role.from_dict(d)
  assert r2.id == r.id
  assert r2.points == r.points
  assert tuple(r2.requirements) == tuple(r.requirements)


def test_playerstate_discounts_empty_and_aggregate():
  # empty purchased cards -> no discounts
  from gems.state import PlayerState
  p0 = PlayerState(seat_id=0, purchased_cards_in=[])
  assert tuple(p0.purchased_cards) == ()
  assert tuple(p0.discounts) == ()

  # purchased cards with bonuses should aggregate discounts per Gem
  c1 = Card(id='a', bonus=Gem.GREEN)
  c2 = Card(id='b', bonus=Gem.GREEN)
  c3 = Card(id='c', bonus=Gem.RED)
  p1 = PlayerState(seat_id=1, purchased_cards_in=[c1, c2, c3])
  # discounts is a tuple of (Gem, count) pairs; convert to dict for easy asserts
  discounts_map = dict(p1.discounts)
  assert discounts_map[Gem.GREEN] == 2
  assert discounts_map[Gem.RED] == 1
