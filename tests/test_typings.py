from gems.typings import Card, Role, Gem


def test_card_roundtrip():
  c = Card(id='c1', name='C', level=2, points=2, bonus=Gem.GREEN, cost=[(Gem.RED, 2)], metadata=[('foo', 'bar')])
  d = c.to_dict()
  c2 = Card.from_dict(d)
  assert c2.id == c.id
  assert c2.level == c.level
  assert c2.points == c.points
  assert c2.bonus == c.bonus
  assert tuple(c2.cost) == tuple(c.cost)


def test_role_roundtrip():
  r = Role(id='r1', name='R', points=3, requirements={Gem.BLUE:3}, metadata=[('k','v')])
  d = r.to_dict()
  r2 = Role.from_dict(d)
  assert r2.id == r.id
  assert r2.points == r.points
  assert tuple(r2.requirements) == tuple(r.requirements)
