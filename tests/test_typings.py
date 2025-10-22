import pytest
from gems.typings import Card, CardIdx, GemList, Role, Gem


def test_gem_list_init():
  gl1 = GemList({Gem.RED: 2, Gem.BLUE: 3})
  assert dict(gl1) == {Gem.RED: 2, Gem.BLUE: 3}

  gl2 = GemList([(Gem.GREEN, 1), (Gem.WHITE, 4)])
  assert dict(gl2) == {Gem.GREEN: 1, Gem.WHITE: 4}

  gl3 = GemList(())
  assert dict(gl3) == {}

  gl4 = GemList()
  assert dict(gl4) == {}


def test_card_init():
  c = Card(id="c1", name="C", level=2, points=2, bonus=Gem.GREEN, cost_in=[(Gem.RED, 2)])
  assert isinstance(c.cost, GemList)


def test_card_roundtrip():
  c = Card(id='c1', name='C', level=2, points=2, bonus=Gem.GREEN, cost_in=[(Gem.RED, 2)], metadata_in=[('foo', 'bar')])
  d = c.to_dict()
  c2 = Card.from_dict(d)
  assert c2.id == c.id
  assert c2.level == c.level
  assert c2.points == c.points
  assert c2.bonus == c.bonus
  assert tuple(c2.cost) == tuple(c.cost)
  assert str(c) == "Card2(<c1>[2]🟢:2🔴)"


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


def test_card_idx():
  a = CardIdx(visible_idx=3)
  assert a.visible_idx == 3
  assert a.reserve_idx is None
  assert a.deck_head_level is None
  assert str(a) == "<[3]>"

  ci2 = CardIdx(reserve_idx=1)
  assert ci2.visible_idx is None
  assert ci2.reserve_idx == 1
  assert ci2.deck_head_level is None
  assert str(ci2) == "<R[1]>"

  ci3 = CardIdx(deck_head_level=3)
  assert ci3.visible_idx is None
  assert ci3.reserve_idx is None
  assert ci3.deck_head_level == 3
  assert str(ci3) == "<D[3]>"

  with pytest.raises(ValueError):
    CardIdx()

  with pytest.raises(ValueError):
    CardIdx(visible_idx=1, reserve_idx=2)

def test_pydantic():
  from pydantic import Field
  from pydantic.dataclasses import dataclass as pydantic_dataclass
  from dataclasses import InitVar
  from pydantic import field_validator

  @pydantic_dataclass(frozen=True)
  class MyList:
    input: InitVar[dict[str, int] | list[tuple[str, int]]] = Field(alias='value')
    value: dict[str, int] = Field(init=False)

    @field_validator('input', mode='before')
    @classmethod
    def validate_value(cls, v: dict[str, int] | list[tuple[str, int]]):
      print(f"Validating input: {v}")
      if isinstance(v, dict):
        return v
      elif isinstance(v, list):
        return dict(v)
      else:
        raise ValueError("Invalid input type for MyList")

    def __post_init__(self, input):
      object.__setattr__(self, 'value', input)


  MyList(value={'a': 1, 'b': 2}) # static typing works
  a = MyList([('a', 1), ('b', 2)]) # static typing failed

  assert a.value == {'a': 1, 'b': 2}
