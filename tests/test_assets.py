from gems.engine import load_assets
from gems.typings import Card, Role


def test_load_assets_default_config():
  cards, roles = load_assets()
  assert isinstance(cards, list)
  assert isinstance(roles, list)
  assert all(isinstance(c, Card) for c in cards)
  assert all(isinstance(r, Role) for r in roles)
  # our sample config contains 2 cards and 2 roles
  assert len(cards) == 90
  assert len(roles) == 10
