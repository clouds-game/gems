from gems.engine import GAME_ASSETS_DEFAULT
from gems.typings import Card, Role


def test_load_assets_default_config():
  assets = GAME_ASSETS_DEFAULT
  cards_by_level = assets.new_decks_by_level()
  cards = [c for deck in cards_by_level.values() for c in deck]
  roles = assets.new_roles_deck()
  assert isinstance(cards, list)
  assert isinstance(roles, list)
  assert all(isinstance(c, Card) for c in cards)
  assert all(isinstance(r, Role) for r in roles)
  # our sample config contains 90 cards and 10 roles
  assert len(cards) == 90
  assert len(roles) == 10
  assert {level: len(deck) for level, deck in cards_by_level.items()} == {1: 40, 2: 30, 3: 20}
