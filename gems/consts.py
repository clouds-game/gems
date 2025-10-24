from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from dataclasses import asdict

from gems.typings import Card, Role

# COIN_MAX_COUNT_PER_PLAYER = 10
# COIN_MIN_COUNT_TAKE2_IN_DECK = 4
# COIN_GOLD_INIT = 5
COIN_DEFAULT_INIT = (4, 4, 5, 7, 7)
# CARD_VISIBLE_COUNT = 4
# CARD_LEVELS = (1, 2, 3)
# CARD_LEVEL_COUNT = len(CARD_LEVELS)
# CARD_VISIBLE_TOTAL_COUNT = CARD_VISIBLE_COUNT * CARD_LEVEL_COUNT
# CARD_MAX_COUNT_RESERVED = 3

DEFAULT_PLAYERS = 4

@pydantic_dataclass(frozen=True)
class GameConfig:
  """Validated immutable configuration for a game.

  Uses pydantic's dataclass wrapper to provide runtime validation while
  retaining a light dataclass footprint. Public attribute API preserved.
  """
  num_players: int = DEFAULT_PLAYERS
  coin_init: int = 0
  coin_gold_init: int = 5
  coin_max_count_per_player: int = 10
  coin_min_count_take2_in_deck: int = 4
  card_visible_count: int = 4
  card_levels: tuple[int, ...] = (1, 2, 3)
  card_max_count_reserved: int = 3
  gem_count: int = 6  # Including gold

  take3_count: int = 3
  take2_count: int = 2

  def __post_init__(self):
    # pydantic already ran basic type validation; now apply domain rules.
    num_players = self.num_players
    if num_players <= 0:
      raise ValueError(f'num_players must be positive, got {num_players}')
    # coin_init: choose default mapping value by number of players capped at 5 index logic
    if not self.coin_init:
      object.__setattr__(self, 'coin_init', COIN_DEFAULT_INIT[max(num_players, 5) - 1])


  @property
  def card_level_count(self) -> int:
    return len(self.card_levels)

  @property
  def card_visible_total_count(self) -> int:
    return self.card_visible_count * self.card_level_count

  def serialize(self) -> dict:
    return asdict(self)

  @classmethod
  def deserialize(cls, data: dict) -> 'GameConfig':
    return cls(**data)


@pydantic_dataclass(frozen=True)
class GameAssets:
  decks_by_level: dict[int, tuple['Card', ...]]
  roles_deck: tuple['Role', ...]

  @classmethod
  def init(cls, cards: list['Card'], roles: list['Role']) -> 'GameAssets':
    decks_by_level: dict[int, list['Card']] = {}
    for card in cards:
      decks_by_level.setdefault(card.level, []).append(card)
    return cls(
      decks_by_level={level: tuple(deck) for level, deck in decks_by_level.items()},
      roles_deck=tuple(roles)
    )

  def new_decks_by_level(self) -> dict[int, list['Card']]:
    return {level: list(deck) for level, deck in self.decks_by_level.items()}

  def new_roles_deck(self) -> list['Role']:
    return list(self.roles_deck)

  @classmethod
  def load_default(cls, path: str | None = None) -> 'GameAssets':
    """Load cards and roles from a JSON config file and return (cards, roles).

    The config file is expected to contain top-level `cards` and `roles` arrays
    matching the `Card.from_dict` / `Role.from_dict` shapes.
    """
    import yaml
    p = Path(path) if path is not None else Path(__file__).parent / "assets" / "config.yaml"
    with p.open('r', encoding='utf8') as fh:
      j = yaml.safe_load(fh)

    cards = [Card.from_dict(c) for c in j.get('cards', [])]
    roles = [Role.from_dict(r) for r in j.get('roles', [])]
    return cls.init(cards, roles)

  def shuffle(self, seed: int | None = None) -> 'GameAssets':
    import random
    rng = random.Random(seed)
    shuffled_decks_by_level = {
      level: tuple(rng.sample(deck, len(deck)))
      for level, deck in self.decks_by_level.items()
    }
    shuffled_roles_deck = tuple(rng.sample(self.roles_deck, len(self.roles_deck)))
    return GameAssets(
      decks_by_level=shuffled_decks_by_level,
      roles_deck=shuffled_roles_deck
    )

GAME_ASSETS_DEFAULT = GameAssets.load_default()
GAME_ASSETS_EMPTY = GameAssets(decks_by_level={}, roles_deck=())
