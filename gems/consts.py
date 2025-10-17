from dataclasses import dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass


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
