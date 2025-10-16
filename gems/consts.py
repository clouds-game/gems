from dataclasses import dataclass

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

@dataclass(frozen=True)
class GameConfig:
  num_players: int = DEFAULT_PLAYERS
  coin_init: int = 0
  coin_gold_init: int = 5
  coin_max_count_per_player: int = 10
  coin_min_count_take2_in_deck: int = 4
  card_visible_count: int = 4
  card_levels: tuple[int, ...] = (1, 2, 3)
  card_max_count_reserved: int = 3

  def __post_init__(self):
    num_players = self.num_players
    if num_players <= 0:
      raise ValueError(f'num_players must be positive, got {num_players}')
    if not self.coin_init:
      object.__setattr__(self, 'coin_init', COIN_DEFAULT_INIT[max(num_players, 5) - 1])

  @property
  def card_level_count(self) -> int:
    return len(self.card_levels)

  @property
  def card_visible_total_count(self) -> int:
    return self.card_visible_count * self.card_level_count
