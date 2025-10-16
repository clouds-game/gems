"""StateSpace helper for building observations from Engine state.

Provides a gymnasium Space-like wrapper with a `make_obs(engine, seat_id)`
method that produces the structured numpy observation dict used by the env.
"""
from __future__ import annotations

from typing import TypeAlias, TypeVar, TypedDict

import numpy as np
from gymnasium import spaces

from ..state import PlayerState
from ..engine import Engine
from ..typings import Gem
from ..consts import GameConfig

from ._common import NDArray1D, NDArray2D, Scalar


GemIndex = {g: i for i, g in enumerate(Gem)}  # order: enum definition order


class CardDict(TypedDict):
  level: NDArray1D[np.int32]  # shape (CARD_VISIBLE_TOTAL_COUNT,)
  points: NDArray1D[np.int32]  # shape (CARD_VISIBLE_TOTAL_COUNT,)
  bonus: NDArray1D[np.int32]  # shape (CARD_VISIBLE_TOTAL_COUNT,)  (0 == none, 1..GEM_COUNT map to GemIndex+1)
  costs: NDArray2D[np.int32]  # shape (CARD_VISIBLE_TOTAL_COUNT, GEM_COUNT)

class StateDict(TypedDict):
  bank: NDArray1D[np.int32]  # shape (GEM_COUNT,)
  player_gems: NDArray1D[np.int32]  # shape (GEM_COUNT,)
  player_discounts: NDArray1D[np.int32]  # shape (GEM_COUNT,)
  player_score: NDArray1D[np.int32]  # shape (1,)
  turn_mod_players: Scalar[np.int32]  # shape (), scalar
  visible_cards: "CardDict"  # structured sub-dict

class StateSpace(spaces.Dict):
  """Helper to build observations from an Engine state.

  This class is also a gymnasium Space by wrapping an internal Dict space so
  it can be assigned directly to `env.observation_space` while providing a
  `make_obs(engine, seat_id)` method that produces the structured numpy
  observation dict.
  """

  def __init__(self, config: GameConfig, *, seed = None):
    # initialize base Space with a dummy low/high; we'll delegate to internal
    self._num_players = config.num_players
    self._visible_card_count = config.card_visible_total_count
    self._gem_count = config.gem_count

    super().__init__({
      'bank': spaces.Box(low=0, high=255, shape=(self._gem_count,), dtype=np.int32),
      'player_gems': spaces.Box(low=0, high=255, shape=(self._gem_count,), dtype=np.int32),
      'player_discounts': spaces.Box(low=0, high=255, shape=(self._gem_count,), dtype=np.int32),
      'player_score': spaces.Box(low=0, high=255, shape=(1,), dtype=np.int32),
      'turn_mod_players': spaces.Discrete(self._num_players),
      'visible_cards': spaces.Dict({
        'level': spaces.Box(low=0, high=config.card_level_count, shape=(self._visible_card_count,), dtype=np.int32),
        'points': spaces.Box(low=0, high=255, shape=(self._visible_card_count,), dtype=np.int32),
        'bonus': spaces.MultiDiscrete([self._gem_count + 1] * self._visible_card_count),
        'costs': spaces.Box(low=0, high=255, shape=(self._visible_card_count, self._gem_count), dtype=np.int32),
      }),
    }, seed=seed)

  def make_obs(self, engine: Engine | None, seat_id: int) -> StateDict:
    bank = np.zeros(self._gem_count, dtype=np.int32)
    player_gems = np.zeros(self._gem_count, dtype=np.int32)
    player_discounts = np.zeros(self._gem_count, dtype=np.int32)
    player_score = np.zeros(1, dtype=np.int32)
    # represent turn_mod_players as a scalar ndarray (0-d) to align with Discrete space
    turn_mod_players = np.array(0, dtype=np.int32)
    # visible cards split into structured fields
    visible_levels = np.zeros((self._visible_card_count,), dtype=np.int32)
    visible_points = np.zeros((self._visible_card_count,), dtype=np.int32)
    # bonus: 0 == none, 1..self._gem_count map to GemIndex+1
    visible_bonus = np.zeros((self._visible_card_count,), dtype=np.int32)
    visible_costs = np.zeros((self._visible_card_count, self._gem_count), dtype=np.int32)
    state = engine.get_state() if engine is not None else None
    if state is None:
      return {
        'bank': bank,
        'player_gems': player_gems,
        'player_discounts': player_discounts,
        'player_score': player_score,
        'turn_mod_players': turn_mod_players,
        'visible_cards': {
          'level': visible_levels,
          'points': visible_points,
          'bonus': visible_bonus,
          'costs': visible_costs,
        },
      }
    player = state.players[seat_id]
    for g, n in state.bank:
      bank[GemIndex[g]] = n
    for g, n in player.gems:
      player_gems[GemIndex[g]] = n
    for g, n in player.discounts:
      player_discounts[GemIndex[g]] = n
    player_score[0] = int(player.score)
    turn_mod_players[...] = int(state.turn % self._num_players)
    cards = list(state.visible_cards)
    cards.sort(key=lambda c: (c.level, c.id))
    limit = min(len(cards), self._visible_card_count)
    for idx in range(limit):
      card = cards[idx]
      visible_levels[idx] = int(card.level) - 1
      visible_points[idx] = int(card.points)
      bonus_index = 0
      if card.bonus is not None:
        bonus_index = GemIndex[card.bonus] + 1
      visible_bonus[idx] = int(bonus_index)
      for gem, cost in card.cost:
        visible_costs[idx, GemIndex[gem]] = int(cost)
    return {
      'bank': bank,
      'player_gems': player_gems,
      'player_discounts': player_discounts,
      'player_score': player_score,
      'turn_mod_players': turn_mod_players,
      'visible_cards': {
        'level': visible_levels,
        'points': visible_points,
        'bonus': visible_bonus,
        'costs': visible_costs,
      },
    }


__all__ = ["StateSpace"]
