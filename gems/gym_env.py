"""Gymnasium environment wrapper for the Gem Merchant engine.

Minimal single-agent view that lets an RL algorithm control a single
seat (player) while all opponents play random legal moves. The action
space is a fixed-size Discrete(N) over the current turn's enumerated
legal actions. Observations are a fixed-length integer vector encoding
public game state features (bank, player gems/discounts, visible cards, etc.).

Design goals:
  - Keep dependency surface tiny (only gymnasium + numpy which are already deps).
  - Deterministic when seeded.
  - Avoid mutating engine objects outside of applying actions.

Limitations / Simplifications:
  - Opponents are random each step (could be replaced by custom agents).
  - Rewards are shaped as delta in the controlled player's score only.
  - Truncation not currently used (always False); episode ends when engine.game_end().

Usage example:
  from gym_env import GemEnv
  import gymnasium as gym
  env = GemEnv(num_players=2, seat_id=0, max_actions=128, seed=42)
  obs, info = env.reset()
  done = False
  while not done:
    action = env.action_space.sample()  # or a policy network output
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
  env.close()
"""
from __future__ import annotations

from typing import Callable, Sequence, Any, TypedDict

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gems.consts import CARD_VISIBLE_TOTAL_COUNT, CARD_LEVEL_COUNT, CARD_MAX_COUNT_RESERVED

from .engine import Engine
from .actions import Action
from .typings import Gem, ActionType, Card, CardIdx
from .consts import COIN_MAX_COUNT_PER_PLAYER
from .agents.random import RandomAgent


GemIndex = {g: i for i, g in enumerate(Gem)}  # order: enum definition order
GEM_COUNT = len(GemIndex)


def default_reward_fn(prev_score: int, new_score: int) -> float:
  """Default reward: delta in controlled player's score."""
  return float(new_score - prev_score)


class StateSpace(spaces.Dict):
  """Helper to build observations from an Engine state.

  This class is also a gymnasium Space by wrapping an internal Dict space so
  it can be assigned directly to `env.observation_space` while providing a
  `make_obs(engine, seat_id)` method that produces the structured numpy
  observation dict.
  """

  class CardDict(TypedDict):
    level: np.ndarray  # shape (CARD_VISIBLE_TOTAL_COUNT,)
    points: np.ndarray  # shape (CARD_VISIBLE_TOTAL_COUNT,)
    bonus: np.ndarray  # shape (CARD_VISIBLE_TOTAL_COUNT,)  (0 == none, 1..GEM_COUNT map to GemIndex+1)
    costs: np.ndarray  # shape (CARD_VISIBLE_TOTAL_COUNT, GEM_COUNT)

  class StateDict(TypedDict):
    bank: np.ndarray  # shape (GEM_COUNT,)
    player_gems: np.ndarray  # shape (GEM_COUNT,)
    player_discounts: np.ndarray  # shape (GEM_COUNT,)
    player_score: np.ndarray  # shape (1,)
    turn_mod_players: np.ndarray  # shape (), scalar
    visible_cards: "StateSpace.CardDict"  # structured sub-dict

  def __init__(self, per_card_feats: int, num_players: int, visible_card_count: int, *, seed = None):
    # initialize base Space with a dummy low/high; we'll delegate to internal
    self._per_card_feats = per_card_feats
    self._num_players = num_players
    self._visible_card_count = visible_card_count

    super().__init__({
      'bank': spaces.Box(low=0, high=255, shape=(GEM_COUNT,), dtype=np.int32),
      'player_gems': spaces.Box(low=0, high=255, shape=(GEM_COUNT,), dtype=np.int32),
      'player_discounts': spaces.Box(low=0, high=255, shape=(GEM_COUNT,), dtype=np.int32),
      'player_score': spaces.Box(low=0, high=255, shape=(1,), dtype=np.int32),
      'turn_mod_players': spaces.Discrete(self._num_players),
      'visible_cards': spaces.Dict({
        'level': spaces.Box(low=0, high=CARD_LEVEL_COUNT, shape=(CARD_VISIBLE_TOTAL_COUNT,), dtype=np.int32),
        'points': spaces.Box(low=0, high=255, shape=(CARD_VISIBLE_TOTAL_COUNT,), dtype=np.int32),
        'bonus': spaces.MultiDiscrete([GEM_COUNT + 1] * CARD_VISIBLE_TOTAL_COUNT),
        'costs': spaces.Box(low=0, high=255, shape=(CARD_VISIBLE_TOTAL_COUNT, GEM_COUNT), dtype=np.int32),
      }),
    }, seed=seed)

  def make_obs(self, engine: Engine | None, seat_id: int) -> StateDict:
    bank = np.zeros(GEM_COUNT, dtype=np.int32)
    player_gems = np.zeros(GEM_COUNT, dtype=np.int32)
    player_discounts = np.zeros(GEM_COUNT, dtype=np.int32)
    player_score = np.zeros(1, dtype=np.int32)
    # represent turn_mod_players as a scalar ndarray (0-d) to align with Discrete space
    turn_mod_players = np.array(0, dtype=np.int32)
    # visible cards split into structured fields
    visible_levels = np.zeros((self._visible_card_count,), dtype=np.int32)
    visible_points = np.zeros((self._visible_card_count,), dtype=np.int32)
    # bonus: 0 == none, 1..GEM_COUNT map to GemIndex+1
    visible_bonus = np.zeros((self._visible_card_count,), dtype=np.int32)
    visible_costs = np.zeros((self._visible_card_count, GEM_COUNT), dtype=np.int32)
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


class ActionSpace(spaces.Dict):
  """Structured encoding of Action objects for gymnasium agents.

  The `type` field selects which subset of data is meaningful for the action.
  Other subsets remain zeroed so a single dict shape can represent all actions.
  """

  _CARD_ID_MAX_LENGTH = 64

  class Take3Dict(TypedDict):
    gems: np.ndarray
    ret: np.ndarray

  class Take2Dict(TypedDict):
    gem: np.ndarray
    count: np.ndarray
    ret: np.ndarray

  class BuyDict(TypedDict):
    # flattened card index (0..visible+reserved-1)
    card_idx: np.ndarray
    payment: np.ndarray

  class ReserveDict(TypedDict):
    card_idx: np.ndarray
    take_gold: np.ndarray
    ret: np.ndarray

  class ActionDict(TypedDict):
    type: np.ndarray
    take3: "ActionSpace.Take3Dict"
    take2: "ActionSpace.Take2Dict"
    buy: "ActionSpace.BuyDict"
    reserve: "ActionSpace.ReserveDict"

  def __init__(self, *, seed = None):
    self._type_order = tuple(ActionType)
    self._type_index = {atype: idx for idx, atype in enumerate(self._type_order)}
    # max index space for cards = visible cards + reserved capacity (3) + deck head levels
    # layout (flattened index):
    # 0..(visible-1) => visible_idx
    # visible..(visible+reserved-1) => reserve_idx (offset by visible)
    # visible+reserved.. => deck_head_level (offset by visible+reserved), one entry per level
    self._visible_count = CARD_VISIBLE_TOTAL_COUNT
    self._reserve_count = CARD_MAX_COUNT_RESERVED
    self._deck_levels = CARD_LEVEL_COUNT
    self._max_card_index = self._visible_count + self._reserve_count + self._deck_levels
    super().__init__({
      'type': spaces.Discrete(len(self._type_order)),
      'take3': spaces.Dict({
        'gems': spaces.Box(low=0, high=1, shape=(GEM_COUNT,), dtype=np.int8),
        'ret': spaces.Box(low=0, high=COIN_MAX_COUNT_PER_PLAYER, shape=(GEM_COUNT,), dtype=np.int8),
      }),
      'take2': spaces.Dict({
        'gem': spaces.MultiBinary(GEM_COUNT),
        'count': spaces.Discrete(3),
        'ret': spaces.Box(low=0, high=COIN_MAX_COUNT_PER_PLAYER, shape=(GEM_COUNT,), dtype=np.int8),
      }),
      'buy': spaces.Dict({
        'card_idx': spaces.Discrete(self._max_card_index),
        'payment': spaces.Box(low=0, high=255, shape=(GEM_COUNT,), dtype=np.int32),
      }),
      'reserve': spaces.Dict({
        'card_idx': spaces.Discrete(self._max_card_index),
        'take_gold': spaces.Discrete(2),
        'ret': spaces.Box(low=0, high=1, shape=(GEM_COUNT,), dtype=np.int8),
      }),
    }, seed=seed)

  def empty(self) -> "ActionSpace.ActionDict":
    return {
      'type': np.array(0, dtype=np.int32),
      'take3': {
        'gems': np.zeros(GEM_COUNT, dtype=np.int8),
        'ret': np.zeros(GEM_COUNT, dtype=np.int8),
      },
      'take2': {
        'gem': np.zeros(GEM_COUNT, dtype=np.int8),
        'count': np.array(0, dtype=np.int8),
        'ret': np.zeros(GEM_COUNT, dtype=np.int8),
      },
      'buy': {
        'card_idx': np.array(0, dtype=np.int32),
        'payment': np.zeros(GEM_COUNT, dtype=np.int32),
      },
      'reserve': {
        'card_idx': np.array(0, dtype=np.int32),
        'take_gold': np.array(0, dtype=np.int8),
        'ret': np.zeros(GEM_COUNT, dtype=np.int8),
      },
    }

  def encode(self, action: Action) -> "ActionSpace.ActionDict":
    data = self.empty()
    data['type'][...] = self._type_index[action.type]
    if action.type == ActionType.TAKE_3_DIFFERENT:
      take3 = data['take3']
      take3['gems'][...] = 0
      for gem in getattr(action, 'gems', ()):  # type: ignore[attr-defined]
        take3['gems'][GemIndex[gem]] = 1
      take3['ret'][...] = 0
      ret = getattr(action, 'ret', None)  # type: ignore[attr-defined]
      if ret:
        for gem, amount in ret:
          take3['ret'][GemIndex[gem]] = int(amount)
      return data
    if action.type == ActionType.TAKE_2_SAME:
      take2 = data['take2']
      take2['gem'][...] = 0
      gem = getattr(action, 'gem')  # type: ignore[attr-defined]
      take2['gem'][GemIndex[gem]] = 1
      count = getattr(action, 'count', 0)  # type: ignore[attr-defined]
      take2['count'][...] = int(count)
      take2['ret'][...] = 0
      ret = getattr(action, 'ret', None)  # type: ignore[attr-defined]
      if ret:
        for gem_ret, amount in ret:
          take2['ret'][GemIndex[gem_ret]] = int(amount)
      return data
    if action.type == ActionType.BUY_CARD:
      buy = data['buy']
      # encode CardIdx into a single integer index using the layout described above
      buy['card_idx'][...] = 0
      if getattr(action, 'idx', None) is not None:
        idx: CardIdx = action.idx  # type: ignore[assignment]
        if idx.visible_idx is not None:
          buy['card_idx'][...] = int(idx.visible_idx)
        elif idx.reserve_idx is not None:
          buy['card_idx'][...] = int(self._visible_count + int(idx.reserve_idx))
        elif idx.deck_head_level is not None:
          # deck_head_level expected 1..CARD_LEVEL_COUNT; map to 0-based
          level = int(idx.deck_head_level) - 1
          buy['card_idx'][...] = int(self._visible_count + self._reserve_count + level)
      buy['payment'][...] = 0
      for gem, amount in getattr(action, 'payment', ()):  # type: ignore[attr-defined]
        buy['payment'][GemIndex[gem]] = int(amount)
      return data
    if action.type == ActionType.RESERVE_CARD:
      reserve = data['reserve']
      reserve['card_idx'][...] = 0
      if getattr(action, 'idx', None) is not None:
        idx: CardIdx = action.idx  # type: ignore[assignment]
        if idx.visible_idx is not None:
          reserve['card_idx'][...] = int(idx.visible_idx)
        elif idx.reserve_idx is not None:
          reserve['card_idx'][...] = int(self._visible_count + int(idx.reserve_idx))
        elif idx.deck_head_level is not None:
          level = int(idx.deck_head_level) - 1
          reserve['card_idx'][...] = int(self._visible_count + self._reserve_count + level)
      take_gold = int(bool(getattr(action, 'take_gold', False)))  # type: ignore[attr-defined]
      reserve['take_gold'][...] = take_gold
      reserve['ret'][...] = 0
      ret = getattr(action, 'ret', None)  # type: ignore[attr-defined]
      if ret is not None:
        reserve['ret'][GemIndex[ret]] = 1
      return data
    if action.type == ActionType.NOOP:
      return data
    raise ValueError(f"Unsupported action type: {action.type}")

  def encode_many(self, actions: Sequence[Action]) -> list["ActionSpace.ActionDict"]:
    return [self.encode(action) for action in actions]

  # Note: card properties are not encoded in the ActionSpace anymore.
  # Actions that reference cards should use a discrete `card_idx` field
  # (0..visible+reserved-1). Mapping from Card -> index must be handled
  # externally by the caller if required.


class GemEnv(gym.Env):
  metadata = {"render_modes": ["human"], "render_fps": 4}

  def __init__(self,
               num_players: int = 2,
               seat_id: int = 0,
               max_actions: int = 128,
               seed: int | None = None,
               reward_fn: Callable[[int, int], float] | None = None,
               opponents: Sequence[Any] | None = None):
    super().__init__()
    if seat_id < 0 or seat_id >= num_players:
      raise ValueError("seat_id must be within number of players")
    self.num_players = num_players
    self.seat_id = seat_id
    self.max_actions = max_actions
    self._base_seed = seed
    self._rng = np.random.default_rng(seed)
    self._engine: Engine | None = None
    self._last_legal: list[Action] = []
    self._reward_fn = reward_fn or default_reward_fn
    # Opponents: default to random agents (one per other seat)
    if opponents is None:
      self._opponents: list[Any] = [RandomAgent(seat) for seat in range(num_players) if seat != seat_id]
    else:
      self._opponents = list(opponents)
    # Observation space: integer vector
    # Layout (all ints):
    # bank[6] + player_gems[6] + player_discounts[6] + player_score[1] + turn_mod_players[1]
    # + visible_cards[CARD_VISIBLE_TOTAL_COUNT * per_card]
    # per_card = level(1) + points(1) + bonus_onehot(GEM_COUNT+1) + cost[6]
    # bonus_onehot length = GEM_COUNT + 1 (0 index means none)
    # per_card_feats = 2 + (GEM_COUNT + 1) + GEM_COUNT  # level + points + bonus_onehot + cost
    # State space helper responsible for building observations and also
    # acts as the env.observation_space (it subclasses spaces.Space).
    self._state_space = StateSpace(2 + (GEM_COUNT + 1) + GEM_COUNT, self.num_players, CARD_VISIBLE_TOTAL_COUNT)
    self.observation_space = self._state_space
    # Action space: pick index into current legal actions; unused tail indices ignored
    self.action_space = spaces.Discrete(max_actions)

  # Gymnasium API -------------------------------------------------
  def seed(self, seed: int | None = None):  # pragma: no cover - compatibility
    if seed is not None:
      self._base_seed = seed
      self._rng = np.random.default_rng(seed)
    return [self._base_seed]

  def reset(self, *, seed: int | None = None, options: dict | None = None):
    if seed is not None:
      self.seed(seed)
    # fresh engine
    self._engine = Engine.new(num_players=self.num_players, seed=self._base_seed)
    # Sync opponents RNG for determinism
    for opp in self._opponents:
      if hasattr(opp, 'reset'):
        opp.reset(seed=self._base_seed)
    # Fast-forward until it's our seat's turn (should be turn 0 anyway)
    self._advance_until_our_turn()
    obs = self._state_space.make_obs(self._engine, self.seat_id)
    info = self._info()
    return obs, info

  def step(self, action: int):
    if self._engine is None:
      raise RuntimeError("Environment not reset")
    self._advance_until_our_turn()
    if self._engine is None:
      raise RuntimeError("Environment unavailable during step")
    state = self._engine.get_state()
    player = state.players[self.seat_id]
    prev_score = player.score
    legal = self._engine.get_legal_actions(self.seat_id)
    self._last_legal = legal
    if not legal:
      # Should not happen (noop fallback exists) but guard anyway
      raise RuntimeError("No legal actions available for current player")
    # Map incoming discrete action index into legal list
    if action < 0 or action >= len(legal):
      chosen = legal[0]
    else:
      chosen = legal[action]
    # Apply chosen action
    new_state = chosen.apply(state)
    self._engine._state = new_state  # internal update (engine is thin wrapper)
    self._engine._action_history.append(chosen)
    self._engine.advance_turn()
    # Opponents play until our next turn or game end
    self._play_opponents_until_our_turn()
    # Observation / reward
    new_player = self._engine.get_state().players[self.seat_id]
    reward = self._reward_fn(prev_score, new_player.score)
    terminated = self._engine.game_end()
    truncated = False
    obs = self._state_space.make_obs(self._engine, self.seat_id)
    info = self._info()
    info['chosen_action_index'] = int(action)
    info['legal_action_count'] = len(legal)
    info['action_applied_type'] = chosen.type.value
    return obs, reward, terminated, truncated, info

  def render(self):  # pragma: no cover - printing side-effect
    if self._engine is None:
      return
    self._engine.print_summary()

  def close(self):  # pragma: no cover - trivial
    self._engine = None

  # Internal helpers ----------------------------------------------
  def _advance_until_our_turn(self):
    # If engine is None, nothing to do
    if self._engine is None:
      return
    while not self._engine.game_end() and (self._engine.get_state().turn % self.num_players) != self.seat_id:
      self._play_single_opponent_turn()

  def _play_opponents_until_our_turn(self):
    if self._engine is None:
      return
    while not self._engine.game_end() and (self._engine.get_state().turn % self.num_players) != self.seat_id:
      self._play_single_opponent_turn()

  def _play_single_opponent_turn(self):
    if self._engine is None:
      return
    state = self._engine.get_state()
    seat = state.turn % self.num_players
    if seat == self.seat_id:
      return
    legal = self._engine.get_legal_actions(seat)
    if not legal:
      # fallback noop
      from gems.actions import Action as _A
      legal = [_A.noop()]
    # pick using matching opponent if indexed, else random
    opp = None
    # Opponents list excludes our seat; find by relative ordering
    filtered = [s for s in range(self.num_players) if s != self.seat_id]
    try:
      idx = filtered.index(seat)
      opp = self._opponents[idx] if idx < len(self._opponents) else None
    except ValueError:
      opp = None
    if opp is not None and hasattr(opp, 'act'):
      try:
        chosen = opp.act(state, legal)
      except Exception:
        chosen = legal[0]
    else:
      chosen = legal[0]
    new_state = chosen.apply(state)
    if self._engine is not None:
      self._engine._state = new_state
      self._engine._action_history.append(chosen)
      self._engine.advance_turn()

  # _make_obs removed: use StateSpace.make_obs instead

  def _info(self) -> dict:
    legal = self._engine.get_legal_actions(self.seat_id) if self._engine is not None else []
    return {
      'legal_action_count': len(legal),
      'max_actions': self.max_actions,
      'action_mask': self._legal_action_mask(len(legal)),
    }

  def _legal_action_mask(self, n: int):
    mask = np.zeros(self.max_actions, dtype=np.int8)
    mask[:min(n, self.max_actions)] = 1
    return mask

__all__ = ["GemEnv", "StateSpace", "ActionSpace"]
