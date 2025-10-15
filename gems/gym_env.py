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
  - Complex action parameters (e.g. payments/returns) are handled by engine's
    legal action enumeration; we just pick by index.

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

from typing import Callable, Sequence, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gems.consts import CARD_VISIBLE_TOTAL_COUNT

from .engine import Engine
from .actions import Action
from .typings import Gem
from .agents.random import RandomAgent


GemIndex = {g: i for i, g in enumerate(Gem)}  # order: enum definition order
GEM_COUNT = len(GemIndex)


def default_reward_fn(prev_score: int, new_score: int) -> float:
  """Default reward: delta in controlled player's score."""
  return float(new_score - prev_score)


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
    # per_card = level(1) + points(1) + bonus_index(1) + cost[6] = 9
    self._per_card_feats = 9
    self._obs_len = 6 + 6 + 6 + 1 + 1 + (CARD_VISIBLE_TOTAL_COUNT * self._per_card_feats)
    self.observation_space = spaces.Box(low=0, high=255, shape=(self._obs_len,), dtype=np.int32)
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
    obs = self._make_obs()
    info = self._info()
    return obs, info

  def step(self, action: int):
    if self._engine is None:
      raise RuntimeError("Environment not reset")
    self._advance_until_our_turn()
    if self._engine is None:
      return
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
    obs = self._make_obs()
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

  def _make_obs(self) -> np.ndarray:
    state = self._engine.get_state() if self._engine is not None else None
    if state is None:
      return np.zeros((self._obs_len,), dtype=np.int32)
    player = state.players[self.seat_id]
    vec: list[int] = []
    # Bank
    bank_counts = {g: 0 for g in Gem}
    for g, n in state.bank:
      bank_counts[g] = n
    vec.extend(bank_counts[g] for g in Gem)
    # Player gems
    player_counts = {g: 0 for g in Gem}
    for g, n in player.gems:
      player_counts[g] = n
    vec.extend(player_counts[g] for g in Gem)
    # Player discounts
    disc_counts = {g: 0 for g in Gem}
    for g, n in player.discounts:
      disc_counts[g] = n
    vec.extend(disc_counts[g] for g in Gem)
    # Score + turn mod players
    vec.append(int(player.score))
    vec.append(int(state.turn % self.num_players))
    # Visible cards (pad to MAX_VISIBLE_TOTAL)
    cards = list(state.visible_cards)
    # Ensure deterministic ordering: sort by level then id
    cards.sort(key=lambda c: (c.level, c.id))
    if len(cards) > CARD_VISIBLE_TOTAL_COUNT:
      cards = cards[:CARD_VISIBLE_TOTAL_COUNT]
    per_card = self._per_card_feats
    for c in cards:
      bonus_index = 0
      if c.bonus is not None:
        bonus_index = list(Gem).index(c.bonus) + 1  # 0 means none
      cost_counts = {g: 0 for g in Gem}
      for g, n in c.cost:
        cost_counts[g] = n
      # level, points, bonus_index + cost[6]
      vec.append(int(c.level))
      vec.append(int(c.points))
      vec.append(int(bonus_index))
      vec.extend(cost_counts[g] for g in Gem)
    # Pad remaining cards
    remaining = CARD_VISIBLE_TOTAL_COUNT - len(cards)
    vec.extend([0] * (remaining * per_card))
    arr = np.array(vec, dtype=np.int32)
    # Safety: pad/trim to expected length
    if arr.shape[0] < self._obs_len:
      pad = self._obs_len - arr.shape[0]
      arr = np.concatenate([arr, np.zeros(pad, dtype=np.int32)])
    elif arr.shape[0] > self._obs_len:
      arr = arr[: self._obs_len]
    return arr

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

__all__ = ["GemEnv"]
