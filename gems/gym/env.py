"""Gymnasium environment wrapper for the Gem Merchant engine (split).

This module contains the `GemEnv` implementation and relies on the
`StateSpace` and `ActionSpace` classes from sibling modules.
"""
from __future__ import annotations

from typing import Callable, Any, Sequence, cast

import numpy as np
import gymnasium as gym

from ..state import PlayerState
from ..engine import Engine
from ..actions import Action
from ..typings import Gem
from ..consts import GameConfig
from .state_space import StateSpace
from .action_space import ActionDict, ActionSpace
from ..agents.random import RandomAgent


def default_reward_fn(prev_state: PlayerState, new_state: PlayerState) -> float:
  """Default reward: delta in controlled player's score."""
  return float(new_state.score - prev_state.score)


class GemEnv(gym.Env):
  metadata = {"render_modes": ["human"], "render_fps": 4}

  def __init__(self,
               game_config: GameConfig | None = None,
               seat_id: int = 0,
               seed: int | None = None,
               reward_fn: Callable[[PlayerState, PlayerState], float] | None = None,
               opponents: Sequence[Any] | None = None):
    super().__init__()
    config = game_config or GameConfig()
    num_players = config.num_players
    if seat_id < 0 or seat_id >= num_players:
      raise ValueError("seat_id must be within number of players")
    self.num_players = num_players
    self.seat_id = seat_id
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
    # State space helper responsible for building observations and also
    # acts as the env.observation_space (it subclasses spaces.Space).
    self._state_space = StateSpace(config, seed=self._base_seed)
    self.observation_space = self._state_space
    # Action space: structured ActionSpace that encodes Action objects
    self._action_space = ActionSpace(config, seed=self._base_seed)
    self.action_space = self._action_space

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

  def step(self, action: int | Action | ActionDict):
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
    chosen_action = None
    penalty = 0
    # Accept integer index (legacy), Action objects, or structured action dicts
    if isinstance(action, int):
      if action >= 0 and action < len(legal):
        chosen_action = legal[action]
    elif isinstance(action, dict):
      # structured ActionSpace encoding provided by agent
      chosen_action = self._action_space.decode(cast(ActionDict, action))  # may raise
      # find matching legal action (by equality)
    else:
      # assume it's an Action instance
      if isinstance(action, Action):
        chosen_action = action
    if chosen_action is None:
      penalty = -0.1  # small penalty for invalid action
      chosen_action = Action.noop()  # fallback noop
    # Apply chosen action
    new_state = chosen_action.apply(state)
    self._engine._state = new_state  # internal update (engine is thin wrapper)
    self._engine._action_history.append(chosen_action)
    self._engine.advance_turn()
    # Opponents play until our next turn or game end
    self._play_opponents_until_our_turn()
    # Observation / reward
    new_player = self._engine.get_state().players[self.seat_id]
    reward = self._reward_fn(player, new_player) + penalty
    terminated = self._engine.game_end()
    truncated = False
    obs = self._state_space.make_obs(self._engine, self.seat_id)
    info = self._info()
    # record the original form when possible; use chosen_index computed above
    info['chosen_action'] = chosen_action
    info['legal_action_count'] = len(legal)
    info['action_applied_type'] = chosen_action.type.value
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
      from ..actions import Action as _A
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

  def _info(self) -> dict:
    legal = self._engine.get_legal_actions(self.seat_id) if self._engine is not None else []
    return {
      'legal_action_count': len(legal),
    }


__all__ = ["GemEnv", "StateSpace", "ActionSpace"]
