"""Agent that samples actions from a gym-style ActionSpace and returns
the first sampled action that is legal for the current state.

This is useful for testing gym-based policies or for simple stochastic
agents that operate in the structured ActionSpace representation.
"""
from collections.abc import Sequence
from typing import cast

from .core import Agent
from ..actions import Action
from ..gym.action_space import ActionDict, ActionSpace


class SpaceSampleAgent(Agent):
  """Agent which samples from a provided ActionSpace and decodes samples
  into engine Actions. It will attempt up to `max_samples` samples and
  return the first one that appears in `legal_actions`. If no sampled
  action matches, it falls back to uniform random choice from
  `legal_actions` using the agent's RNG.

  Contract:
  - __init__(seat_id, action_space: ActionSpace, seed: int | None = None,
    max_samples: int = 16)
  - act(state, legal_actions, *, timeout=None) -> Action
  """

  def __init__(self, seat_id: int, action_space: ActionSpace, seed: int | None = None, max_samples: int = 16):
    super().__init__(seat_id, seed=seed)
    if action_space is None:
      raise ValueError("action_space is required")
    self.action_space = action_space
    self.max_samples = int(max_samples)

  def _reset(self) -> None:
    # Re-seed the ActionSpace RNG where supported to keep sampling
    # deterministic across agent.reset(seed) calls.
    try:
      # gym spaces provide a `seed` method; call it with the agent seed.
      self.action_space.seed(self._seed)
    except Exception:
      # If the provided ActionSpace doesn't support reseeding, ignore.
      pass

  def act(self, state, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:
    if not legal_actions:
      raise ValueError("No legal actions available")

    # Try sampling from the structured ActionSpace and decode to Action.
    # Return the first sample that equals one of the provided legal actions.
    for _ in range(self.max_samples):
      try:
        sample = cast(ActionDict, self.action_space.sample())
        action = self.action_space.decode(sample)
      except Exception:
        # Sampling/decoding may fail for invalid intermediate samples; try again.
        continue

      # Use dataclass equality on Action objects (works for action subclasses).
      for la in legal_actions:
        if action == la:
          return action

    # Fallback: pick uniformly among legal actions using agent RNG.
    return self.rng.choice(legal_actions)


__all__ = ["SpaceSampleAgent"]
