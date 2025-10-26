"""Agent that samples actions from a gym-style ActionSpace and returns
the first sampled action that is legal for the current state.

This is useful for testing gym-based policies or for simple stochastic
agents that operate in the structured ActionSpace representation.
"""
from collections.abc import Sequence
from typing import cast


from ..state import GameState
from ..actions import Action, BuyCardActionGold
from ..gym.action_space import ActionDict, ActionSpace
from ..typings import ActionType

from .core import Agent


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

  def __init__(self, seat_id: int, *, action_space: ActionSpace, seed: int | None = None, name: str | None = None, max_samples: int = 100):
    super().__init__(seat_id, seed=seed, name=name)
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

  def act(self, state: GameState, legal_actions: Sequence[Action], *, timeout: float | None = None) -> Action:
    if not legal_actions:
      raise ValueError("No legal actions available")

    state.print_summary()
    legal = set(legal_actions)

    # Try sampling from the structured ActionSpace and decode to Action.
    # Return the first sample that equals one of the provided legal actions.
    actions: list[Action] = []
    for i in range(self.max_samples):
      try:
        sample = cast(ActionDict, self.action_space.sample())
        # if actions is not None:
        #   sample['type'][...] = 2  # BUY_CARD
        action = self.action_space.decode(sample)
        if isinstance(action, BuyCardActionGold):
          assert action.idx is not None
          card = state.get_card(action.idx, seat_id=self.seat_id)
          action = action.normalize(card)
      except Exception as e:
        print(e)
        # Sampling/decoding may fail for invalid intermediate samples; try again.
        continue

      # print(f"Sampled action: [{i}] {action}")
      # if action not in legal:
      #   continue

      if action.type == ActionType.NOOP:
        continue

      if not action.check(state):
        if action.type == ActionType.BUY_CARD:
          print(f"Sampled action failed legality check: {action}")
        continue

      actions.append(action)

    for action in actions:
      if action.type == ActionType.BUY_CARD:
        return action

    for action in actions:
      return action

    # Fallback: pick uniformly among legal actions using agent RNG.
    return Action.noop()


__all__ = ["SpaceSampleAgent"]
