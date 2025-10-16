"""Compatibility shim: re-export new gym package symbols.

This file preserves the original module path `gems.gym_env` by importing
the split implementations from `gems.gym.*` modules. External imports
that reference `gems.gym_env` will continue to work.
"""
from .env import GemEnv
from .state_space import StateSpace
from .action_space import ActionSpace

__all__ = ["GemEnv", "StateSpace", "ActionSpace"]
