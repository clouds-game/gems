"""Top-level package exports for the gems project.

Expose a small, stable API so callers can `from gems import Engine, init_game`.
"""

from .engine import Engine

__all__ = ["Engine"]
