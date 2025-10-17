# %%
"""Sampling helpers used by gym action spaces.

This small module provides `sample_exact` which performs masked,
weighted sampling over integer indices. It mirrors the behaviour used in
the action space sampling helpers: supports a mask (bool-like), a
probability/weight vector `p`, optional seed via `x`, and replacement
semantics.

Function contract:
- sample_exact(n, mask, p, *, x, replacement) -> ndarray[int]
  - n: number of samples requested (int)
  - mask: 1-D bool-like array where True marks allowed candidates
  - p: 1-D numeric array of same length as mask giving weights
  - x: optional integer seed for RNG (if None, uses numpy default RNG)
  - replacement: bool - if True sampling is with replacement

Returns an ndarray of selected flat indices (int). When replacement is
False the returned array has length min(n, available) where available is
the number of True entries in mask.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TypeVar
from collections.abc import Sequence

T = TypeVar("T")

# %%
def sample_exact(n: int, mask, p, *, x: Sequence[T] | None = None, replacement: bool = False, seed: Optional[int] = None) -> np.ndarray:
  """Sample indices or population elements from a weighted distribution with mask support.

  Behaviour details:
  - `mask` is interpreted as a boolean array-like. Entries where mask is
    truthy are eligible for sampling.
  - `p` provides non-negative weights for every index. We zero-out
    weights where mask is False.
  - If the (masked) total weight is non-finite or <= 0, uniform weights
    are used over the available indices.
  - If `replacement` is False, at most `available` distinct indices are
    returned (size = min(n, available)). When `replacement` is True the
    returned length equals `n`.
  - If `x` is provided it must be a sequence of the same length as
    `mask`/`p` and the function returns elements from `x` corresponding
    to the chosen indices. If `x` is None the function returns the
    chosen indices (dtype int).
  """
  mask_arr = np.asarray(mask, dtype=bool)
  p_arr = np.asarray(p, dtype=float)
  if mask_arr.ndim != 1 or p_arr.ndim != 1 or mask_arr.shape[0] != p_arr.shape[0]:
    raise ValueError("mask and p must be 1-D arrays of the same length")

  # validate population length matches mask/p if provided
  if x is not None:
    try:
      x_len = len(x)
    except Exception:
      raise ValueError("x must be a sequence with a length")
    if x_len != mask_arr.shape[0]:
      raise ValueError("x (population) must have the same length as mask and p")

  available = np.flatnonzero(mask_arr)
  if available.size == 0:
    # nothing available -> return empty (indices or population dtype)
    if x is None:
      return np.array([], dtype=np.int64)
    x_arr = np.asarray(x)
    return np.array([], dtype=x_arr.dtype)

  weights = p_arr[available].copy()
  total = float(np.nansum(weights))
  if not np.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights, dtype=float) / weights.size
  else:
    weights = weights / total

  rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

  if replacement:
    chosen_idx = rng.choice(available, size=int(n), replace=True, p=weights)
  else:
    take_count = int(min(n, available.size))
    if take_count <= 0:
      # return empty array of appropriate dtype
      if x is None:
        return np.array([], dtype=np.int64)
      x_arr = np.asarray(x)
      return np.array([], dtype=x_arr.dtype)
    # If there are fewer positive-weight entries than requested without
    # replacement, return as many positive-weight items as possible
    positive = int(np.count_nonzero(weights > 0.0))
    if positive < take_count:
      take_count = positive
    chosen_idx = rng.choice(available, size=take_count, replace=False, p=weights)

  # If x provided, map chosen indices back to population elements.
  if x is None:
    return np.asarray(chosen_idx, dtype=np.int64)
  x_arr = np.asarray(x)
  result = x_arr[chosen_idx]
  return np.asarray(result)

__all__ = ["sample_exact"]
