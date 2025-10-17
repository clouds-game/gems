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
from typing import Optional

# %%
def sample_exact(n: int, mask, p, *, x: Optional[int] = None, replacement: bool = False) -> np.ndarray:
  """Sample indices from a weighted distribution with mask support.

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
  - `x` if provided seeds a local RNG (numpy.default_rng(x)).
  """
  mask_arr = np.asarray(mask, dtype=bool)
  p_arr = np.asarray(p, dtype=float)
  if mask_arr.ndim != 1 or p_arr.ndim != 1 or mask_arr.shape[0] != p_arr.shape[0]:
    raise ValueError("mask and p must be 1-D arrays of the same length")

  available = np.flatnonzero(mask_arr)
  if available.size == 0:
    # nothing available -> return empty
    return np.array([], dtype=np.int64)

  weights = p_arr[available].copy()
  total = float(np.nansum(weights))
  if not np.isfinite(total) or total <= 0.0:
    weights = np.ones_like(weights, dtype=float) / weights.size
  else:
    weights = weights / total

  rng = np.random.default_rng(x) if x is not None else np.random.default_rng()

  if replacement:
    chosen = rng.choice(available, size=int(n), replace=True, p=weights)
  else:
    non_zero_count = np.count_nonzero(weights) if weights is not None else available.size
    take_count = int(min(n, non_zero_count))
    if take_count <= 0:
      return np.array([], dtype=np.int64)
    # when not replacing, use choice with replace=False. numpy will raise if
    # take_count > available.size but we already capped it.
    chosen = rng.choice(available, size=take_count, replace=False, p=weights)

  return np.asarray(chosen, dtype=np.int64)

__all__ = ["sample_exact"]
