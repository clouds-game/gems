import numpy as np
from typing import TypeVar, overload
from collections.abc import Sequence

from gems.gym._common import NDArray1D, _ScalarT

T = _ScalarT

@overload
def sample_exact(n: int, mask: Sequence[bool], p: Sequence[float], *, x: Sequence[T], replacement: bool = False, seed: int | None = None, rng: np.random.Generator | None = None) -> NDArray1D[T]: ...
@overload
def sample_exact(n: int, mask: Sequence[bool], p: Sequence[float], *, x: None = None, replacement: bool = False, seed: int | None = None, rng: np.random.Generator | None = None) -> NDArray1D[np.int64]: ...

def sample_exact(n: int, mask: Sequence[bool], p: Sequence[float], *, x: Sequence[T] | None = None, replacement: bool = False, seed: int | None = None, rng: np.random.Generator | None = None):
  """Sample indices or population elements from a weighted distribution with mask support.

  Behaviour details:
  - `mask` is interpreted as a boolean array-like. Entries where mask is
    truthy are eligible for sampling.
  - `p` provides non-negative weights for every index. We zero-out
    weights where mask is False.
  - If the (masked) total weight is non-finite or <= 0, ValueError is raised.
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
  x_arr = np.asarray(x) if x is not None else None
  rng = rng or np.random.default_rng(seed)
  if x_arr is not None and x_arr.shape != mask_arr.shape:
    raise ValueError("x (population) must have the same length as mask and p")
  chosen_idx = sample_exact_idx(n, mask_arr, p_arr, replacement=replacement, rng=rng)
  if x_arr is None:
    return chosen_idx
  result = x_arr[chosen_idx]
  return np.asarray(result)

def sample_exact_idx(n: int, mask: np.ndarray, p: np.ndarray, *, replacement: bool = False, rng: np.random.Generator) -> NDArray1D[np.int64]:
  mask_arr = np.asarray(mask, dtype=bool)
  p_arr = np.asarray(p, dtype=float)
  if mask_arr.ndim != 1 or p_arr.ndim != 1 or mask_arr.shape[0] != p_arr.shape[0]:
    raise ValueError("mask and p must be 1-D arrays of the same length")

  p_arr = p_arr * mask_arr
  available = np.flatnonzero(p_arr)
  if available.size == 0:
    return np.array([], dtype=np.int64)

  weights = p_arr[available]
  total = float(np.nansum(weights))
  if not np.isfinite(total) or total <= 0.0:
    raise ValueError("Weights must be non-negative and not all zero")

  if replacement:
    chosen_idx = rng.choice(available, size=int(n), replace=True, p=weights)
  else:
    # If there are fewer positive-weight entries than requested without
    # replacement, return as many positive-weight items as possible
    positive = int(np.count_nonzero(weights > 0.0))
    take_count = int(min(n, positive))
    chosen_idx = rng.choice(available, size=take_count, replace=False, p=weights)

  return chosen_idx
