import numpy as np
from typing import TypeVar, overload
from collections.abc import Sequence

from gems.gym._common import NDArray1D, _ScalarT

T = _ScalarT

def sample_exact(total: int, n: int, *, dtype: type[T] = np.int64, mask: Sequence[bool] | np.ndarray | None = None, p: Sequence[float] | np.ndarray | None = None, replacement: bool = False, seed: int | None = None, rng: np.random.Generator | None = None) -> NDArray1D[T]:
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
  p_arr = np.asarray(p, dtype=float) if p is not None else np.ones(total, dtype=float)
  if p_arr.size != total:
    raise ValueError("p length does not match total")
  if mask is not None:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != p_arr.shape:
      raise ValueError("mask length does not match p")
    p_arr = p_arr.copy()
    p_arr[~mask_arr] = 0.0
  rng = rng or np.random.default_rng(seed)
  chosen_idx = sample_exact_idx(n, p_arr, replacement=replacement, rng=rng)
  # result should be an array of length `total` (counts per index). Using
  # np.zeros_like(p) is incorrect when `p` is None or not the same length as
  # `total`. Create an explicit zeros array and use np.add.at to correctly
  # accumulate counts when `chosen_idx` contains duplicates (replacement=True).
  result = np.zeros(total, dtype=dtype)
  if chosen_idx.size > 0:
    np.add.at(result, chosen_idx, 1)
  return result

def sample_exact_idx(n: int, p: np.ndarray, *, replacement: bool = False, rng: np.random.Generator) -> NDArray1D[np.int64]:
  if any(p == np.inf):
    # all +inf weights treated as uniform
    p = np.where(p == np.inf, 1.0, 0.0)
  available = np.flatnonzero(p > 0)
  if available.size == 0:
    return np.array([], dtype=np.int64)

  weights = p[available]
  total = float(weights.sum())
  if total <= 0.0:
    raise ValueError("Weights must be non-negative and not all zero")
  weights /= total

  if replacement:
    chosen_idx = rng.choice(available, size=int(n), replace=True, p=weights)
  else:
    # If there are fewer positive-weight entries than requested without
    # replacement, return as many positive-weight items as possible
    positive = int(np.count_nonzero(weights))
    take_count = int(min(n, positive))
    chosen_idx = rng.choice(available, size=take_count, replace=False, p=weights)

  return chosen_idx

def sample_single(total: int, *, dtype: type[T] = np.int64, mask: Sequence[bool] | np.ndarray | None = None, p: Sequence[float] | np.ndarray | None = None, seed: int | None = None, rng: np.random.Generator | None = None) -> T:
  """Sample a single index or population element from a weighted distribution with mask support.

  Behaviour details:
  - `mask` is interpreted as a boolean array-like. Entries where mask is
    truthy are eligible for sampling.
  - `p` provides non-negative weights for every index. We zero-out
    weights where mask is False.
  - If the (masked) total weight is non-finite or <= 0, ValueError is raised.
  - The function returns a single sampled index (dtype int) or element from
    `x` if provided.
  """
  p_arr = np.asarray(p, dtype=float) if p is not None else np.ones(total, dtype=float)
  if p_arr.size != total:
    raise ValueError("p length does not match total")
  if mask is not None:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != p_arr.shape:
      raise ValueError("mask length does not match p")
    p_arr = p_arr.copy()
    p_arr[~mask_arr] = 0.0
  rng = rng or np.random.default_rng(seed)
  chosen_idx = rng.choice(total, size=1, replace=False, p=p_arr / p_arr.sum()).astype(dtype)
  return chosen_idx[0]
