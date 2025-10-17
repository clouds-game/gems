import numpy as np

from gems.gym.sampling import sample_exact


def test_sample_exact_reproducible_seed():
  pop = [10, 11, 12, 13]
  mask = [True, True, True, True]
  p = [0.1, 0.2, 0.3, 0.4]
  a = sample_exact(3, mask, p, x=pop, replacement=False, seed=42)
  b = sample_exact(3, mask, p, x=pop, replacement=False, seed=42)
  assert np.array_equal(a, b)


def test_sample_exact_replacement_length():
  pop = [0, 1]
  mask = [True, True]
  p = [0.5, 0.5]
  s = sample_exact(10, mask, p, x=pop, replacement=True, seed=1)
  assert len(s) == 10


def test_sample_exact_no_replacement_capped():
  pop = [4, 5, 6]
  mask = [True, False, True]
  p = [1.0, 0.0, 0.0]
  s = sample_exact(5, mask, p, x=pop, replacement=False, seed=3)
  # available=1 => length capped to 1
  assert len(s) == 1


def test_sample_exact_empty_mask_returns_empty():
  pop = [7, 8]
  mask = [False, False]
  p = [1.0, 1.0]
  s = sample_exact(3, mask, p, x=pop, replacement=True, seed=0)
  assert s.size == 0


def test_sample_exact_nonfinite_weights_uses_uniform():
  pop = [20, 21, 22]
  mask = [True, True, True]
  p = [np.nan, np.inf, -np.inf]
  s = sample_exact(2, mask, p, x=pop, replacement=False, seed=7)
  assert len(s) == 2


def test_sample_exact_shape_mismatch_raises():
  pop = [1, 2]
  mask = [True, True]
  p = [1.0]
  try:
    sample_exact(1, mask, p, x=pop)
  except ValueError:
    return
  raise AssertionError("Expected ValueError for mask/p length mismatch")


def test_sample_exact_indices_reproducible():
  mask = [True, True, True, True]
  p = [0.1, 0.2, 0.3, 0.4]
  a = sample_exact(3, mask, p, x=None, replacement=False, seed=42)
  b = sample_exact(3, mask, p, x=None, replacement=False, seed=42)
  assert np.array_equal(a, b)
  assert a.dtype == np.int64 or np.issubdtype(a.dtype, np.integer)


def test_sample_exact_indices_replacement_length():
  mask = [True, True]
  p = [0.5, 0.5]
  s = sample_exact(10, mask, p, x=None, replacement=True, seed=1)
  assert len(s) == 10
  assert s.dtype == np.int64 or np.issubdtype(s.dtype, np.integer)


def test_sample_exact_indices_empty_mask_returns_empty():
  mask = [False, False]
  p = [1.0, 1.0]
  s = sample_exact(3, mask, p, x=None, replacement=True, seed=0)
  assert s.size == 0
  assert s.dtype == np.int64 or np.issubdtype(s.dtype, np.integer)
