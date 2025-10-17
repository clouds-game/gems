import numpy as np

from gems.gym.sampling import sample_exact


def test_sample_exact_reproducible_seed():
  mask = [True, True, True, True]
  p = [0.1, 0.2, 0.3, 0.4]
  a = sample_exact(3, mask, p, x=42, replacement=False)
  b = sample_exact(3, mask, p, x=42, replacement=False)
  assert np.array_equal(a, b)


def test_sample_exact_replacement_length():
  mask = [True, True]
  p = [0.5, 0.5]
  s = sample_exact(10, mask, p, x=1, replacement=True)
  assert len(s) == 10


def test_sample_exact_no_replacement_capped():
  mask = [True, False, True]
  p = [1.0, 0.0, 0.0]
  s = sample_exact(5, mask, p, x=3, replacement=False)
  # available=1 => length capped to 1
  assert len(s) == 1


def test_sample_exact_empty_mask_returns_empty():
  mask = [False, False]
  p = [1.0, 1.0]
  s = sample_exact(3, mask, p, x=0, replacement=True)
  assert s.size == 0


def test_sample_exact_nonfinite_weights_uses_uniform():
  mask = [True, True, True]
  p = [np.nan, np.inf, -np.inf]
  s = sample_exact(2, mask, p, x=7, replacement=False)
  assert len(s) == 2


def test_sample_exact_shape_mismatch_raises():
  mask = [True, True]
  p = [1.0]
  try:
    sample_exact(1, mask, p)
  except ValueError:
    return
  raise AssertionError("Expected ValueError for mask/p length mismatch")
