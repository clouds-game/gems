import numpy as np

from gems.gym.sampling import sample_exact, sample_exact_idx


def test_sample_exact_reproducible_seed():
  mask = [True, True, True, True]
  p = [0.1, 0.2, 0.3, 0.4]
  # new API: first arg is total (population size), second is number to draw
  a = sample_exact(4, 3, mask=mask, p=p, replacement=False, seed=42)
  b = sample_exact(4, 3, mask=mask, p=p, replacement=False, seed=42)
  assert np.array_equal(a, b)


def test_sample_exact_replacement_counts_sum():
  p = [0.5, 0.5]
  s = sample_exact(2, 10, mask=[True, True], p=p, replacement=True, seed=1)
  # With replacement the total count equals requested n
  assert int(s.sum()) == 10


def test_sample_exact_no_replacement_capped():
  p = [1.0, 0.0, 0.0]
  mask = [True, False, True]
  s = sample_exact(3, 5, mask=mask, p=p, replacement=False, seed=3)
  # available=1 => total count capped to 1
  assert int(s.sum()) == 1


def test_sample_exact_empty_mask_returns_zero_counts():
  p = [1.0, 1.0]
  mask = [False, False]
  s = sample_exact(2, 3, mask=mask, p=p, replacement=True, seed=0)
  # no available items => zero counts across the population
  assert s.size == 2
  assert int(s.sum()) == 0


def test_sample_exact_nonfinite_weights_raises():
  mask = [True, True, True]
  p = [np.nan, np.inf, -np.inf]
  with np.testing.assert_raises(ValueError):
    sample_exact(3, 2, mask=mask, p=p, replacement=False, seed=7)


def test_sample_exact_shape_mismatch_raises():
  # incompatible shapes for mask/p should raise during array ops
  mask = [True, True]
  p = [1.0, 1.0, 1.0]
  with np.testing.assert_raises(ValueError):
    sample_exact(2, 1, mask=mask, p=p)


def test_sample_exact_indices_reproducible():
  p = np.array([0.1, 0.2, 0.3, 0.4])
  rng1 = np.random.default_rng(42)
  rng2 = np.random.default_rng(42)
  a = sample_exact_idx(3, p, replacement=False, rng=rng1)
  b = sample_exact_idx(3, p, replacement=False, rng=rng2)
  assert np.array_equal(a, b)
  assert a.dtype == np.int64 or np.issubdtype(a.dtype, np.integer)


def test_sample_exact_indices_replacement_length():
  p = np.array([0.5, 0.5])
  s = sample_exact_idx(10, p, replacement=True, rng=np.random.default_rng(1))
  assert len(s) == 10
  assert s.dtype == np.int64 or np.issubdtype(s.dtype, np.integer)


def test_sample_exact_indices_empty_mask_returns_empty():
  p = np.array([0.0, 0.0])
  s = sample_exact_idx(3, p, replacement=True, rng=np.random.default_rng(0))
  assert s.size == 0
  assert s.dtype == np.int64 or np.issubdtype(s.dtype, np.integer)
