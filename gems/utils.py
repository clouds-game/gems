from typing import TypeVar
from collections.abc import Sequence

T = TypeVar('T')

def _to_kv_tuple(v: Sequence):
  """Normalize a dict or iterable of pairs into a stable tuple of pairs.

  Kept as a small utility to be shared by typing and state helpers.
  """
  if isinstance(v, tuple):
    return v
  if isinstance(v, dict):
    # sort dict items by stringified key so callers may pass either
    # string keys or Gem enum keys without causing a TypeError from
    # comparing Enum instances.
    return tuple(sorted(v.items(), key=lambda kv: str(kv[0])))
  return tuple(v)


def _replace_tuple(v: tuple[T, ...], i: int, d: T) -> tuple[T, ...]:
  """Return a new tuple where index `i` is replaced with `d`.

  This helper keeps code that needs to update a single element of an
  immutable tuple concise and avoids the common pattern `lst = list(t); lst[i]=d; t=tuple(lst)`.
  """
  if not (0 <= i < len(v)):
    raise IndexError("index out of range")
  # build new tuple via slicing for efficiency
  return v[:i] + (d,) + v[i+1:]
