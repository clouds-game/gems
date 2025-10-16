
from typing import TypeAlias, TypeVar

import numpy as np


_ScalarT = TypeVar('_ScalarT', bound=np.generic)
NDArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[_ScalarT]]
NDArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_ScalarT]]
Scalar: TypeAlias = np.ndarray[tuple[()], np.dtype[_ScalarT]]
