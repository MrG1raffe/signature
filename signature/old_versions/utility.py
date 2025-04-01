from numba import jit
import numpy as np
from numpy.typing import NDArray
from numpy import float64

from signature.old_versions.alphabet import Alphabet


@jit(nopython=True)
def get_lengths_array(alphabet: Alphabet, trunc: int) -> NDArray[float64]:
    return alphabet.index_to_length(np.arange(alphabet.number_of_elements(trunc)))
