import jax
import jax.numpy as jnp
from numba import jit
import numpy as np
from numpy.typing import NDArray
from numpy import float64

from signature.old_versions.alphabet import Alphabet
from .jax_signature.words import number_of_words_up_to_trunc, index_to_word_len


@jit(nopython=True)
def get_lengths_array(alphabet: Alphabet, trunc: int) -> NDArray[float64]:
    return alphabet.index_to_length(np.arange(alphabet.number_of_elements(trunc)))


@jax.jit
def get_lengths_array_jax(trunc: int, dim: int) -> jax.Array:
    return index_to_word_len(index=jnp.arange(number_of_words_up_to_trunc(trunc, dim)), dim=dim)
