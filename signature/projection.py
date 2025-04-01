import jax
import jax.numpy as jnp
from .tensor_sequence import TensorSequence
from .words import word_len, word_to_index, index_to_word_len


@jax.jit
def proj(ts: TensorSequence, word: int) -> TensorSequence:
    """
    Calculates the projection of TensorSequence with respect to the given word.

    :param ts: A tensor sequence which projection should be computed.
    :param word: The word (as integer) to calculate the projection.
    :return: A new TensorSequence representing the projection.
    """
    indices = jnp.arange(len(ts))
    word_length = word_len(word)
    word_index = word_to_index(word, dim=ts.dim)

    # A mask for indices to keep
    indices_mask = (((indices - word_index) % ts.dim**word_length) == 0) & (indices >= word_index)

    # Compute new indices
    length_arr = index_to_word_len(indices, dim=ts.dim)
    new_indices = (indices - ts.dim ** length_arr + 1) // ts.dim ** word_length + \
                  ts.dim ** (length_arr - word_length) - 1
    # Set out-of-bounds index for non-valid ones
    new_indices = jnp.where(indices_mask, new_indices, len(ts) + 1)

    array = jnp.zeros_like(ts.array)
    array = array.at[new_indices].set(jnp.where(jnp.einsum("i..., i -> i...", jnp.ones_like(ts.array), indices_mask),
                                                jnp.einsum("i..., i -> i...", ts.array, indices_mask), 0))

    return TensorSequence(array=array, trunc=ts.trunc, dim=ts.dim)