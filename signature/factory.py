import jax
import jax.numpy as jnp

from .tensor_sequence import TensorSequence
from .words import number_of_words_up_to_trunc, word_to_index


@jax.jit
def zero_like(ts: TensorSequence) -> TensorSequence:
    """
    Creates an instance of TensorSequenceJAX with no indices and the same sizes of other axis.

    :return: A new zero TensorSequenceJAX.
    """
    return TensorSequence(array=jnp.zeros_like(ts.array), trunc=ts.trunc, dim=ts.dim)


@jax.jit
def unit_like(ts: TensorSequence) -> TensorSequence:
    """
    Creates an instance of TensorSequenceJAX with index 1 corresponding to the word Ø.

    :return: A unit element as TensorSequenceJAX.
    """
    array = jnp.zeros_like(ts.array)
    return TensorSequence(array=array.at[0].set(1), trunc=ts.trunc, dim=ts.dim)


def zero(trunc: int, dim: int) -> TensorSequence:
    """
    Creates an instance of TensorSequenceJAX with no indices and the same sizes of other axis.

    :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequenceJAX.
    :param dim: Path space dimension.

    :return: A new zero TensorSequenceJAX.
    """
    array = jnp.zeros(number_of_words_up_to_trunc(trunc=trunc, dim=dim))
    return TensorSequence(array=array, trunc=trunc, dim=dim)


def unit(trunc: int, dim: int) -> TensorSequence:
    """
    Creates an instance of TensorSequenceJAX with index 1 corresponding to the word Ø.

    :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequenceJAX.
    :param dim: Path space dimension.

    :return: A unit element as TensorSequenceJAX.
    """
    array = jnp.zeros(number_of_words_up_to_trunc(trunc=trunc, dim=dim))
    return TensorSequence(array=array.at[0].set(1), trunc=trunc, dim=dim)


def from_word(word: int, trunc: int, dim: int) -> TensorSequence:
    """
    Creates a TensorSequence from a given word and a truncation level.

    :param word: A word to transform into a tensor sequence.
    :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
    :param dim: Path space dimension.
    :return: A TensorSequence constructed from the provided word and truncation level.
    """
    index = word_to_index(word=word, dim=dim)
    array = jnp.zeros(number_of_words_up_to_trunc(trunc=trunc, dim=dim))
    return TensorSequence(array=array.at[index].set(1), trunc=trunc, dim=dim)


def from_dict(word_dict: dict, trunc: int, dim: int) -> TensorSequence:
    """
    Creates a TensorSequence from a given word dictionary and a truncation level.

    :param word_dict: A dictionary where keys are words (as integers) and values are their coefficients.
    :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
    :param dim: Path space dimension.

    :return: A TensorSequence constructed from the provided word dictionary and truncation level.
    """
    if not word_dict:
        return zero(trunc=trunc, dim=dim)
    else:
        # sort the arrays with respect to indices
        indices, values = zip(*sorted(zip(
            [word_to_index(word=word, dim=dim) for word in word_dict.keys()],
            list(word_dict.values())
        )))
        indices = jnp.array(indices)
        values = jnp.array(values)

        n_elem = number_of_words_up_to_trunc(trunc=trunc, dim=dim)
        array = jnp.zeros((n_elem,) + values.shape[1:])
        array = array.at[indices].set(values)

    return TensorSequence(array=array, trunc=trunc, dim=dim)


def from_array(array: jax.Array, trunc: int, dim: int) -> TensorSequence:
    """
    Creates a TensorSequence from a given array and (optionally) indices. If indices are not given, takes the
    indices of the first elements of the tensor algebra.

    :param array: An array of coefficients.
    :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
    :param dim: Path space dimension.

    :return: A TensorSequence constructed from the provided array and indices.
    """
    array_ts = jnp.zeros((number_of_words_up_to_trunc(trunc=trunc, dim=dim),) + array.shape[1:])
    array_ts = array_ts.at[jnp.arange(array.shape[0])].set(array)

    return TensorSequence(array=array_ts, trunc=trunc, dim=dim)
