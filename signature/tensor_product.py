import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from typing import Union

from .tensor_sequence import TensorSequence
from .words import word_len, word_to_base_dim_number, index_to_word_len
from .factory import zero_like, unit_like


@jax.jit
def tensor_prod_word(ts: TensorSequence, word: int) -> TensorSequence:
    """
    Performs the tensor product of the current TensorSequence with a given word and
    multiply the result by `coefficient`.

    :param ts: Tensor sequence to multiply by word.
    :param word: The word to tensor multiply with.
    :return: A new TensorSequence representing the tensor product.
    """
    indices = jnp.arange(len(ts))

    word_length = word_len(word)
    word_dim_base = word_to_base_dim_number(word, dim=ts.dim)
    length_indices = index_to_word_len(indices, dim=ts.dim)
    new_indices = (ts.dim ** length_indices * ts.dim ** word_length - 1) + \
                   ts.dim ** word_length * (indices - ts.dim ** length_indices + 1) + word_dim_base

    array = jnp.zeros_like(ts.array)
    array = array.at[new_indices].set(ts.array)
    return TensorSequence(array=array, trunc=ts.trunc, dim=ts.dim)


@jax.jit
def _tensor_prod_index(ts: TensorSequence, index: int, coefficient: Union[float, jax.Array] = 1) -> TensorSequence:
    """
    Performs the tensor product of the current TensorSequence with a given index and
    multiply the result by `coefficient`.

    :param ts: Tensor sequence to multiply by word.
    :param index: The index of word to tensor multiply with.
    :param coefficient: The coefficient to multiply the resulting tensor product.
    :return: A new TensorSequence representing the tensor product.
    """
    indices = jnp.arange(len(ts))

    dim = ts.dim
    other_len = index_to_word_len(jnp.array([index]), dim=dim)
    other_dim_base = index - dim ** other_len + 1
    length_indices = index_to_word_len(indices, dim=dim)
    new_indices = (dim ** length_indices * dim ** other_len - 1) + \
                   dim ** other_len * (indices - dim ** length_indices + 1) + other_dim_base

    array = jnp.zeros_like(ts.array)
    array = array.at[new_indices].set(ts.array * coefficient)

    return TensorSequence(array=array, trunc=ts.trunc, dim=dim)


@jax.jit
def tensor_prod(ts1: TensorSequence, ts2: TensorSequence) -> TensorSequence:
    """
    Performs the tensor product of one TensorSequence with another TensorSequence.

    :param ts1: The first TensorSequence.
    :param ts2: The second TensorSequence.
    :return: A new TensorSequence representing the tensor product.
    """
    other_array = ts2.array

    def body_fun(i, acc):
        coefficient = other_array[i]
        return jax.lax.cond(
            jnp.allclose(coefficient, 0),
            lambda: acc,
            lambda: acc + _tensor_prod_index(ts1, i, coefficient)
        )

    res = jax.lax.fori_loop(lower=0, upper=len(ts2), body_fun=body_fun, init_val=zero_like(ts1))
    return res


@jax.jit
def tensor_pow(ts: TensorSequence, p: int) -> TensorSequence:
    """
    Raises the TensorSequence to a tensor power p.

    :param ts:
    :param p: The power to which the TensorSequence is raised.
    :return: A new TensorSequence representing the shuffle power.
    """
    def body_fun(i, acc):
        return tensor_prod(ts1=acc, ts2=ts)

    return jax.lax.fori_loop(lower=0, upper=p, body_fun=body_fun, init_val=unit_like(ts))


@jax.jit
def tensor_exp(ts: TensorSequence, N_trunc: int) -> TensorSequence:
    """
    Computes the shuffle exponential of the TensorSequence up to a specified truncation level.

    :param ts:
    :param N_trunc: The truncation level for the exponential.
    :return: A new TensorSequence representing the shuffle exponential.
    """
    def body_fun(i, acc):
        return acc + tensor_pow(ts=ts, p=i) / jsp.factorial(i)

    return jax.lax.fori_loop(lower=1, upper=N_trunc + 1, body_fun=body_fun, init_val=unit_like(ts))


def resolvent(ts: TensorSequence, N_trunc) -> TensorSequence:
    """
    Computes the resolvent of the TensorSequence up to a specified truncation level.
    The resolvent is defined as the series of the TensorSequence's tensor powers.

    :param ts:
    :param N_trunc: The truncation level for the resolvent.
    :return: A new TensorSequence representing the resolvent.
    """
    def body_fun(i, acc):
        return acc + tensor_pow(ts=ts, p=i)

    return jax.lax.fori_loop(lower=1, upper=N_trunc + 1, body_fun=body_fun, init_val=unit_like(ts))