import jax
import jax.numpy as jnp
from jax import lax
from typing import Union
jax.config.update("jax_enable_x64", True)


@jax.jit
def word_len(word: Union[int, jax.Array]):
    """
    Computes the length (number of digits) of an integer `word` in decimal representation.

    :param word: The integer whose length (number of digits) is to be computed.
    :return: The number of digits in the integer `word`.
    """
    return jnp.where(word == 0, 0, jnp.floor(jnp.log10(word) + 1e-10) + 1).astype(int)


@jax.jit
def number_of_words_up_to_trunc(trunc: Union[int, jax.Array], dim: int = 2):
    """
    Calculates the number of words of length up to trunc.

    :param dim: the dimension.
    :param trunc: maximal word length.
    :return: Number of words v such that |v| <= trunc.
    """
    return (dim ** (trunc + 1) - 1) // (dim - 1)


@jax.jit
def index_to_length_of_word(index: Union[int, jax.Array], dim: int = 2) -> jax.Array:
    """
    Computes the length of the word corresponding to a given index.

    :param dim: the base dimension.
    :param index: An integer or array of integers representing the index or indices to evaluate.
    :return: An integer or array of integers representing the length of
             the word(s) corresponding to the given index/indices.
    """
    return (jnp.log2(index * (dim - 1) + 1) / jnp.log2(dim) + 1e-10).astype(int)


@jax.jit
def index_to_word(index: int, dim: int = 2) -> jnp.int64:
    """
    Converts an index back to its corresponding word in the alphabet.

    :param dim: the base dimension.
    :param index: An integer representing the index to convert.
    :return: An integer representing the word corresponding to the given index.
    """
    length = jnp.floor(jnp.log2(index + 1) / jnp.log2(dim) + 1e-10).astype(jnp.int64)
    index = index - (dim ** length - 1)

    def body_fun(i, state):
        res_tmp, index_inner = state
        p = dim ** (length - 1 - i)
        digit = index_inner // p
        index_inner = index_inner % p
        res_tmp = 10 * res_tmp + (jnp.array(digit, dtype=jnp.int64) + 1)
        return jnp.array((res_tmp, index_inner), dtype=jnp.int64)

    # Initial state is (0, index)
    res, _ = lax.fori_loop(lower=0, upper=length, body_fun=body_fun, init_val=jnp.array((0, index), dtype=jnp.int64))
    return res


@jax.jit
def word_to_base_dim_number(word: int, dim: int = 2) -> int:
    """
    Converts a word to the corresponding number with base `dim`.

    :param word: An integer representing the word to convert.
    :param dim: The base of the number system (default: 2).
    :return: An integer corresponding to the word if the word is written under base `dim`
             in alphabet {1, 2, ..., dim}.
    """
    def cond_fun(state):
        word_innder, _, _ = state
        return word_innder > 0

    def body_fun(state):
        word_innder, res, p = state
        res += ((word_innder % 10) - 1) * dim ** p
        word_innder //= 10
        p += 1
        return word_innder, res, p

    # Loop state: (word, res, p)
    _, result, _ = jax.lax.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=(word, 0, 0))
    return result


@jax.jit
def word_to_index(word: int, dim: int = 2) -> int:
    """
    Converts a word to its corresponding one-dimensional index in the tensor algebra.

    :param dim: the base dimension.
    :param word: A string representing the word to convert.
    :return: An integer representing the index of the word.
    """
    return dim ** word_len(word) - 1 + word_to_base_dim_number(word, dim)


word_to_base_dim_number_vect = jax.jit(jax.vmap(word_to_base_dim_number, in_axes=(0, None)))
word_to_index_vect = jax.jit(jax.vmap(word_to_index, in_axes=(0, None)))
index_to_word_vect = jax.jit(jax.vmap(index_to_word, in_axes=(0, None)))
