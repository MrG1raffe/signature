import jax.numpy as jnp
import numpy as np
import numba as nb
from numpy import int64
from typing import Tuple

from .words import number_of_words_up_to_trunc, index_to_word_len, index_to_word, word_to_index_vect
from .numba_utility import combinations


def get_shuffle_table(table_trunc: int, dim: int):
    n_words = number_of_words_up_to_trunc(table_trunc, dim)

    arrays_to_stack = []
    table_len = 0

    ij_array = np.array(np.meshgrid(np.arange(n_words), np.arange(n_words))).T.reshape(-1, 2)
    ij_array = ij_array[index_to_word_len(jnp.array(ij_array), dim=dim).sum(axis=1) <= table_trunc]

    for i, j in ij_array:
        words_res, counts = shuffle_product_words(int(index_to_word(i, dim=dim)), int(index_to_word(j, dim=dim)))
        indices_res = word_to_index_vect(jnp.array(words_res, dtype=jnp.int64), dim)

        arr_to_add = np.zeros((len(counts), 4), dtype=int)
        arr_to_add[:, 0] = i
        arr_to_add[:, 1] = j
        arr_to_add[:, 2] = indices_res
        arr_to_add[:, 3] = counts

        table_len += arr_to_add.shape[0]
        arrays_to_stack.append(arr_to_add)

    shuffle_table = np.vstack(arrays_to_stack)
    return shuffle_table.T


@nb.jit(nopython=True)
def shuffle_product_words(word_1: int, word_2: int) -> Tuple:
    """
    Computes the shuffle product of two words, resulting in a dictionary where the keys
    are the possible shuffles (as int) of the two words, and the values are the counts of each shuffle.

    The shuffle product interleaves the letters of `word_1` and `word_2` in all possible ways while maintaining
    the relative order of letters within each word.

    :param word_1: The first word as an integer.
    :param word_2: The second word as an integer.
    :return: A pair of resulting words from the shuffle product and counts of each word.
    """
    if word_1 == 0:
        return np.array([word_2], dtype=int64), np.ones(1, dtype=int64)
    if word_2 == 0:
        return np.array([word_1], dtype=int64), np.ones(1, dtype=int64)

    l1, l2 = (np.log10(np.array([word_1, word_2], dtype=int64)) + 1).astype(int64)
    word_concat = word_1 * 10**l2 + word_2
    letters = np.array([word_concat // 10**k % 10 for k in range(l1 + l2 - 1, -1, -1)])

    indices_left = combinations(np.arange(l1 + l2), l1)
    indices_right = combinations(np.arange(l1 + l2), l2)[::-1]

    indices = np.zeros((indices_left.shape[0], l1 + l2), dtype=int64)
    indices[:, :l1] = indices_left
    indices[:, l1:] = indices_right

    powers = 10**(l1 + l2 - 1 - indices)
    shuffle = np.sum(powers * letters, axis=1)

    sorted_shuffle = np.zeros(shuffle.size + 1, dtype=int64)
    sorted_shuffle[:shuffle.size] = np.sort(shuffle)
    sorted_shuffle[shuffle.size] = -1

    change_indices = np.where(np.diff(sorted_shuffle) != 0)[0]
    counts = np.zeros(change_indices.size, dtype=int64)
    counts[0] = change_indices[0] + 1
    counts[1:] = np.diff(change_indices)

    return sorted_shuffle[change_indices], counts
