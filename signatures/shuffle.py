import numpy as np
from numpy import int64, float64
from numba import jit
from typing import Tuple

from signatures.numba_utility import combinations


@jit(nopython=True)
def shuffle_product(word_1: str, word_2: str) -> Tuple:
    """
    Computes the shuffle product of two words, resulting in a dictionary where the keys
    are the possible shuffles (as strings) of the two words, and the values are the counts of each shuffle.

    The shuffle product interleaves the letters of `word_1` and `word_2` in all possible ways while maintaining
    the relative order of letters within each word.

    :param word_1: The first word as a string.
    :param word_2: The second word as a string.
    :return: A pair of resulting words from the shuffle product and counts of each word.
    """
    if word_1 in ["", "∅", "Ø"]:
        return [word_2], np.ones(1, dtype=int64)
    if word_2 in ["", "∅", "Ø"]:
        return [word_1], np.ones(1, dtype=int64)

    letters = np.array([ord(a) - ord("0") for a in list(word_1 + word_2)])
    l1, l2 = len(word_1), len(word_2)

    indices_left = np.array(list(combinations(np.arange(l1 + l2), l1)))
    indices_right = np.array(list(combinations(np.arange(l1 + l2), l2))[::-1])

    indices = np.zeros((indices_left.shape[0], l1 + l2))
    indices[:, :l1] = indices_left
    indices[:, l1:] = indices_right

    powers = l1 + l2 - 1 - indices
    shuffle = ((10 ** powers).astype(float64) @ letters.astype(float64)).astype(int64)

    sorted_shuffle = np.zeros(shuffle.size + 1, dtype=int64)
    sorted_shuffle[:shuffle.size] = np.sort(shuffle)
    sorted_shuffle[shuffle.size] = -1

    change_indices = np.where(np.diff(sorted_shuffle) != 0)[0]
    counts = np.zeros(change_indices.size, dtype=int64)
    counts[0] = change_indices[0] + 1
    counts[1:] = np.diff(change_indices)

    words = [str(word) for word in sorted_shuffle[change_indices]]
    return words, counts
