import numpy as np
from typing import Union
from numpy.typing import NDArray
from numpy import int64
from math import floor
import numba as nb
from numba.experimental import jitclass

spec = [
    ('__dim', nb.int64)
]


@jitclass(spec)
class Alphabet:
    __dim: int

    def __init__(self, dim: int):
        """
        Initializes an Alphabet instance with a specified dimension and letters '1', '2', ..., 'dim'.

        :param dim: The dimension, representing the number of distinct letters.
        """
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Provided dimension should be a positive integer.")
        self.__dim = dim

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the alphabet.

        :return: The number of distinct letters in the alphabet.
        """
        return self.__dim

    def word_to_base_dim_number(self, word: str) -> int:
        """
        Converts a word to the corresponding number with base dim.

        :param word: A string representing the word to convert.
        :return: An integer representing the number of the word.
        """
        if word in ["", "∅", "Ø"]:
            return 0
        length = len(str(word))
        num = np.sum((np.array([ord(a) - ord("0") for a in word], dtype=int64) - 1) *
                     self.__dim**(np.arange(length - 1, -1, -1)))
        return num

    def word_to_index(self, word: str) -> int:
        """
        Converts a word to its corresponding one-dimensional index in the tensor algebra.

        :param word: A string representing the word to convert.
        :return: An integer representing the index of the word.
        """
        if word in ["", "∅", "Ø"]:
            return 0
        index = self.__dim ** len(str(word)) - 1 + self.word_to_base_dim_number(word=word)
        return index

    def index_to_word(self, index: int) -> str:
        """
        Converts an index back to its corresponding word in the alphabet.

        :param index: An integer representing the index to convert.
        :return: A string representing the word corresponding to the given index.
        """
        if not index:
            return "∅"
        length = floor(np.log2(index + 1) / np.log2(self.__dim))
        index = index - (self.__dim ** length - 1)
        word_list = [""] * length
        for i in range(length):
            p = self.__dim ** (length - 1 - i)
            digit = index // p
            index = index % p
            word_list[i] = str(int(digit) + 1)
        return "".join(word_list)

    def index_to_length(self, index: NDArray[int64]) -> NDArray[int64]:
        """
        Computes the length of the word corresponding to a given index.

        :param index: An integer or array of integers representing the index or indices to evaluate.
        :return: An integer or array of integers representing the length of
                 the word(s) corresponding to the given index/indices.
        """
        return (np.log2(index + 1) / np.log2(self.__dim)).astype(int64)
