from numpy.typing import NDArray
from numpy import int64, complex128
from numba.experimental import jitclass
import numba as nb
import numpy as np

from .tensor_sequence import TensorSequence
from .alphabet import Alphabet
from .shuffle import shuffle_product

spec = [
    ('__alphabet', Alphabet.class_type.instance_type),
    ('__trunc', nb.int64),
    ('__shuffle_table', nb.int64[:, :]),
]


@jitclass(spec)
class ShuffleOperator:
    def __init__(self, trunc, alphabet):
        self.__alphabet = alphabet
        self.__trunc = trunc

        n_words = self.alphabet.number_of_elements(trunc)

        arrs_to_stack = []
        table_len = 0
        for i in range(n_words):
            for j in range(n_words):
                if self.alphabet.index_to_length(np.array([i])) + self.alphabet.index_to_length(np.array([j])) <= trunc:
                    words_res, counts = shuffle_product(self.alphabet.index_to_int(i), self.alphabet.index_to_int(j))
                    indices_res = [self.alphabet.int_to_index(word) for word in words_res]

                    arr_to_add = np.zeros((len(counts), 4), dtype=int64)
                    arr_to_add[:, 0] = i
                    arr_to_add[:, 1] = j
                    arr_to_add[:, 2] = indices_res
                    arr_to_add[:, 3] = counts

                    table_len += arr_to_add.shape[0]
                    arrs_to_stack.append(arr_to_add)

        shuffle_table = np.zeros((table_len, 4), dtype=int64)
        iter = 0
        for arr_to_stack in arrs_to_stack:
            arr_len = arr_to_stack.shape[0]
            shuffle_table[iter:iter+arr_len, :] = arr_to_stack
            iter += arr_len

        self.__shuffle_table = shuffle_table.T

    @property
    def alphabet(self) -> Alphabet:
        """
        Returns the alphabet.

        :return: An alphabet for TS indices.
        """
        return self.__alphabet

    @property
    def shuffle_table(self) -> NDArray[complex128]:
        """
        Returns the shuffle table.

        :return: The table of precomputed shuffle products for words.
        """
        return self.__shuffle_table

    @property
    def trunc(self) -> int:
        """
        Returns the truncation level of the TensorSequence.

        :return: The truncation level as an integer.
        """
        return self.__trunc

    def __get_extended_array(self, ts: TensorSequence):
        n_elements = self.alphabet.number_of_elements(self.trunc)

        new_array = np.zeros((n_elements,) + ts.shape[1:], dtype=complex128)
        new_array[:min(n_elements, ts.shape[0])] = ts.array[:min(n_elements, ts.shape[0])]
        return new_array

    def shuffle_prod(
        self,
        ts1: TensorSequence,
        ts2: TensorSequence,
    ):
        index_left, index_right, index_result, count = self.shuffle_table

        if ts1.trunc < self.trunc:
            array_1 = self.__get_extended_array(ts1)
        else:
            array_1 = ts1.array

        if ts2.trunc < self.trunc:
            array_2 = self.__get_extended_array(ts2)
        else:
            array_2 = ts2.array

        source = count * array_1[index_left, 0, 0] * array_2[index_right, 0, 0]
        linear_result = np.zeros(index_result[-1] + 1, dtype=complex128)
        for i in range(len(index_result)):
            linear_result[index_result[i]] = linear_result[index_result[i]] + source[i]
        return TensorSequence(self.alphabet, self.trunc, linear_result)

    def shuffle_prod_2d(
        self,
        ts1: TensorSequence,
        ts2: TensorSequence
    ):
        index_left, index_right, index_result, count = self.shuffle_table
        left = ts1.array[:, :, 0][index_left]
        right = ts2.array[:, :, 0][index_right]
        source = np.reshape(np.ascontiguousarray(count), (count.size, 1)) * left * right
        source = source.T
        linear_result = np.zeros(shape=source.shape[:1] + (index_result[-1] + 1,), dtype=complex128)
        for i in range(len(index_result)):
            linear_result[:, index_result[i]] += source[:, i]
        return TensorSequence(self.alphabet, self.trunc, linear_result.T)

    def shuffle_pow(self, ts: TensorSequence, p: int) -> TensorSequence:
        """
        Raises the TensorSequence to a shuffle power p.

        :param ts: the input tensor sequence.
        :param p: The power to which the TensorSequence is raised.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the shuffle power.
        """

        if p == 0:
            return ts.unit(self.alphabet, self.trunc)

        res = ts.unit(self.alphabet, self.trunc)
        # TODO: think about more efficient implementation (with log_2(p) operations)
        for _ in range(p):
            res.update(self.shuffle_prod(res, ts))
        return res