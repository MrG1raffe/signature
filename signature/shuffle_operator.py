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

        self.__shuffle_table = shuffle_table

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

    def shuffle_prod_timedep(
        self,
        ts1: TensorSequence,
        ts2: TensorSequence,
        trunc: int = -1,
    ) -> TensorSequence:
        """
        Computes the shuffle product of two TensorSequences.
        :param ts1: The first TensorSequence.
        :param ts2: The second TensorSequence.
        :param trunc: Truncation order of the resulting TensorSequence.
        :return: A new TensorSequence representing the shuffle product of ts1 and ts2.
        """
        if trunc == -1:
            trunc = min(self.trunc, ts1.trunc + ts2.trunc)

        max_len = self.alphabet.number_of_elements(trunc)

        if self.shuffle_table[-1, 2] < max_len - 1:
            raise ValueError(f"Cannot compute the shuffle product with given shuffle_table with maximal"
                             f" word {self.shuffle_table[-1, 2]}. Recompute the shuffle_table or decrease trunc.")

        if self.shuffle_table[-1, 2] > max_len - 1:
            shuffle_table = self.shuffle_table[self.shuffle_table[:, 2] < max_len]
        else:
            shuffle_table = self.shuffle_table

        left_words_idx, right_words_idx, result_idx, counts = shuffle_table.T

        arr_left = np.zeros((max_len,) + ts1.array.shape[1:], dtype=complex128)
        arr_left[ts1.indices] = ts1.array

        arr_right = np.zeros((max_len,) + ts2.array.shape[1:], dtype=complex128)
        arr_right[ts2.indices] = ts2.array
        np.reshape(np.ascontiguousarray(counts), counts.shape)
        source = arr_left[left_words_idx] * arr_right[right_words_idx] * np.reshape(np.ascontiguousarray(counts),
                                                                                    (counts.size, 1, 1))
        result = np.zeros((max_len,) + source.shape[1:], dtype=complex128)

        for i in range(len(result_idx)):
            result[result_idx[i]] += source[i]

        return TensorSequence(self.alphabet, trunc, result, np.arange(max_len))

    def shuffle_prod(
        self,
        ts1: TensorSequence,
        ts2: TensorSequence,
    ):
        index_left, index_right, index_result, count = self.shuffle_table.T

        max_len = self.alphabet.number_of_elements(self.trunc)
        linear_left = np.zeros((max_len,), dtype=complex128)
        linear_left[ts1.indices] = ts1.array[:, 0, 0]

        linear_right = np.zeros((max_len,), dtype=complex128)
        linear_right[ts2.indices] = ts2.array[:, 0, 0]

        source = count * linear_left[index_left] * linear_right[index_right]
        linear_result = np.zeros(shape=index_result[-1] + 1, dtype=complex128)
        for i in range(len(index_result)):
            linear_result[index_result[i]] = linear_result[index_result[i]] + source[i]
        return TensorSequence(self.alphabet, self.trunc, linear_result, np.arange(max_len))

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