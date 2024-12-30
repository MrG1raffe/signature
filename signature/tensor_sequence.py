from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numpy import int64, complex128
from typing import Union, Tuple
from numba.experimental import jitclass
import numba as nb

from .alphabet import Alphabet
from .shuffle import shuffle_product
from .numba_utility import factorial

spec = [
    ('__alphabet', Alphabet.class_type.instance_type),
    ('__trunc', nb.int64),
    ('__array', nb.complex128[:, :, :]),
]


@jitclass(spec)
class TensorSequence:
    def __init__(
        self,
        alphabet: Alphabet,
        trunc: int,
        array: NDArray[complex128],
    ):
        """
        Initializes a TensorSequence object, which represents a collection of coefficients indexed
        by words from a specified alphabet, truncated at a certain length `trunc`.

        :param alphabet: An Alphabet object that defines the dimension and convertion functions.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.
        :param array: A one-dimensional numpy array of tensor coefficients corresponding to the indices.
        """
        self.__alphabet = alphabet
        self.__trunc = trunc

        if array.ndim == 1:
            reshaped_array = np.reshape(a=array, newshape=(-1, 1, 1)).astype(complex128)
        elif array.ndim == 2:
            reshaped_array = np.reshape(a=array, newshape=array.shape + (1,)).astype(complex128)
        elif array.ndim == 3:
            reshaped_array = array.astype(complex128)
        else:
            raise ValueError("Array can be at most 3-dimensional.")

        n_elements = alphabet.number_of_elements(trunc)

        self.__array = np.zeros((n_elements,) + reshaped_array.shape[1:], dtype=complex128)
        self.__array[:min(n_elements, reshaped_array.shape[0])] = reshaped_array[:min(n_elements, reshaped_array.shape[0])]

    def update_trunc(self, new_trunc):
        if new_trunc != self.trunc:
            self.__trunc = new_trunc
            n_elements = self.alphabet.number_of_elements(new_trunc)

            new_array = np.zeros((n_elements,) + self.array.shape[1:], dtype=complex128)
            new_array[:min(n_elements, self.array.shape[0])] = self.__array[:min(n_elements, self.array.shape[0])]
            self.__array = new_array

    def __getitem__(self, key: Union[str, int]) -> Union[complex128, NDArray[complex128], TensorSequence]:
        """
        Retrieves the coefficient associated with a given word if key is a string.
        If key is int, returns a tensor sequence corresponding in the time dimension (axis 1 of self.array).

        :param key: A string representing a word formed from the alphabet or integer representing the time index.
        :return: The tensor coefficient corresponding to the given word if type(key) == str
            or an instance of TensorSequence with the time index key if type(key) == int.
        """
        if isinstance(key, str):
            word_index = self.__alphabet.word_to_index(key)
            return self.__array[word_index]
        else:
            indexed_arr = np.ascontiguousarray(self.array[:, key, :])
            return TensorSequence(self.alphabet, self.trunc, indexed_arr)

    def __rmul__(self, c: Union[float, complex, NDArray[complex128]]) -> TensorSequence:
        """
        Performs right multiplication of the TensorSequence by a scalar or a numpy array.

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        new_array = self.__array * c
        return TensorSequence(self.__alphabet, self.__trunc, new_array)

    def __mul__(self, c: Union[float, complex, NDArray[complex128]]) -> TensorSequence:
        """
        Performs left multiplication of the TensorSequence by a scalar or a numpy array
        of shape (self.__array.shape[0], 1).

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        return self.__rmul__(c)

    def __truediv__(self, c: Union[float, complex, NDArray[complex128]]):
        """
        Divides the TensorSequence by a scalar or numpy array of shape (self.__array.shape[0], 1).

        :param c: A scalar or numpy array by which to divide the TensorSequence.
        :return: A new TensorSequence that is the result of the division.
        """
        return self * (1 / c)

    def __add__(self, ts: TensorSequence) -> TensorSequence:
        """
        Adds another TensorSequence to the current one.

        :param ts: The TensorSequence to add.
        :return: A new TensorSequence that is the result of the addition.
        """
        if not bool(self):
            return ts
        if not bool(ts):
            return self
        if self.__array.shape[1] != ts.array.shape[1]:
            raise ValueError("Time grids of sequences should be the same.")

        trunc = max(self.trunc, ts.trunc)
        new_len = self.alphabet.number_of_elements(trunc)

        array_1 = np.zeros((new_len,) + self.array.shape[1:], dtype=complex128)
        array_1[:min(new_len, self.array.shape[0])] = self.__array[:min(new_len, self.array.shape[0])]

        array_2 = np.zeros((new_len,) + ts.array.shape[1:], dtype=complex128)
        array_2[:min(new_len, ts.array.shape[0])] = ts.__array[:min(new_len, ts.array.shape[0])]

        new_array = array_1 + array_2
        return TensorSequence(self.__alphabet, trunc, new_array)

    def __sub__(self, ts: TensorSequence) -> TensorSequence:
        """
        Subtracts another TensorSequence from the current one.

        :param ts: The TensorSequence to subtract.
        :return: A new TensorSequence that is the result of the subtraction.
        """
        return self + ts * (-1.0)

    def __matmul__(self, ts: TensorSequence) -> Union[float, NDArray[complex128]]:
        """
        Computes the inner product (dot product) of the current TensorSequence with another.

        :param ts: The TensorSequence with which to compute the inner product.
        :return: The inner product as a scalar.
        """
        n_min = min(len(self), len(ts))
        return (self.array[:n_min] * ts.array[:n_min]).sum(axis=0)

    def __len__(self) -> int:
        """
        Returns the number of non-zero coefficients in the TensorSequence.

        :return: The number of non-zero coefficients.
        """
        return self.array.shape[0]

    def __bool__(self) -> bool:
        """
        Returns whether the TensorSequence is non-empty (has non-zero coefficients).

        :return: True if the TensorSequence has non-zero coefficients, False otherwise.
        """
        return not np.allclose(self.array, 0)

    def zero_like(self) -> TensorSequence:
        """
        Creates an instance of TensorSequence with no indices and the same sizes of other axis.

        :return: A new zero TensorSequence.
        """
        self_shape = (0,) + self.array.shape[1:]
        return TensorSequence(self.__alphabet, self.__trunc, np.zeros(self_shape))

    def unit_like(self) -> TensorSequence:
        """
        Creates an instance of TensorSequence with index 1 corresponding to the word Ø.

        :return: A unit element as TensorSequence.
        """
        self_shape = (1,) + self.array.shape[1:]
        return TensorSequence(self.__alphabet, self.__trunc, np.ones(self_shape))

    @staticmethod
    def zero(alphabet, trunc) -> TensorSequence:
        """
        Creates an instance of TensorSequence with no indices and the same sizes of other axis.

        :param alphabet: An Alphabet object that defines the dimension and convertion functions.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.

        :return: A new zero TensorSequence.
        """
        self_shape = (0, 1, 1)
        return TensorSequence(alphabet, trunc, np.zeros(self_shape))

    @staticmethod
    def unit(alphabet, trunc) -> TensorSequence:
        """
        Creates an instance of TensorSequence with index 1 corresponding to the word Ø.

        :param alphabet: An Alphabet object that defines the dimension and convertion functions.
        :param trunc: The truncation level, i.e., the maximum length of words considered in this TensorSequence.

        :return: A unit element as TensorSequence.
        """
        self_shape = (1, 1, 1)
        return TensorSequence(alphabet, trunc, np.ones(self_shape))

    @property
    def alphabet(self) -> Alphabet:
        """
        Returns the array of tensor coefficients.

        :return: A numpy array of tensor coefficients.
        """
        return self.__alphabet

    @property
    def array(self) -> NDArray[complex128]:
        """
        Returns the array of tensor coefficients.

        :return: A numpy array of tensor coefficients.
        """
        return self.__array

    @property
    def trunc(self) -> int:
        """
        Returns the truncation level of the TensorSequence.

        :return: The truncation level as an integer.
        """
        return self.__trunc

    def update(self, ts: TensorSequence) -> None:
        """
        Updates the attributes of the instance copying the attributes of ts.

        :param ts: tensor sequence which attributes will be used as new attributes of self.
        """
        self.__alphabet = ts.alphabet
        self.__trunc = ts.trunc
        self.__array = ts.array

    def proj(self, word: str) -> TensorSequence:
        """
        Calculates the projection of TensorSequence with respect to the given word.

        :param word: The to calculate the projection.
        :return: A new TensorSequence representing the projection.
        """
        indices = np.arange(len(self))

        dim = self.__alphabet.dim
        word_index = self.__alphabet.word_to_index(word)
        indices_mask = (((indices - word_index) % dim**len(word)) == 0) & (indices >= word_index)
        indices_to_keep = indices[indices_mask]
        length_arr = self.__alphabet.index_to_length(indices_to_keep)
        new_indices = (indices_to_keep - dim**length_arr + 1) // dim**(len(word)) + dim**(length_arr - len(word)) - 1

        array = np.zeros_like(self.array, dtype=complex128)
        array[new_indices] = self.__array[indices_mask]
        return TensorSequence(self.__alphabet, self.__trunc, array)

    def tensor_prod_word(self, word: str, coefficient: float = 1, trunc: int = -1) -> TensorSequence:
        """
        Performs the tensor product of the current TensorSequence with a given word and
        multiply the result by `coefficient`.

        :param word: The word to tensor multiply with.
        :param coefficient: The coefficient to multiply the resulting tensor product.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the tensor product.
        """
        indices = np.arange(len(self))

        dim = self.__alphabet.dim
        word_dim_base = self.__alphabet.word_to_base_dim_number(word)
        length_indices = self.__alphabet.index_to_length(indices)
        new_indices = (dim**length_indices * dim**len(word) - 1) + \
            dim**len(word) * (indices - dim**length_indices + 1) + word_dim_base

        array = np.zeros_like(self.array, dtype=complex128)
        array[new_indices[new_indices < len(self)]] = self.array[new_indices < len(self)] * coefficient
        if trunc == -1:
            trunc = self.trunc
        return TensorSequence(self.__alphabet, trunc, array)

    def tensor_prod_index(self, index: int, coefficient: Union[float, NDArray[complex128]] = 1, trunc: int = -1) -> TensorSequence:
        """
        Performs the tensor product of the current TensorSequence with a given index and
        multiply the result by `coefficient`.

        :param index: The index of word to tensor multiply with.
        :param coefficient: The coefficient to multiply the resulting tensor product.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the tensor product.
        """
        indices = np.arange(len(self))

        dim = self.__alphabet.dim
        other_len = self.__alphabet.index_to_length(np.array([index], dtype=int64))
        other_dim_base = index - dim**other_len + 1
        length_indices = self.__alphabet.index_to_length(indices)
        new_indices = (dim**length_indices * dim**other_len - 1) + \
            dim**other_len * (indices - dim**length_indices + 1) + other_dim_base

        array = np.zeros_like(self.array, dtype=complex128)
        array[new_indices[new_indices < len(self)]] = self.array[new_indices < len(self)] * coefficient

        if trunc == -1:
            trunc = self.trunc
        return TensorSequence(self.__alphabet, trunc, array)

    def tensor_prod(self, ts: TensorSequence, trunc: int = -1) -> TensorSequence:
        """
        Performs the tensor product of the current TensorSequence with another TensorSequence.

        :param ts: The other TensorSequence to tensor multiply with.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, max(self.trunc, ts.trunc).
        :return: A new TensorSequence representing the tensor product.
        """
        if trunc == -1:
            trunc = max(self.trunc, ts.trunc)

        other_array = ts.array
        res = self.zero_like()
        for i in range(len(ts)):
            coefficient = other_array[i]
            if not np.allclose(coefficient, 0):
                res.update(res + self.tensor_prod_index(i, coefficient, trunc))
        return res

    def shuffle_prod(self, ts: TensorSequence, trunc: int = -1) -> TensorSequence:
        """
        Performs the shuffle product of the current TensorSequence with another TensorSequence.

        :param ts: The other TensorSequence to shuffle multiply with.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, max(self.trunc, ts.trunc).
        :return: A new TensorSequence representing the shuffle product.
        """
        if trunc == -1:
            trunc = max(self.trunc, ts.trunc)

        res_array = np.zeros((self.alphabet.number_of_elements(trunc),) + self.array.shape[1:], dtype=complex128)

        for i_self in range(len(self)):
            for i_other in range(len(ts)):
                if not np.allclose(self.array[i_self] * ts.array[i_other], 0):
                    word_self = self.__alphabet.index_to_int(i_self)
                    word_other = self.__alphabet.index_to_int(i_other)
                    if len(str(word_self)) * (word_self > 0) + len(str(word_other)) * (word_other > 0) <= trunc:
                        shuffle_words, counts = shuffle_product(word_self, word_other)
                        shuffle_indices = np.array([self.__alphabet.int_to_index(word) for word in shuffle_words], dtype=int64)
                        coefficients = self.array[i_self] * ts.array[i_other]
                        shuffle_array = (np.reshape(counts, (-1, 1, 1)) *
                                         np.reshape(coefficients, (1,) + coefficients.shape)).astype(complex128)
                        res_array[shuffle_indices] += shuffle_array

        return TensorSequence(self.__alphabet, trunc, res_array)

    def tensor_pow(self, p: int, trunc: int = -1) -> TensorSequence:
        """
        Raises the TensorSequence to a tensor power p.

        :param p: The power to which the TensorSequence is raised.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the tensor power.
        """
        if trunc == -1:
            trunc = self.trunc

        if p == 0:
            return self.unit(self.alphabet, trunc)

        res = self * 1
        # TODO: think about more efficient implementation (with log_2(p) operations)
        for _ in range(p - 1):
            res.update(res.tensor_prod(self, trunc))
        return res

    def shuffle_pow(self, p: int, trunc: int = -1) -> TensorSequence:
        """
        Raises the TensorSequence to a shuffle power p.

        :param p: The power to which the TensorSequence is raised.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the shuffle power.
        """
        if trunc == -1:
            trunc = self.trunc

        if p == 0:
            return self.unit(self.alphabet, trunc)

        res = self.unit_like()
        res.update_trunc(trunc)
        # TODO: think about more efficient implementation (with log_2(p) operations)
        for _ in range(p):
            res.update(res.shuffle_prod(self, trunc))
        return res

    def tensor_exp(self, N_trunc: int, trunc: int = -1) -> TensorSequence:
        """
        Computes the tensor exponential of the TensorSequence up to a specified truncation level.

        :param N_trunc: The truncation level for the exponential.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the tensor exponential.
        """
        if trunc == -1:
            trunc = self.trunc

        res = self.unit_like()
        res.update_trunc(trunc)

        for n in range(1, N_trunc):
            res.update(res + self.tensor_pow(n, trunc) / factorial(n))
        return res

    def shuffle_exp(self, N_trunc: int, trunc: int = -1) -> TensorSequence:
        """
        Computes the shuffle exponential of the TensorSequence up to a specified truncation level.

        :param N_trunc: The truncation level for the exponential.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the shuffle exponential.
        """
        if trunc == -1:
            trunc = self.trunc

        res = self.unit_like()
        res.update_trunc(trunc)

        for n in range(1, N_trunc):
            res.update(res + self.shuffle_pow(n, trunc) / factorial(n))
        return res

    def resolvent(self, N_trunc, trunc: int = -1):
        """
        Computes the resolvent of the TensorSequence up to a specified truncation level.
        The resolvent is defined as the series of the TensorSequence's tensor powers.

        :param N_trunc: The truncation level for the resolvent.
        :param trunc: truncation level of the resulting TensorSequence instance. By default, self.trunc.
        :return: A new TensorSequence representing the resolvent.
        :raises ValueError: If the coefficient corresponding to the empty word exceeds or equals 1.
        """
        if np.max(np.abs(self[""])) >= 1:
            raise ValueError("Resolvent cannot be calculated. The tensor sequence l should have |l^∅| < 1.")

        if trunc == -1:
            trunc = self.trunc

        res = self.unit_like()
        res.update_trunc(trunc)

        for n in range(1, N_trunc):
            res.update(res + self.tensor_pow(n, trunc))
        return res
