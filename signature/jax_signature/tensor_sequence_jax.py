from __future__ import annotations
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt

from .words import number_of_words_up_to_trunc, index_to_word, word_len, word_to_index, index_to_word_len


@jdc.pytree_dataclass
class TensorSequenceJAX:
    array: jax.Array
    trunc: int
    dim: int

    # def __post_init__(self):
    #     if self.array.shape[0] != number_of_words_up_to_trunc(trunc=self.trunc, dim=self.dim):
    #         raise ValueError("The array's shape should be consistent with given `trunc` and `dim` values.")

    def __repr__(self):
        return str(self.array)

    def __str__(self):
        res = ""
        is_first = True
        for i in range(len(self)):
            coefficient = self.array[i].squeeze()
            if not np.allclose(coefficient, 0):
                if not is_first:
                    res += " + "
                if np.all(np.isclose(coefficient.imag, 0)):
                    coefficient = coefficient.real
                res += str(coefficient) + "*" + str(index_to_word(i, self.dim))
                is_first = False
        return res

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

    def __getitem__(self, key):
        return self.array[key]

    def subsequence(self, key: Tuple):
        return TensorSequenceJAX(array=self.array[(slice(None), *key)], trunc=self.trunc, dim=self.dim)

    def __rmul__(self, c: Union[float, complex, jax.Array]) -> TensorSequenceJAX:
        """
        Performs right multiplication of the TensorSequence by a scalar or a numpy array.

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        return TensorSequenceJAX(array=self.array * c, trunc=self.trunc, dim=self.dim)

    def __mul__(self, c: Union[float, complex, jax.Array]) -> TensorSequenceJAX:
        """
        Performs left multiplication of the TensorSequence by a scalar or a numpy array
        of shape (self.__array.shape[0], 1).

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        return self.__rmul__(c)

    def __truediv__(self, c: Union[float, complex, jax.Array]):
        """
        Divides the TensorSequence by a scalar or numpy array of shape (self.__array.shape[0], 1).

        :param c: A scalar or numpy array by which to divide the TensorSequence.
        :return: A new TensorSequence that is the result of the division.
        """
        return self * (1 / c)

    def __add__(self, ts: TensorSequenceJAX) -> TensorSequenceJAX:
        """
        Adds another TensorSequence to the current one.

        :param ts: The TensorSequence to add.
        :return: A new TensorSequence that is the result of the addition.
        """
        return TensorSequenceJAX(array=self.array + ts.array, trunc=max(self.trunc, ts.trunc), dim=self.dim)

    def __sub__(self, ts: TensorSequenceJAX) -> TensorSequenceJAX:
        """
        Subtracts another TensorSequence from the current one.

        :param ts: The TensorSequence to subtract.
        :return: A new TensorSequence that is the result of the subtraction.
        """
        return TensorSequenceJAX(array=self.array - ts.array, trunc=max(self.trunc, ts.trunc), dim=self.dim)

    def __matmul__(self, ts: TensorSequenceJAX) -> Union[float, jax.Array]:
        """
        Computes the inner product (dot product) of the current TensorSequence with another.

        :param ts: The TensorSequence with which to compute the inner product.
        :return: The inner product as a scalar.
        """
        return jnp.sum(self.array * ts.array, axis=0)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the TensorSequence array.

        :return: Shape of self.array.
        """
        return self.array.shape

    @jax.jit
    def proj(self, word: int) -> TensorSequenceJAX:
        """
        Calculates the projection of TensorSequence with respect to the given word.

        :param word: The to calculate the projection.
        :return: A new TensorSequence representing the projection.
        """
        indices = jnp.arange(len(self))
        word_length = word_len(word)
        word_index = word_to_index(word, dim=self.dim)
        indices_mask = (((indices - word_index) % self.dim**word_length) == 0) & (indices >= word_index)
        # indices_to_keep = indices[indices_mask]
        length_arr = index_to_word_len(indices[indices_mask], dim=self.dim)
        new_indices = (indices[indices_mask] - self.dim**length_arr + 1) // self.dim**word_length + \
            self.dim**(length_arr - word_length) - 1

        array = jnp.zeros_like(self.array)
        array = array.at[new_indices].set(self.array[indices_mask])
        return TensorSequenceJAX(array=array, trunc=self.trunc, dim=self.dim)

    def plot(self, trunc: int = None, ax: plt.axis = None, **kwargs) -> None:
        """
        Plots the coefficients of the tensor sequence.

        :param trunc: truncation order, the coefficients of order <= trunc will be plotted. By default, equals to self.trunc.
        :param ax: plt axis to plot on.
        """
        if trunc is None:
            trunc = self.trunc

        n_coefficients = number_of_words_up_to_trunc(trunc, dim=self.dim)

        indices = np.arange(n_coefficients)
        coefficients = np.zeros(n_coefficients)
        coefficients[:min(n_coefficients, len(self))] = self.array[:min(n_coefficients, len(self))]

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(coefficients, "o", **kwargs)
        ax.grid("on")
        ax.set_xticks(ticks=np.arange(coefficients.size),
                      labels=[str(index_to_word(i, dim=self.dim)) for i in indices],
                      rotation=-90)
        ax.plot()
