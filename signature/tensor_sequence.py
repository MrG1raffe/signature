from __future__ import annotations
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt

from .words import (number_of_words_up_to_trunc, index_to_word, word_len, word_to_index,
                    index_to_word_len, index_to_lam_sum_vect, index_to_lam_prod_vect)


@jdc.pytree_dataclass
class TensorSequence:
    array: jax.Array
    trunc: int
    dim: int

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
        return TensorSequence(array=self.array[(slice(None), *key)], trunc=self.trunc, dim=self.dim)

    def __rmul__(self, c: Union[float, complex, jax.Array]) -> TensorSequence:
        """
        Performs right multiplication of the TensorSequence by a scalar or a numpy array.

        :param c: A scalar or numpy array by which to multiply the TensorSequence.
        :return: A new TensorSequence that is the result of the multiplication.
        """
        return TensorSequence(array=self.array * c, trunc=self.trunc, dim=self.dim)

    def __mul__(self, c: Union[float, complex, jax.Array]) -> TensorSequence:
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

    def __add__(self, ts: TensorSequence) -> TensorSequence:
        """
        Adds another TensorSequence to the current one.

        :param ts: The TensorSequence to add.
        :return: A new TensorSequence that is the result of the addition.
        """
        return TensorSequence(array=self.array + ts.array, trunc=jnp.maximum(self.trunc, ts.trunc), dim=self.dim)

    def __sub__(self, ts: TensorSequence) -> TensorSequence:
        """
        Subtracts another TensorSequence from the current one.

        :param ts: The TensorSequence to subtract.
        :return: A new TensorSequence that is the result of the subtraction.
        """
        return TensorSequence(array=self.array - ts.array, trunc=jnp.maximum(self.trunc, ts.trunc), dim=self.dim)

    def __matmul__(self, ts: TensorSequence) -> Union[float, jax.Array]:
        """
        Computes the inner product (dot product) of the current TensorSequence with another.

        :param ts: The TensorSequence with which to compute the inner product.
        :return: The inner product as a scalar.
        """
        return jnp.einsum("i..., i... -> ...", self.array, ts.array)


    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the TensorSequence array.

        :return: Shape of self.array.
        """
        return self.array.shape

    @jax.jit
    def proj(self, word: int) -> TensorSequence:
        """
        Calculates the projection (right shift) of TensorSequence with respect to the given word.

        Only the words ``u`` that *end* with ``word`` are kept, and the suffix ``word`` is removed
        from each of them, so that the resulting sequence satisfies ``result^u = self^{u word}``.
        See :meth:`left_proj` for the corresponding left shift. The implementation works for any
        ``dim >= 1``; see ``doc/indexation_and_projections.md`` for the derivation.

        :param word: The word (as integer) to calculate the projection.
        :return: A new TensorSequence representing the projection.
        """
        indices = jnp.arange(len(self))
        word_length = word_len(word)
        word_index = word_to_index(word, dim=self.dim)

        # Length |u| of the word carried by each index, and the length |u| - |word| of the word
        # obtained once the suffix is stripped (clamped to stay non-negative; the shorter words
        # are discarded by the mask below).
        length_arr = index_to_word_len(indices, dim=self.dim)
        rest_length = jnp.maximum(length_arr - word_length, 0)

        # Base-dim numbers: position of each word u and of ``word`` within their length blocks.
        b_u = indices - number_of_words_up_to_trunc(length_arr - 1, dim=self.dim)
        b_w = word_index - number_of_words_up_to_trunc(word_length - 1, dim=self.dim)
        word_pow = self.dim ** word_length

        # A mask for indices to keep: u must be at least as long as word and end with word,
        # i.e. the trailing |word| letters of u (the low-order base-dim digits) equal word.
        indices_mask = (length_arr >= word_length) & ((b_u % word_pow) == b_w)

        # Compute new indices: drop the suffix, mapping u to its prefix u' of length |u| - |word|.
        new_indices = number_of_words_up_to_trunc(rest_length - 1, dim=self.dim) + b_u // word_pow
        # Set out-of-bounds index for non-valid ones
        new_indices = jnp.where(indices_mask, new_indices, len(self) + 1)

        array = jnp.zeros_like(self.array)
        array = array.at[new_indices].set(jnp.where(jnp.einsum("i..., i -> i...", jnp.ones_like(self.array), indices_mask),
                                                    jnp.einsum("i..., i -> i...", self.array, indices_mask), 0))

        return TensorSequence(array=array, trunc=self.trunc, dim=self.dim)

    @jax.jit
    def left_proj(self, word: int) -> TensorSequence:
        """
        Calculates the left projection (left shift) of TensorSequence with respect to the given word.

        Whereas :meth:`proj` strips ``word`` from the right (so that ``result^u = self^{u word}``),
        this method strips ``word`` from the left: the resulting sequence satisfies
        ``result^u = self^{word u}``. Only the words ``u`` that start with ``word`` are kept, and
        the prefix ``word`` is removed from each of them.

        :param word: The word (as integer) to calculate the left projection.
        :return: A new TensorSequence representing the left projection.
        """
        indices = jnp.arange(len(self))
        word_length = word_len(word)
        word_index = word_to_index(word, dim=self.dim)

        # Length |u| of the word carried by each index, and the length |u| - |word| of the
        # word obtained once the prefix is stripped (clamped to stay non-negative; the shorter
        # words are discarded by the mask below).
        length_arr = index_to_word_len(indices, dim=self.dim)
        rest_length = jnp.maximum(length_arr - word_length, 0)

        # Base-dim numbers: position of each word and of ``word`` within their length blocks.
        b_u = indices - number_of_words_up_to_trunc(length_arr - 1, dim=self.dim)
        b_w = word_index - number_of_words_up_to_trunc(word_length - 1, dim=self.dim)
        rest_pow = self.dim ** rest_length

        # A mask for indices to keep: u must be at least as long as word and start with word,
        # i.e. the leading |word| letters of u (the high-order base-dim digits) equal word.
        indices_mask = (length_arr >= word_length) & ((b_u // rest_pow) == b_w)

        # Compute new indices: drop the prefix, mapping u to its suffix u' of length |u| - |word|.
        new_indices = number_of_words_up_to_trunc(rest_length - 1, dim=self.dim) + b_u - b_w * rest_pow
        # Set out-of-bounds index for non-valid ones
        new_indices = jnp.where(indices_mask, new_indices, len(self) + 1)

        array = jnp.zeros_like(self.array)
        array = array.at[new_indices].set(jnp.where(jnp.einsum("i..., i -> i...", jnp.ones_like(self.array), indices_mask),
                                                    jnp.einsum("i..., i -> i...", self.array, indices_mask), 0))

        return TensorSequence(array=array, trunc=self.trunc, dim=self.dim)

    @jax.jit
    def seminorm(self, x: TensorSequence) -> jax.Array:
        """
        Computes the seminorm ``|| . ||_X`` of this tensor sequence with respect to another
        tensor sequence ``x`` representing a (group-like) signature X, as defined by equation
        (3.2) of the paper:

            ||p||_X = sum_{n >= 0} | sum_{|v| = n} p^v X^v |,

        i.e. the words are grouped by their length ``n``, the level-``n`` pairing
        ``sum_{|v| = n} p^v X^v`` is formed, and the seminorm is the sum over levels of the
        absolute values of these pairings. This is the quantity controlling the convergence of
        the signature series ``<p, X>`` and is shuffle-compatible (3.5).

        :param x: A TensorSequence X (same ``dim``) against which the seminorm is computed.
            Its trailing (e.g. batch/time) axes may differ from those of ``self`` and are
            broadcast against them, so a single coefficient sequence can be evaluated against a
            whole batch of signatures X at once.
        :return: The seminorm as a real-valued array, broadcast over the trailing axes of the
            coefficient arrays.
        """
        n_min = min(self.array.shape[0], x.array.shape[0])
        a = self.array[:n_min]
        b = x.array[:n_min]
        # Align the trailing (batch/time) axes so the coefficient axis 0 lines up; this lets a
        # single sequence (shape (n,)) broadcast against a batch of signatures (shape (n, size)).
        while a.ndim < b.ndim:
            a = a[..., None]
        while b.ndim < a.ndim:
            b = b[..., None]
        # element-wise products p^v X^v for every word v up to the common length
        products = a * b
        # length |v| of the word carried by each index
        lengths = index_to_word_len(jnp.arange(n_min), self.dim)
        # sum the products level by level: level_pairings[n] = sum_{|v| = n} p^v X^v
        level_pairings = jax.ops.segment_sum(products, lengths, num_segments=n_min)
        # sum of the absolute values of the per-level pairings
        return jnp.sum(jnp.abs(level_pairings), axis=0)

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

    @jax.jit
    def get_lengths_array(self) -> jax.Array:
        return index_to_word_len(index=jnp.arange(len(self)), dim=self.dim)

    @jax.jit
    def get_lambdas_sum_array(self, lam: jax.Array) -> jax.Array:
        return index_to_lam_sum_vect(jnp.arange(len(self)), self.dim, lam)

    @jax.jit
    def get_lambdas_prod_array(self, lam: jax.Array) -> jax.Array:
        return index_to_lam_prod_vect(jnp.arange(len(self)), self.dim, lam)
