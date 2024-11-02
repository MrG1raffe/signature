from numba import jit
import numpy as np

from signatures.tensor_sequence import TensorSequence

@jit(nopython=True)
def G(ts: TensorSequence) -> TensorSequence:
    """
    An operator multiplying the coefficients of tensor sequence by the lengths of the corresponding words.

    :param ts: tensor sequence to transform.

    :return: G(ts) as a new instance of TensorSequence.
    """

    return TensorSequence(ts.alphabet, ts.trunc,
                          ts.array * np.reshape(ts.alphabet.index_to_length(ts.indices), ts.array.shape),
                          ts.indices)

@jit(nopython=True)
def discount_ts(ts: TensorSequence, dt: float, lam: float) -> TensorSequence:
    """
    A discounting operator with discounting rate lambda and discounting period dt.
    Multiplies the coefficient l^v by exp(-lam * |v| * dt).

    :param ts: tensor sequence to discount.
    :param dt: length of the discounting period.
    :param lam: discounting rate.

    :return: Discounted tensor sequence.
    """
    return TensorSequence(ts.alphabet, ts.trunc,
                          ts.array * np.reshape(np.exp(-ts.alphabet.index_to_length(ts.indices) * lam * dt), ts.array.shape),
                          ts.indices)

@jit(nopython=True)
def semi_integrated_scheme(ts: TensorSequence, dt: float, lam: float) -> TensorSequence:
    """
    A numerical scheme for integration of the equation
    psi' = -lam *G(psi) + F
    Given by an operator (lam * G)^{-1}(Id - D_h^lam).

    :param ts: tensor sequence to transform.
    :param dt: time step.
    :param lam: mean-reversion rate.

    :return: A result of application of the operator to ts.
    """
    return TensorSequence(ts.alphabet, ts.trunc,
                          ts.array * np.reshape(
                              np.where(ts.indices != 0, (1 - np.exp(-ts.alphabet.index_to_length(ts.indices) * lam * dt)) /
                              (lam * ts.alphabet.index_to_length(ts.indices)), dt),
                              ts.array.shape),
                          ts.indices)

