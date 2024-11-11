from numba import jit
import numpy as np
from numpy.typing import NDArray
from numpy import float64, complex128
from typing import Union

from signatures.tensor_sequence import TensorSequence
from signatures.alphabet import Alphabet
from signatures.numba_utility import factorial

def stationary_signature_from_path(
    path: NDArray[float64],
    trunc: int,
    t_grid: NDArray[float64],
    lam: float
) -> TensorSequence:
    """
    Computes the stationary signature with coefficient lam corresponding to a given d-dimensional path on the
    time grid t_grid up to the order trunc.

    :param path: Path as NDArray of shape (len(t_grid), d).
    :param trunc: Truncation order, i.e. maximal order of coefficients to be calculated.
    :param t_grid: Time grid as NDArray. An increasing time grid from T0 < 0 to T > 0. The signature is calculated
        only for the positive values of grid.
    :param lam: signature mean reversion coefficient.

    :return: TensorSequence objet corresponding to a trajectory of signature of the path on t_grid corresponding
        to the positive values t_grid[t_grid >= 0]
    """
    dim = path.shape[1]
    alphabet = Alphabet(dim)

    dt = np.diff(t_grid)
    path_inc = (path[1:] - path[:-1]) / np.reshape(dt, (-1, 1))

    n_indices = alphabet.number_of_elements(trunc)
    array_steps = np.zeros((n_indices, t_grid.size - 1))
    array_steps[0] = 1

    # Calculates step arrays: each array array_steps[:, k] corresponds to the signature bb{X}_{t_k, t_{k + 1}}.
    # n-th level of this signature is the tensor product of the path increments path[k + 1] - path[k] multiplied
    # by a signature of the linear path given by the function __h.
    for n in range(1, trunc + 1):
        tp_n = path_inc
        for i in range(n - 1):
            inc_shape = (path_inc.shape[0],) + (1,) * (len(tp_n.shape) - 1) + (path_inc.shape[1],)
            tp_n = tp_n.reshape(tp_n.shape + (1,)) * path_inc.reshape(inc_shape)

        idx_start = alphabet.number_of_elements(n - 1)
        array_steps[idx_start:idx_start + dim ** n] = tp_n.reshape((path_inc.shape[0], -1)).T * __h(dt=dt, n=n, lam=lam)

    return __sum_steps_stat(array_steps, trunc, alphabet, t_grid, lam)

@jit(nopython=True)
def __h(dt: Union[float, NDArray[float64]], n: int, lam: float):
    """
    Calculates a stationary lambda-signature of order n of the path X_t = t on [0, dt].
    """
    return ((1 - np.exp(-lam * dt)) / lam)**n / factorial(n)

@jit(nopython=True)
def __sum_steps_stat(
    array_steps: NDArray[float64],
    trunc: int,
    alphabet: Alphabet,
    t_grid: NDArray[float64],
    lam: float
) -> TensorSequence:
    """
    Transforms the signature bb{X}_{t_k, t_{k + 1}} of linear paths into the path signature bb{X}_{t_{k + 1}}
    using the Chen's identity.
    """
    dt = t_grid[1:] - t_grid[:-1]
    n_indices = alphabet.number_of_elements(trunc)
    indices = np.arange(array_steps.shape[0])
    ts_steps = [TensorSequence(alphabet, trunc,
                               np.ascontiguousarray(array_steps[:, i]).reshape((n_indices, 1, 1)), indices)
                for i in range(array_steps.shape[1])]
    for i in range(len(ts_steps) - 1):
        # Stationary Chan's identity
        ts_steps[i + 1].update(discount_ts(ts=ts_steps[i], dt=dt[i + 1], lam=lam).tensor_prod(ts_steps[i + 1]))

    n_pos_points = np.sum(t_grid >= 0)
    array_res = np.zeros((n_indices, n_pos_points), dtype=complex128)

    for i, ts in enumerate(ts_steps[-n_pos_points:]):
        # Creating an array corresponding to the positive time points.
        array_res[ts.indices, i] = ts.array[:, 0, 0]

    return TensorSequence(alphabet, trunc, array_res, indices)

### Operators ###

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
def G_inv(ts: TensorSequence) -> TensorSequence:
    """
    An operator dividing the coefficients of tensor sequence by the lengths of the corresponding words (pseudo-inverse of G).

    :param ts: tensor sequence to transform.

    :return: G^{-1}(ts) as a new instance of TensorSequence.
    """

    return TensorSequence(ts.alphabet, ts.trunc,
                          ts.array * np.where(ts.indices != 0, 1 /ts.alphabet.index_to_length(ts.indices), 0).reshape(ts.array.shape),
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

@jit(nopython=True)
def G_resolvent(ts: TensorSequence, lam: float) -> TensorSequence:
    """
    Calculates the operator (Id + lam * G)^{-1}

    :param ts: tensor sequence to transform.
    :param lam: mean-reversion rate.

    :return: A result of application of the operator to ts.
    """
    return TensorSequence(ts.alphabet, ts.trunc,
                          ts.array * np.reshape(1 / (1 + lam * ts.alphabet.index_to_length(ts.indices)), ts.array.shape),
                          ts.indices)

