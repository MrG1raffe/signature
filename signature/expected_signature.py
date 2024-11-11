import numpy as np
from numpy.typing import NDArray
from numpy import float64, complex128
from typing import Union
from numba import jit

from signature.tensor_sequence import TensorSequence
from signature.alphabet import Alphabet
from signature.stationary_signature import G_inv
from signature.ode_integration import ode_stat_pece

def expected_signature(t: Union[float, NDArray[float64]], trunc: int) -> TensorSequence:
    """
    Calculates the expected signature of X_t = (t, W_t).

    :param t: time grid to calculate the expected signature.
    :param trunc: truncation order.

    :return: expected signature evaluated at t as a TensorSequence instance.
    """
    array = np.reshape([1, 0.5], (2, 1, 1)) * np.reshape(t, (1, -1, 1))
    # w = (1 + 0.5 * 22) * t
    w = TensorSequence(Alphabet(2), trunc, array.astype(complex128), np.array([1, 6]))
    return w.tensor_exp(trunc)

def expected_stationary_signature(trunc: int, lam: int, t: float = None, n_points: int = 100) -> TensorSequence:
    """
    Computes expected stationary lambda-signature of X_t = (t, W_t). If t is not specified,
    computes stationary expected signature E^lam = E[SigX^lam]. Otherwise, computes E_t^lam = E[SigX_{0, t}^lam].

    :param trunc: truncation order of the result.
    :param lam: stationary signature parameter.
    :param t: time index of the expected signature. By default, t = inf, which corresponds to stationary signature.
    :param n_points: number of points to be used to solve an SDE on E_t^lam.

    :return: expected signature as a TensorSequence instance.
    """
    # w = 1 + 0.5 * 22
    w = TensorSequence(Alphabet(2), trunc, np.array([1, 0.5], dtype=complex128), np.array([1, 6]))

    if t is None:
        res = TensorSequence(Alphabet(2), trunc, np.ones((1, 1, 1), dtype=complex128), np.array([0]))
        v = TensorSequence(Alphabet(2), trunc, np.ones((1, 1, 1), dtype=complex128), np.array([0]))
        for _ in range(trunc):
            v.update(G_inv(v.tensor_prod(w)) / lam)
            res.update(res + v)
        return res
    else:
        t_grid = np.linspace(0, t, n_points)
        return ode_stat_pece(func=__expected_sig_ode_func, t_grid=t_grid, u=TensorSequence(w.alphabet, trunc, np.ones((1, 1, 1)), np.zeros(1)), lam=lam)

@jit(nopython=True)
def __expected_sig_ode_func(l: TensorSequence):
    # w = 1 + 0.5 * 22
    w = TensorSequence(Alphabet(2), 2, np.array([1, 0.5], dtype=complex128), np.array([1, 6]))
    return l.tensor_prod(w)