import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Union
from numba import jit

from signature.old_versions.tensor_sequence import TensorSequence
from signature.old_versions.alphabet import Alphabet
from signature.old_versions.stationary_signature import G_inv
from signature.old_versions.ode_integration import ode_stat_pece


def expected_signature(t: Union[float, NDArray[float64]], trunc: int) -> TensorSequence:
    """
    Calculates the expected signature of X_t = (t, W_t).

    :param t: time grid to calculate the expected signature.
    :param trunc: truncation order.

    :return: expected signature evaluated at t as a TensorSequence instance.
    """
    alphabet = Alphabet(2)
    # w = (1 + 0.5 * 22) * t
    w = get_1_22(alphabet, trunc) * np.reshape(t, (1, -1, 1))
    return w.tensor_exp(trunc)


def expected_stationary_signature(trunc: int, lam: float, t: float = None, n_points: int = 100) -> TensorSequence:
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
    alphabet = Alphabet(2)
    w = get_1_22(alphabet, trunc)

    if t is None:
        res = TensorSequence.unit(alphabet, trunc)
        v = TensorSequence.unit(alphabet, trunc)
        for _ in range(trunc):
            v.update(G_inv(v.tensor_prod(w)) / lam)
            res.update(res + v)
        return res
    else:
        t_grid = np.linspace(0, t, n_points)
        return ode_stat_pece(func=__expected_sig_ode_func, t_grid=t_grid, u=TensorSequence.unit(alphabet, trunc), lam=lam)


@jit(nopython=True)
def get_1_22(alphabet: Alphabet, trunc: int):
    array = np.zeros(alphabet.number_of_elements(trunc))
    array[1] = 1
    array[6] = 0.5
    return TensorSequence(alphabet, trunc, array)


@jit(nopython=True)
def __expected_sig_ode_func(l: TensorSequence):
    # w = 1 + 0.5 * 22
    w = get_1_22(Alphabet(2), l.trunc)
    return l.tensor_prod(w)
