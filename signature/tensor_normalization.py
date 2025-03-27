from numba import jit
import numpy as np
from scipy.optimize import minimize

from signature.old_versions.tensor_sequence import TensorSequence
from .utility import get_lengths_array


@jit(nopython=True)
def dilation(ts: TensorSequence, c: float) -> TensorSequence:
    """
    Multiplies n-th level of tensor sequence by c**n.

    :param ts: input tensor sequence.
    :param c: multiplicative coefficient.

    :return: Result of the application of the dilation map to ts.
    """
    return TensorSequence(ts.alphabet, ts.trunc,
                          ts.array * np.reshape(c**get_lengths_array(ts.alphabet, ts.trunc), (len(ts), 1, 1)))


def __psi(x, c: float = 4, alpha: float = 1):
    return x * (x <= c) + (c + c**(1 + alpha) * (c**(-alpha) - x**(-alpha)) / alpha) * (x > c)


def tensor_normalization(ts: TensorSequence) -> TensorSequence:
    if ts.shape[1:] != (1, 1):
        raise NotImplemented("Please provide one element of the tensor algebra of shape (:, 1, 1).")

    # TODO: implement for multiple elements:
    # Create a new array and fill it in a for loop

    norm_squared = np.sum(np.abs(ts.array)**2)

    psi_norm_squared = __psi(norm_squared)
    if np.isclose(psi_norm_squared, psi_norm_squared):
        return ts

    def loss(x):
        norm_squared_scaled = np.sum(np.abs(dilation(ts, x).array)**2)
        return (norm_squared_scaled - psi_norm_squared)**2

    res = minimize(loss, x0=np.ones(1), method="BFGS")
    return dilation(ts, res.x)


