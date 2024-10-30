import numpy as np
from numpy.typing import NDArray
from numpy import float64, complex128
from numba import jit, prange

from signatures.tensor_sequence import TensorSequence

@jit(nopython=True)
def func_psi(psi: TensorSequence):
    return psi.proj("2").shuffle_pow(2) / 2 + psi.proj("22") / 2 + psi.proj("1")

@jit(nopython=True)
def func_xi(xi: TensorSequence):
    return xi.proj("22") / 2 + xi.proj("1")

@jit(nopython=True)
def psi_riccati(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    dt = np.diff(t_grid)

    psi = u * 1
    for i in range(len(dt)):
        psi.update(psi + func_psi(psi) * dt[i])

    return psi

@jit(nopython=True)
def psi_riccati_rk4(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    dt = np.diff(t_grid)

    psi = u * 1

    k1 = func_psi(psi) * 0
    k2 = func_psi(psi) * 0
    k3 = func_psi(psi) * 0
    k4 = func_psi(psi) * 0

    for i in range(len(dt)):
        k1.update(func_psi(psi))
        k2.update(func_psi(psi + k1 * (dt[i] / 2)))
        k3.update(func_psi(psi + k2 * (dt[i] / 2)))
        k4.update(func_psi(psi + k3 * dt[i]))
        psi.update(psi + (k1 + k2 * 2 + k3 * 2 + k4) * (dt[i] / 6))

    return psi

@jit(nopython=True)
def xi_riccati(
    t_grid: NDArray[float64],
    xi_0: TensorSequence,
) -> TensorSequence:
    dt = np.diff(t_grid)

    xi = xi_0 * 1
    for i in range(len(dt)):
        xi.update(xi + func_xi(xi) * dt[i])

    return xi