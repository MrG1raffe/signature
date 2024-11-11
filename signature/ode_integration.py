import numpy as np
from numpy.typing import NDArray
from numpy import float64
from numba import jit
from typing import Callable

from signatures.tensor_sequence import TensorSequence
from signatures.stationary_signature import G, semi_integrated_scheme, discount_ts

@jit(nopython=True)
def ode_stat_pece(
    func: Callable,
    t_grid: NDArray[float64],
    u: TensorSequence,
    lam: float
) -> TensorSequence:
    dt = t_grid[1:] - t_grid[:-1]

    psi = u * 1
    psi_pred = u * 1
    for i in range(len(dt)):
        psi_pred.update(discount_ts(ts=psi, dt=dt[i], lam=lam) + semi_integrated_scheme(ts=func(psi), dt=dt[i], lam=lam))
        psi.update(discount_ts(ts=psi, dt=dt[i], lam=lam) + semi_integrated_scheme(ts=(func(psi_pred) + func(psi)) * 0.5, dt=dt[i], lam=lam))

    return psi