import numpy as np
from numpy.typing import NDArray
from numpy import float_
from typing import Callable
from scipy.interpolate import interp1d
from math import ceil

from models.model_params import HistoricalParams
from models.hjm_model.contracts import ForwardContract
from utility.utility import to_numpy


def forward_vol_from_hist_params(
    hist_params: HistoricalParams,
    contract: ForwardContract,
    T_grid: NDArray[float_],
    g_arr: NDArray[float_],
    h_func: Callable = lambda x: np.ones_like(x),
    is_kemna_vorst: bool = True,
    is_interpolate: bool = True
) -> Callable[[NDArray[float_]], NDArray[float_]]:
    """
    Calculates an HJM deterministic volatility component of the given forward contract.

    :param hist_params:
    :param contract: a forward contract the volatility of which to be calculated.
    :param T_grid: delivery date T integration grid.
    :param g_arr: the values of the function g on the integration grid T_grid.
    :param h_func: function h as a callable object.
    :param is_kemna_vorst: whether to use the Kemna-Vorst approximation or create a three-dimensional function sigmas
        with the volatilities of the instantaneous forwards with delivery T for T in T_grid.
    :param is_interpolate: whether to use the linear interpolation of sigmas instead of the function itself.
        May be useful for fine T-grids to accelerate the evaluation of sigma for pricing.
    :return: a callable object `sigmas` which is the deterministic volatility of the forward.
    """
    T_s, T_e = contract.time_to_delivery_start, contract.time_to_delivery_end
    inf_tau_idx = np.isinf(hist_params.taus)
    taus = np.ones_like(hist_params.taus)
    taus[~inf_tau_idx] = hist_params.taus[~inf_tau_idx]
    if np.isclose(T_s, T_e):
        def sigmas(t):
            t_grid = to_numpy(t)
            return (g_arr * np.ones((t_grid.size, hist_params.sigmas.size)) * hist_params.sigmas *
                    (~inf_tau_idx * np.exp(-(T_s - np.reshape(t_grid, (-1, 1))) / taus) +
                    inf_tau_idx * 1)) * np.reshape((t_grid <= T_s) * h_func(t_grid), (-1, 1))
    else:
        if not is_kemna_vorst:
            def sigmas(t):
                t_grid = to_numpy(t)
                return np.einsum('j,i,kij -> kij',
                                 g_arr,
                                 hist_params.sigmas,
                                 np.exp(-np.einsum('kj,i->kij',
                                                   T_grid[None, :] - t_grid[:, None],
                                                   1 / hist_params.taus)
                                        )
                                 ) * np.reshape((t_grid <= T_s) * h_func(t_grid), (t_grid.size,  1, 1))
        else:
            def sigmas(t):
                t_grid = to_numpy(t)
                res = (np.einsum("j,i,kij -> ki",
                                  g_arr,
                                  hist_params.sigmas * hist_params.taus,
                                  (np.exp(-np.einsum('kj,i->kij',
                                                     T_grid[None, :-1] - t_grid[:, None],
                                                     1 / hist_params.taus)
                                          ) -
                                   np.exp(-np.einsum('kj,i->kij',
                                                     T_grid[None, 1:] - t_grid[:, None],
                                                     1 / hist_params.taus)
                                          )
                                   )) * np.reshape((t_grid <= T_s) * h_func(t_grid), (t_grid.size,  1))
                        ) / (T_e - T_s)
                return res
            if is_interpolate:
                dt = 0.001
                t_grid = np.linspace(0, T_s, ceil(T_s / dt))
                sigmas = interp1d(x=t_grid, y=sigmas(t_grid), axis=0, bounds_error=False, fill_value=0)
    return sigmas

