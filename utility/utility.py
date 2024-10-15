import numpy as np
import pandas as pd
from typing import Any, List
from numpy.typing import NDArray
from numpy import float_
from scipy.stats import norm


DEFAULT_SEED = 42
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def is_number(x: Any) -> bool:
    """
    Checks whether x is int or float.
    """
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, complex)


def to_numpy(x: Any) -> NDArray[float_]:
    """
    Converts x to numpy array.
    """
    if is_number(x):
        return np.array([x])
    else:
        return np.array(x)


def is_call(flag: str) -> bool:
    """
    Checks whether the flag corresponds to vanilla call option.

    Args:
        flag: flag value.

    Returns:
         True if flag corresponds to call.
    """
    return flag in ["Call", "call", "C", "c"]


def is_put(flag: str) -> bool:
    """
    Checks whether the flag corresponds to vanilla put option.

    Args:
        flag: flag value.

    Returns:
         True if flag corresponds to put.
    """
    return flag in ["Put", "put", "P", "p"]


def get_months_grid(
    T_s: pd.Timestamp,
    T_e: pd.Timestamp
) -> List[str]:
    """
    Creates a month grid (of the form "MONYY") between the dates T_s and T_e
    including the month of T_s and excluding the month of T_e.

    :param T_s: grid start date.
    :param T_e: grid end date.
    :return: a list containing the month grid between T_s and T_e.
    """
    year_min = (T_s.year % 100)
    year_max = (T_e.year % 100)
    months_grid = []

    for year in range(year_min, year_max + 1):
        if year == year_min:
            mon_list = MONTHS[T_s.month - 1:]
        elif year == year_max:
            mon_list = MONTHS[:T_e.month]
        else:
            mon_list = MONTHS
        months_grid += [mon + str(year) for mon in mon_list]

    return months_grid


def from_delta_call_to_strike(
    deltas: NDArray[float_],
    F0: NDArray[float_],
    sigma: NDArray[float_],
    ttm: NDArray[float_]
) -> NDArray[float_]:
    """
    Transforms the delta-strikes in the absolute strikes.

    :param deltas: array of delta-strikes.
    :param F0: underlying price at t = 0.
    :param sigma: array of the implied volatilities corresponding to `K_deltas`.
    :param ttm: array of times to maturity corresponding to `K_deltas`.
    :return: an array of absolute strikes corresponding to `K_deltas`.
    """
    return F0 * np.exp(0.5 * sigma**2 * ttm - sigma * np.sqrt(ttm) * norm.ppf(deltas))


def from_strike_to_pseudo_1_delta_call(
    K: NDArray[float_],
    F0: NDArray[float_],
    ttm: NDArray[float_],
    sigma: NDArray[float_]
) -> NDArray[float_]:
    """
    Transforms the absolute strikes in the one minus pseudo-delta strikes.

    :param K: array of absolute strikes.
    :param F0: underlying price at t = 0.
    :param ttm: array of times to maturity corresponding to `K`.
    :param sigma: array of the implied volatilities corresponding to `K`.
    :return:  an array of one minus pseudo-delta strikes corresponding to `K`.
    """
    d = np.log(F0/K) / (sigma * np.sqrt(ttm))
    return 1 - norm.cdf(d)


def from_strike_and_iv_to_1_delta_call(
    K: NDArray[float_],
    F0: NDArray[float_],
    sigma: NDArray[float_],
    ttm: NDArray[float_]
) -> NDArray[float_]:
    """
    Transforms the absolute strikes in the one minus delta strikes.

    :param K: array of absolute strikes.
    :param F0: underlying price at t = 0.
    :param sigma: array of the implied volatilities corresponding to `K`.
    :param ttm: array of times to maturity corresponding to `K`.
    :return: an array of one minus delta strikes corresponding to `K`.
    """
    d = np.log(F0/K) / (sigma * np.sqrt(ttm)) + 0.5*sigma*np.sqrt(ttm)
    return 1 - norm.cdf(d)


def from_1_delta_call_to_strike(
    deltas: NDArray[float_],
    F0: NDArray[float_],
    sigma: NDArray[float_],
    ttm: NDArray[float_],
) -> NDArray[float_]:
    """
    Transforms the one minus delta-strikes into theabsolute strikes.

    :param deltas: array of one minus delta-strikes.
    :param F0: underlying price at t = 0.
    :param sigma: array of the implied volatilities corresponding to `deltas`.
    :param ttm: array of times to maturity corresponding to `deltas`.
    :return: an array of absolute strikes corresponding to `deltas`.
    """
    return F0 * np.exp(0.5 * sigma**2 * ttm - sigma * np.sqrt(ttm) * norm.ppf(1-deltas))
