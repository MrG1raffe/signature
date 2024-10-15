import sys
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from numpy import float_
from typing import Union, Callable

from utility.utility import is_put, is_call, to_numpy

PATH_TO_SITE_PACKAGES = 'C:\\Users\\DM6579\\Anaconda3\\envs\\SmileHJM\\Lib\\site-packages'
if PATH_TO_SITE_PACKAGES not in sys.path:
    sys.path.append(PATH_TO_SITE_PACKAGES)
# "Let's be rational" IV computation method by Peter Jäckel
from py_vollib.black.implied_volatility import implied_volatility
iv_lets_be_rational = np.vectorize(implied_volatility)


def black_iv(
    option_price: Union[float, NDArray[float_]],
    T: Union[float, NDArray[float_]],
    K: Union[float, NDArray[float_]],
    F: Union[float, NDArray[float_]],
    r: Union[float, NDArray[float_]],
    flag: str,
) -> Union[float, NDArray[float_]]:
    """
    Calculates implied vol in the Black-76 model given the option price and parameters.

    Args:
        option_price: option prices.
        T: times to maturity.
        K: strikes.
        F: forward prices at t = 0.
        r: the risk-free interest rate.
        flag: 'c' for calls, 'p' for puts.

    Returns:
        Implied volatility or an array of implied volatilities corresponding to the prices.
    """
    T = np.reshape(T, (-1, 1))
    K = to_numpy(K)
    if len(K.shape) < 2:
        K = np.reshape(K, (1, -1))

    if is_call(flag):
        iv_flag = 'c'
    elif is_put(flag):
        iv_flag = 'p'
    else:
        raise ValueError("Wrong `flag` value was given.")
    return iv_lets_be_rational(option_price, F, K, r, T, iv_flag)


def black_vanilla_price(
    sigma: Union[float, NDArray[float_]],
    T: Union[float, NDArray[float_]],
    K: Union[float, NDArray[float_]],
    F: Union[float, NDArray[float_]],
    r: Union[float, NDArray[float_]],
    flag: str,
) -> Union[float, NDArray[float_]]:
    """
    Calculates prices in the Black-76 model given the volatility and parameters.

    Args:
        sigma: black-76 volatility.
        T: times to maturity.
        K: strikes.
        F: forward prices at t = 0.
        r: the risk-free interest rate.
        flag: 'c' for calls, 'p' for puts.

    Returns:
        Option price or an array of option prices corresponding to the given parameters.
    """
    d1 = (np.log(F / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    T = np.reshape(T, (-1, 1))
    K = to_numpy(K)
    if len(K.shape) < 2:
        K = np.reshape(K, (1, -1))
    if is_call(flag):
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif is_put(flag):
        price = np.exp(-r * T) * (-F * norm.cdf(-d1) + K * norm.cdf(-d2))
    else:
        raise ValueError("Incorrect flag value was given.")
    return price.squeeze()


def get_log_contract_price(
    ttm: NDArray[float_],
    smile_func: Callable,
    F: float = 1
) -> NDArray[float_]:
    """
    Calculates the price of the European log-contract -2 log(F_T) for T = ttm.

    :param ttm: time to maturity.
    :param smile_func: IV smile as a callable function of log-moneyness.
    :param F: underlying price at t = 0.
    :return: the log-contract price implied from the smile by the Carr's formula.
    """
    K_puts = np.linspace(F / 10, F, 5000)
    puts = black_vanilla_price(sigma=smile_func(np.log(K_puts / F)), T=ttm, K=K_puts, F=F, flag='p', r=0)

    K_calls = np.linspace(F, 10 * F, 5000)
    calls = black_vanilla_price(sigma=smile_func(np.log(K_calls / F)), T=ttm, K=K_calls, F=F, flag='c', r=0)

    return 2 * np.trapz(y=puts / K_puts ** 2, x=K_puts) + 2 * np.trapz(y=calls / K_calls ** 2, x=K_calls)
