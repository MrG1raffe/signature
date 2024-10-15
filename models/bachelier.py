import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_
from typing import Union
from scipy.stats import norm
from dataclasses import dataclass

from models.analytic_model import AnalyticModel
from models.characteristic_function_model import CharacteristicFunctionModel
from simulation.diffusion import Diffusion
from utility.utility import is_put, is_call, to_numpy
from volatility_surface.volatility_surface import black_iv


@dataclass
class Bachelier(AnalyticModel, CharacteristicFunctionModel):
    sigma: float

    def __init__(
            self,
            sigma : float) -> None:
        self.sigma = sigma
        self.model_type = "normal"

    def get_vanilla_option_price_analytic(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F0: float,
        flag: str = "call",
        is_vol_surface: bool = False
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates analytically the prices of the European vanilla with the Bachelier formula.

        :param T: option maturities, a number or a 1-dimensional array.
        :param K: options strikes. Either a number, or a 1D array of strikes, or a 2D array of shape
            (len(T), len(strikes)) containing in the i-th raw the strikes corresponding to maturity T[i].
        :param F0: initial value of the underlying price.
        :param flag: determines the option type: "c" or "call" for calls, "p" or "put" for puts.
        :param is_vol_surface: whether to return the Black implied volatility value instead of option prices.
        :return: an array of shape (T.size, K.shape[-1]) with the option prices or implied vols.
        """
        T = np.reshape(T, (-1, 1))
        K = to_numpy(K)
        if len(K.shape) < 2:
            K = np.reshape(K, (1, -1))
        d1 = (F0 - K) / (self.sigma * np.sqrt(T))
        if is_call(flag):
            prices = (F0 - K) * norm.cdf(d1) + self.sigma * np.sqrt(T) * norm.pdf(d1)
        elif is_put(flag):
            prices = (K - F0) * norm.cdf(-d1) + self.sigma * np.sqrt(T) * norm.pdf(d1)
        else:
            raise ValueError("Incorrect flag value was given.")
        if is_vol_surface:
            prices = black_iv(option_price=prices, T=T, K=K,
                              F=F0, r=0, flag=flag)
        return prices.squeeze()

    def get_price_trajectory(
        self,
        t_grid: NDArray[float_],
        size: int,
        F0: Union[float, NDArray[float_]],
        rng: np.random.Generator = None,
        *args,
        **kwargs
    ) -> NDArray[float_]:
        """
        Simulates the trajectories of price in the Bachelier model on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price.
        :param rng: random number generator to simulate the trajectories with.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories
        """
        diffusion = Diffusion(t_grid=to_numpy(t_grid).reshape((-1,)), dim=1, size=size, rng=rng)
        F_traj = diffusion.brownian_motion(init_val=F0, vol=self.sigma, squeeze=True)
        return F_traj

    def quadratic_variation(
            self,
            T: float,
    ) -> float:
        """
        Computes the quadratic variation <F>_T at t = T.

        :param T: the date the quadratic variation to be calculated on.
        :return: the value of numerical approximation of <X>_T.
        """
        return self.sigma**2 * T

    def _char_func(
        self,
        T: float,
        x: float,
        u1: Union[complex, NDArray[complex_]],
        u2: Union[complex, NDArray[complex_]] = 0,
        f1: Union[complex, NDArray[complex_]] = 0,
        f2: Union[complex, NDArray[complex_]] = 0,
        **kwargs
    ) -> complex_:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * F_T + i * u2 * V_T + i * f1 * ∫ F_s ds + i * f2 * ∫ V_s ds}]     (1)

        for the Bachelier model, where V_t = σ.

        :param u1: F_T coefficient in the characteristic function, see (1).
        :param u2: V_T coefficient in the characteristic function, see (1).
        :param f1: ∫ F_s ds coefficient in the characteristic function, see (1).
        :param f2: ∫ V_s ds coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: F_0, initial underlying price.
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        # TODO: verify the joint CF for Asian options.
        return np.exp(1j * u1 * x - 0.5 * self.sigma**2 * T * (u1**2 + (f1 * T)**2 / 3 + u1 * f1 * T) +
                      1j * u2 + 1j * f2 * T)
